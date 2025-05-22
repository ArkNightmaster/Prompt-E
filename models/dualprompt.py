import logging
import os.path

import numpy as np
import torch
import wandb
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from utils.inc_net import PromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, order_num2label, print_file, count_parameters, show_images

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 10

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self.network1 = PromptVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.grad_clip = args["grad_clip"]

        # Freeze the parameters for ViT.
        if self.args["freeze"]:

            # freeze original_backbone
            for p in self.network1.original_backbone.parameters():
                p.requires_grad = False
            # freeze args.freeze[blocks, patch_embed, cls_token, norm, pos_embed] parameters
            for n, p in self.network1.backbone.named_parameters():
                if n.startswith(tuple(self.args["freeze"])):
                    p.requires_grad = False

        self.global_step = 0  

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._all_task = data_manager.nb_tasks
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        print(f"Learning on {self._known_classes}-{self._total_classes}")

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        
        # 加载验证集
        val_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="val",
                                              mode="val")
        self.val_dataset = val_dataset
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        total_params = round(count_parameters(self.network1, trainable=False, component=None), 2)
        trainable_params = round(count_parameters(self.network1, trainable=True, component=None), 2)
        trainable_prompt_params = round(count_parameters(self.network1, trainable=True, component='prompt'), 2)
     
        print(f"Total Parameters: {total_params}M, Trainable Parameters: {trainable_params}M, Trainable Prompt Parameters: {trainable_prompt_params}M")

        # Log parameters in table format
        wandb.log({
            f"Param/Parameters Table": wandb.Table(
                columns=["Parameter Type", "Parameter Count"],
                data=[
                    ["Total Parameters", total_params],
                    ["Trainable Parameters", trainable_params],
                    ["Trainable Prompt Parameters", trainable_prompt_params],
                ]
            )
        })

        self._train(self.train_loader, self.val_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self.network1 = self.network1.module

    def _train(self, train_loader, val_loader, test_loader):
        self.network1.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        if self._cur_task > 0:
            self._init_prompt(optimizer)

        if self._cur_task > 0 and self.args["reinit_optimizer"]:
            optimizer = self.get_optimizer()

        self._init_train(train_loader, val_loader, test_loader, optimizer, scheduler)

    def get_optimizer(self):

        optimizer_parameters = [
            {'params': [p for n, p in self.network1.named_parameters() if p.requires_grad],
             'lr': self.init_lr,
             'weight_decay': self.weight_decay}
        ]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.network1.parameters()),
                momentum=0.9,
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(optimizer_parameters)

        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(optimizer_parameters)

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['epochs'],
                                                             eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"],
                                                       gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_prompt(self, optimizer):
        args = self.args
        model = self.network1.backbone
        task_id = self._cur_task

        # Transfer previous learned prompt params to the new prompt
        if args["prompt_pool"] and args["shared_prompt_pool"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args[
                    "use_prefix_tune_for_e_prompt"] else (slice(None), slice(cur_start, cur_end))
                prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args[
                    "use_prefix_tune_for_e_prompt"] else (slice(None), slice(prev_start, prev_end))

                with torch.no_grad():
                    model.e_prompt.prompt.grad.zero_()
                    model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()

        # Transfer previous learned prompt param keys to the new prompt
        if args["prompt_pool"] and args["shared_prompt_key"]:
            prev_start = (task_id - 1) * args["top_k"]
            prev_end = task_id * args["top_k"]

            cur_start = prev_end
            cur_end = (task_id + 1) * args["top_k"]

            if (prev_end > args["size"]) or (cur_end > args["size"]):
                pass
            else:
                cur_idx = (slice(cur_start, cur_end))
                prev_idx = (slice(prev_start, prev_end))

            with torch.no_grad():
                model.e_prompt.prompt_key.grad.zero_()
                model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                optimizer.param_groups[0]['params'] = model.parameters()

    def _init_train(self, train_loader, val_loader, test_loader, optimizer, scheduler):
        scaler = torch.cuda.amp.GradScaler()
        prog_bar = tqdm(range(self.args['epochs']))
        
        # 记录最佳验证准确率和对应的epoch
        best_val_acc = 0.0
        best_epoch = 0
        
        for _, epoch in enumerate(prog_bar):
            self.network1.backbone.train()
            self.network1.original_backbone.eval()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, names) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.long().to(self._device)
                with torch.cuda.amp.autocast():
                    output = self.network1(inputs, task_id=self._cur_task, train=True)
                    logits = output["logits"][:, :self._total_classes]
                    logits[:, :self._known_classes] = float('-inf')

                    loss = F.cross_entropy(logits, targets.long())
                    if self.args["pull_constraint"] and 'reduce_sim' in output:
                        loss = loss - self.args["pull_constraint_coeff"] * output['reduce_sim']

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.network1.parameters(), max_norm=self.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            self.global_step += 1  

            val_acc = self._compute_accuracy(self.network1, val_loader)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                # torch.save(self.network1.state_dict(), f"best_model_task_{self._cur_task}.pth")

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self.network1, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Val_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['epochs'],
                    losses / len(train_loader),
                    train_acc,
                    val_acc,
                    test_acc,
                )
                wandb.log({
                    "Loss/Train Loss": losses / len(train_loader),
                    "Accuracy/Train Accuracy": train_acc,
                    "Accuracy/Validation Accuracy": val_acc,
                    "Accuracy/Test Accuracy": test_acc,
                    "Task": self._cur_task,
                    "Epoch": epoch + 1
                }, step=self.global_step)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Val_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['epochs'],
                    losses / len(train_loader),
                    train_acc,
                    val_acc,
                )
                wandb.log({
                    "Loss/Train Loss": losses / len(train_loader),
                    "Accuracy/Train Accuracy": train_acc,
                    "Accuracy/Validation Accuracy": val_acc,
                    "Task": self._cur_task,
                    "Epoch": epoch + 1
                }, step=self.global_step)
            prog_bar.set_description(info)
            
        print(f"Task {self._cur_task} - Best Validation Accuracy: {best_val_acc:.2f} at Epoch {best_epoch}")
        wandb.log({
            "Best/Validation Accuracy": best_val_acc,
            "Best/Epoch": best_epoch,
            "Task": self._cur_task
        }, step=self.global_step)

        if self._cur_task == self._all_task - 1 and self.args['if_tsne']:
            self.umap(showcenters=False, Normalize=False)
        print(info)

    def _eval_cnn(self, loader):
        self.network1.eval()
        y_pred, y_true = [], []
        for i, (_, inputs, targets, names) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self.network1(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets, names) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs, task_id=self._cur_task)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def umap(self, showcenters=False, Normalize=False):
        import umap
        import matplotlib.pyplot as plt
        print('now draw tsne results of extracted features.')
        tot_classes = self._total_classes
        test_dataset = self.data_manager.get_dataset(np.arange(tot_classes, step=(tot_classes / self._all_task) - 2),
                                                     source='test', mode='test')
        valloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        vectors, y_true = self._extract_vectors(valloader)

        if Normalize:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        embedding = umap.UMAP(n_neighbors=5,
                              min_dist=0.3,
                              metric='correlation').fit_transform(vectors)
        np.save('./fig/numpy_array/{}_{}_tsne.npy'.format(self.args['model_name'], self.args['dataset']), embedding)
        # scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_true, s=20, cmap=plt.cm.get_cmap("tab20"))
        # plt.legend(*scatter.legend_elements())
        # plt.savefig(os.path.join('./fig/tsne', str(self.args['model_name']) + '_' + str(self.args['dataset']) + '_tsne.jpg'), dpi=300)
        # plt.close()

    def _extract_vectors(self, loader):
        self.network1.backbone.eval()
        self.network1.original_backbone.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets, _ in loader:
                _targets = _targets.numpy()
                _vectors = tensor2numpy(
                    self.network1.extract_vector(_inputs.to(self._device))
                )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
        
    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        
        val_y_pred, val_y_true = self._eval_cnn(self.val_loader)
        val_cnn_accy = self._evaluate(val_y_pred, val_y_true)
        
        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
            val_y_pred, val_y_true = self._eval_nme(self.val_loader, self._class_means)
            val_nme_accy = self._evaluate(val_y_pred, val_y_true)
        else:
            nme_accy = None
            val_nme_accy = None
            
        return cnn_accy, nme_accy, val_cnn_accy, val_nme_accy
