import logging
import os
import sys
import wandb
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.optim import Optimizer
import math
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CodaPromptVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, count_parameters, print_file

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self.network1 = CodaPromptVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.grad_clip = args["grad_clip"] if args["grad_clip"] is not None else 1.0
        self.args = args
        self.global_step = 0
    
        total_params = round(count_parameters(self.network1), 2)
        vit_params = round(count_parameters(self.network1), 2)
        trainable_params = round(count_parameters(self.network1, True), 2)
        trainable_vit_params = round(count_parameters(self.network1, True), 2)
    
        print(f"Total Parameters: {total_params}M, ViT Parameters: {vit_params}M")
        print(f"Trainable Parameters: {trainable_params}M, ViT Parameters: {trainable_vit_params}M")

        # Log parameters in table format
        wandb.log({
            f"Param/Parameters Table": wandb.Table(
                columns=["Parameter Type", "Parameter Count"],
                data=[
                    ["Total Parameters", total_params],
                    ["ViT Parameters", vit_params],
                    ["Trainable Parameters", trainable_params],
                    ["Trainable ViT Parameters", trainable_vit_params],
                ]
            )
        })

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._all_task = data_manager.nb_tasks

        if self._cur_task > 0:
            try:
                if self.network1.module.prompt is not None:
                    self.network1.module.prompt.process_task_count()
            except:
                if self.network1.prompt is not None:
                    self.network1.prompt.process_task_count()

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        
        # Load validation set
        val_dataset = data_manager.get_dataset(np.arange(self._known_classes, self.total_classes), source="val", mode="val")
        self.val_dataset = val_dataset
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, 
                                    drop_last=False, num_workers=num_workers)
        
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self.network1 = nn.DataParallel(self.network1, self._multiple_gpus)
        self._train(self.train_loader, self.val_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self.network1 = self.network1.module

    def _train(self, train_loader, val_loader, test_loader):
        self.network1.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        self.data_weighting()
        self._init_train(train_loader, val_loader, test_loader, optimizer, scheduler)

    def data_weighting(self):
        self.dw_k = torch.tensor(np.ones(self._total_classes + 1, dtype=np.float32))
        self.dw_k = self.dw_k.to(self._device)

    def get_optimizer(self):
        if len(self._multiple_gpus) > 1:
            params = list(self.network1.module.prompt.parameters()) + list(self.network1.module.fc.parameters())
        else:
            params = list(self.network1.prompt.parameters()) + list(self.network1.fc.parameters())
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = CosineSchedule(optimizer, K=self.args["epochs"])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, val_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['epochs']))
        scaler = torch.cuda.amp.GradScaler()
        
        # Record best validation accuracy and corresponding epoch
        best_val_acc = 0.0
        best_epoch = 0
        
        for _, epoch in enumerate(prog_bar):
            self.network1.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, names) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.long().to(self._device)
            
                # logits
                with torch.cuda.amp.autocast():
                    logits, prompt_loss = self.network1(inputs, train=True)
                    logits = logits[:, :self._total_classes]
                    logits[:, :self._known_classes] = float('-inf')
                    
                    dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                    loss_supervised = (F.cross_entropy(logits, targets.long()) * dw_cls).mean()

                    # ce loss
                    loss = loss_supervised + prompt_loss.sum()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.network1.parameters(), self.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            
            # Evaluate validation set performance
            val_acc = self._compute_accuracy(self.network1, val_loader)
            
            # Record best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                # Can save best model here
                # torch.save(self.network1.state_dict(), f"best_model_task_{self._cur_task}.pth")
            
            self.global_step += 1  # Increment global step at the end of each epoch

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
                
                # 如果使用wandb，记录日志
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
        
        # 在训练结束后输出最佳验证准确率信息
        logging.info(f"Task {self._cur_task} - Best Validation Accuracy: {best_val_acc:.2f} at Epoch {best_epoch}")
        
        # 如果使用wandb，记录最佳验证准确率
        wandb.log({
            "Best/Validation Accuracy": best_val_acc,
            "Best/Epoch": best_epoch,
            "Task": self._cur_task
        }, step=self.global_step)

        if self._cur_task == self._all_task - 1 and self.args['if_tsne']:
            self.umap(showcenters=False, Normalize=False)

        logging.info(info)

    def _eval_cnn(self, loader):
        self.network1.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets, names) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self.network1(inputs)[:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets, names) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def umap(self, showcenters=False, Normalize=False):
        import umap
        import matplotlib.pyplot as plt
        print('now draw tsne results of extracted features.')
        tot_classes=self._total_classes
        test_dataset = self.data_manager.get_dataset(np.arange(tot_classes, step=(tot_classes/self._all_task)-2), source='test', mode='test')
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
        
        # 评估验证集性能
        val_y_pred, val_y_true = self._eval_cnn(self.val_loader)
        val_accy = self._evaluate(val_y_pred, val_y_true)
        
        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
            y_val_pred, y_val_true = self._eval_nme(self.val_loader, self._class_means)
            val_nme_accy = self._evaluate(y_val_pred, y_val_true)
        else:
            nme_accy = None
            val_nme_accy = None
            
        return cnn_accy, nme_accy, val_accy, val_nme_accy

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K-1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]