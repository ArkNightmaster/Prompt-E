#!/usr/bin/env Python
# coding=utf-8
import os
import numpy as np
import torch
import wandb
import random
from typing import List
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import clip4l2p, CLIP_Encoder, Bert_Encoder, T5_Encoder, RoBERTa_Encoder
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, order_num2label, count_parameters

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 10
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.network1 = clip4l2p(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.grad_clip = args["grad_clip"] if args["grad_clip"] is not None else 1.0
        self.args = args
        self.alpha = args["alpha"]
        self.beta = args["beta"]
        self.text_encoder = args["text_encoder"]
        self.train_cur_classes = []
        self.train_known_classes = []
        self.task_features = []
        self.class_features = []
        self.class_mask = []  

        init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
        self.logfilename = "logs/{}/{}/{}/{}/{}_{}_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["alpha"],
            args["beta"],
            args["batch_size"],
            args["backbone_type"],
        )

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

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        
        # Load validation set
        val_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), 
                                               source="val", mode="val")
        self.val_dataset = val_dataset
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, 
                                    num_workers=num_workers, pin_memory=True)
        
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        # Store current task class mask
        self.class_mask.append(list(range(self._known_classes, self._total_classes)))
        
        # ————————————————————————————————————
        # get current task train-class labels
        # ————————————————————————————————————
        self.train_map = order_num2label(train_dataset.labels, train_dataset.names)  # Get pairs {number: label}
        self.train_cur_classes = [self.train_map[i + self._known_classes] for i in range(len(self.train_map))]  # Get label order list
        self.train_known_classes.append(self.train_cur_classes) # [[task1], [task2], [task3], ...]


        # embedding The Text
        tasks_prompts = [self.task_classes_prompts(task) for task in self.train_known_classes]
        classes_prompts = [item for sublist in self.train_known_classes for item in sublist]
        # ————————————————————————————————————
        # Embedding Tasks Text
        # ————————————————————————————————————
        for prompts in tasks_prompts:
            if self.text_encoder == "bert":
                self.network2 = Bert_Encoder(self.args, self._device)
                feature = self.network2.encode_text(prompts).to(self._device)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                self.task_features.append(feature.detach())
            elif self.text_encoder == "t5":
                self.network2 = T5_Encoder(self.args, self._device)
                feature = self.network2.encode_text(prompts).to(self._device)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                self.task_features.append(feature.detach())
            elif self.text_encoder == "roberta":
                self.network2 = RoBERTa_Encoder(self.args, self._device)
                feature = self.network2.encode_text(prompts).to(self._device)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                self.task_features.append(feature.detach())
            elif self.text_encoder == "clip":
                self.network2 = CLIP_Encoder(self.args, self._device, classnames=None)
                feature = self.network2.encode_text(prompts).to(self._device)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                self.task_features.append(feature.detach())
            else:
                raise ValueError(f"Unknown text encoder: {self.text_encoder}")
        
        # ————————————————————————————————————
        # Embedding Classes Text
        # ————————————————————————————————————
        if self.text_encoder == "bert":
            self.network2 = Bert_Encoder(self.args, self._device)
            feature = self.network2.Bert_textweight(classes_prompts).to(self._device)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            self.class_features.append(feature.detach())
        elif self.text_encoder == "t5":
            self.network2 = T5_Encoder(self.args, self._device)
            feature = self.network2.T5_textweight(classes_prompts).to(self._device)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            self.class_features.append(feature.detach())
        elif self.text_encoder == "roberta":
            self.network2 = RoBERTa_Encoder(self.args, self._device)
            feature = self.network2.RoBERTa_textweight(classes_prompts).to(self._device)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            self.class_features.append(feature.detach())
        elif self.text_encoder == "clip":
            self.network2 = CLIP_Encoder(self.args, self._device, classnames=None)
            feature = self.network2.clip_textweight(classes_prompts).to(self._device)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            self.class_features.append(feature.detach())
        else:
            raise ValueError(f"Unknown text encoder: {self.text_encoder}")
        # self.task_features = [feature_1, feature_2, feature_3, ..., feature_n] (Known Task)

        # Freeze parameters
        for name, param in self.network2.named_parameters():
            param.requires_grad_(False)

        total_params = round(count_parameters(self.network1) + count_parameters(self.network2), 2)
        vit_params = round(count_parameters(self.network1), 2)
        text_encoder_params = round(count_parameters(self.network2), 2)
        trainable_params = round(count_parameters(self.network1, True) + count_parameters(self.network2, True), 2)
        trainable_vit_params = round(count_parameters(self.network1, True), 2)
        trainable_text_encoder_params = round(count_parameters(self.network2, True), 2)

        print(f"Total Parameters: {total_params}M, ViT Parameters: {vit_params}M, {self.text_encoder} Text Encoder Parameters: {text_encoder_params}M")
        print(f"Trainable Parameters: {trainable_params}M, ViT Parameters: {trainable_vit_params}M, {self.text_encoder} Text Encoder Parameters: {trainable_text_encoder_params}M")

        # Log parameters in table format
        wandb.log({
            f"Param/{self.text_encoder} Parameters Table": wandb.Table(
                columns=["Parameter Type", "Parameter Count"],
                data=[
                    ["Total Parameters", total_params],
                    ["ViT Parameters", vit_params],
                    [f"{self.text_encoder} Text Encoder Parameters", text_encoder_params],
                    ["Trainable Parameters", trainable_params],
                    ["Trainable ViT Parameters", trainable_vit_params],
                    [f"Trainable {self.text_encoder} Text Encoder Parameters", trainable_text_encoder_params]
                ]
            )
        })

        self._train(self.train_loader, self.val_loader, self.test_loader)

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
             'lr': self.args['init_lr'],
             'weight_decay': self.args['weight_decay']}
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

        if 'dual' in args['backbone_type']:
            # Transfer previous learned prompt params to the new prompt
            if args["prompt_pool"] and args["shared_prompt_pool"]:
                prev_start = (task_id - 1) * args["top_k"]
                prev_end = task_id * args["top_k"]

                cur_start = prev_end
                cur_end = (task_id + 1) * args["top_k"]

                if (prev_end > args["size"]) or (cur_end > args["size"]):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args["use_prefix_tune_for_e_prompt"] else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args["use_prefix_tune_for_e_prompt"] else (slice(None), slice(prev_start, prev_end))

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
        elif 'l2p' in args['backbone_type']:
            # Transfer previous learned prompt params to the new prompt
            if args["prompt_pool"] and args["shared_prompt_pool"]:
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
                        model.prompt.prompt.grad.zero_()
                        model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
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
                    model.prompt.prompt_key.grad.zero_()
                    model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                    optimizer.param_groups[0]['params'] = model.parameters()

    def _init_train(self, train_loader, val_loader, test_loader, optimizer, scheduler):
        scaler = torch.cuda.amp.GradScaler()
        prog_bar = tqdm(range(self.args['epochs']))
        
        # 记录最佳验证准确率和对应的epoch
        best_val_acc = 0.0
        best_epoch = 0
        
        for _, epoch in enumerate(prog_bar):
            self.set_training(mode=True)
            
            alpha = self.alpha
            beta = self.beta
            losses = 0.0
            losses_origin = 0.0
            losses_task = 0.0
            losses_class = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, names) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.long().to(self._device)
                
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    output = self.network1(inputs, task_id=self._cur_task, train=True)
                    logits = output["logits"][:, :self._total_classes]
                    
                    if self._cur_task == 0:
                        # 对于第一个任务，只计算原始交叉熵损失
                        loss_total = F.cross_entropy(logits, targets)
                    else:
                        # 对于后续任务，计算综合损失
                        # 原始交叉熵损失
                        logits[:, :self._known_classes] = float('-inf')
                        loss_origin = F.cross_entropy(logits, targets)
                        
                        # 任务级别损失
                        keys = output['selected_key']
                        task_features_pos = self.task_features[self._cur_task].expand(inputs.size(0), -1)
                        task_features_neg = self.get_negative_task_features(inputs.size(0))
                        loss_task = self.task_level_loss(keys, task_features_pos, task_features_neg)
                        
                        # 类别级别损失
                        visual_features = output["visual_features"]
                        class_features_pos = self.class_features[self._cur_task][targets]
                        class_features_neg = self.get_negative_class_features(inputs.size(0))
                        loss_class = self.class_level_loss(visual_features, class_features_pos, class_features_neg)
                        
                        # 综合总损失
                        loss_total = loss_origin + alpha * loss_task + beta * loss_class

                # 反向传播
                scaler.scale(loss_total).backward()
                torch.nn.utils.clip_grad_norm_(self.network1.parameters(), self.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                
                losses += loss_total.item()
                if self._cur_task > 0:
                    losses_origin += loss_origin.item()
                    losses_task += loss_task.item()
                    losses_class += loss_class.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            self.global_step += 1  # 每个 epoch 结束时递增全局步数
            
            # 评估验证集性能
            val_acc = self._compute_accuracy(self.network1, val_loader)
            
            # 记录最佳验证准确率
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                # 可以在这里保存最佳模型
                # torch.save(self.network1.state_dict(), f"best_model_task_{self._cur_task}.pth")

            # 日志记录
            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self.network1, test_loader)
                info = f"Task {self._cur_task}, Epoch {epoch + 1}/{self.args['epochs']} => Loss {losses / len(train_loader):.3f}, Loss_Origin {losses_origin / len(train_loader):.3f}, Loss_Task {losses_task / len(train_loader):.3f}, Loss_Class {losses_class / len(train_loader):.3f}, Train_accy {train_acc:.2f}, Val_accy {val_acc:.2f}, Test_accy {test_acc:.2f}"
                wandb.log({
                    "Loss/Train Loss": losses / len(train_loader),
                    "Loss/Train Loss Origin": losses_origin / len(train_loader),
                    "Loss/Loss Task": losses_task / len(train_loader),
                    "Loss/Loss Class": losses_class / len(train_loader),
                    "Accuracy/Train Accuracy": train_acc,
                    "Accuracy/Validation Accuracy": val_acc,
                    "Accuracy/Test Accuracy": test_acc,
                    "Task": self._cur_task,
                    "Epoch": epoch + 1
                }, step=self.global_step)
            else:
                info = f"Task {self._cur_task}, Epoch {epoch + 1}/{self.args['epochs']} => Loss {losses / len(train_loader):.3f}, Loss_Origin {losses_origin / len(train_loader):.3f}, Loss_Task {losses_task / len(train_loader):.3f}, Loss_Class {losses_class / len(train_loader):.3f}, Train_accy {train_acc:.2f}, Val_accy {val_acc:.2f}"
                wandb.log({
                    "Loss/Train Loss": losses / len(train_loader),
                    "Loss/Train Loss Origin": losses_origin / len(train_loader),
                    "Loss/Loss Task": losses_task / len(train_loader),
                    "Loss/Loss Class": losses_class / len(train_loader),
                    "Accuracy/Train Accuracy": train_acc,
                    "Accuracy/Validation Accuracy": val_acc,
                    "Task": self._cur_task,
                    "Epoch": epoch + 1
                }, step=self.global_step)
            prog_bar.set_description(info)
        
        # 在训练结束后输出最佳验证准确率信息
        print(f"Task {self._cur_task} - Best Validation Accuracy: {best_val_acc:.2f} at Epoch {best_epoch}")
        wandb.log({
            "Best/Validation Accuracy": best_val_acc,
            "Best/Epoch": best_epoch,
            "Task": self._cur_task
        }, step=self.global_step)

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
        print('now draw umap results of extracted features.')
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
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_true, s=20, cmap=plt.cm.get_cmap("tab20"))
        plt.legend(*scatter.legend_elements())
        plt.savefig(os.path.join('./fig/tsne', str(self.args['model_name']) + '_' + str(self.args['dataset']) + '_tsne.jpg'), dpi=300)
        plt.close()

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

    def set_training(self, mode=True):
        self.network1.backbone.train(True)
        self.network1.original_backbone.train(False)

    def task_classes_prompts(self, classnames: List[str]):
        # 准备模板
        template = 'a photo of {}.'
        classes_str = ' or '.join([c.replace('_', ' ') for c in classnames])
        prompts = [template.format(classes_str)]
        return prompts

    def task_level_loss(self, keys, task_features_pos, task_features_neg):
        """
        任务级别三元组损失
        keys: [batch_size, num_prompts, embed_dim]
        task_features_pos: [batch_size, embed_dim]
        task_features_neg: [batch_size, embed_dim]
        """
        # 扩展 task_features_pos 和 task_features_neg 的维度以匹配 keys
        task_features_pos = task_features_pos.unsqueeze(1)  # [batch_size, 1, embed_dim]
        task_features_neg = task_features_neg.unsqueeze(1)  # [batch_size, 1, embed_dim]

        similarity_pos = F.cosine_similarity(keys, task_features_pos, dim=2)  # [batch_size, num_prompts]
        similarity_neg = F.cosine_similarity(keys, task_features_neg, dim=2)  # [batch_size, num_prompts]

        # 计算每个提示的损失并取平均
        loss = (1 - similarity_pos + similarity_neg).mean()
        return loss

    def class_level_loss(self, visual_features, class_features_pos, class_features_neg):
        """
        类别级别三元组损失
        visual_features: [batch_size, embed_dim]
        class_features_pos: [batch_size, embed_dim]
        class_features_neg: [batch_size, embed_dim]
        """
        similarity_pos = F.cosine_similarity(visual_features, class_features_pos, dim=1)  # [batch_size]
        similarity_neg = F.cosine_similarity(visual_features, class_features_neg, dim=1)  # [batch_size]

        loss = (1 - similarity_pos + similarity_neg).mean()
        return loss

    def get_negative_task_features(self, batch_size: int):
        """
        从之前的任务中随机选择语言特征作为负样本
        """
        # 对于第一个任务，没有先前任务作为负样本
        if self._cur_task == 0:
            # 返回当前任务的特征（实际上在第一个任务不会用到这个负样本）
            return self.task_features[0].expand(batch_size, -1)
        
        # 为每个样本随机选择一个来自先前任务的任务特征
        selected_features = []
        for _ in range(batch_size):
            # 随机选择一个先前的任务索引（不包括当前任务）
            neg_task_idx = random.randint(0, self._cur_task - 1)
            selected_features.append(self.task_features[neg_task_idx])
        
        # 将特征堆叠成一个批次
        return torch.cat(selected_features, dim=0)

    def get_negative_class_features(self, batch_size: int):
        """
        从之前的类别中随机选择语言特征作为负样本，确保选择的样本不来自当前任务类别
        """
        # 对于第一个任务，由于没有之前的类别，返回随机的当前任务类别特征（不影响损失计算）
        if self._cur_task == 0:
            random_indices = torch.randint(0, self.args['increment'], (batch_size,)).to(self._device)
            return self.class_features[0][random_indices]
        
        # 收集所有先前任务的类别
        all_task_classes = []
        for task_idx in range(self._cur_task):
            all_task_classes.extend(self.class_mask[task_idx])
        
        # 为每个样本选择一个负样本类别
        selected_features = []
        current_task_classes = set(self.class_mask[self._cur_task])
        
        for b in range(batch_size):
            # 继续选择直到找到一个不在当前任务类别中的类别
            while True:
                neg_class_idx = random.choice(all_task_classes)
                if neg_class_idx not in current_task_classes:
                    break
            
            # 确定该类别所属的任务和在该任务中的索引
            for task_idx in range(self._cur_task):
                if neg_class_idx in self.class_mask[task_idx]:
                    # 在对应任务的类别特征中找到该类别的特征
                    class_within_task_idx = list(self.class_mask[task_idx]).index(neg_class_idx)
                    selected_features.append(self.class_features[task_idx][class_within_task_idx])
                    break
        
        return torch.stack(selected_features, dim=0)

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