import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
import copy
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from models.base import BaseLearner
from utils.inc_net import AdaptiveNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy

num_workers=8
EPSILON = 1e-8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._old_base = None
        self.network1 = AdaptiveNet(args, args['pretrained'])
        logging.info(f'>>> train generalized blocks:{self.args["train_base"]} train_adaptive:{self.args["train_adaptive"]}')

    def after_task(self):
        self._known_classes = self._total_classes
        if self._cur_task == 0:
            if self.args['train_base']:
                logging.info("Train Generalized Blocks...")
                self.network1.TaskAgnosticExtractor.train()
                for param in self.network1.TaskAgnosticExtractor.parameters():
                    param.requires_grad = True
            else:
                logging.info("Fix Generalized Blocks...")
                self.network1.TaskAgnosticExtractor.eval()
                for param in self.network1.TaskAgnosticExtractor.parameters():
                    param.requires_grad = False
        
        logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.network1.update_fc(self._total_classes)

        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        if self._cur_task>0:
            for i in range(self._cur_task):
                for p in self.network1.AdaptiveExtractors[i].parameters():
                    if self.args['train_adaptive']:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

        logging.info('All params: {}'.format(count_parameters(self.network1)))
        logging.info('Trainable params: {}'.format(count_parameters(self.network1, True)))
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source='train',
            mode='train', 
            appendent=self._get_memory()
        )
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args["batch_size"], 
            shuffle=True, 
            num_workers=num_workers
        )
        
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), 
            source='test', 
            mode='test'
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.args["batch_size"],
            shuffle=False, 
            num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self.network1 = nn.DataParallel(self.network1, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self.network1 = self.network1.module
    
    def set_network(self):
        if len(self._multiple_gpus) > 1:
            self.network1 = self.network1.module
        self.network1.train()                   #All status from eval to train
        if self.args['train_base']:
            self.network1.TaskAgnosticExtractor.train()
        else:
            self.network1.TaskAgnosticExtractor.eval()
        
        # set adaptive extractor's status
        self.network1.AdaptiveExtractors[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                if self.args['train_adaptive']:
                    self.network1.AdaptiveExtractors[i].train()
                else:
                    self.network1.AdaptiveExtractors[i].eval()
        if len(self._multiple_gpus) > 1:
            self.network1 = nn.DataParallel(self.network1, self._multiple_gpus)
            
    def _train(self, train_loader, test_loader):
        self.network1.to(self._device)
        if self._cur_task==0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.network1.parameters()),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"]
            )
            if self.args['scheduler'] == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer, 
                    milestones=self.args['init_milestones'], 
                    gamma=self.args['init_lr_decay']
                )
            elif self.args['scheduler'] == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['init_epoch']
                ) 
            else:
                raise NotImplementedError
            
            if not self.args['skip']:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
            else:
                if isinstance(self.network1, nn.DataParallel):
                    self.network1 = self.network1.module
                load_acc = self.network1.load_checkpoint(self.args)
                self.network1.to(self._device)

                if len(self._multiple_gpus) > 1:
                    self.network1 = nn.DataParallel(self.network1, self._multiple_gpus)
                
                cur_test_acc = self._compute_accuracy(self.network1, self.test_loader)
                logging.info(f"Loaded_Test_Acc:{load_acc} Cur_Test_Acc:{cur_test_acc}")
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self.network1.parameters()),
                lr=self.args['lrate'], 
                momentum=0.9, 
                weight_decay=self.args['weight_decay']
            )
            if self.args['scheduler'] == 'steplr':
                scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=self.args['milestones'], 
                    gamma=self.args['lrate_decay']
                )
            elif self.args['scheduler'] == 'cosine':
                assert self.args['t_max'] is not None
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.args['t_max']
                )
            else:
                raise NotImplementedError
            self._update_representation(train_loader, test_loader, optimizer, scheduler)
            # if len(self._multiple_gpus) > 1:
            #     self._network.module.weight_align(self._total_classes-self._known_classes)
            # else:
            #     self._network.weight_align(self._total_classes-self._known_classes)

            
    def _init_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        scaler = GradScaler()
        for _, epoch in enumerate(prog_bar):
            self.network1.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets, _) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.long().to(self._device)
                with torch.cuda.amp.autocast():
                    logits = self.network1(inputs)['logits']
                    loss=F.cross_entropy(logits,targets)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self.network1, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args['init_epoch'], losses/len(train_loader), train_acc)
            prog_bar.set_description(info)
        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        scaler = GradScaler()
        for _, epoch in enumerate(prog_bar):
            self.set_network()
            losses = 0.
            losses_clf=0.
            losses_aux=0.
            correct, total = 0, 0
            for i, (_, inputs, targets, _) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.long().to(self._device)
                with torch.cuda.amp.autocast():
                    outputs= self.network1(inputs)
                    logits,aux_logits=outputs["logits"],outputs["aux_logits"]
                    loss_clf=F.cross_entropy(logits,targets)
                    aux_targets = targets.clone()
                    aux_targets=torch.where(aux_targets-self._known_classes+1>0,  aux_targets-self._known_classes+1,0)
                    loss_aux=F.cross_entropy(aux_logits,aux_targets)
                    loss=loss_clf+self.args['alpha_aux']*loss_aux

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses += loss.item()
                losses_aux+=loss_aux.item()
                losses_clf+=loss_clf.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            if epoch%5==0:
                test_acc = self._compute_accuracy(self.network1, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux  {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader),losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc, test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["epochs"], losses/len(train_loader), losses_clf/len(train_loader),losses_aux/len(train_loader),train_acc)
            prog_bar.set_description(info)
        logging.info(info)
    
    def save_checkpoint(self, test_acc):
        assert self.args['model_name'] == 'finetune'
        checkpoint_name = f"checkpoints/finetune_{self.args['csv_name']}"
        _checkpoint_cpu = copy.deepcopy(self.network1)
        if isinstance(_checkpoint_cpu, nn.DataParallel):
            _checkpoint_cpu = _checkpoint_cpu.module
        _checkpoint_cpu.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "backbone": _checkpoint_cpu.backbone.state_dict(),
            "fc":_checkpoint_cpu.fc.state_dict(),
            "test_acc": test_acc
        }
        torch.save(save_dict, "{}_{}.pkl".format(checkpoint_name, self._cur_task))
    
    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, names, idx_dataset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                idx_loader = DataLoader(
                    idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4
                )
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)

                # Select
                selected_exemplars = []
                exemplars_names = []
                exemplar_vectors = []  # [n, feature_dim]
                for k in range(1, m + 1):
                    S = np.sum(
                        exemplar_vectors, axis=0
                    )  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                    selected_exemplars.append(
                        np.array(data[i])
                    )  # New object to avoid passing by inference
                    exemplar_vectors.append(
                        np.array(vectors[i])
                    )  # New object to avoid passing by inference

                    vectors = np.delete(
                        vectors, i, axis=0
                    )  # Remove it to avoid duplicative selection
                    data = np.delete(
                        data, i, axis=0
                    )  # Remove it to avoid duplicative selection

                    if len(vectors) == 0:
                        break
                # uniques = np.unique(selected_exemplars, axis=0)
                # print('Unique elements: {}'.format(len(uniques)))
                selected_exemplars = np.array(selected_exemplars)
                # exemplar_targets = np.full(m, class_idx)
                exemplar_targets = np.full(selected_exemplars.shape[0], class_idx)
                exemplars_names = np.full(m, names[0])
                self._data_memory = (
                    np.concatenate((self._data_memory, selected_exemplars))
                    if len(self._data_memory) != 0
                    else selected_exemplars
                )
                self._targets_memory = (
                    np.concatenate((self._targets_memory, exemplar_targets))
                    if len(self._targets_memory) != 0
                    else exemplar_targets
                )
                self._names_memory = (
                    np.concatenate((self._names_memory, exemplars_names))
                    if len(self._names_memory) != 0
                    else exemplars_names
                )
                # Exemplar mean
                idx_dataset = data_manager.get_dataset(
                    [],
                    source="train",
                    mode="test",
                    appendent=(selected_exemplars, exemplar_targets, exemplars_names),
                )
                idx_loader = DataLoader(
                    idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4
                )
                vectors, _ = self._extract_vectors(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean