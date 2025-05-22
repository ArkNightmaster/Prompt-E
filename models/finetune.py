import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.network1 = IncrementalNet(args, True)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._all_task = data_manager.nb_tasks
        self.data_manager = data_manager
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self.network1.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers, pin_memory=True
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True
        )

        if len(self._multiple_gpus) > 1:
            self.network1 = nn.DataParallel(self.network1, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self.network1 = self.network1.module

    def _train(self, train_loader, test_loader):
        self.network1.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.AdamW(
                self.network1.parameters(),
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = None
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.AdamW(
                self.network1.parameters(),
                lr=self.args["lrate"],
                weight_decay=self.args["weight_decay"],
            )  # 1e-5
            scheduler = None
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        scaler = torch.cuda.amp.GradScaler()
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self.network1.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, _) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                with torch.cuda.amp.autocast():
                    logits = self.network1(inputs)["logits"]
                    loss = F.cross_entropy(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self.network1, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["init_epoch"],
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        scaler = torch.cuda.amp.GradScaler()
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.network1.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets, _) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.long().to(self._device)
                with torch.cuda.amp.autocast():
                    logits = self.network1(inputs)["logits"]
                    fake_targets = targets - self._known_classes
                    loss_clf = F.cross_entropy(logits[:, self._known_classes :], fake_targets)
                    loss = loss_clf

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self.network1, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args["epochs"],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        if self._cur_task == self._all_task - 1 and self.args['if_tsne']:
            self.umap(showcenters=False, Normalize=False)

        logging.info(info)

    def umap(self, showcenters=False, Normalize=False):
        import umap
        import matplotlib.pyplot as plt
        print('now draw tsne results of extracted features.')
        tot_classes=self._total_classes
        test_dataset = self.data_manager.get_dataset(np.arange(tot_classes, step=(tot_classes/self._all_task)-2), source='test', mode='test')
        valloader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=8)
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
        self.network1.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets, _ in loader:
                _targets = _targets.numpy()
                _vectors = tensor2numpy(
                    self.network1.backbone(_inputs.to(self._device))
                )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
