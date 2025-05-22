import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR224, iImageNetR,iImageNetA,CUB, objectnet, omnibenchmark, vtab


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
            
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y, z = self._train_data, self._train_targets, self.train_data_classes
        elif source == "val":
            x, y, z = self._val_data, self._val_targets, self.val_data_classes
        elif source == "test":
            x, y, z = self._test_data, self._test_targets, self.test_data_classes
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "val":
            trsf = transforms.Compose([*self._val_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets, names = [], [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets, class_name = self._select(
                    x, y, z, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets, class_name = self._select_rmm(
                    x, y, z, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)
            names.append(class_name)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets, appendent_names = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)
            names.append(appendent_names)

        data, targets, names = np.concatenate(data), np.concatenate(targets), np.concatenate(names)

        if ret_data:
            return data, targets, names, DummyDataset(data, targets, names, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, names, trsf, self.use_path)

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y, z = self._train_data, self._train_targets, self.train_data_classes
        elif source == "val":
            x, y, z = self._val_data, self._val_targets, self.val_data_classes
        elif source == "test":
            x, y, z = self._test_data, self._test_targets, self.test_data_classes
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "val":
            trsf = transforms.Compose([*self._val_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets, train_names = [], [], []
        val_data, val_targets, val_names = [], [], []
        for idx in indices:
            class_data, class_targets, class_names = self._select(
                x, y, z, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            val_names.append(class_names[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])
            train_names.append(class_names[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets, appendent_names = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets, append_names = self._select(
                    appendent_data, appendent_targets, appendent_names, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                val_names.append(append_names[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])
                train_names.append(append_names[train_indx])

        train_data, train_targets, train_names = np.concatenate(train_data), np.concatenate(train_targets), np.concatenate(train_names)
        val_data, val_targets, val_names = np.concatenate(val_data), np.concatenate(val_targets), np.concatenate(val_names)

        return DummyDataset(train_data, train_targets, train_names, trsf, self.use_path),\
               DummyDataset(val_data, val_targets, val_names, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        '''
        获取数据集：图像、标签、类别
        获取数据集Transforms：训练集、测试集
        获取数据集template：
        对标签进行Order重排
        '''
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self.train_data_classes = idata.train_data_classes
        self._val_data, self._val_targets = idata.val_data, idata.val_targets
        self.val_data_classes = idata.val_data_classes
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.test_data_classes = idata.test_data_classes
        self.use_path = idata.use_path
        
        # Transforms
        self._train_trsf = idata.train_trsf
        self._val_trsf = idata.val_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._val_targets = _map_new_class_index(self._val_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)


    def _select(self, x, y, z, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]

        z = [z[i] for i in list(idxes)]
        if type(x) is np.ndarray:
            x = x[idxes]
        else:
            x = [x[i] for i in list(idxes)]
        return x, y[idxes], z

    def _get_template(self, dataset_name):
        if dataset_name == "cifar100":
            self.template = ['a photo of a {}.']

        elif dataset_name == "imagenetr":
            self.template = ['a photo of a {}.']
        else:
            self.template = ['a photo of a {}.']

    def _select_rmm(self, x, y, z, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int(m_rate * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]

        new_index = list(y[new_idxes])
        z = [z[i] for i in new_index]

        return x[new_idxes], y[new_idxes], z

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, names, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.names = names
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        name = self.names[idx]

        return idx, image, label, name


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "cifar224":
        return iCIFAR224()
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == "imageneta":
        return iImageNetA()
    elif name == "cub":
        return CUB()
    elif name == "objectnet":
        return objectnet()
    elif name == "omnibenchmark":
        return omnibenchmark()
    elif name == "vtab":
        return vtab()
    elif name == "domainnet":
        from utils.data import iDomainNet
        return iDomainNet(args)

    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
