import torch
import torchvision
import torchvision.transforms as T
import torch_geometric as geom
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader
from tu_dataset import drop_nodes
import numpy as np
from RandAugment import RandAugment
from PIL import ImageOps


def convert_features_mnist(data):
    data.pos /= 28.0  # normalize the mnist position dimensions
    data.x = torch.cat([data.x, data.pos], dim=-1)
    return data


def convert_features_cifar(data):
    data.pos /= 32.0  # normalize the mnist position dimensions
    data.x = torch.cat([data.x, data.pos], dim=-1)
    return data


def colorize(img):
    # randomly recolor MNIST. leave background color as black.
    r, g, b = np.random.randint(low=50, high=255, size=3)
    img = ImageOps.colorize(img, black="black", white=(r, g, b))
    return img


def rgb(img):
    img = img.convert("RGB")
    return img


class aug_drop_nodes(object):
    def __init__(self, aug_ratio):
        self.aug_ratio = aug_ratio

    def __call__(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num * self.aug_ratio)

        idx_perm = np.random.permutation(node_num)

        idx_drop = idx_perm[:drop_num]
        idx_nondrop = idx_perm[drop_num:]
        idx_nondrop.sort()
        idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

        edge_index = data.edge_index.numpy()
        adj = torch.zeros((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 1
        adj = adj[idx_nondrop, :][:, idx_nondrop]
        edge_index = adj.nonzero().t()

        try:
            data.edge_index = edge_index
            data.x = data.x[idx_nondrop]
        except:
            data = data
        return data

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.aug_ratio)


class MNISTImageTransform:
    def __init__(self):
        self.transform = T.Compose(
            [
                T.RandomAffine(
                    degrees=30, translate=(0.1, 0.1), scale=None, shear=[0, 0.2, 0, 0.2]
                ),
                T.RandomApply([T.Compose([T.CenterCrop(10), T.Resize(28)])], p=0.7),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
                geom.transforms.ToSLIC(n_segments=75, enforce_connectivity=True),
                geom.transforms.KNNGraph(k=8, loop=True),
                geom.transforms.Distance(cat=False),
            ]
        )

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


class MNISTColorizeTransform:
    def __init__(self):
        self.transform = T.Compose(
            [
                T.RandomAffine(
                    degrees=30, translate=(0.1, 0.1), scale=None, shear=[0, 0.2, 0, 0.2]
                ),
                T.RandomApply([colorize], p=0.7),
                rgb,
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
                geom.transforms.ToSLIC(
                    n_segments=75, enforce_connectivity=True, convert2lab=True
                ),
                geom.transforms.KNNGraph(k=8, loop=True),
                convert_features_mnist,
                geom.transforms.Distance(cat=False),
            ]
        )

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


class CIFARImageTransform:
    def __init__(self):
        # self.transform = T.Compose(
        #     [
        #         T.RandomResizedCrop(32, scale=(0.2, 1.0)),
        #         T.RandomHorizontalFlip(),
        #         T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        #         T.RandomGrayscale(p=0.2),
        #         T.ToTensor(),
        #         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #         geom.transforms.ToSLIC(n_segments=150, enforce_connectivity=True),
        #         geom.transforms.KNNGraph(k=8, loop=True),
        #         convert_features_cifar,
        #         geom.transforms.Distance(cat=False),
        #     ]
        # )
        self._CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
        self._CIFAR_STD = (0.2023, 0.1994, 0.2010)
        self.transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self._CIFAR_MEAN, self._CIFAR_STD),
                geom.transforms.ToSLIC(
                    n_segments=150, enforce_connectivity=True, convert2lab=True
                ),
                geom.transforms.KNNGraph(k=8, loop=True),
                # convert_features_cifar,
                geom.transforms.Distance(cat=False),
            ]
        )

        # 3,5 -- 256 batchsize
        self.transform.transforms.insert(0, RandAugment(3, 5))

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


class MNISTGraphTransform:
    def __init__(self, aug_ratio):
        self.aug_ratio = aug_ratio
        self.transform = T.Compose(
            [
                rgb,
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
                geom.transforms.ToSLIC(n_segments=75, enforce_connectivity=True, convert2lab=True),
                geom.transforms.KNNGraph(k=8, loop=True),
                convert_features_mnist,
                geom.transforms.Distance(cat=False),
                aug_drop_nodes(self.aug_ratio),
            ]
        )

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


class CIFARGraphTransform:
    def __init__(self, aug_ratio):
        self.aug_ratio = aug_ratio
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(_CIFAR_MEAN, _CIFAR_STD),
                geom.transforms.ToSLIC(n_segments=150, enforce_connectivity=True),
                geom.transforms.KNNGraph(k=8, loop=True),
                convert_features_cifar,
                geom.transforms.Distance(cat=False),
                aug_drop_nodes(self.aug_ratio),
            ]
        )

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2


class MNISTColorEvalTransform:
    def __init__(self):
        self.transform = T.Compose(
            [
                rgb,
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
                geom.transforms.ToSLIC(
                    n_segments=75, enforce_connectivity=True, convert2lab=True
                ),
                geom.transforms.KNNGraph(k=8, loop=True),
                convert_features_mnist,
                geom.transforms.Distance(cat=False),
            ]
        )

    def __call__(self, x):
        x1 = self.transform(x)
        return x1


class MNISTEvalTransform:
    def __init__(self):
        self.transform = T.Compose(
            [
                rgb,
                T.ToTensor(),
                T.Normalize((0.1307,), (0.3081,)),
                geom.transforms.ToSLIC(
                    n_segments=75, enforce_connectivity=True, convert2lab=True
                ),
                geom.transforms.KNNGraph(k=8, loop=True),
                convert_features_mnist,
                geom.transforms.Distance(cat=False),
            ]
        )

    def __call__(self, x):
        x1 = self.transform(x)
        return x1


class CIFAREvalTransform:
    def __init__(self):
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(_CIFAR_MEAN, _CIFAR_STD),
                geom.transforms.ToSLIC(
                    n_segments=150, enforce_connectivity=True, convert2lab=True
                ),
                geom.transforms.KNNGraph(k=8, loop=True),
                # convert_features_cifar,
                geom.transforms.Distance(cat=False),
            ]
        )

    def __call__(self, x):
        x1 = self.transform(x)
        return x1
