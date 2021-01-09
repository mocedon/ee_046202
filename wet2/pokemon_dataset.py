"""
Author: Tal Daniel
"""

# imports
import numpy as np
import pandas as pd
import os
import torch
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import warnings


def int_to_one_hot(val, length):
    """
    Converts number to one-hot vector given the vector length
    :param val: value to convert
    :param length: length of the vector
    :return: one_hot_vec
    """
    one_hot = np.zeros(length)
    one_hot[val] = 1
    return one_hot


class VisionDataset(Dataset):
    _repr_indent = 4

    def __init__(self, root, transforms=None, transform=None, target_transform=None):
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root

        has_transforms = transforms is not None
        has_separate_transform = transform is not None or target_transform is not None
        if has_transforms and has_separate_transform:
            raise ValueError("Only transforms or transform/target_transform can "
                             "be passed as argument")

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

        if has_separate_transform:
            transforms = transforms.StandardTransform(transform, target_transform)
        self.transforms = transforms

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""


class PokemonDataset(VisionDataset):
    def __init__(self, root, rgb=False):
        warnings.filterwarnings(
            category=UserWarning, action="ignore")
        self.poke_list = pd.read_csv(os.path.join(root, 'pokemon.csv'))
        self.unique_types = sorted(pd.unique(self.poke_list['Type1']))
        self.type_to_one_hot = OrderedDict(
            [(self.unique_types[i], int_to_one_hot(i, len(self.unique_types))) for i in range(len(self.unique_types))])
        self.poke_list = sorted(self.poke_list.values, key=lambda x: x[0])
        self.name_to_type = OrderedDict(
            [(self.poke_list[i][0], self.type_to_one_hot[self.poke_list[i][1]]) for i in range(len(self.poke_list))])
        self.file_list = os.listdir(os.path.join(root, 'images'))
        self.data = []
        self.targets = []
        for i in range(len(self.file_list)):
            name = self.file_list[i].split('.')[0]
            poke_type = self.name_to_type[name]
            self.targets.append(poke_type)
            rgba_image = Image.open(os.path.join(root, 'images', self.file_list[i]))
            rgb_image = rgba_image.convert('RGB')
            self.data.append(np.array(rgb_image))

        self.data = np.array(self.data)
        self.targets = np.array(self.targets).astype(np.float32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if rgb:
            self.transform = transforms.Compose([transforms.Resize(60), transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.Grayscale(), transforms.Resize(60), transforms.ToTensor()])

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.poke_list)


if __name__ == '__main__':
    data = PokemonDataset(root='./data/pokemon')
    dl = DataLoader(dataset=data, batch_size=5, shuffle=False)
    print(next(iter(dl)))
