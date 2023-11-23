import os
import torch
import numpy as np
import torchvision.datasets as dset
from . import data_utils as utils


def define_dataloader(dataset='cifar100', dataset_path='data',cutout=False, cutout_length=16, cutout_prob=1.0, split_portion=1.0, batch_size=64):
    #### data
    if dataset == 'cifar10':
        train_transform, _ = utils._data_transforms_cifar10(cutout, cutout_length, cutout_prob)
        train_data = dset.CIFAR10(root=dataset_path, train=True, download=True, transform=train_transform)
        n_classes=10
        input_size = (3, 32, 32)
    elif dataset == 'cifar100':
        train_transform, _ = utils._data_transforms_cifar100(cutout, cutout_length, cutout_prob)
        train_data = dset.CIFAR100(root=dataset_path, train=True, download=True, transform=train_transform)
        n_classes=100
        input_size = (3, 32, 32)
    elif dataset == 'svhn':
        train_transform, _ = utils._data_transforms_svhn(cutout, cutout_length, cutout_prob)
        train_data = dset.SVHN(root=dataset_path, split='train', download=True, transform=train_transform)
        n_classes=10
        input_size = (3, 32, 32)
    elif dataset == 'imagenet16-120':
        import torchvision.transforms as transforms
        from .DownsampledImageNet import ImageNet16
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(root=os.path.join(utils.data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700
        n_classes=120
        input_size = (3, 16, 16)
    elif dataset == 'imagenet':
        import torchvision.transforms as transforms
        from .hdf5 import H5Dataset
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.2),
                transforms.ToTensor(),
                normalize,
        ])

        train_data = H5Dataset(os.path.join(dataset_path, 'imagenet-train-256.h5'), transform=train_transform)
        n_classes=1000
        input_size = (3, 224, 224)
    else:
        raise Exception('Not recognized dataset name {}, please use dataset from[cifar10, cifar100, svhn, imagenet16-120, imagenet] or mannully designed your dataloader'.format(dataset))
        

    num_train = len(train_data)
    indices = list(range(num_train))
    #split = int(np.floor(split_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:batch_size]),
        pin_memory=True)
    
    return train_queue, n_classes, input_size