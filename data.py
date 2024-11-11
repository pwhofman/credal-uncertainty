import numpy as np
import torchvision.transforms as T
import torch
import torchvision
from torch.utils.data import DataLoader
import PIL
import os


MNIST_SIZE = 28
SVHN_SIZE = 32
CIFAR_SIZE = 224
PATH = "./data/"

def get_data_ood(data_id, data_ood, batch_size=1024):
    num = 10000
    if data_id == "mnist":
        transforms = T.Compose([T.Resize(MNIST_SIZE),
                                T.ToTensor()])
        id = torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=transforms)
    elif data_id == "cifar10":
        transforms = T.Compose([T.Resize((CIFAR_SIZE, CIFAR_SIZE)),
                                T.ToTensor(),
                                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        id = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True, transform=transforms)
    elif data_id == "cifar100":
        transforms = T.Compose([T.Resize((CIFAR_SIZE, CIFAR_SIZE)),
                                T.ToTensor(),
                                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        id = torchvision.datasets.CIFAR100(root=PATH, train=False, download=True, transform=transforms)
    elif data_id == "food101":
        transforms = T.Compose([T.Resize((CIFAR_SIZE, CIFAR_SIZE)), T.ToTensor()])
        id = torchvision.datasets.Food101(root=PATH, split="test", download=True,
                                            transform=transforms)
    elif data_id == "fmnist":
        transforms = T.Compose([T.Resize((MNIST_SIZE, MNIST_SIZE)), T.ToTensor()])
        id = torchvision.datasets.FashionMNIST(root=PATH, train=False, download=True, transform=transforms)
    else:
        raise ValueError("Invalid id dataset name")

    if data_ood == "mnist":
        # transforms.transforms.insert(0, T.Grayscale(3))
        ood = torchvision.datasets.MNIST(root=PATH, train=False, download=True, transform=transforms)
    elif data_ood == "fmnist":
        ood = torchvision.datasets.FashionMNIST(root=PATH, train=False, download=True, transform=transforms)
    elif data_ood == "kmnist":
        ood = torchvision.datasets.KMNIST(root=PATH, train=False, download=True, transform=transforms)
    elif data_ood == "cifar10":
        ood = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True, transform=transforms)
    elif data_ood == "cifar100":
        ood = torchvision.datasets.CIFAR100(root=PATH, train=False, download=True, transform=transforms)
    elif data_ood == 'svhn':
        ood = torchvision.datasets.SVHN(root=PATH, split="test", download=True, transform=transforms)
    else:
        raise ValueError("Invalid ood dataset name")
    # balance the datasets
    random_id = torch.utils.data.RandomSampler(id, num_samples=num)
    random_ood = torch.utils.data.RandomSampler(ood, num_samples=num)
    id_loader = torch.utils.data.DataLoader(id, batch_size=batch_size, shuffle=False, num_workers=4, sampler=random_id)
    ood_loader = torch.utils.data.DataLoader(ood, batch_size=batch_size, shuffle=False, num_workers=4, sampler=random_ood)
    return id_loader, ood_loader


def get_data(dataset, batch_size=64, validate=False):
    if dataset == "mnist":
        train = torchvision.datasets.MNIST(root=PATH, train=True, download=True,
                                           transform=T.ToTensor())
        test = torchvision.datasets.MNIST(root=PATH, train=False, download=True,
                                          transform=T.ToTensor())
    elif dataset == "fmnist":
        train = torchvision.datasets.FashionMNIST(root=PATH, train=True, download=True,
                                           transform=T.ToTensor())
        test = torchvision.datasets.FashionMNIST(root=PATH, train=False, download=True,
                                          transform=T.ToTensor())
    elif dataset == "kmnist":
        train = torchvision.datasets.KMNIST(root=PATH, train=True, download=True,
                                                  transform=T.ToTensor())
        test = torchvision.datasets.KMNIST(root=PATH, train=False, download=True,
                                                 transform=T.ToTensor())

    elif dataset == "cifar10":
        train = torchvision.datasets.CIFAR10(root=PATH, train=True, download=True,
                                                  transform=T.Compose([T.Resize(CIFAR_SIZE),
                                                                       T.ToTensor(),
                                                                       T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        test = torchvision.datasets.CIFAR10(root=PATH, train=False, download=True,
                                                 transform=T.Compose([T.Resize(CIFAR_SIZE),
                                                                      T.ToTensor(),
                                                                      T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

    elif dataset == "svhn":
        train = torchvision.datasets.SVHN(root=PATH, split="train", download=True,
                                                  transform=T.ToTensor())
        test = torchvision.datasets.SVHN(root=PATH, split="test", download=True,
                                                 transform=T.ToTensor())
    elif dataset == "cifar100":
        train = torchvision.datasets.CIFAR100(root=PATH, train=True, download=True,
                                            transform=T.Compose([T.Resize(CIFAR_SIZE), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        test = torchvision.datasets.CIFAR100(root=PATH, train=False, download=True,
                                            transform=T.Compose([T.Resize(CIFAR_SIZE), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    else:
        raise ValueError("Invalid dataset name")

    if validate:
        train, val = torch.utils.data.random_split(train, [0.9, 0.1])
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, val_loader, test_loader
    else:
        random_sampler = torch.utils.data.RandomSampler(test, num_samples=10000)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, sampler=random_sampler)
        return train_loader, test_loader
