import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import itertools

EPS = 1e-10


@torch.no_grad()
def accuracy(model, loader, device='cpu'):
    model.eval().to(device)
    if hasattr(model, 'members'):
        for m in model.members:
            m.eval().to(device)
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model.predict(inputs)
        outputs = outputs.mean(axis=2)
        correct += torch.sum(torch.argmax(outputs, dim=1) == targets)
        total += targets.shape[0]
    return correct/total


@torch.no_grad()
def torch_get_outputs(model, loader, device, samples=5):
    outputs = torch.empty(0, device=device)
    targets = torch.empty(0, device=device)
    for input, target in tqdm(loader):
        input, target = input.to(device), target.to(device)
        targets = torch.cat((targets, target), dim=0)
        outputs = torch.cat((outputs, model.predict(input, samples)), dim=0)
    return outputs, targets

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device("cpu")
    return device

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)))
