import scipy.stats
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms

import utils
from utils import accuracy
import sklearn.metrics as sm
import torch.optim as optim
from tqdm import tqdm
import time
from data import get_data
import models as mds
import os
import experiments
import sklearn.model_selection as sms
import uncertainty as unc
import sklearn.tree as st
import random
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt

np.random.seed(7)
torch.random.manual_seed(7)
random.seed(7)

DATA = "mnist"

if DATA == "mnist":
    train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                       transform=T.Compose([T.ToTensor(), torch.flatten]))
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                      transform=T.Compose([T.ToTensor(), torch.flatten]))
    test_loader = torch.utils.data.DataLoader(test, batch_size=1000, shuffle=False)

RUNS = 5
ROUNDS = 20
START_SAMPLES = 200
EPOCHS = 5
NUM_MEMBERS = 10
MEASURE = 'log'
NEW_SAMPLES = 200
DEVICE = 'cuda'
accs = np.empty((RUNS, ROUNDS))
for run in tqdm(range(RUNS)):
    # print(accs)
    # print(f"Run {run}")
    x = torch.flatten(train.data, start_dim=1).float()
    y = train.targets
    perm = torch.randperm(x.shape[0])
    x = x[perm]
    y = y[perm]
    x_train = x[:START_SAMPLES]
    y_train = y[:START_SAMPLES]
    x_pool = x[START_SAMPLES:]
    y_pool = y[START_SAMPLES:]
    criterion = nn.CrossEntropyLoss()
    for round in range(ROUNDS):
        # print(f"Training data {x_train.shape}")
        # print(f"Pool data {x_pool.shape}")
        train_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.TensorDataset(x_train, y_train),
                                                   batch_size=64, shuffle=True)
        model = mds.ActiveEnsemble(NUM_MEMBERS)
        for i in range(len(model.members)):
            optimizer = optim.Adam(model.members[i].parameters())
            for epoch in range(EPOCHS):
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model.members[i](inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
        acc = accuracy(model, test_loader)
        accs[run, round] = acc
        # print(f"Accuracy for round {round}: {acc}")
        preds_pool = model(x_pool.float())
        unc_pool = unc.epistemic_uncertainty(preds_pool.detach().cpu().numpy(), MEASURE)

        indices = np.argsort(unc_pool)
        selected = indices[-NEW_SAMPLES:]
        x_train = torch.cat((x_train, x_pool[selected]), dim=0)
        y_train = torch.cat((y_train, y_pool[selected]), dim=0)
        x_pool = x_pool[indices[:-NEW_SAMPLES]]
        y_pool = y_pool[indices[:-NEW_SAMPLES]]
np.save(f"./output/al_{DATA}_{MEASURE}.npy", accs)