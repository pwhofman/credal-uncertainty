import numpy as np
import torch
from scipy.stats import entropy
from itertools import product
from tqdm import tqdm
import utils

EPS = 1e-25

def total_uncertainty(probs, measure):
    return aleatoric_uncertainty(probs, measure) + epistemic_uncertainty(probs, measure)


def aleatoric_uncertainty(probs, measure):
    if measure == "log":
        au = np.zeros((probs.shape[2], probs.shape[0]))
        for i in range(probs.shape[2]):
            au[i] = entropy(probs[:, :, i], axis=1, base=2)
        au_low = np.min(au, axis=0)
        au_up = np.max(au, axis=0)
    elif measure == "brier":
        au = np.zeros((probs.shape[2], probs.shape[0]))
        for i in range(probs.shape[2]):
            au[i] = 1 - np.sum(probs[:, :, i] ** 2, axis=1)
        au_low = np.min(au, axis=0)
        au_up = np.max(au, axis=0)
    elif measure == "01":
        au = np.zeros((probs.shape[2], probs.shape[0]))
        for i in range(probs.shape[2]):
            au[i] = 1 - np.max(probs[:, :, i], axis=1)
        au_low = np.min(au, axis=0)
        au_up = np.max(au, axis=0)
    elif measure == "spherical":
        au = np.zeros((probs.shape[2], probs.shape[0]))
        for i in range(probs.shape[2]):
            au += 1 - np.linalg.norm(probs[:, :, i], axis=1)
        au_low = np.min(au, axis=0)
        au_up = np.max(au, axis=0)
    elif measure == "entropy":
        au = entropy(probs, axis=1, base=2)
        au = np.mean(au, axis=1)
        au_low = au
        au_up = au
    else:
        raise ValueError("Invalid uncertainty measure")
    return au_low, au_up


def epistemic_uncertainty(probs, measure):
    if measure == "log":
        eu = np.zeros((probs.shape[2]**2, probs.shape[0]))
        for i in range(probs.shape[2]):
            for j in range(probs.shape[2]):
                eu[i * probs.shape[2] + j] = entropy(probs[:, :, i], probs[:, :, j], axis=1, base=2)
        eu = np.max(eu, axis=0)
    elif measure == "brier":
        eu = np.zeros((probs.shape[2]**2, probs.shape[0]))
        for i in range(probs.shape[2]):
            for j in range(probs.shape[2]):
                eu[i * probs.shape[2] + j] = np.sum((probs[:, :, i] - probs[:, :, j]) ** 2, axis=1)
        eu = np.max(eu, axis=0)
    elif measure == "01":
        eu = np.zeros((probs.shape[2]**2, probs.shape[0]))
        for i in range(probs.shape[2]):
            for j in range(probs.shape[2]):
                eu[i * probs.shape[2] + j] = (np.max(probs[:, :, i], axis=1) -
                       probs[np.arange(probs.shape[0]), np.argmax(probs[:, :, j], axis=1), i])
        eu = np.max(eu, axis=0)
    elif measure == "spherical":
        eu = np.zeros((probs.shape[2]**2, probs.shape[0]))
        for i in range(probs.shape[2]):
            for j in range(probs.shape[2]):
                eu[i * probs.shape[2] + j] = np.linalg.norm(probs[:, :, i], axis=1) - np.sum(probs[:, :, i] * probs[:, :, j], axis=1) / np.linalg.norm(probs[:, :, i], axis=1)
        eu = np.max(eu, axis=0)
    elif measure == "entropy":
        probs_mean = np.mean(probs, axis=2)
        probs_mean = np.repeat(np.expand_dims(probs_mean, 2), repeats=probs.shape[2], axis=2)
        eu = entropy(probs, probs_mean, axis=1, base=2)
        eu = np.mean(eu, axis=1)
    elif measure == "gh":
        if probs.shape[1] > 20:
            eu = np.empty(probs.shape[0])
            for i in tqdm(range(probs.shape[0])):
                prob = np.expand_dims(probs[i], 0)
                mask = np.all(prob < 0.1, axis=2)
                prob = np.expand_dims(prob[~mask], 0)
                eu[i] = generalised_hartley(prob)
        else:
            eu = generalised_hartley(probs)
    else:
        raise ValueError("Invalid uncertainty measure")
    return eu

def capacity(Q, A):
    """Computes the capacity of a set Q given a set A"""
    sum = np.sum(Q[:, A, :], axis=1)
    min = np.min(sum, axis=1)
    return min


def moebius(Q, A):
    """Computes the Moebius function of a set Q given a set A,
    however here it's done for all outputs at the same time.
    Q: array of shape (num_samples, num_classes, num_members)
    A: set of indices
    """
    ps_B = utils.powerset(A)  # powerset of A
    ps_B.pop(0)  # remove empty set
    m_A = np.zeros(Q.shape[0])
    for B in ps_B:
        l = len(set(A) - set(B))
        m_A += ((-1) ** l) * capacity(Q, B)
    return m_A


def generalised_hartley(outputs):
    """Computes the generalised Hartley measure given a 'credal' set of probability distributions

    outputs: array of shape (num_samples, num_classes, num_members)
    """
    if (isinstance(outputs, torch.Tensor)):
        outputs = outputs.detach().numpy()

    gh = np.zeros(outputs.shape[0])
    idxs = list(range(outputs.shape[1]))  # list of class indices
    ps_A = utils.powerset(idxs)  # powerset of all indices
    ps_A.pop(0)  # remove empty set
    for A in ps_A:
        m_A = moebius(outputs, A)
        gh += m_A * np.log2(len(A))
    return gh


def total_uncertainty_entropy(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    t_u = entropy(np.mean(probs, axis=2), axis=1, base=2) / np.log2(probs.shape[1])
    return t_u


def epistemic_uncertainty_entropy(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    mean_probs = np.mean(probs, axis=2)
    mean_probs = np.repeat(np.expand_dims(mean_probs, 2), repeats=probs.shape[2], axis=2)
    mean_probs = np.clip(mean_probs, 1e-25, 1)
    probs = np.clip(probs, 1e-25, 1)
    e_u = entropy(probs, mean_probs, axis=1, base=2) / np.log2(probs.shape[1])
    e_u = np.mean(e_u, axis=1)
    return e_u


def aleatoric_uncertainty_entropy(probs):
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().numpy()
    a_u = entropy(probs, axis=1, base=2) / np.log2(probs.shape[1])
    a_u = np.mean(a_u, axis=1)
    return a_u


def remove_rejected(y_pred, y_true, reject_portion, uncertainties):
    if reject_portion == 0:
        return y_pred, y_true
    num = int(y_pred.shape[0] * reject_portion)
    indices = np.argsort(uncertainties)
    y_pred = y_pred[indices]
    y_true = y_true[indices]
    y_pred = y_pred[:-num]
    y_true = y_true[:-num]
    return y_pred, y_true


def remove_random(y_pred, y_true, reject_portion):
    if reject_portion == 0:
        return y_pred, y_true
    num = int(y_pred.shape[0] * reject_portion)
    indices = np.random.permutation(y_pred.shape[0])
    y_pred = y_pred[indices]
    y_true = y_true[indices]
    y_pred = y_pred[:-num]
    y_true = y_true[:-num]
    return y_pred, y_true

def remove_incorrect(y_pred, y_true, reject_portion):
    if reject_portion == 0:
        return y_pred, y_true
    num = int(y_pred.shape[0] * reject_portion)
    indices = np.argsort(y_pred != y_true)
    y_pred = y_pred[indices]
    y_true = y_true[indices]
    y_pred = y_pred[:-num]
    y_true = y_true[:-num]
    return y_pred, y_true

