import numpy as np
import torch.cuda
import torch.nn as nn
import sklearn.metrics as sm
import uncertainty as unc
from typing import Union
import models as mds
import utils

def accuracy_rejection(model: Union[nn.Module, mds.RandomForest], x, y, measure, device='cpu'):
    portion_vals = np.linspace(0, 1, 50, endpoint=False)

    acc_tu = np.empty(len(portion_vals))
    acc_eu = np.empty(len(portion_vals))
    acc_au = np.empty(len(portion_vals))
    acc_ra = np.empty(len(portion_vals))
    acc_bt = np.empty(len(portion_vals))

    if isinstance(model, (mds.RandomForest, mds.CalibratedForest)):
        preds = model.predict(x)
    elif isinstance(model, nn.Module):
        preds, y = utils.torch_get_outputs(model, x, device, samples=20)
        preds = preds.cpu().detach().numpy()
        y = y.cpu().detach().numpy()

    tu = unc.total_uncertainty(preds, measure)
    eu = unc.epistemic_uncertainty(preds, measure)
    print(f"Epistemic uncertainty: {eu.mean()}")
    au = unc.aleatoric_uncertainty(preds, measure)

    preds = preds.mean(axis=2).argmax(axis=1)
    for i, portion in enumerate(portion_vals):
        acc_preds, acc_y = unc.remove_rejected(preds, y, portion, tu)
        acc_tu[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_rejected(preds, y, portion, au)
        acc_au[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_rejected(preds, y, portion, eu)
        acc_eu[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_random(preds, y, portion)
        acc_ra[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_incorrect(preds, y, portion)
        acc_bt[i] = sm.accuracy_score(acc_y, acc_preds)
    return np.asarray([acc_tu, acc_eu, acc_au, acc_ra, acc_bt])

def accuracy_rejection_with_preds(preds, y, measure, device='cpu'):
    portion_vals = np.linspace(0, 1, 50, endpoint=False)

    acc_tu_low = np.empty(len(portion_vals))
    acc_tu_up = np.empty(len(portion_vals))
    acc_eu = np.empty(len(portion_vals))
    acc_au_low = np.empty(len(portion_vals))
    acc_au_up = np.empty(len(portion_vals))
    acc_ra = np.empty(len(portion_vals))
    acc_bt = np.empty(len(portion_vals))


    preds = preds.cpu().detach().numpy()
    y = y.cpu().detach().numpy()

    eu = unc.epistemic_uncertainty(preds, measure)
    # print(f"Epistemic uncertainty: {eu.mean()}")
    au_low, au_up = unc.aleatoric_uncertainty(preds, measure)
    tu_low = au_low + eu
    tu_up = au_up + eu

    # print(np.isclose(tu, au + eu, rtol=1e-04, atol=1e-07).sum() / len(tu))

    preds = preds.mean(axis=2).argmax(axis=1)
    for i, portion in enumerate(portion_vals):
        acc_preds, acc_y = unc.remove_rejected(preds, y, portion, tu_low)
        acc_tu_low[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_rejected(preds, y, portion, tu_up)
        acc_tu_up[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_rejected(preds, y, portion, au_low)
        acc_au_low[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_rejected(preds, y, portion, au_up)
        acc_au_up[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_rejected(preds, y, portion, eu)
        acc_eu[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_random(preds, y, portion)
        acc_ra[i] = sm.accuracy_score(acc_y, acc_preds)
        acc_preds, acc_y = unc.remove_incorrect(preds, y, portion)
        acc_bt[i] = sm.accuracy_score(acc_y, acc_preds)
    return np.asarray([acc_tu_low, acc_tu_up, acc_au_low, acc_au_up, acc_eu, acc_ra, acc_bt])


def out_of_distribution_with_preds(preds_id, preds_ood, unc_measure, device='cpu'):
    # print(f"Mean ID confidence: {preds_id.mean(dim=2).max(dim=1)[0].mean()}, and {preds_id.std(dim=2).mean(dim=(0,1))}")
    # print(f"Mean OOD confidence: {preds_ood.mean(dim=2).max(dim=1)[0].mean()}, and {preds_id.std(dim=2).mean(dim=(0,1))}")
    preds_id = preds_id.cpu().detach().numpy()
    preds_ood = preds_ood.cpu().detach().numpy()
    preds_id = np.clip(preds_id, 1e-25, 1)
    preds_ood = np.clip(preds_ood, 1e-25, 1)
    uncertainties_id = unc.epistemic_uncertainty(preds_id, unc_measure)
    uncertainties_ood = unc.epistemic_uncertainty(preds_ood, unc_measure)

    labels = np.concatenate((np.zeros(len(uncertainties_id)), np.ones(len(uncertainties_ood))))
    uncertainties = np.concatenate((uncertainties_id, uncertainties_ood))

    auroc = sm.roc_auc_score(labels, uncertainties)
    return uncertainties_id, uncertainties_ood, auroc


def out_of_distribution(model, loader_id, loader_ood, unc_measure, device='cpu'):
    preds_id, _ = utils.torch_get_outputs(model, loader_id, device=device, samples=20)
    preds_ood, _ = utils.torch_get_outputs(model, loader_ood, device=device, samples=20)
    print(f"Mean ID confidence: {preds_id.mean(dim=2).max(dim=1)[0].mean()}, and {preds_id.std(dim=2).mean(dim=(0,1))}")
    print(f"Mean OOD confidence: {preds_ood.mean(dim=2).max(dim=1)[0].mean()}, and {preds_id.std(dim=2).mean(dim=(0,1))}")
    preds_id = preds_id.cpu().detach().numpy()
    preds_ood = preds_ood.cpu().detach().numpy()
    preds_id = np.clip(preds_id, 1e-25, 1)
    preds_ood = np.clip(preds_ood, 1e-25, 1)
    uncertainties_id = unc.epistemic_uncertainty(preds_id, unc_measure)
    uncertainties_ood = unc.epistemic_uncertainty(preds_ood, unc_measure)

    labels = np.concatenate((np.zeros(len(uncertainties_id)), np.ones(len(uncertainties_ood))))
    uncertainties = np.concatenate((uncertainties_id, uncertainties_ood))

    auroc = sm.roc_auc_score(labels, uncertainties)
    return uncertainties_id, uncertainties_ood, auroc


