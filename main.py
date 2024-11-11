import os
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import utils
from data import get_data, get_data_ood
import models as mds
import experiments
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
from py_experimenter.experimenter import PyExperimenter

# EU_MEASURES = ["entropy", "log", "brier", "spherical", "01", "gh", "diff"]
EU_MEASURES = ["gh"]

AU_MEASURES = ["entropy", "log", "brier", "spherical", "01"]
CHECK_DIR = "./checkpoints/"

def get_best_gpu():
    u0 = torch.cuda.utilization("cuda:0")
    u1 = torch.cuda.utilization("cuda:1")
    if u0 < u1:
        return "cuda:0"
    else:
        return "cuda:1"


def main(args, result_processor, _):
    if torch.cuda.is_available():
        device = torch.device(get_best_gpu())
    else:
        device = torch.device("cpu")

    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.mps.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])

    print("Experiment with following config:", args, "on device:", device)
    if args['exp'] == "arc":
        if args['uncertainty'] == "rf":
            accs = np.empty((args['runs'], 5, 50))
            for i in tqdm(range(args['runs'])):
                train, test = get_data(args['data'])
                x_train, y_train = train
                # x_train, x_cal, y_train, y_cal = train_test_split(x_train, y_train, test_size=0.2)
                x_test, y_test = test
                model = mds.RandomForest(n_estimators=args['num_members'],
                                         max_depth=args['max_depth'],
                                         max_features=None,
                                         )
                model.fit(x_train, y_train)
                # cal_model = mds.CalibratedForest(model)
                # cal_model.fit(x_cal, y_cal)
                preds = model.predict(x_test)
                print("RF test accuracy:", sm.accuracy_score(y_test, preds.mean(axis=2).argmax(axis=1)))
                acc = experiments.accuracy_rejection(model, x_test, y_test, args['measure'])
                # acc = experiments.accuracy_rejection(cal_model, x_test, y_test, args['measure'])
                accs[i, :, :] = acc
            np.save(f"./output/{args['exp']}_{args['data']}_{args['measure']}_{args['uncertainty']}.npy", accs)
        else:
            paths = os.listdir(CHECK_DIR)
            paths = [s for s in paths if f"{args['data']}_{args['base_model']}_{args['uncertainty']}" in s]
            if args['uncertainty'] != "dropout" and len(paths) < args['runs']:
                raise ValueError("Not enough models")
            if args['measure'] == 'all':
                accs = np.empty((args['runs'], 7, 50, len(AU_MEASURES)))
                for i in tqdm(range(args['runs'])):
                    train_loader, test_loader = get_data(args['data'], validate=False, batch_size=128)
                    print("DONE LOADING DATA")
                    if args['uncertainty'] == "dropout":
                        model = torch.load(CHECK_DIR + paths[0], map_location=device).to(device)
                        print(f"Model {paths[0]} loaded")
                    else:
                        model = torch.load(CHECK_DIR + paths[i], map_location=device).to(device)
                        print(f"Model {paths[i]} loaded")
                    if args['uncertainty'] == "ensemble":
                        for m in model.members:
                            m.eval()
                            m.to(device)
                    model.eval()
                    preds, y = utils.torch_get_outputs(model, test_loader, device, samples=5)
                    for j in range(len(AU_MEASURES)):
                        acc = experiments.accuracy_rejection_with_preds(preds, y, AU_MEASURES[j], device=device)
                        print(acc.shape)
                        accs[i, :, :, j] = acc
                        print(accs.shape)
                        # print(accs)
                for i in range(len(AU_MEASURES)):
                    print(accs[:, :, :, i].shape)
                    np.save(f"./output/{args['exp']}_{args['data']}_{AU_MEASURES[i]}_{args['uncertainty']}.npy", accs[:, :, :, i])
            else:
                accs = np.empty((args['runs'], 5, 50))
                for i in tqdm(range(args['runs'])):
                    train_loader, test_loader = get_data(args['data'], validate=False, batch_size=128)
                    if args['uncertainty'] == "dropout":
                        model = torch.load(CHECK_DIR + paths[0], map_location=device).to(device)
                    else:
                        model = torch.load(CHECK_DIR + paths[i], map_location=device).to(device)
                    if args['uncertainty'] == "ensemble":
                        for m in model.members:
                            m.eval()
                            m.to(device)
                    model.eval()
                    acc = experiments.accuracy_rejection(model, test_loader, None, measure=args['measure'], device=device)
                    accs[i, :, :] = acc
                np.save(f"./output/{args['exp']}_{args['data']}_{args['measure']}_{args['uncertainty']}.npy", accs)

    elif args['exp'] == "ood":
        paths = os.listdir(CHECK_DIR)
        paths = [s for s in paths if f"{args['data']}_{args['base_model']}_{args['uncertainty']}" in s]
        if args['uncertainty'] != "dropout" and len(paths) < args['runs']:
            raise ValueError("Not enough models")
        if args['measure'] == 'all':
            aurocs = np.empty((len(EU_MEASURES), args['runs']))
            for i in tqdm(range(args['runs'])):
                loader_id, loader_ood = get_data_ood(args['data'], args['ood_data'])
                if args['uncertainty'] == "dropout":
                    model = torch.load(CHECK_DIR + paths[0], map_location=device).to(device)
                    print(f"Model {paths[0]} loaded")
                else:
                    model = torch.load(CHECK_DIR + paths[i], map_location=device).to(device)
                    print(f"Model {paths[i]} loaded")
                model.eval()
                if args['uncertainty'] == "ensemble":
                    for m in model.members:
                        m.eval().to(device)
                preds_id, _ = utils.torch_get_outputs(model, loader_id, device, samples=5)
                preds_ood, _ = utils.torch_get_outputs(model, loader_ood, device, samples=5)
                for j in range(len(EU_MEASURES)):
                    u_id, u_ood, auroc = experiments.out_of_distribution_with_preds(preds_id, preds_ood, EU_MEASURES[j], device=device)
                    aurocs[j, i] = auroc
            for i in range(len(EU_MEASURES)):
                np.save(f"./output/{args['exp']}_{args['data']}_{args['ood_data']}_{EU_MEASURES[i]}_{args['uncertainty']}.npy", aurocs[i])
        else:
            aurocs = np.empty(args['runs'])
            for i in tqdm(range(args['runs'])):
                loader_id, loader_ood = get_data_ood(args['data'], args['ood_data'])
                if args['uncertainty'] == "dropout":
                    model = torch.load(CHECK_DIR + paths[0], map_location=device).to(device)
                    print(f"Model {paths[0]} loaded")
                else:
                    model = torch.load(CHECK_DIR + paths[i], map_location=device).to(device)
                    print(f"Model {paths[i]} loaded")
                if args['uncertainty'] == "ensemble":
                    for m in model.members:
                        m.eval().to(device)
                model.eval()
                u_id, u_ood, auroc = experiments.out_of_distribution(model, loader_id, loader_ood, args['measure'], device=device)
                aurocs[i] = auroc
            print(f"AUROC: mean={np.round(np.mean(aurocs), 3)}, std={np.round(np.std(aurocs), 3)}")
            results = json.JSONEncoder().encode(aurocs.tolist())
            result_processor.process_results({'aurocs': results})

    else:
        raise ValueError("Invalid experiment name")


if __name__ == '__main__':
    pyexp = PyExperimenter('./expconfig.yml', use_codecarbon=False)
    pyexp.fill_table_from_config()
    pyexp.reset_experiments('error')
    pyexp.reset_experiments('created')
    pyexp.reset_experiments('running')
    pyexp.execute(main)
