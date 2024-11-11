import numpy as np

MEASURES = ['log', 'brier', 'spherical', '01', 'gh', 'diff']
data = 'food101'
ood_data = 'cifar100'
unc = 'ensemble'
for m in MEASURES:
    aurocs = np.load(f"./output/ood_{data}_{ood_data}_{m}_{unc}.npy")
    print(f"$ {aurocs.mean().round(3)} \scriptstyle{{\pm {aurocs.std().round(3)}}} $", end=' & ')