import numpy as np
import matplotlib.pyplot as plt, mpld3
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import datetime
import seaborn as sns
tab10 = plt.get_cmap('tab10')
sns.set_theme(context='paper', style='whitegrid', font_scale=1.2)
# fix the size to be squrae
# plt.rcParams['figure.figsize'] = [5, 5]
plt.rcParams['axes.edgecolor'] = 'black'
# make axis text larger, but not legend
plt.rcParams['axes.labelsize'] = 12
# same for title
plt.rcParams['axes.titlesize'] = 12
# TIME = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

title_dict = {"kmnist" : "KMNIST", "mnist" : "MNIST", "dna" : "DNA", "vehicle" : "Vehicle", "imagenet": "ImageNet", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "dropout": "Dropout", "edl" : "Evidential", "laplace" : "Laplace", "fmnist" : "FashionMNIST", "food101" : "Food101", "rf" : "RandomForest", "svhn" : "SVHN", "places365" : "Places",
              "electricity" : "Electricity"}
colors_dict = {"log" : "#ff7f0e", "brier" : "#2ca02c", "spherical" : "#d62728", "zero-one" : "#9467bd", "entropy" : "#8c564b", "random" : "#7f7f7f"}


data = "mnist"
accs = np.load(f"./output/al_{data}_log.npy")
accs3 = np.load(f"./output/al_{data}_01.npy")

print(accs.shape)

portion_vals = np.linspace(100, 2000, 15, endpoint=True)
# portion_vals = np.linspace(100, 2000, 15, endpoint=False)


# methods = np.full(portion_vals.shape[0], 0.7)
# baselines = np.full(portion_vals.shape[0], 0.7)
# plt.plot(portion_vals, methods, label=r"Ours", linestyle="-", alpha=0)
plt.plot(portion_vals, accs.mean(axis=0), label=r"log", linestyle="-", color=colors_dict["log"])
plt.fill_between(portion_vals, accs.mean(axis=0) - accs.std(axis=0), accs.mean(axis=0) + accs.std(axis=0), alpha=0.2, color=colors_dict["log"])
plt.plot(portion_vals, accs3.mean(axis=0), label=r"zero-one", linestyle="-", color=colors_dict["zero-one"])
plt.fill_between(portion_vals, accs3.mean(axis=0) - accs3.std(axis=0), accs3.mean(axis=0) + accs3.std(axis=0), alpha=0.2, color=colors_dict["zero-one"])

plt.legend()
# ax = plt.gca()
# ax.yaxis.set_major_locator(MultipleLocator(0.05))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xlabel("Train Instances")
plt.ylabel("Accuracy")
legend = plt.legend()
# legend.get_texts()[0].set_fontweight('bold')
# legend.get_texts()[5].set_fontweight('bold')
plt.title(f"Active Learning on {title_dict[data]}")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout(pad=0)
plt.savefig(f"./plots/al_{data}.pdf", bbox_inches='tight', pad_inches=0)
mpld3.show()

