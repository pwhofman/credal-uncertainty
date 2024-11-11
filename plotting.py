import numpy as np
import matplotlib.pyplot as plt, mpld3
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import datetime
import seaborn as sns
tab10 = plt.get_cmap('tab10')
sns.set_theme(context='paper', style='whitegrid', font_scale=1.2)
# plt.rcParams['figure.figsize'] = [5, 5]
plt.rcParams['axes.edgecolor'] = 'black'
# make axis text larger, but not legend
plt.rcParams['axes.labelsize'] = 12
# same for title
plt.rcParams['axes.titlesize'] = 12
# TIME = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

title_dict = {"imagenet": "ImageNet", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "dropout": "Dropout", "edl" : "Evidential", "laplace" : "Laplace", "fmnist" : "FashionMNIST", "food101" : "Food101", "rf" : "RandomForest", "svhn" : "SVHN", "places365" : "Places"}
colors_dict = {"log" : "#ff7f0e", "brier" : "#2ca02c", "spherical" : "#d62728", "zero-one" : "#9467bd", "entropy" : "#8c564b", "random" : "#7f7f7f"}

data = "food101"
uncertainty = "ensemble"
accs = np.load(f"./output/arc_{data}_entropy_{uncertainty}.npy")
accs1 = np.load(f"./output/arc_{data}_log_{uncertainty}.npy")
# accs1 = np.load(f"./output/arc_{data}_ogbrier_{uncertainty}.npy")
accs2 = np.load(f"./output/arc_{data}_brier_{uncertainty}.npy")
accs3 = np.load(f"./output/arc_{data}_spherical_{uncertainty}.npy")
# accs3 = np.load(f"./output/arc_{data}_og01_{uncertainty}.npy")
accs4 = np.load(f"./output/arc_{data}_01_{uncertainty}.npy")

tu = accs[:, 1, :]
eu = accs[:, 4, :]
au = accs[:, 3, :]
rand = accs[:, 5, :]
methods = np.ones(tu.shape[1])
baselines = np.ones(tu.shape[1])
# best = accs[:, 4, :]
# best = np.ones((tu.shape[0], tu.shape[1]))

tu1 = accs1[:, 1, :]
eu1 = accs1[:, 4, :]
au1 = accs1[:, 3, :]

tu2 = accs2[:, 1, :]
eu2 = accs2[:, 4, :]
au2 = accs2[:, 3, :]

tu3 = accs3[:, 1, :]
eu3 = accs3[:, 4, :]
au3 = accs3[:, 3, :]

tu4 = accs4[:, 1, :]
eu4 = accs4[:, 4, :]
au4 = accs4[:, 3, :]


portion_vals = np.linspace(0, 1, 50, endpoint=False)

# fig, (plt, plt, plt) = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(20, 4.5))

plt.plot(portion_vals * 100, methods, label=r"Ours", linestyle="-", alpha=0)
plt.plot(portion_vals * 100, au1.mean(axis=0), label=r"log", linestyle="-", color=colors_dict["log"])
plt.fill_between(portion_vals * 100, au1.mean(axis=0) - au1.std(axis=0),
                 au1.mean(axis=0) + au1.std(axis=0), alpha=0.2, color=colors_dict["log"])
plt.plot(portion_vals * 100, au2.mean(axis=0), label=r"brier", linestyle="-", color=colors_dict["brier"])
plt.fill_between(portion_vals * 100, au2.mean(axis=0) - au2.std(axis=0),
                 au2.mean(axis=0) + au2.std(axis=0), alpha=0.2, color=colors_dict["brier"])
plt.plot(portion_vals * 100, au3.mean(axis=0), label=r"spherical", linestyle="-", color=colors_dict["spherical"])
plt.fill_between(portion_vals * 100, au3.mean(axis=0) - au3.std(axis=0),
                 au3.mean(axis=0) + au3.std(axis=0), alpha=0.2, color=colors_dict["spherical"])
plt.plot(portion_vals * 100, au4.mean(axis=0), label=r"zero-one", linestyle="-", color=colors_dict["zero-one"])
plt.fill_between(portion_vals * 100, au4.mean(axis=0) - au4.std(axis=0),
                 au4.mean(axis=0) + au4.std(axis=0), alpha=0.2, color=colors_dict["zero-one"])
# plt.plot(portion_vals * 100, best.mean(axis=0), label=r"oracle", linestyle="--")
plt.plot(portion_vals * 100, baselines, label=r"Baselines", linestyle="-", alpha=0)
# plt.plot(portion_vals * 100, au.mean(axis=0), label=r"entropy", linestyle="--", color=colors_dict["entropy"])
# plt.fill_between(portion_vals * 100, au.mean(axis=0) - au.std(axis=0),
#                  au.mean(axis=0) + au.std(axis=0), alpha=0.2, color=colors_dict["entropy"])
plt.plot(portion_vals * 100, rand.mean(axis=0), label=r"random", linestyle="--", color=colors_dict["random"])
plt.fill_between(portion_vals * 100, rand.mean(axis=0) - rand.std(axis=0), rand.mean(axis=0) + rand.std(axis=0),
                 alpha=0.2, color=colors_dict["random"])
plt.legend()
ax = plt.gca()
# ax.yaxis.set_major_locator(MultipleLocator(0.05))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.xlabel("Percentage Rejected")
plt.ylabel("Accuracy")
legend = plt.legend()
legend.get_texts()[0].set_fontweight('bold')
legend.get_texts()[5].set_fontweight('bold')
plt.title(f"Upper AU on {title_dict[data]}")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout(pad=0)
plt.savefig(f"./plots/arc_au_upper_{data}_{uncertainty}.pdf", bbox_inches='tight', pad_inches=0)
# mpld3.show()
# fig3 = mpld3.fig_to_html(plt.figure())
plt.show()
#
# mpld3._server.serve(fig1+fig2+fig3)