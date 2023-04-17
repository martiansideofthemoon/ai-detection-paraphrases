import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.rcParams.update({'font.size': 16})

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
max_fpr = 100


for i, detector in enumerate(["detectgpt", "gptzero", "openai", "watermark", "s2"]):
    with open(f"detect-plots/{detector}.pkl", 'rb') as f:
        stats1, stats2 = pickle.load(f)
    stats1_x = [100 * x for x in stats1[0]]
    stats1_y = [100 * x for x in stats1[1]]
    stats2_x = [100 * x for x in stats2[0]]
    stats2_y = [100 * x for x in stats2[1]]

    print(f"{detector} has {len(stats2_x)} points")
    if detector != "s2":
        plt.plot(stats1_x, stats1_y, label=f"{detector}", color=colors[i], linewidth=2)
    else:
        detector = "retrieval"

    plt.plot(stats2_x, stats2_y, label=f"{detector} (pp)", linestyle='--', color=colors[i], linewidth=2)

    if detector == "openai":
        # plot y = x from x = 0 to x = max_fpr
        plt.plot([0, 100], [0, 100], label="random", color='black', linestyle=':')

max_fpr = 100

plt.ylim([0, 100])
plt.xlim([0, max_fpr])
plt.legend(loc='upper left', bbox_to_anchor=(-0.03, 1.3), ncol=3, prop={'size': 11})

plt.xlabel('FPR (%)')
plt.ylabel('TPR (%)')
plt.savefig(f"detect-plots/roc_fpr_{max_fpr}.pdf", bbox_inches='tight')