import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import ColorConverter
import seaborn as sns

from models.memory.similarity.lca import LCA
from utils import savefig


plt.rcParams['font.size'] = 14


results_dir = Path("./experiments/RL/Noise/figures/ValueMemoryGRU")

exp_names = []
exp_names_nostepbetween = []
for noise in ["0", "02", "04", "06", "08", "1"]:
    for gamma in ["00", "03", "06", "09"]:
        exp_names.append("setup_noise{}_gamma{}".format(noise, gamma))
        exp_names_nostepbetween.append("setup_noise{}_gamma{}_nostepbetween".format(noise, gamma))

""" count accuracy of each hyperparam setting """
for noise in ["0", "02", "04", "06", "08", "1"]:
    for gamma1 in ["00", "03", "06", "09"]:
        exp_name = "setup_noise{}_gamma{}".format(noise, gamma1)
        accuracy = 0.0
        forward_asymmetry = 0.0
        temporal_factor = 0.0
        cnt = 0
        for i in range(10):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    if float(row[0]) >= 0.5:
                        cnt += 1
                        accuracy += float(row[0])
                        forward_asymmetry += float(row[1])
                        temporal_factor += float(row[2])
        if cnt == 0:
            cnt = 1
        print(exp_name, np.round(accuracy / cnt, 4), np.round(forward_asymmetry / cnt, 4), np.round(temporal_factor / cnt, 4))
    print()

    for gamma2 in ["00", "03", "06", "09"]:
        exp_name = "setup_noise{}_gamma{}_nostepbetween".format(noise, gamma2)
        accuracy = 0.0
        forward_asymmetry = 0.0
        temporal_factor = 0.0
        cnt = 0
        for i in range(10):
            with open(results_dir / exp_name / str(i) / "contiguity_effect.csv") as f:
                reader = csv.reader(f)
                for row in reader:
                    if float(row[0]) >= 0.5:
                        cnt += 1
                        accuracy += float(row[0])
                        forward_asymmetry += float(row[1])
                        temporal_factor += float(row[2])
        if cnt == 0:
            cnt = 1
        print(exp_name, np.round(accuracy / cnt, 4), np.round(forward_asymmetry / cnt, 4), np.round(temporal_factor / cnt, 4))
    print()
    print()
