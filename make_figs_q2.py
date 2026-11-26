"""Figures for Q2 (reads results/q2c.json and runs/q2/perm_final_*.json)."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figs", exist_ok=True)
R = json.load(open("results/q2c.json"))
conds = [c for c in R if not c.startswith("_")]
fig, ax = plt.subplots(1, 3, figsize=(17, 4))
w = 0.38; x = np.arange(len(conds))
for i, name in enumerate(["MLP", "CNN"]):
    ax[0].bar(x + (i - .5) * w, [100 * R[c][name]["acc"] for c in conds], w, label=f"{name} accuracy")
    ax[1].bar(x + (i - .5) * w, [R[c][name]["conf_mean"] for c in conds], w, label=f"{name} mean confidence")
    ax[2].bar(x + (i - .5) * w, [R[c][name]["ece"] for c in conds], w, label=f"{name} ECE")
for a, t in zip(ax, ["test accuracy (%)", "mean max-softmax confidence", "expected calibration error"]):
    a.set_xticks(x); a.set_xticklabels(conds, rotation=35, ha="right"); a.set_title(t); a.grid(alpha=.3, axis="y"); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figs/q2c_conditions.pdf")
pc = R["_per_class_acc_CNN"]; cls = ["plane","car","bird","cat","deer","dog","frog","horse","ship","truck"]
plt.figure(figsize=(9, 4))
for i, c in enumerate(pc): plt.bar(np.arange(10) + (i - 1) * .27, pc[c], .27, label=c)
plt.xticks(range(10), cls, rotation=30); plt.ylabel("CNN per-class test accuracy"); plt.legend(); plt.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.savefig("figs/q2c_per_class.pdf")
