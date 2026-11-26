"""Figures for Q3 (reads saved logs / results only)."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figs", exist_ok=True)

fs = sorted(glob.glob("runs/q3/ae_*.json"), key=lambda f: json.load(open(f))["z"])
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for f in fs:
    r = json.load(open(f)); e = [h["epoch"] for h in r["hist"]]
    ax[0].plot(e, [h["train_mse"] for h in r["hist"]], label=f"z={r['z']} ({r['params']/1e6:.2f}M)")
    ax[1].plot(e, [h["val_mse"] for h in r["hist"]], label=f"z={r['z']}")
for a, t in zip(ax, ["train MSE", "validation MSE"]): a.set_yscale("log"); a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig("figs/q3a_curves.pdf"); plt.close()

if os.path.exists("results/q3_examples.npz"):
    D = np.load("results/q3_examples.npz"); names = ["clean", "noisy", "clean_ae", "dae"]
    titles = ["clean", "corrupted", "clean-AE(corrupted)", "DAE(corrupted)"]
    fig, ax = plt.subplots(4, 10, figsize=(14, 6))
    for i, n in enumerate(names):
        for j in range(10):
            ax[i, j].imshow(np.clip(D[n][j].transpose(1, 2, 0), 0, 1)); ax[i, j].axis("off")
        ax[i, 0].set_title(titles[i], fontsize=8, loc="left")
    plt.tight_layout(); plt.savefig("figs/q3b_examples.pdf"); plt.close()

if os.path.exists("results/q3.json"):
    R = json.load(open("results/q3.json")); P = R["probes"]
    zs = sorted(int(z) for z in R["test_mse_clean_ae"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([R["test_mse_clean_ae"][str(z)] for z in zs], [100 * P[f"AE z={z}"]["test_acc"] for z in zs], "o-")
    for z in zs: ax[0].annotate(f"z={z}", (R["test_mse_clean_ae"][str(z)], 100 * P[f"AE z={z}"]["test_acc"]))
    ax[0].set_xlabel("test reconstruction MSE (lower = better)"); ax[0].set_ylabel("linear-probe test acc (%)"); ax[0].grid(alpha=.3)
    names = list(P); ax[1].barh(names, [100 * P[n]["test_acc"] for n in names]); ax[1].set_xlabel("linear-probe test acc (%)")
    plt.tight_layout(); plt.savefig("figs/q3c_recon_vs_probe.pdf")
