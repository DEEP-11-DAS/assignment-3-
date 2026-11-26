"""Figures for Q4 (reads results/*.csv|json only)."""
import csv, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("figs", exist_ok=True)
CLS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

rows = list(csv.DictReader(open("results/q4a_sweep.csv")))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, k in enumerate(sorted({r["kernel"] for r in rows})):
    R = sorted([r for r in rows if r["kernel"] == k], key=lambda r: float(r["C"])); C = [float(r["C"]) for r in R]
    ax[0].plot(C, [100 * float(r["val_acc"]) for r in R], "o-", color=f"C{i}", label=f"{k} val")
    ax[0].plot(C, [100 * float(r["train_acc"]) for r in R], "o--", color=f"C{i}", alpha=.5, label=f"{k} train")
    ax[1].plot(C, [float(r["sv_frac"]) for r in R], "o-", color=f"C{i}", label=k)
    ax[2].plot(C, [float(r["bounded_sv_frac"]) for r in R], "o-", color=f"C{i}", label=k)
for a, t in zip(ax, ["accuracy (%)", "support-vector fraction", "bounded-SV fraction (alpha = C)"]):
    a.set_xscale("log"); a.set_xlabel("C"); a.set_title(t); a.grid(alpha=.3); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("figs/q4a_sweep.pdf"); plt.close()

if os.path.exists("results/q4c_final.json"):
    R = json.load(open("results/q4c_final.json")); cont = np.array(R["contingency"])
    plt.figure(figsize=(1 + .5 * cont.shape[1], 5)); plt.imshow(cont / cont.sum(1, keepdims=True), cmap="viridis", aspect="auto")
    plt.yticks(range(10), CLS); plt.xlabel("cluster"); plt.colorbar(label="fraction of class"); 
    for i in range(10):
        for j in range(cont.shape[1]): plt.text(j, i, cont[i, j], ha="center", va="center", color="w", fontsize=6)
    plt.title(f"{R['method']} k={R['k']}: test ARI {R['test_ari']:.3f}, NMI {R['test_nmi']:.3f}")
    plt.tight_layout(); plt.savefig("figs/q4c_contingency.pdf")
