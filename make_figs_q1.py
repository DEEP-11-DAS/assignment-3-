"""Figures and summary tables for Q1 (reads only saved logs)."""
import glob, json, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True); os.makedirs("results", exist_ok=True)

# ---- Q1(a): train / val curves at the selected LR (seed 26004) + train-vs-val gap table
fs = sorted(glob.glob("runs/q1a/final_*.json"))
if fs:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    gap = ["| model | final train acc (clean, 5k subset) | final val acc | gap |", "|---|---|---|---|"]
    for f in fs:
        r = json.load(open(f)); h = r["hist_seed0"]; e = [x["epoch"] for x in h]
        lab = f"{r['name']} ({r['params']/1e6:.2f}M)"
        ax[0].plot(e, [x["train_acc_clean"] for x in h], label=lab)
        ax[1].plot(e, [x["val_acc"] for x in h], label=lab)
        ax[2].plot(e, [x["val_loss"] for x in h], label=lab)
        t, v = h[-1]["train_acc_clean"], h[-1]["val_acc"]
        gap.append(f"| {r['name']} | {100*t:.1f} | {100*v:.1f} | {100*(t-v):.1f} |")
    for a, t in zip(ax, ["train acc (no aug, 5k subset)", "validation acc", "validation loss"]):
        a.set_title(t); a.set_xlabel("epoch"); a.grid(alpha=.3)
    ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig("figs/q1a_curves.pdf"); plt.close()
    open("results/q1a_gap.md", "w").write("\n".join(gap) + "\n"); print("\n".join(gap))

# ---- Q1(b): memorisation trajectories, mean +- std over seeds
runs = {}
for f in glob.glob("runs/q1b/p*.json"):
    r = json.load(open(f)); runs.setdefault(r["p"], []).append(r)
if runs:
    ps = sorted(runs)
    fig, ax = plt.subplots(2, len(ps), figsize=(5 * len(ps), 7), squeeze=False)
    ms = lambda a: f"{np.nanmean(a):.3f}±{np.nanstd(a):.3f}" if not np.all(np.isnan(a)) else "n/a"
    ms_e = lambda a: f"{np.mean(a):.1f}±{np.std(a):.1f}" if len(a) else "n/a"
    first = lambda h, k, t: next((x["epoch"] for x in h if x[k] >= t), np.nan)
    L = ["| p | seeds | peak val acc | peak val epoch | val final | onset epoch (fit_corrupt>=0.2) | epoch fit_corrupt>=0.9 | peak true_corrupt | peak true_corrupt epoch | true_corrupt final | fit_corrupt final |", "|" + "---|" * 11]
    E1 = [1, 2, 3, 4, 5, 6, 8, 10]
    L2 = ["| p | " + " | ".join(f"ep{e}" for e in E1) + " |", "|" + "---|" * (len(E1) + 1)]
    for c, p in enumerate(ps):
        rs = runs[p]; H = [r["hist"] for r in rs]; e = np.array([x["epoch"] for x in H[0]])
        for k, lab in [("val_acc", "val acc (true labels)"), ("fit_clean", "train: intact-label examples"),
                       ("fit_corrupt", "train: corrupted, vs RANDOM label (memorised)"),
                       ("true_corrupt", "train: corrupted, vs TRUE label")]:
            A = np.array([[x[k] for x in h] for h in H], dtype=float)
            if np.all(np.isnan(A)): continue
            mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
            col = {"val_acc": "C0", "fit_clean": "C1", "fit_corrupt": "C2", "true_corrupt": "C3"}[k]  # fixed colour per curve in every panel
            ax[0, c].plot(e, mu, label=lab, color=col); ax[0, c].fill_between(e, mu - sd, mu + sd, alpha=.2, color=col)
        ax[0, c].axhline(0.1, color="gray", ls=":", lw=1)
        ax[0, c].set_title(f"label noise p={p} ({len(rs)} seeds, {rs[0]['params']/1e6:.1f}M params)")
        ax[0, c].set_xlabel("epoch"); ax[0, c].set_ylabel("accuracy"); ax[0, c].grid(alpha=.3)
        ax[0, c].set_xscale("log")
        for k, lab in [("train_loss", "train loss (noisy labels)"), ("val_loss", "val loss")]:
            A = np.array([[x[k] for x in h] for h in H]); mu, sd = A.mean(0), A.std(0)
            ax[1, c].plot(e, mu, label=lab); ax[1, c].fill_between(e, mu - sd, mu + sd, alpha=.2)
        ax[1, c].set_xlabel("epoch"); ax[1, c].set_ylabel("loss"); ax[1, c].grid(alpha=.3); ax[1, c].set_xscale("log")
        if c == 0: ax[0, c].legend(fontsize=7); ax[1, c].legend(fontsize=7)
        va = [[x["val_acc"] for x in h] for h in H]
        pk = [int(np.argmax(v)) + 1 for v in va]; pv = [max(v) for v in va]
        tc = [[x["true_corrupt"] for x in h] for h in H]
        has = not np.all(np.isnan(tc))
        tpk = [int(np.nanargmax(t)) + 1 for t in tc] if has else []
        tpv = [np.nanmax(t) for t in tc] if has else [np.nan]
        ons = [first(h, "fit_corrupt", .2) for h in H] if has else []
        e90 = [first(h, "fit_corrupt", .9) for h in H] if has else []
        L.append(f"| {p} | {len(rs)} | {ms(pv)} | {ms_e(pk)} | {ms([v[-1] for v in va])} | {ms_e(ons)} | {ms_e(e90)} | "
                 f"{ms(tpv)} | {ms_e(tpk)} | {ms([t[-1] for t in tc]) if has else 'n/a'} | "
                 f"{ms([h[-1]['fit_corrupt'] for h in H]) if has else 'n/a'} |")
        if has:
            L2.append(f"| {p} | " + " | ".join(ms([h[e_ - 1]["fit_corrupt"] for h in H]) for e_ in E1) + " |")
    plt.tight_layout(); plt.savefig("figs/q1b_memorisation.pdf"); plt.savefig("figs/q1b_memorisation.png", dpi=130)
    out = "\n".join(L) + "\n\nfit_corrupt (accuracy on corrupted examples vs their random label) in the first epochs:\n\n" + "\n".join(L2) + "\n"
    open("results/q1b_summary.md", "w").write(out); print(out)
