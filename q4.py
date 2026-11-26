"""Q4 (bonus): SVMs and clustering on ONE fixed representation.

Representation (fixed before any SVM result; env Q4_REP):
  ae  (default): 'bottleneck of the selected clean Q3 autoencoder' (label-free, so clustering it is not circular)
  cnn          : 'penultimate (global-pool) features of the Q1(a) CNN' (supervised features)
Features are standardised with the mean/std of the 45k TRAIN features (label-free).
SVM subset: ONE stratified 10,000-image subset of the 45k train split, drawn in phase `features` (before any model
selection, seed 26004), used for EVERY SVM comparison. Validation = the full 5,000 val split.
Clustering is fit on all 45k training representations.

Phases (in order):
  features        : extract + standardise representation, draw the SVM subset (test features saved separately)
  sweep           : (a) C x kernel sweep, train/val acc + support-vector fraction; selection by val acc only
  cluster_select  : (c) KMeans / diagonal GMM, several k; selection on val (rule stated below)
  final           : test evaluation of the selected SVM (once), hard pairs (b), clustering on test (c), comparison
Usage: CUDA_VISIBLE_DEVICES=0 NJOBS=8 python q4.py features|sweep|cluster_select|final
"""
import sys, csv, time, warnings
from common import *
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (confusion_matrix, adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, davies_bouldin_score)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels
from joblib import Parallel, delayed

REP = os.environ.get("Q4_REP", "ae")
SUB_N = 10000
CS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]                 # 6 values, 5 orders of magnitude
KERNELS = {"linear": dict(kernel="linear"),
           "rbf": dict(kernel="rbf", gamma="scale"),
           "poly": dict(kernel="poly", degree=3, gamma="scale", coef0=1.0)}
MAXIT = 400_000                                             # libsvm iteration cap; non-converged runs are flagged
KS = [5, 8, 10, 12, 15, 20, 30]
CLS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
PAIRS = [(i, j) for i in range(10) for j in range(i + 1, 10)]
PIDX = {p: k for k, p in enumerate(PAIRS)}                  # column of sklearn's 'ovo' decision_function
phase = sys.argv[1]
for p in ("runs/q4", "results", "figs"): os.makedirs(p, exist_ok=True)
FT, FTT = f"runs/q4/feats_{REP}.npz", f"runs/q4/feats_test_{REP}.npz"


def load_trval():
    z = np.load(FT)
    return z["Xtr"], z["ytr"], z["Xva"], z["yva"], z["sub"]


def run_one(kname, C, Xs, ys, Xv, yv):                      # top-level, numpy/sklearn only (joblib worker)
    t = time.time()
    clf = SVC(C=C, cache_size=1500, max_iter=MAXIT, **KERNELS[kname]).fit(Xs, ys)
    nsv = int(clf.n_support_.sum()); it = int(np.max(clf.n_iter_))
    bounded = (np.abs(clf.dual_coef_) >= C - 1e-6).any(0).sum()
    return dict(kernel=kname, C=C, train_acc=float((clf.predict(Xs) == ys).mean()), val_acc=float((clf.predict(Xv) == yv).mean()),
                sv_frac=nsv / len(ys), bounded_sv_frac=float(bounded) / len(ys), n_iter=it, converged=it < MAXIT,
                fit_seconds=time.time() - t)


def fit_selected(Xs, ys):
    s = json.load(open("runs/q4/selected_svm.json"))
    clf = SVC(C=s["C"], cache_size=1500, max_iter=MAXIT, decision_function_shape="ovo", **KERNELS[s["kernel"]]).fit(Xs, ys)
    return s, clf


if phase == "features":
    d = get_data(with_test=True)
    if REP == "ae":
        z = json.load(open("runs/q3/selected.json"))["z"]
        net = ConvAE(z).to(DEV); net.load_state_dict(torch.load(f"ckpt/q3/ae_{z}.pt")); net.eval()
        f = lambda xb: net.encode(to01(xb))
    else:
        net = MODELS["CNN"][0]().to(DEV); net.load_state_dict(torch.load(f"ckpt/q1a/CNN_s{SEED}.pt")); net.eval()
        f = lambda xb: net.features(xb).mean((2, 3))

    @torch.no_grad()
    def extract(x): return torch.cat([f(x[i:i + 2000]).float().cpu() for i in range(0, len(x), 2000)]).numpy()

    Xtr, Xva, Xte = extract(d.xtr), extract(d.xva), extract(d.xte)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    ytr, yva, yte = d.ytr.cpu().numpy(), d.yva.cpu().numpy(), d.yte.cpu().numpy()
    sub = np.sort(train_test_split(np.arange(len(ytr)), train_size=SUB_N, stratify=ytr, random_state=SEED)[0])
    np.savez(FT, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, sub=sub)
    np.savez(FTT, Xte=Xte, yte=yte)                         # only loaded in phase `final`
    np.save("splits/q4_subset_idx.npy", sub)
    print(f"representation={REP} dim={Xtr.shape[1]} subset={len(sub)} (per-class {np.bincount(ytr[sub])})")

elif phase == "sweep":
    Xtr, ytr, Xva, yva, sub = load_trval(); Xs, ys = Xtr[sub], ytr[sub]
    jobs = [(k, C) for k in KERNELS for C in CS]
    rows = Parallel(n_jobs=int(os.environ.get("NJOBS", 6)), verbose=5)(delayed(run_one)(k, C, Xs, ys, Xva, yva) for k, C in jobs)
    with open("results/q4a_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    best = max(rows, key=lambda r: (r["val_acc"], -r["C"]))       # val acc only; ties -> smaller C
    save_json(dict(kernel=best["kernel"], C=best["C"], val_acc=best["val_acc"], rep=REP), "runs/q4/selected_svm.json")
    with open("results/q4a_sweep.md", "w") as f:
        f.write("| kernel | C | train acc | val acc | SV fraction | bounded-SV fraction | converged |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['kernel']} | {r['C']:g} | {100*r['train_acc']:.2f} | {100*r['val_acc']:.2f} | {r['sv_frac']:.3f} | {r['bounded_sv_frac']:.3f} | {r['converged']} |\n")
    print(open("results/q4a_sweep.md").read()); print("selected:", best["kernel"], best["C"], best["val_acc"])

elif phase == "cluster_select":
    Xtr, ytr, Xva, yva, sub = load_trval(); rows = []
    for method in ("kmeans", "gmm_diag"):
        for k in KS:
            m = (KMeans(k, n_init=10, random_state=SEED) if method == "kmeans" else
                 GaussianMixture(k, covariance_type="diag", random_state=SEED, max_iter=200, reg_covar=1e-3))
            m.fit(Xtr)                                      # labels never used in fitting
            lv = m.predict(Xva)
            sil = silhouette_score(Xva, lv) if len(set(lv)) > 1 else float("nan")
            rows.append(dict(method=method, k=k, val_silhouette=float(sil), val_davies_bouldin=float(davies_bouldin_score(Xva, lv)),
                             val_ari=float(adjusted_rand_score(yva, lv)), val_nmi=float(normalized_mutual_info_score(yva, lv))))
            print(rows[-1], flush=True)
    # Selection rule (declared before looking): highest validation ARI. ARI uses VAL labels only for scoring a clustering
    # that was fitted without labels; unlike NMI it does not reward simply increasing k.
    best = max(rows, key=lambda r: r["val_ari"])
    save_json(dict(rows=rows, selected=dict(method=best["method"], k=best["k"]), rule="max validation ARI"), "runs/q4/cluster_select.json")
    with open("results/q4c_select.md", "w") as f:
        f.write("| method | k | val silhouette | val Davies-Bouldin | val ARI | val NMI |\n|---|---|---|---|---|---|\n")
        for r in rows: f.write(f"| {r['method']} | {r['k']} | {r['val_silhouette']:.3f} | {r['val_davies_bouldin']:.3f} | {r['val_ari']:.4f} | {r['val_nmi']:.4f} |\n")
    print(open("results/q4c_select.md").read()); print("selected:", best["method"], best["k"])

elif phase == "final":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = get_data(with_test=True)                            # raw images for the figures only
    Xtr, ytr, Xva, yva, sub = load_trval(); Xs, ys = Xtr[sub], ytr[sub]
    T = np.load(FTT); Xte, yte = T["Xte"], T["yte"]
    img = lambda x: np.clip(to01(x[None])[0].permute(1, 2, 0).cpu().numpy(), 0, 1)

    # ---------------- (a) selected SVM, ONE test evaluation
    sel, clf = fit_selected(Xs, ys)
    pred = clf.predict(Xte); test_acc = float((pred == yte).mean())
    conf = confusion_matrix(yte, pred, labels=range(10)); recall = conf.diagonal() / conf.sum(1)
    dec_te = clf.decision_function(Xte)                     # (n, 45) pairwise scores, >0 means the first class of the pair
    save_json(dict(selected=sel, test_acc=test_acc, per_class_recall=recall.tolist(), confusion=conf.tolist()), "results/q4a_final.json")
    print(f"selected SVM {sel}  TEST ACC {100*test_acc:.2f}")

    # ---------------- (b) hard pairs
    off = conf + conf.T; np.fill_diagonal(off, 0)
    order = sorted(PAIRS, key=lambda p: -off[p]); HARD = order[:4]
    C = sel["C"]; kp = KERNELS[sel["kernel"]]
    gamma = getattr(clf, "_gamma", 1.0 / (Xs.shape[1] * Xs.var()))
    kfun = {"linear": dict(metric="linear"), "rbf": dict(metric="rbf", gamma=gamma),
            "poly": dict(metric="poly", degree=3, gamma=gamma, coef0=1.0)}[sel["kernel"]]
    start = np.concatenate([[0], np.cumsum(clf.n_support_)])

    def pair_svs(i, j):                                     # support vectors of the (i,j) sub-classifier
        out = []
        for a, b in ((i, j), (j, i)):
            blk = np.arange(start[a], start[a + 1]); row = b - 1 if b > a else b
            coef = clf.dual_coef_[row, blk]; nz = np.abs(coef) > 1e-12
            out.append((clf.support_[blk[nz]], coef[nz]))
        return out

    lines = ["| pair | test pair-errors | mean abs score of errors | errors with abs score<0.5 | #pair SVs | bounded SVs | margin 1/||w|| | centroid dist | mean radius (i,j) | dist/(r_i+r_j) | kNN cross-class rate |",
             "|" + "---|" * 11]
    pair_stats = {}
    for (i, j) in HARD:
        p = PIDX[(i, j)]; (svi, ci), (svj, cj) = pair_svs(i, j)
        coefs = np.concatenate([ci, cj]); allsv = np.concatenate([svi, svj])
        K = pairwise_kernels(Xs[allsv], **kfun); margin = 1.0 / np.sqrt(max(coefs @ K @ coefs, 1e-12))
        bounded = float((np.abs(coefs) >= C - 1e-6).mean())
        mask = (yte == i) | (yte == j); s = dec_te[mask, p]; yt = yte[mask]
        ppred = np.where(s > 0, i, j); err = ppred != yt
        Xi, Xj = Xs[ys == i], Xs[ys == j]; mi, mj = Xi.mean(0), Xj.mean(0)
        ri, rj = np.linalg.norm(Xi - mi, axis=1).mean(), np.linalg.norm(Xj - mj, axis=1).mean(); dist = np.linalg.norm(mi - mj)
        tr_mask = (ys == i) | (ys == j); nn_ = NearestNeighbors(n_neighbors=10).fit(Xs[tr_mask])
        nb = nn_.kneighbors(Xte[mask], return_distance=False); cross = float((ys[tr_mask][nb] != yt[:, None]).mean())
        pair_stats[f"{CLS[i]}-{CLS[j]}"] = dict(pair_err=int(err.sum()), n=int(mask.sum()), margin=float(margin), bounded=bounded,
                                                n_sv=int(len(allsv)), dist=float(dist), ri=float(ri), rj=float(rj), knn_cross=cross)
        lines.append(f"| {CLS[i]} vs {CLS[j]} | {err.sum()}/{mask.sum()} | {np.abs(s[err]).mean() if err.any() else float('nan'):.3f} | "
                     f"{(np.abs(s[err]) < .5).mean() if err.any() else float('nan'):.2f} | {len(allsv)} | {bounded:.2f} | {margin:.3f} | "
                     f"{dist:.2f} | {ri:.2f},{rj:.2f} | {dist/(ri+rj):.3f} | {cross:.3f} |")
        # figure: support vectors of each class (largest |alpha| first) and the test images closest to the boundary
        fig = plt.figure(figsize=(16, 8.5)); gs = fig.add_gridspec(4, 8)
        def put(r, c, x, title, col="k"):
            a = fig.add_subplot(gs[r, c]); a.imshow(img(x)); a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=7, color=col)
        for r, (cls_, sv, cf) in enumerate(((i, svi, ci), (j, svj, cj))):
            o = np.argsort(-np.abs(cf))[:8]
            sc = clf.decision_function(Xs[sv[o]])[:, p] if len(o) else []
            for c, (idx, a_, s_) in enumerate(zip(sv[o], cf[o], sc)):
                put(r, c, d.xtr[sub[idx]], f"SV {CLS[cls_]}\nscore {s_:+.2f}" + (" B" if abs(a_) >= C - 1e-6 else ""))
        ids = np.where(mask)[0]; near = ids[np.argsort(np.abs(dec_te[ids, p]))[:16]]
        for n_, ti in enumerate(near):
            ok = pred[ti] == yte[ti]
            put(2 + n_ // 8, n_ % 8, d.xte[ti], f"test: {CLS[yte[ti]]}\npred {CLS[pred[ti]]}  s={dec_te[ti, p]:+.2f}", "g" if ok else "r")
        fig.suptitle(f"{CLS[i]} (score>0) vs {CLS[j]}: rows 1-2 training support vectors (B = bounded, alpha=C); rows 3-4 test images nearest the boundary (green=correct, red=wrong, multi-class prediction)", fontsize=9)
        plt.tight_layout(); plt.savefig(f"figs/q4b_{CLS[i]}_{CLS[j]}.pdf"); plt.close()
    open("results/q4b_pairs.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
    save_json(dict(hard_pairs=[f"{CLS[i]}-{CLS[j]}" for i, j in HARD], stats=pair_stats), "results/q4b_pairs.json")

    # ---------------- (c) clustering fit on TRAIN reps, assign TEST reps, labels used only afterwards
    cs = json.load(open("runs/q4/cluster_select.json"))["selected"]; k = cs["k"]
    m = (KMeans(k, n_init=10, random_state=SEED) if cs["method"] == "kmeans" else
         GaussianMixture(k, covariance_type="diag", random_state=SEED, max_iter=200, reg_covar=1e-3))
    m.fit(Xtr); lab = m.predict(Xte)
    ari, nmi = float(adjusted_rand_score(yte, lab)), float(normalized_mutual_info_score(yte, lab))
    cont = np.zeros((10, k), dtype=int)
    for a, b in zip(yte, lab): cont[a, b] += 1
    np.savetxt("results/q4c_contingency.csv", cont, fmt="%d", delimiter=",")
    # per-class: SVM recall vs clustering (majority cluster) recall / precision
    maj = cont.argmax(1); crec = cont[np.arange(10), maj] / cont.sum(1); cprec = cont[np.arange(10), maj] / cont[:, maj].sum(0)
    # pair level: SVM pairwise accuracy vs best possible pairwise accuracy using ONLY the cluster id
    P = cont / cont.sum(1, keepdims=True); pair_rows = []
    for (i, j) in PAIRS:
        mk = (yte == i) | (yte == j); sp = np.where(dec_te[mk, PIDX[(i, j)]] > 0, i, j); svm_pa = float((sp == yte[mk]).mean())
        clu_pa = 1.0 - np.minimum(P[i], P[j]).sum() / 2.0   # 0.5 = clusters cannot tell i from j, 1.0 = perfectly separate
        pair_rows.append(dict(pair=f"{CLS[i]}-{CLS[j]}", svm_pair_acc=svm_pa, cluster_pair_acc=float(clu_pa), gap=svm_pa - float(clu_pa),
                              shared_majority_cluster=bool(maj[i] == maj[j])))
    pair_rows.sort(key=lambda r: -r["gap"])
    save_json(dict(method=cs["method"], k=k, test_ari=ari, test_nmi=nmi, svm_test_acc=test_acc, contingency=cont.tolist(),
                   per_class=[dict(cls=CLS[c], svm_recall=float(recall[c]), majority_cluster=int(maj[c]), cluster_recall=float(crec[c]),
                                   cluster_precision=float(cprec[c])) for c in range(10)], pairs=pair_rows), "results/q4c_final.json")
    with open("results/q4c_summary.md", "w") as f:
        f.write(f"clustering: {cs['method']} k={k} (fit on train reps)  TEST ARI {ari:.4f}  NMI {nmi:.4f}   | SVM test acc {100*test_acc:.2f}\n\n")
        f.write("| class | SVM recall | majority cluster | cluster recall | cluster precision |\n|---|---|---|---|---|\n")
        for c in range(10): f.write(f"| {CLS[c]} | {recall[c]:.3f} | {maj[c]} | {crec[c]:.3f} | {cprec[c]:.3f} |\n")
        f.write("\n| class pair | SVM pairwise acc | best acc from cluster id alone | gap | same majority cluster |\n|---|---|---|---|---|\n")
        for r in pair_rows[:10]: f.write(f"| {r['pair']} | {r['svm_pair_acc']:.3f} | {r['cluster_pair_acc']:.3f} | {r['gap']:+.3f} | {r['shared_majority_cluster']} |\n")
        f.write("\ncontingency (rows = true class, cols = cluster):\n\n```\n" + "\n".join(" ".join(f"{v:4d}" for v in row) for row in cont) + "\n```\n")
    print(open("results/q4c_summary.md").read())
