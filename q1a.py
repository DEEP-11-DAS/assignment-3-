"""Q1(a): isolate 'MLP lacks spatial inductive bias (locality + weight sharing)' from two rivals:
   R1 'the MLP simply has too little capacity'    -> MLP-wide (5.3x params)
   R2 'the MLP simply had too little training'     -> MLP-long (3x epochs)
   Core matched pair: MLP vs CNN (same params +-0.5%, same optimiser, epochs, augmentation, wd,
   same LR-grid tuned on VALIDATION only). CNN-tiny (4% of the params) is a further capacity probe:
   if it beats the big MLPs, parameter count cannot be the explanation.

Phase 'tune'  : every model x LR grid, one seed, validation only.   (no test data loaded)
Phase 'final' : best LR per model (by val acc from tune logs) x 3 seeds; evaluate the
                best-val-epoch checkpoint ONCE on the official test set.
LR grid extended to 1e-4..1e-2 because the first grid's best LRs sat on its edges (see report).
Usage: CUDA_VISIBLE_DEVICES=0 python q1a.py tune ; python q1a.py final
"""
import sys, glob
from common import *

phase = sys.argv[1]
os.makedirs("runs/q1a", exist_ok=True); os.makedirs("ckpt/q1a", exist_ok=True)

if phase == "tune":
    d = get_data()
    for name, (mk, ep) in MODELS.items():
        for lr in LRS:
            f = f"runs/q1a/tune_{name}_lr{lr}.json"
            if os.path.exists(f): continue
            set_seed(SEED); m = mk()
            h, _, be = train(m, d.xtr, d.ytr, d.xva, d.yva, ep, lr, WD, BS, tag=f"{name} lr{lr}")
            save_json(dict(name=name, lr=lr, params=count_params(m), epochs=ep, best_epoch=be,
                           final_val=h[-1]["val_acc"], best_val=max(r["val_acc"] for r in h), hist=h), f)

elif phase == "final":
    d = get_data(with_test=True)
    rows = []
    for name, (mk, ep) in MODELS.items():
        runs = [json.load(open(f)) for f in glob.glob(f"runs/q1a/tune_{name}_lr*.json")]
        assert len(runs) == len(LRS), f"run tune first ({name})"
        lr = max(runs, key=lambda r: r["best_val"])["lr"]      # selection on validation only
        vals, tests, hists = [], [], []
        for s in (SEED, SEED + 1, SEED + 2):
            set_seed(s); m = mk()
            h, st, be = train(m, d.xtr, d.ytr, d.xva, d.yva, ep, lr, WD, BS, seed=s, tag=f"{name} s{s}")
            m.load_state_dict(st); m.to(DEV)
            _, te = loss_acc(predict(m, d.xte), d.yte)           # single test evaluation per model/seed
            vals.append(max(r["val_acc"] for r in h)); tests.append(te); hists.append(h)
            torch.save(st, f"ckpt/q1a/{name}_s{s}.pt")
        save_json(dict(name=name, lr=lr, params=count_params(m), epochs=ep, val=vals, test=tests,
                       hist_seed0=hists[0], hists=hists), f"runs/q1a/final_{name}.json")
        rows.append((name, count_params(m), ep, lr, np.mean(vals), np.std(vals), np.mean(tests), np.std(tests)))
    os.makedirs("results", exist_ok=True)
    with open("results/q1a_table.md", "w") as f:
        f.write("| model | params | epochs | lr | val acc (mean±std, 3 seeds) | test acc (mean±std) |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]:,} | {r[2]} | {r[3]} | {100*r[4]:.2f}±{100*r[5]:.2f} | {100*r[6]:.2f}±{100*r[7]:.2f} |\n")
    print(open("results/q1a_table.md").read())
