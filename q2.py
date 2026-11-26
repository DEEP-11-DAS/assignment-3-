"""Q2. Reuses MLP and CNN (and their checkpoints/LRs) from Q1(a) -- first introduced there.
Phases (run in this order):
  perm_tune   : (b) retrain MLP and CNN on spatially permuted pixels, LR grid, validation only.
  perm_final  : (b) best LR per model by val, 3 seeds, single test evaluation per run.
  conditions  : (c) needs q1a final. Test-time interventions on the Q1(a) CNN/MLP checkpoints (seed 26004),
                + one CNN trained on grayscale. Writes accuracy and confidence/error statistics + Q2 table.
Predictions are in q2_predictions.md and must be committed before running.
Usage: CUDA_VISIBLE_DEVICES=0 python q2.py perm_tune|perm_final|conditions
"""
import sys, glob
from common import *

phase = sys.argv[1]
PERM = np.random.RandomState(SEED).permutation(1024)      # ONE fixed permutation, saved below
NAMES = ["MLP", "CNN"]
os.makedirs("runs/q2", exist_ok=True); os.makedirs("ckpt/q2", exist_ok=True); os.makedirs("results", exist_ok=True)
os.makedirs("splits", exist_ok=True); np.save("splits/perm_pixels.npy", PERM)


def perm_model(name):
    return Permuted(MODELS[name][0](), PERM)


if phase == "perm_tune":
    d = get_data()
    for name in NAMES:
        for lr in LRS:
            f = f"runs/q2/tune_perm_{name}_lr{lr}.json"
            if os.path.exists(f): continue
            set_seed(SEED); m = perm_model(name)
            h, _, be = train(m, d.xtr, d.ytr, d.xva, d.yva, E, lr, WD, BS, tag=f"perm {name} lr{lr}")
            save_json(dict(name=name, lr=lr, params=count_params(m), best_val=max(r["val_acc"] for r in h),
                           best_epoch=be, hist=h), f)

elif phase == "perm_final":
    d = get_data(with_test=True)
    for name in NAMES:
        runs = [json.load(open(f)) for f in glob.glob(f"runs/q2/tune_perm_{name}_lr*.json")]
        assert len(runs) == len(LRS), "run perm_tune first"
        lr = max(runs, key=lambda r: r["best_val"])["lr"]
        vals, tests, hs = [], [], []
        for s in (SEED, SEED + 1, SEED + 2):
            set_seed(s); m = perm_model(name)
            h, st, be = train(m, d.xtr, d.ytr, d.xva, d.yva, E, lr, WD, BS, seed=s, tag=f"perm {name} s{s}")
            m.load_state_dict(st); m.to(DEV)
            tests.append(loss_acc(predict(m, d.xte), d.yte)[1]); vals.append(max(r["val_acc"] for r in h)); hs.append(h)
            torch.save(st, f"ckpt/q2/perm_{name}_s{s}.pt")
        save_json(dict(name=name, lr=lr, params=count_params(m), val=vals, test=tests, hist_seed0=hs[0]),
                  f"runs/q2/perm_final_{name}.json")
        print(name, "permuted test", np.mean(tests), np.std(tests))

elif phase == "conditions":
    d = get_data(with_test=True)
    g = torch.Generator(device=DEV); g.manual_seed(SEED)

    def block_shuffle(x, k):
        n, c, h, w = x.shape; b = h // k
        t = x.view(n, c, k, b, k, b).permute(0, 2, 4, 1, 3, 5).reshape(n, k * k, c, b, b)
        idx = torch.rand(n, k * k, device=x.device, generator=g).argsort(1)
        t = t[torch.arange(n, device=x.device)[:, None], idx]
        return t.reshape(n, k, k, c, b, b).permute(0, 3, 1, 4, 2, 5).reshape(n, c, h, w)

    def occlude_centre(x):
        x = x.clone(); x[:, :, 8:24, 8:24] = 0; return x          # 0 = dataset mean after normalisation

    def keep_centre(x):
        o = torch.zeros_like(x); o[:, :, 8:24, 8:24] = x[:, :, 8:24, 8:24]; return o

    CONDS = {"clean": lambda x: x, "gray": to_gray_norm, "shuffle2x2": lambda x: block_shuffle(x, 2),
             "shuffle4x4": lambda x: block_shuffle(x, 4), "shuffle8x8": lambda x: block_shuffle(x, 8),
             "centre_occluded": occlude_centre, "centre_only": keep_centre}
    XC = {k: f(d.xte) for k, f in CONDS.items()}                # each corrupted test set built once, shared by all models

    def stats(lg, y):
        p = lg.softmax(1); conf, pred = p.max(1); ok = pred == y
        ece, edges = 0.0, torch.linspace(0, 1, 16, device=lg.device)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mk = (conf > lo) & (conf <= hi)
            if mk.any(): ece += mk.float().mean().item() * abs(ok[mk].float().mean().item() - conf[mk].mean().item())
        cnt = torch.bincount(pred, minlength=10).float()
        return dict(acc=ok.float().mean().item(), conf_mean=conf.mean().item(),
                    conf_correct=conf[ok].mean().item() if ok.any() else float("nan"),
                    conf_wrong=conf[~ok].mean().item() if (~ok).any() else float("nan"),
                    wrong_conf_gt90=((~ok) & (conf > .9)).sum().item() / max(1, (~ok).sum().item()),
                    ece=ece, nll=F.cross_entropy(lg, y).item(),
                    entropy=-(p * p.clamp_min(1e-12).log()).sum(1).mean().item(),
                    top_pred_class=int(cnt.argmax()), top_pred_share=(cnt.max() / len(y)).item())

    models = {}
    for name in NAMES:
        m = MODELS[name][0]().to(DEV); m.load_state_dict(torch.load(f"ckpt/q1a/{name}_s{SEED}.pt")); models[name] = m
    # control: CNN trained on grayscale only; same LR as the Q1(a)-selected CNN LR (chosen on validation there)
    lr = json.load(open("runs/q1a/final_CNN.json"))["lr"]
    set_seed(SEED); gm = Gray(MODELS["CNN"][0]())
    h, st, be = train(gm, d.xtr, d.ytr, d.xva, d.yva, E, lr, WD, BS, tag="CNN-gray-trained")
    gm.load_state_dict(st); gm.to(DEV); torch.save(st, "ckpt/q2/CNN_gray_trained.pt")
    gray_val = max(r["val_acc"] for r in h)

    res = {c: {} for c in CONDS}
    for c in CONDS:
        for name, m in models.items():
            res[c][name] = stats(predict(m, XC[c]), d.yte)
    res["clean"]["CNN-gray-trained"] = stats(predict(gm, XC["clean"]), d.yte)   # it converts to gray internally
    res["gray"]["CNN-gray-trained"] = stats(predict(gm, XC["gray"]), d.yte)
    res["_meta"] = dict(gray_trained_val_acc=gray_val, gray_trained_params=count_params(gm), cnn_lr=lr)
    # per-class accuracy of the colour-trained CNN under block shuffle vs clean (error analysis)
    pc = {}
    for c in ("clean", "shuffle4x4", "gray"):
        pr = predict(models["CNN"], XC[c]).argmax(1)
        pc[c] = [((pr == d.yte) & (d.yte == k)).sum().item() / (d.yte == k).sum().item() for k in range(10)]
    res["_per_class_acc_CNN"] = pc
    save_json(res, "results/q2c.json")

    with open("results/q2c.md", "w") as f:
        f.write("| condition | model | test acc | mean conf | conf (correct) | conf (wrong) | wrong with conf>0.9 | ECE | NLL | top predicted class (share) |\n|" + "---|" * 10 + "\n")
        for c in CONDS:
            for name, s in res[c].items():
                f.write(f"| {c} | {name} | {100*s['acc']:.2f} | {s['conf_mean']:.3f} | {s['conf_correct']:.3f} | {s['conf_wrong']:.3f} | "
                        f"{100*s['wrong_conf_gt90']:.1f}% | {s['ece']:.3f} | {s['nll']:.3f} | {s['top_pred_class']} ({100*s['top_pred_share']:.1f}%) |\n")
    # Q2 main table: params + clean / permuted test (3 seeds, from Q1(a) and perm_final) + seed-0 interventions
    with open("results/q2_table.md", "w") as f:
        f.write("| model | params | clean test (3 seeds) | permuted-pixel test (3 seeds) | gray | shuffle4x4 | centre_occluded |\n|---|---|---|---|---|---|---|\n")
        for name in NAMES:
            a = json.load(open(f"runs/q1a/final_{name}.json")); b = json.load(open(f"runs/q2/perm_final_{name}.json"))
            f.write(f"| {name} | {a['params']:,} | {100*np.mean(a['test']):.2f}±{100*np.std(a['test']):.2f} | "
                    f"{100*np.mean(b['test']):.2f}±{100*np.std(b['test']):.2f} | {100*res['gray'][name]['acc']:.2f} | "
                    f"{100*res['shuffle4x4'][name]['acc']:.2f} | {100*res['centre_occluded'][name]['acc']:.2f} |\n")
        f.write(f"| CNN trained on gray | {count_params(gm):,} | gray test {100*res['gray']['CNN-gray-trained']['acc']:.2f} (1 seed) | | | | |\n")
    print(open("results/q2_table.md").read()); print(open("results/q2c.md").read())
