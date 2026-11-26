"""Q3: convolutional autoencoders.  Reuses the Q1(a) CNN (ckpt/q1a/CNN_s26004.pt) as the FIXED classifier.
Phases (in order):
  ae    : (a) 3 AEs, bottleneck z in {32,128,512}; LR tuned on val MSE per AE; model chosen by val MSE only.
  dae   : (b) denoising AE with the selected bottleneck; Gaussian noise sigma=0.2 (fixed a priori), fresh noise every
          epoch, clean images as targets, NO labels.
  final : test-set evaluation: selected-AE test MSE, classifier on corrupted / clean-AE / DAE reconstructions,
          (c) frozen-encoder linear probes (+ raw-pixel and random-encoder controls).
All training is on the 45k train split; selection on the 5k validation split; test only in `final`.
Usage: CUDA_VISIBLE_DEVICES=0 python q3.py ae|dae|final
"""
import sys
from common import *

BOTTLENECKS = [32, 128, 512]
SIGMA = 0.2                       # corruption level, fixed before any result was seen
EP, BSZ = 50, 256
LRS_AE = [3e-4, 1e-3, 3e-3]
phase = sys.argv[1]
for p in ("runs/q3", "ckpt/q3", "results"): os.makedirs(p, exist_ok=True)


@torch.no_grad()
def recon(m, x, bs=2000):
    m.eval(); return torch.cat([m(x[i:i + bs]) for i in range(0, len(x), bs)])


@torch.no_grad()
def encode(m, x, bs=2000):
    m.eval(); return torch.cat([m.encode(x[i:i + bs]) for i in range(0, len(x), bs)])


def train_ae(model, xtr, xva_in, xva_tgt, lr, sigma=None, seed=SEED, tag=""):
    model.to(DEV)
    o = torch.optim.Adam(model.parameters(), lr=lr)
    spe = math.ceil(len(xtr) / BSZ); total = EP * spe
    sch = torch.optim.lr_scheduler.LambdaLR(o, lambda s: min(1.0, (s + 1) / spe) * 0.5 * (1 + math.cos(math.pi * s / total)))
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    hist, best, bv = [], None, 1e9
    for ep in range(1, EP + 1):
        model.train(); perm = torch.randperm(len(xtr), generator=g, device=DEV)
        tl = torch.zeros((), device=DEV); n = 0
        for k in range(spe):
            idx = perm[k * BSZ:(k + 1) * BSZ]
            if len(idx) < 2: continue
            xb = xtr[idx]; xin = corrupt(xb, sigma, g) if sigma else xb
            loss = F.mse_loss(model(xin), xb)
            o.zero_grad(set_to_none=True); loss.backward(); o.step(); sch.step()
            tl += loss.detach() * len(idx); n += len(idx)
        v = F.mse_loss(recon(model, xva_in), xva_tgt).item()
        hist.append(dict(epoch=ep, train_mse=(tl / n).item(), val_mse=v))
        if v < bv:
            bv = v; best = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1: print(f"[{tag}] ep {ep} train {hist[-1]['train_mse']:.5f} val {v:.5f}", flush=True)
    return hist, best, bv


def val_corrupted(xva):
    g = torch.Generator(device=DEV); g.manual_seed(SEED + 1)     # fixed noisy validation set
    return corrupt(xva, SIGMA, g)


if phase == "ae":
    d = get_data(); xtr, xva = to01(d.xtr), to01(d.xva)
    for z in BOTTLENECKS:
        best = None; allr = []
        for lr in LRS_AE:
            set_seed(SEED); m = ConvAE(z)
            h, st, bv = train_ae(m, xtr, xva, xva, lr, tag=f"AE z={z} lr={lr}")
            allr.append(dict(lr=lr, best_val_mse=bv))
            if best is None or bv < best[0]: best = (bv, lr, h, st)
        torch.save(best[3], f"ckpt/q3/ae_{z}.pt")
        save_json(dict(z=z, params=count_params(ConvAE(z)), lr=best[1], best_val_mse=best[0], hist=best[2], lr_grid=allr),
                  f"runs/q3/ae_{z}.json")
    res = {z: json.load(open(f"runs/q3/ae_{z}.json"))["best_val_mse"] for z in BOTTLENECKS}
    sel = min(res, key=res.get)                                   # reconstruction error is the ONLY criterion here
    save_json(dict(z=sel, val_mse=res), "runs/q3/selected.json"); print("selected bottleneck:", sel, res)

elif phase == "dae":
    d = get_data(); xtr, xva = to01(d.xtr), to01(d.xva)
    z = json.load(open("runs/q3/selected.json"))["z"]; xvn = val_corrupted(xva)
    best = None; allr = []
    for lr in LRS_AE:
        set_seed(SEED); m = ConvAE(z)
        h, st, bv = train_ae(m, xtr, xvn, xva, lr, sigma=SIGMA, tag=f"DAE z={z} lr={lr}")
        allr.append(dict(lr=lr, best_val_mse=bv))
        if best is None or bv < best[0]: best = (bv, lr, h, st)
    torch.save(best[3], f"ckpt/q3/dae_{z}.pt")
    save_json(dict(z=z, sigma=SIGMA, params=count_params(ConvAE(z)), lr=best[1], best_val_mse_noisy_to_clean=best[0],
                   hist=best[2], lr_grid=allr), "runs/q3/dae.json")

elif phase == "final":
    d = get_data(with_test=True); xtr, xva, xte = to01(d.xtr), to01(d.xva), to01(d.xte)
    sel = json.load(open("runs/q3/selected.json"))["z"]
    aes = {}
    for z in BOTTLENECKS:
        m = ConvAE(z).to(DEV); m.load_state_dict(torch.load(f"ckpt/q3/ae_{z}.pt")); aes[z] = m.eval()
    dae = ConvAE(sel).to(DEV); dae.load_state_dict(torch.load(f"ckpt/q3/dae_{sel}.pt")); dae.eval()
    cnn = MODELS["CNN"][0]().to(DEV); cnn.load_state_dict(torch.load(f"ckpt/q1a/CNN_s{SEED}.pt")); cnn.eval()
    acc = lambda x: (predict(cnn, norm01(x)).argmax(1) == d.yte).float().mean().item()
    mse = lambda x: F.mse_loss(x, xte).item()

    # ---- (a) test reconstruction error, selected clean AE (all three reported for completeness)
    test_mse = {z: mse(recon(aes[z], xte)) for z in BOTTLENECKS}
    # ---- (b) classification on corrupted images and reconstructions (same fixed CNN)
    rows = []
    def add(name, x): rows.append(dict(setting=name, test_acc=acc(x), mse_vs_clean=mse(x)))
    for s in (SIGMA, 0.1, 0.3):                                   # sigma=0.2 is the trained-for level; others are extra
        g = torch.Generator(device=DEV); g.manual_seed(SEED + 2); xn = corrupt(xte, s, g)
        if s == SIGMA:
            add("clean images", xte); add("clean-AE(clean)", recon(aes[sel], xte)); add("DAE(clean)", recon(dae, xte))
        add(f"sigma={s}: corrupted", xn); add(f"sigma={s}: clean-AE(corrupted)", recon(aes[sel], xn))
        add(f"sigma={s}: DAE(corrupted)", recon(dae, xn))
        if s == SIGMA: keep = xn
    A = {r["setting"]: r["test_acc"] for r in rows}
    cl, co, de = A["clean images"], A[f"sigma={SIGMA}: corrupted"], A[f"sigma={SIGMA}: DAE(corrupted)"]
    recovered = (de - co) / (cl - co) if cl > co else float("nan")
    # example images for the figure script
    np.savez("results/q3_examples.npz", clean=xte[:10].cpu().numpy(), noisy=keep[:10].cpu().numpy(),
             clean_ae=recon(aes[sel], keep[:10]).cpu().numpy(), dae=recon(dae, keep[:10]).cpu().numpy())

    # ---- (c) linear probes on frozen encoders
    def probe(ftr, fva, fte, wds=(0.0, 1e-4, 1e-3, 1e-2), epochs=80):
        mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
        ftr, fva, fte = (ftr - mu) / sd, (fva - mu) / sd, (fte - mu) / sd
        best = None
        for wd in wds:
            set_seed(SEED); lin = nn.Linear(ftr.shape[1], 10).to(DEV)
            o = torch.optim.AdamW(lin.parameters(), lr=2e-3, weight_decay=wd)
            g = torch.Generator(device=DEV); g.manual_seed(SEED)
            for _ in range(epochs):
                perm = torch.randperm(len(ftr), generator=g, device=DEV)
                for k in range(0, len(ftr), 1024):
                    idx = perm[k:k + 1024]
                    loss = F.cross_entropy(lin(ftr[idx]), d.ytr[idx]); o.zero_grad(); loss.backward(); o.step()
            with torch.no_grad():
                va = (lin(fva).argmax(1) == d.yva).float().mean().item()
                if best is None or va > best[0]:
                    best = (va, wd, (lin(fte).argmax(1) == d.yte).float().mean().item())
        return dict(val_acc=best[0], wd=best[1], test_acc=best[2])      # wd chosen on VAL; test just reported

    P = {}
    for z in BOTTLENECKS:
        P[f"AE z={z}"] = probe(encode(aes[z], xtr), encode(aes[z], xva), encode(aes[z], xte))
    P[f"DAE z={sel}"] = probe(encode(dae, xtr), encode(dae, xva), encode(dae, xte))
    set_seed(SEED + 7); rnd = ConvAE(sel).to(DEV).eval()           # untrained encoder, BN in eval mode with default stats
    P[f"random encoder z={sel}"] = probe(encode(rnd, xtr), encode(rnd, xva), encode(rnd, xte))
    P["raw pixels (3072-d)"] = probe(xtr.flatten(1), xva.flatten(1), xte.flatten(1))

    out = dict(selected_z=sel, test_mse_clean_ae=test_mse, classifier_rows=rows, recovered_fraction=recovered, probes=P,
               params={z: count_params(ConvAE(z)) for z in BOTTLENECKS})
    for z in BOTTLENECKS: out.setdefault("val_mse", {})[z] = json.load(open(f"runs/q3/ae_{z}.json"))["best_val_mse"]
    save_json(out, "results/q3.json")
    with open("results/q3.md", "w") as f:
        f.write("### (a) bottleneck sizes (selected by val MSE only)\n\n| z | params | val MSE | test MSE | selected |\n|---|---|---|---|---|\n")
        for z in BOTTLENECKS:
            f.write(f"| {z} | {out['params'][z]:,} | {out['val_mse'][z]:.5f} | {test_mse[z]:.5f} | {'yes' if z == sel else ''} |\n")
        f.write(f"\n### (b) fixed CNN (Q1(a) CNN) on corrupted images and reconstructions, test set (sigma={SIGMA} is the training noise)\n\n| setting | test acc | MSE vs clean |\n|---|---|---|\n")
        for r in rows: f.write(f"| {r['setting']} | {100*r['test_acc']:.2f} | {r['mse_vs_clean']:.5f} |\n")
        f.write(f"\nfraction of the corruption-induced accuracy loss recovered by the DAE: {recovered:.3f}\n")
        f.write("\n### (c) frozen-encoder linear probe vs reconstruction quality (test)\n\n| representation | probe val acc | probe test acc | test recon MSE |\n|---|---|---|---|\n")
        for k, v in P.items():
            mz = test_mse.get(int(k.split('=')[1]), float('nan')) if k.startswith("AE z=") else float('nan')
            f.write(f"| {k} | {100*v['val_acc']:.2f} | {100*v['test_acc']:.2f} | {mz:.5f} |\n")
    print(open("results/q3.md").read())
