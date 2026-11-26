"""Q1(b): memorisation with an over-parameterised MLP (3072-2048-2048-10, 10.5M params, no BN).
A fraction p of TRAINING labels is replaced by uniformly random labels (Zhang et al. style; a random
label equals the true one 10% of the time by chance). No augmentation, no weight decay, constant LR,
so nothing prevents memorisation. Validation set keeps its true labels and is used only for reporting.

Logged every epoch (eval mode, full 45k train set):
  fit_clean     : accuracy on examples whose label was left intact
  fit_corrupt   : accuracy on corrupted examples w.r.t. the RANDOM label they were trained on (= memorisation)
  true_corrupt  : accuracy on corrupted examples w.r.t. their TRUE label (= useful structure still recovered)
  val_acc/val_loss : clean held-out performance
Question answered from these curves: does true_corrupt / val_acc rise before fit_corrupt does?
Runs 3 seeds (SEED, SEED+1, SEED+2); the seed also controls WHICH examples are corrupted and which random labels they get.
Existing runs/q1b/p{p}.json (seed 26004) are kept and skipped.
Usage: CUDA_VISIBLE_DEVICES=0 python q1b.py [epochs=200]
"""
import sys
from common import *

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PS = [0.0, 0.4, 1.0]
LR = 3e-4

d = get_data()
ytrue = d.ytr.clone()
n = len(ytrue)
for s_ in (SEED, SEED + 1, SEED + 2):
  for p in PS:
    out = f"runs/q1b/p{p}.json" if s_ == SEED else f"runs/q1b/p{p}_s{s_}.json"
    if os.path.exists(out): continue
    rng = np.random.RandomState(s_)
    cidx = rng.permutation(n)[:int(round(p * n))]
    ynoisy = ytrue.clone()
    if len(cidx):
        ynoisy[torch.from_numpy(cidx).to(DEV)] = torch.from_numpy(rng.randint(0, 10, len(cidx))).long().to(DEV)
    cm = torch.zeros(n, dtype=torch.bool, device=DEV)
    if len(cidx): cm[torch.from_numpy(cidx).to(DEV)] = True

    def extra(model):
        pr = predict(model, d.xtr).argmax(1)
        acc = lambda m, y: (pr[m] == y[m]).float().mean().item() if m.any() else float("nan")
        return dict(fit_clean=acc(~cm, ynoisy), fit_corrupt=acc(cm, ynoisy), true_corrupt=acc(cm, ytrue))

    set_seed(s_); m = MLP(2048, 2, bn=False)
    h, _, be = train(m, d.xtr, ynoisy, d.xva, d.yva, EPOCHS, LR, wd=0.0, bs=256, aug=False,
                     sched="const", seed=s_, extra_eval=extra, tag=f"p={p} s={s_}")
    save_json(dict(p=p, seed=s_, params=count_params(m), lr=LR, epochs=EPOCHS, n_corrupt=int(cm.sum()), hist=h), out)
