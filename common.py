"""Shared code for all questions (introduced in Q1; reused by Q2-Q4).

- SEED = all numeric characters of the roll number, in order (m26mac004 -> 26004).
- One stratified 45k/5k split, saved to splits/ so it is identical in every script.
- The official test set is only loaded when with_test=True (final evaluation only).
"""
import copy, json, math, os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ROLL = "m26mac004"
SEED = int("".join(ch for ch in ROLL if ch.isdigit()))  # 26004
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ----------------------------------------------------------------- data
def _load_raw(root="data"):
    if os.environ.get("FAKE") == "1":  # smoke test only, never for real runs
        r = np.random.RandomState(0)
        return (r.randint(0, 256, (10000, 32, 32, 3), dtype=np.uint8), r.randint(0, 10, 10000),
                r.randint(0, 256, (2000, 32, 32, 3), dtype=np.uint8), r.randint(0, 10, 2000))
    from torchvision.datasets import CIFAR10
    tr, te = CIFAR10(root, train=True, download=True), CIFAR10(root, train=False, download=True)
    return tr.data, np.array(tr.targets), te.data, np.array(te.targets)


def get_split(ytrain):
    """Stratified 45,000 / 5,000 split, seed = SEED. Saved and reloaded for reproducibility."""
    from sklearn.model_selection import train_test_split
    path = f"splits/split_{SEED}.npz"
    if os.path.exists(path) and os.environ.get("FAKE") != "1":
        d = np.load(path); return d["train_idx"], d["val_idx"]
    n = len(ytrain)
    tr, va = train_test_split(np.arange(n), test_size=5000 if n == 50000 else n // 10,
                              stratify=ytrain, random_state=SEED)
    if os.environ.get("FAKE") != "1":
        os.makedirs("splits", exist_ok=True)
        np.savez(path, train_idx=np.sort(tr), val_idx=np.sort(va))
        tr, va = np.sort(tr), np.sort(va)
    return tr, va


def to_tensor(x_uint8_nhwc):
    x = torch.from_numpy(x_uint8_nhwc).permute(0, 3, 1, 2).float().div(255)
    return ((x - MEAN) / STD).to(DEV)


class Data:  # plain container
    pass


def get_data(with_test=False):
    xtr_all, ytr_all, xte, yte = _load_raw()
    tr, va = get_split(ytr_all)
    d = Data()
    d.xtr, d.ytr = to_tensor(xtr_all[tr]), torch.from_numpy(ytr_all[tr]).long().to(DEV)
    d.xva, d.yva = to_tensor(xtr_all[va]), torch.from_numpy(ytr_all[va]).long().to(DEV)
    d.train_idx, d.val_idx = tr, va
    if with_test:  # call only after every choice is fixed
        d.xte, d.yte = to_tensor(xte), torch.from_numpy(yte).long().to(DEV)
    return d


def augment(x, g=None):
    """GPU random horizontal flip + reflect-pad-4 random crop (same for every model)."""
    n = x.size(0)
    flip = torch.rand(n, device=x.device, generator=g) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (n,), device=x.device, generator=g)
    j = torch.randint(0, 9, (n,), device=x.device, generator=g)
    ar = torch.arange(32, device=x.device)
    rows, cols = i[:, None] + ar[None], j[:, None] + ar[None]
    return xp[torch.arange(n, device=x.device)[:, None, None, None],
              torch.arange(3, device=x.device)[None, :, None, None],
              rows[:, None, :, None], cols[:, None, None, :]]


# --------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, width=512, depth=3, bn=True, nc=10):
        super().__init__()
        L, d = [nn.Flatten()], 3072
        for _ in range(depth):
            L += [nn.Linear(d, width)] + ([nn.BatchNorm1d(width)] if bn else []) + [nn.ReLU()]
            d = width
        L.append(nn.Linear(d, nc))
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """VGG-style: [conv-BN-ReLU x2, maxpool] per stage, global average pool, linear head."""
    def __init__(self, widths=(64, 128, 256), nc=10):
        super().__init__()
        L, c = [], 3
        for w in widths:
            for _ in range(2):
                L += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU()]
                c = w
            L.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*L)
        self.head = nn.Linear(c, nc)

    def forward(self, x):
        return self.head(self.features(x).mean((2, 3)))


# ------------------------------------------------------------- training
@torch.no_grad()
def predict(model, x, bs=4096):
    model.eval()
    return torch.cat([model(x[i:i + bs]).float() for i in range(0, len(x), bs)])


def loss_acc(logits, y):
    return F.cross_entropy(logits, y).item(), (logits.argmax(1) == y).float().mean().item()


def train(model, xtr, ytr, xva, yva, epochs, lr, wd=0.05, bs=256, aug=True, opt="adamw",
          sched="cosine", seed=SEED, extra_eval=None, tag="", verbose=True):
    """Returns (history, best_val_state_dict, best_epoch). Selection of the checkpoint uses
    validation accuracy only. 'train_acc_clean' is measured in eval mode without augmentation
    on a fixed random 5,000-image subset of the training set, so it is comparable with val."""
    model.to(DEV)
    o = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) if opt == "adamw" else
         torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd))
    spe = math.ceil(len(xtr) / bs); total = epochs * spe
    if sched == "cosine":
        f = lambda s: min(1.0, (s + 1) / spe) * 0.5 * (1 + math.cos(math.pi * s / total))
    else:
        f = lambda s: 1.0
    sch = torch.optim.lr_scheduler.LambdaLR(o, f)
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    sub = torch.randperm(len(xtr), generator=g, device=DEV)[:5000]
    hist, best_state, best_ep, best_acc = [], None, -1, -1.0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=g, device=DEV)
        tl = torch.zeros((), device=DEV); n = 0
        for k in range(spe):
            idx = perm[k * bs:(k + 1) * bs]
            if len(idx) < 2: continue
            xb, yb = xtr[idx], ytr[idx]
            if aug: xb = augment(xb, g)
            loss = F.cross_entropy(model(xb), yb)
            o.zero_grad(set_to_none=True); loss.backward(); o.step(); sch.step()
            tl += loss.detach() * len(idx); n += len(idx)
        vl, va = loss_acc(predict(model, xva), yva)
        _, ta = loss_acc(predict(model, xtr[sub]), ytr[sub])
        rec = dict(epoch=ep, train_loss=(tl / n).item(), train_acc_clean=ta, val_loss=vl, val_acc=va)
        if extra_eval: rec.update(extra_eval(model))
        hist.append(rec)
        if va > best_acc:
            best_acc, best_ep = va, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose and (ep % 5 == 0 or ep == 1 or ep == epochs):
            print(f"[{tag}] ep {ep:3d} loss {rec['train_loss']:.3f} train {ta:.3f} val {va:.3f}", flush=True)
    return hist, best_state, best_ep


# ------------------------------------------------ shared experiment config (Q1 -> reused in Q2, Q3, Q4)
E = 60
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]       # tuned on validation only
WD, BS = 0.05, 256
MODELS = {  # name: (constructor, epochs); first defined in Q1(a)
    "MLP":      (lambda: MLP(448, 3),                 E),
    "CNN":      (lambda: CNN((80, 160, 320)),         E),
    "MLP-wide": (lambda: MLP(1536, 3),                E),      # capacity control
    "MLP-long": (lambda: MLP(448, 3),                 3 * E),  # budget control
    "CNN-tiny": (lambda: CNN((16, 32, 64)),           E),      # fewer params than every MLP
}


class Permuted(nn.Module):
    """Applies ONE fixed spatial permutation of the 32x32 pixel positions (same for all 3 channels,
    so every pixel keeps its colour triplet) as the first op of the model. Because it lives inside the
    model, train / val / test inputs get exactly the same transformation. Augmentation is applied
    BEFORE it, so the permuted task carries the same information as the original. 0 extra parameters."""
    def __init__(self, m, perm):
        super().__init__(); self.m = m
        self.register_buffer("perm", torch.as_tensor(np.asarray(perm), dtype=torch.long))

    def forward(self, x):
        n, c, h, w = x.shape
        return self.m(x.flatten(2)[:, :, self.perm].view(n, c, h, w))


def to_gray_norm(x):
    m, s = MEAN.to(x.device), STD.to(x.device)
    r = x * s + m
    g = (0.299 * r[:, 0:1] + 0.587 * r[:, 1:2] + 0.114 * r[:, 2:3]).expand(-1, 3, -1, -1)
    return (g - m) / s


class Gray(nn.Module):
    """Model that only ever sees luminance (replicated to 3 channels). 0 extra parameters."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(to_gray_norm(x))


# ------------------------------------------------ autoencoder pieces (introduced in Q3, reused in Q4)
def to01(x):
    """normalised tensor -> pixel space [0,1] (autoencoders work in [0,1] with a sigmoid output)"""
    return x * STD.to(x.device) + MEAN.to(x.device)


def norm01(x):
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def corrupt(x01, sigma, g):
    """additive Gaussian noise (std sigma in [0,1] pixel units), clipped to [0,1]; g = torch.Generator"""
    return (x01 + sigma * torch.randn(x01.shape, device=x01.device, generator=g)).clamp(0, 1)


class ConvAE(nn.Module):
    """Conv encoder (3 stride-2 convs: 32->16->8->4) -> flat linear bottleneck of size z -> mirrored decoder.
    Only z changes between the three autoencoders; every other layer is identical."""
    def __init__(self, z):
        super().__init__()
        cb = lambda i, o: [nn.Conv2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU()]
        db = lambda i, o: [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU()]
        self.enc_conv = nn.Sequential(*cb(3, 64), *cb(64, 128), *cb(128, 256))
        self.to_z = nn.Linear(256 * 16, z)
        self.from_z = nn.Linear(z, 256 * 16)
        self.dec_conv = nn.Sequential(*db(256, 128), *db(128, 64), nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        return self.to_z(self.enc_conv(x).flatten(1))

    def decode(self, z):
        return self.dec_conv(F.relu(self.from_z(z)).view(-1, 256, 4, 4))

    def forward(self, x):
        return self.decode(self.encode(x))
