"""Training and prediction for the vendored DSAFLite predictor.

The recipe is fixed a priori, verified against the sibling project's Experiments/exp16_ress.py
lines 94-121. It is never tuned here, and never on calibration or test assets.

Uncertainty is MC dropout: the dropout layers alone are returned to training mode (batch-norm
keeps its eval statistics) and 30 passes are averaged. Per-window means and standard deviations
are then aggregated to the minute grid.

Validation for early stopping uses every VAL_STRIDE-th window of the fit assets themselves. That
is leak-free: no calibration or test asset is involved.
"""
import numpy as np
import torch
import torch.nn as nn

from .dsaflite import DANNModule, DSAFLite, PGMCLoss

RECIPE = {
    "feat_dim": 128, "dropout": 0.2, "batch": 32,
    "lr": 3e-4, "weight_decay": 1e-4, "epochs": 120, "patience": 30,
    "lambda_pgmc": 0.08, "pgmc_warmup": 10, "pgmc_chunk": 32,
    "lambda_dann": 0.08, "dann_gamma": 10.0, "dann_warmup": 20,
    "mc_passes": 30,
}
VAL_STRIDE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_criterion(cfg):
    """PGMC in sequential mode.

    The default, mode="soft", ignores the time and bearing indices and sorts the whole minibatch
    by target instead, penalising pairs drawn from different bearings. Only the sequential mode
    is the within-bearing temporal constraint meant here.
    """
    return PGMCLoss(lambda_max=cfg["lambda_pgmc"], warmup_epochs=cfg["pgmc_warmup"],
                     mode="sequential")


def window_targets(asset):
    """Normalized remaining life repeated for every window of each minute."""
    return np.repeat(asset.rul, asset.per_min).astype(np.float32)


def domain_labels(assets, allowed=None):
    """Binary domain label per fit asset: condition 1 against conditions 2 and 3.

    The textbook transductive alternative - source = fit assets, target = calibration assets -
    is forbidden: it would put calibration features into the predictor, so calibration and test
    assets would stop being exchangeable and the conformal guarantee would not hold.
    """
    names = {a.name for a in assets}
    if allowed is not None and not set(allowed) <= names:
        raise ValueError("domain labels may only be built from the fit assets of this fold")
    return {a.name: (0 if a.condition == 1 else 1) for a in assets}


def to_minutes(mu_w, sigma_w, per_min):
    """Per-window statistics -> minute grid; a partial final minute is dropped.

    The minute's variance is the full law-of-total-variance decomposition: the mean of the
    windows' own variances plus the variance of their means. Keeping only the first term would
    discard the disagreement between the windows inside a minute, which is real predictive
    uncertainty, and would report a scale that is too small for the rules that consume it.
    """
    n = len(mu_w) // per_min
    mu = np.asarray(mu_w, float)[:n * per_min].reshape(n, per_min)
    sd = np.asarray(sigma_w, float)[:n * per_min].reshape(n, per_min)
    return mu.mean(1), np.sqrt((sd ** 2).mean(1) + mu.var(1))


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _stack(assets):
    """-> X (N, 1, samples), y (N,), asset index (N,), time index within asset (N,)."""
    X = np.concatenate([a.windows for a in assets])[:, None, :].astype(np.float32)
    y = np.concatenate([window_targets(a) for a in assets])
    aid = np.concatenate([np.full(len(a.windows), i) for i, a in enumerate(assets)])
    tid = np.concatenate([np.arange(len(a.windows)) for a in assets])
    return X, y, aid, tid


def train(assets, seed, epochs=None, dann=False, device=None, recipe=None):
    """Train on the given fit assets only. Returns the model in eval mode."""
    cfg = {**RECIPE, **(recipe or {})}
    epochs = cfg["epochs"] if epochs is None else epochs
    device = device or DEVICE
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X, y, aid, tid = _stack(assets)
    val = np.zeros(len(X), bool)
    val[::VAL_STRIDE] = True
    tr = ~val

    model = DSAFLite(feat_dim=cfg["feat_dim"], dropout=cfg["dropout"]).to(device)
    crit = make_criterion(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=1e-6)

    dom, dom_opt, dom_lab = None, None, None
    if dann:
        dom = DANNModule(feat_dim=cfg["feat_dim"], gamma=cfg["dann_gamma"]).to(device)
        dom_opt = torch.optim.Adam(dom.parameters(), lr=cfg["lr"] * 3)
        lab = domain_labels(assets)
        dom_lab = np.array([lab[assets[i].name] for i in aid], np.float32)

    idx_tr = np.flatnonzero(tr)
    best, best_state, waited = np.inf, None, 0
    for epoch in range(1, epochs + 1):
        crit.set_epoch(epoch)
        if dann:
            dom.set_progress(max(0.0, epoch - cfg["dann_warmup"]) /
                              max(1, epochs - cfg["dann_warmup"]))
        model.train()
        order = rng.permutation(idx_tr)
        for s in range(0, len(order), cfg["batch"]):
            b = order[s:s + cfg["batch"]]
            xb = torch.from_numpy(X[b]).to(device)
            yb = torch.from_numpy(y[b]).to(device)
            pred, _, fused, *_ = model(xb)
            loss, _, _ = crit(pred, yb,
                               torch.from_numpy(tid[b]).to(device),
                               torch.from_numpy(aid[b]).to(device))
            if dann:
                dl = dom.domain_loss(dom(fused), torch.from_numpy(dom_lab[b]).to(device))
                loss = loss + cfg["lambda_dann"] * dl
                dom_opt.zero_grad()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if dann:
                dom_opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vp = []
            vi = np.flatnonzero(val)
            for s in range(0, len(vi), 256):
                b = vi[s:s + 256]
                p, *_ = model(torch.from_numpy(X[b]).to(device))
                vp.append(p.cpu().numpy())
            vmae = float(np.abs(np.concatenate(vp) - y[vi]).mean())
        if vmae < best - 1e-5:
            best, waited = vmae, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_minutes(model, asset, mc_passes=None, device=None, chunk=256):
    """One asset -> (mu (n_min,), sigma (n_min,)) on the minute grid."""
    passes = RECIPE["mc_passes"] if mc_passes is None else mc_passes
    device = device or DEVICE
    X = asset.windows[:, None, :].astype(np.float32)
    # The model must be moved to the input's device, not assumed to be there. A freshly trained
    # model is already on the GPU, but one reloaded from the model cache comes back on the CPU,
    # so without this every cache hit - which is to say every resumed run - crashes here.
    model.to(device)
    model.eval()
    enable_mc_dropout(model)
    draws = []
    for _ in range(passes):
        out = []
        for s in range(0, len(X), chunk):
            p, *_ = model(torch.from_numpy(X[s:s + chunk]).to(device))
            out.append(p.cpu().numpy())
        draws.append(np.concatenate(out))
    model.eval()
    draws = np.stack(draws)
    return to_minutes(draws.mean(0), draws.std(0), asset.per_min)
