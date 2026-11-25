"""Tests for the vendored DSAFLite architecture."""
import numpy as np
import pytest
import torch

from certalarm.predictors import dsaflite
from certalarm.predictors import dsaf_train


def _x(B=4, L=2560, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 1, L, generator=g)


def test_forward_returns_the_documented_five_outputs():
    m = dsaflite.DSAFLite()
    rul, alpha, fused, fa, fb = m(_x())
    assert rul.shape == (4,)
    assert alpha.shape == (4,)
    assert fused.shape == (4, 128)
    assert fa.shape == (4, 128) and fb.shape == (4, 128)


def test_the_fusion_gate_stays_in_the_unit_interval():
    _, alpha, *_ = dsaflite.DSAFLite()(_x())
    assert torch.all((alpha >= 0) & (alpha <= 1))


def test_the_envelope_front_end_is_scale_invariant():
    e = dsaflite.EnvelopeSpectrumExtractor()
    x = _x(B=2)
    assert torch.allclose(e(x), e(x * 7.0), atol=1e-5)


def test_the_model_stays_lightweight():
    n = sum(p.numel() for p in dsaflite.DSAFLite().parameters())
    assert n < 3_000_000


def test_pgmc_penalises_life_that_rises_with_time():
    loss = dsaflite.PGMCLoss(lambda_max=1.0, warmup_epochs=1, mode="sequential")
    loss.set_epoch(5)
    t = torch.arange(6)
    b = torch.zeros(6, dtype=torch.long)
    falling = torch.linspace(1.0, 0.0, 6)
    rising = torch.linspace(0.0, 1.0, 6)
    y = torch.linspace(1.0, 0.0, 6)
    _, _, pen_ok = loss(falling, y, t, b)
    _, _, pen_bad = loss(rising, y, t, b)
    assert pen_bad > pen_ok


def test_pgmc_is_inert_before_warmup():
    loss = dsaflite.PGMCLoss(lambda_max=1.0, warmup_epochs=10, mode="sequential")
    loss.set_epoch(0)
    total, base, pen = loss(torch.linspace(0, 1, 5), torch.linspace(1, 0, 5))
    assert total == pytest.approx(base.item())
    assert pen.item() == pytest.approx(0.0)


def test_gradient_reversal_flips_the_gradient():
    grl = dsaflite.GradientReversalLayer(gamma=10.0)
    grl.set_progress(0.5)
    x = torch.ones(3, 2, requires_grad=True)
    grl(x).sum().backward()
    assert torch.all(x.grad < 0)


def test_the_domain_head_is_binary():
    logit = dsaflite.DomainClassifier(feat_dim=128)(torch.randn(5, 128))
    assert logit.shape == (5,)


class FakeAsset:
    def __init__(self, name, condition, n_min, per_min=6, samples=256, seed=0):
        rng = np.random.default_rng(seed)
        self.name, self.condition, self.per_min = name, condition, per_min
        self.windows = rng.standard_normal((n_min * per_min, samples)).astype(np.float32)

    @property
    def n_min(self):
        return len(self.windows) // self.per_min

    @property
    def rul(self):
        return np.linspace(1.0, 0.0, self.n_min, dtype=np.float32)


def test_the_recipe_matches_the_published_one():
    assert dsaf_train.RECIPE["epochs"] == 120
    assert dsaf_train.RECIPE["lr"] == 3e-4
    assert dsaf_train.RECIPE["weight_decay"] == 1e-4
    assert dsaf_train.RECIPE["patience"] == 30
    assert dsaf_train.RECIPE["lambda_pgmc"] == 0.08
    assert dsaf_train.RECIPE["mc_passes"] == 30


def test_window_targets_repeat_each_minute_label():
    a = FakeAsset("Bearing1_1", 1, n_min=3)
    y = dsaf_train.window_targets(a)
    assert y.shape == (18,)
    assert np.allclose(y[:6], 1.0)
    assert np.allclose(y[-6:], 0.0)


def test_domain_labels_split_condition_one_from_the_rest():
    a = [FakeAsset("Bearing1_1", 1, 2), FakeAsset("Bearing2_1", 2, 2),
         FakeAsset("Bearing3_1", 3, 2)]
    assert dsaf_train.domain_labels(a) == {"Bearing1_1": 0, "Bearing2_1": 1, "Bearing3_1": 1}


def test_domain_labels_refuse_assets_outside_the_fit_set():
    fit = [FakeAsset("Bearing1_1", 1, 2)]
    with pytest.raises(ValueError, match="fit"):
        dsaf_train.domain_labels(fit, allowed=["Bearing9_9"])


def test_minute_aggregation_averages_means_and_pools_variances():
    # Total variance = mean of the windows' variances + variance of their means.
    mu_w = np.array([1.0, 3.0, 2.0, 4.0])
    sd_w = np.array([1.0, 1.0, 3.0, 4.0])
    mu, sd = dsaf_train.to_minutes(mu_w, sd_w, per_min=2)
    assert np.allclose(mu, [2.0, 3.0])
    assert np.allclose(sd, [np.sqrt(1.0 + 1.0), np.sqrt((9 + 16) / 2 + 1.0)])


def test_minute_uncertainty_keeps_the_disagreement_between_windows():
    # Windows that individually claim no uncertainty but disagree with each other still leave
    # the minute uncertain; dropping the between-window term would report zero here.
    mu, sd = dsaf_train.to_minutes(np.array([0.2, 0.8]), np.array([0.0, 0.0]), per_min=2)
    assert mu[0] == pytest.approx(0.5)
    assert sd[0] == pytest.approx(0.3)


def test_minute_aggregation_drops_a_partial_final_minute():
    mu, sd = dsaf_train.to_minutes(np.arange(7.0), np.ones(7), per_min=3)
    assert mu.shape == (2,) and sd.shape == (2,)


def test_mc_dropout_enables_dropout_but_not_batchnorm():
    m = dsaflite.DSAFLite().eval()
    dsaf_train.enable_mc_dropout(m)
    assert any(mod.training for mod in m.modules() if isinstance(mod, torch.nn.Dropout))
    assert not any(mod.training for mod in m.modules()
                   if isinstance(mod, torch.nn.BatchNorm1d))


def test_the_monotonicity_penalty_is_the_within_bearing_one():
    # mode="soft" silently ignores the time and bearing indices and pairs windows across
    # different bearings instead. It must not be the mode this trainer uses.
    fit = [FakeAsset("Bearing1_1", 1, n_min=3, seed=1)]
    crit = dsaf_train.make_criterion(dsaf_train.RECIPE)
    assert crit.mode == "sequential"


def test_training_and_prediction_run_end_to_end_at_a_tiny_budget():
    fit = [FakeAsset(f"Bearing1_{i}", 1, n_min=4, seed=i) for i in range(1, 4)]
    model = dsaf_train.train(fit, seed=42, epochs=1)
    mu, sd = dsaf_train.predict_minutes(model, fit[0], mc_passes=2)
    assert mu.shape == (4,) and sd.shape == (4,)
    assert np.all(np.isfinite(mu)) and np.all(sd >= 0)


def test_the_dann_arm_trains_without_touching_other_assets():
    fit = [FakeAsset("Bearing1_1", 1, 4, seed=1), FakeAsset("Bearing2_1", 2, 4, seed=2),
           FakeAsset("Bearing1_2", 1, 4, seed=3), FakeAsset("Bearing3_1", 3, 4, seed=4)]
    model = dsaf_train.train(fit, seed=42, epochs=1, dann=True)
    mu, _ = dsaf_train.predict_minutes(model, fit[0], mc_passes=2)
    assert mu.shape == (4,)
