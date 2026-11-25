"""DSAFLite, vendored from the Dual-Stream Vibration-Vision project.

Source: that project's `src/dsaf_v2_lite.py` (architecture, PGMC loss) and `src/mini_dann.py`
(gradient reversal, domain classifier, DANN module), copied on 2026-09-20 and not modified.
It is vendored rather than imported so that these results cannot change when that project
is edited. It is a demonstration vehicle here, not a contribution of this study: the certification
machinery is predictor-agnostic and CT-GRU is certified the same way.

Two streams read the same raw 0.1 s vibration window: a temporal 1D CNN, and a 1D CNN over the
envelope spectrum obtained by a fixed Hilbert front-end. An adaptive gate mixes them per sample
and a small head regresses normalized remaining life. PGMC penalises predicted life that rises
with time within a bearing. The DANN head is used only in a pre-registered ablation arm.

Note on the domain head: `DomainClassifier` emits a single logit under `BCEWithLogitsLoss`, so it
is strictly binary. It must never be given a domain label derived from calibration or test
bearings; see `dsaf_train.domain_labels` for the only construction permitted here.
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Envelope Spectrum Extractor (CPU/GPU differentiable via analytical Hilbert)
# ─────────────────────────────────────────────────────────────────────────────

class EnvelopeSpectrumExtractor(nn.Module):
    """
    Converts raw 1D vibration signal to its envelope spectrum.

    Steps:
      1. Hilbert transform via FFT (analytical signal)
      2. Instantaneous amplitude (envelope)
      3. FFT of envelope → magnitude spectrum (first half only)
      4. Normalize by max value for scale invariance

    This is a FIXED (non-learnable) front-end. It is deterministic and
    grounded in bearing fault physics (demodulation theory).
    """
    def __init__(self, signal_len: int = 2560, n_fft_out: int = 512):
        super().__init__()
        self.signal_len = signal_len
        self.n_fft_out  = n_fft_out  # output spectrum length (one-sided)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, L) raw vibration signal
        Returns:
            envelope_spectrum: (B, 1, n_fft_out)
        """
        # Remove channel dim for FFT ops
        x_sq = x.squeeze(1)  # (B, L)
        B, L = x_sq.shape

        # Step 1: Analytical signal via Hilbert (FFT-based)
        X = torch.fft.fft(x_sq, n=L, dim=-1)          # (B, L) complex
        # Zero negative frequencies, double positive (Hilbert convention)
        h = torch.zeros(L, device=x.device, dtype=x.dtype)
        if L % 2 == 0:
            h[0] = h[L // 2] = 1
            h[1:L // 2] = 2
        else:
            h[0] = 1
            h[1:(L + 1) // 2] = 2
        analytic = torch.fft.ifft(X * h, dim=-1)       # (B, L) complex

        # Step 2: Instantaneous amplitude (envelope)
        envelope = analytic.abs()                       # (B, L) real

        # Step 3: FFT of envelope — one-sided magnitude
        E = torch.fft.rfft(envelope, dim=-1)            # (B, L//2+1) complex
        mag = E.abs()                                   # (B, L//2+1) real
        mag = mag[:, :self.n_fft_out]                   # (B, n_fft_out)

        # Step 4: Normalize per sample
        max_val = mag.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        mag = mag / max_val                             # (B, n_fft_out)

        return mag.unsqueeze(1)                         # (B, 1, n_fft_out)


# ─────────────────────────────────────────────────────────────────────────────
# Stream A: Temporal Vibration CNN (unchanged from original Variant A — it works)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalVibrStream(nn.Module):
    """
    3-layer 1D CNN on raw vibration. Identical to Ablation Variant A that
    achieved MAE=0.0949 — the strongest single-stream result. Keep as-is.
    Approx 0.5M parameters.
    """
    def __init__(self, out_features: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1,   64,  kernel_size=15, stride=4,  padding=7),
            nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64,  128, kernel_size=7,  stride=4,  padding=3),
            nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3,  stride=2,  padding=1),
            nn.BatchNorm1d(128), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(128, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 2560)
        feat = self.cnn(x)          # (B, 128, T)
        feat = self.pool(feat).squeeze(-1)  # (B, 128)
        return self.proj(feat)      # (B, out_features)


# ─────────────────────────────────────────────────────────────────────────────
# Stream B: Envelope Spectrum CNN (NEW — replaces 2D STFT branch)
# ─────────────────────────────────────────────────────────────────────────────

class EnvelopeSpectrStream(nn.Module):
    """
    3-layer 1D CNN on envelope spectrum. Lightweight spectral counterpart.
    Input: (B, 1, 512) envelope spectrum.
    Approx 0.3M parameters.
    """
    def __init__(self, out_features: int = 128, n_fft_out: int = 512):
        super().__init__()
        self.extractor = EnvelopeSpectrumExtractor(signal_len=2560, n_fft_out=n_fft_out)
        self.cnn = nn.Sequential(
            nn.Conv1d(1,  64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),  nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),  nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, out_features)

    def forward(self, x_1d: torch.Tensor) -> torch.Tensor:
        # x_1d: (B, 1, L) — raw vibration fed in; we extract envelope internally
        spec = self.extractor(x_1d)   # (B, 1, 512)
        feat = self.cnn(spec)         # (B, 64, T')
        feat = self.pool(feat).squeeze(-1)  # (B, 64)
        return self.proj(feat)        # (B, out_features)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Fusion Gate
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveFusionGate(nn.Module):
    """
    Learns a scalar α ∈ [0,1] per sample to weight stream A vs B.
    fused = α * feat_A + (1-α) * feat_B
    α is computed from both features jointly (cross-stream attention).
    ~0.01M parameters.
    """
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        combined = torch.cat([feat_a, feat_b], dim=-1)  # (B, 2*feat_dim)
        alpha = self.gate(combined)                      # (B, 1)
        fused = alpha * feat_a + (1 - alpha) * feat_b   # (B, feat_dim)
        return fused, alpha.squeeze(-1)                  # (B, feat_dim), (B,)


# ─────────────────────────────────────────────────────────────────────────────
# DSAF-Lite: Full Dual-Stream Adaptive Fusion Model
# ─────────────────────────────────────────────────────────────────────────────

class DSAFLite(nn.Module):
    """
    Lightweight Dual-Stream Adaptive Fusion Network.

    BOTH streams operate on raw 1D vibration (x_1d). The x_2d input is
    IGNORED — we no longer need STFT spectrograms. This simplifies the
    data pipeline (no image preprocessing needed).

    Forward signature compatible with existing Trainer and experiment scripts.
    Returns: (rul_pred, gate_alpha, fused_feat, feat_a, feat_b)
    """
    def __init__(self, feat_dim: int = 128, dropout: float = 0.2,
                 n_fft_out: int = 512):
        super().__init__()
        self.stream_a  = TemporalVibrStream(out_features=feat_dim)
        self.stream_b  = EnvelopeSpectrStream(out_features=feat_dim, n_fft_out=n_fft_out)
        self.fusion    = AdaptiveFusionGate(feat_dim=feat_dim)
        self.regressor = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1)
        )

    def forward(self, x_1d, x_2d=None, use_checkpointing=False):
        """
        Args:
            x_1d: (B, 1, L) raw vibration signal — REQUIRED
            x_2d:  ignored (kept for API compatibility with Trainer)
        Returns:
            rul_pred:   (B,) scalar RUL predictions
            gate_alpha: (B,) fusion gate weights (diagnostic)
            fused:      (B, feat_dim) fused representation
            feat_a:     (B, feat_dim) temporal stream features
            feat_b:     (B, feat_dim) envelope stream features
        """
        feat_a = self.stream_a(x_1d)                    # (B, feat_dim)
        feat_b = self.stream_b(x_1d)                    # (B, feat_dim)
        fused, alpha = self.fusion(feat_a, feat_b)       # (B, feat_dim), (B,)
        rul_pred = self.regressor(fused).squeeze(-1)     # (B,)
        return rul_pred, alpha, fused, feat_a, feat_b


# ─────────────────────────────────────────────────────────────────────────────
# PGMC Loss — FIXED to operate on temporal sequences
# ─────────────────────────────────────────────────────────────────────────────

class PGMCLoss(nn.Module):
    """
    Physics-Guided Monotonic Constraint Loss — CORRECTED VERSION.

    CRITICAL FIX from original PINNLoss:
    The original loss sorted arbitrary minibatch samples by target value and
    penalized non-monotonic PAIRS ACROSS DIFFERENT BEARINGS. This is physically
    meaningless — a bearing at RUL=0.8 from Bearing1_1 has no temporal
    relationship to a bearing at RUL=0.6 from Bearing1_3.

    THIS VERSION requires that the dataloader provides SEQUENTIAL windows from
    a SINGLE bearing within each batch segment. It penalizes cases where
    prediction[t] > prediction[t-1] (RUL increasing over time = physically wrong).

    How to use:
      - DataLoader must return (x_1d, x_2d, targets, bearing_id, time_idx)
        OR the loss can be called with an explicit sequence flag.
      - For RANDOM BATCHES (standard dataloader): use mode='soft' which applies
        a softer directional penalty on targets that are adjacent in sorted order
        within the batch — still better than the original cross-bearing penalty.
      - For SEQUENTIAL BATCHES: use mode='sequential' for exact temporal constraint.

    Default: mode='soft' (works with existing dataloader, no changes required).
    The 'sequential' mode requires the bearing-sequential dataloader (Phase 3B).

    Linear warmup schedule: lambda ramps from 0 to lambda_max over warmup_epochs.
    """
    def __init__(self, lambda_max: float = 0.10, warmup_epochs: int = 10,
                 mode: str = 'soft'):
        super().__init__()
        assert mode in ('soft', 'sequential'), f"mode must be 'soft' or 'sequential'"
        self.mae           = nn.L1Loss()
        self.lambda_max    = lambda_max
        self.warmup_epochs = warmup_epochs
        self.mode          = mode
        self._lambda_eff   = 0.0

    def set_epoch(self, epoch: int):
        if epoch < self.warmup_epochs:
            self._lambda_eff = self.lambda_max * (epoch / self.warmup_epochs)
        else:
            self._lambda_eff = self.lambda_max

    def forward(self, preds, targets, time_indices=None, bearing_ids=None):
        """
        Args:
            preds:        (B,) predicted RUL values
            targets:      (B,) ground truth RUL values
            time_indices: (B,) integer time step within bearing (optional, for sequential mode)
            bearing_ids:  (B,) bearing identity (required for sequential mode to skip cross-bearing pairs)
        Returns:
            total_loss, base_loss, pgmc_penalty
        """
        base_loss = self.mae(preds, targets)

        if self._lambda_eff == 0.0:
            return base_loss, base_loss, torch.tensor(0.0, device=preds.device)

        if self.mode == 'sequential' and time_indices is not None and bearing_ids is not None:
            # Penalize preds[t] > preds[t-1] for consecutive time steps WITHIN same bearing
            bearing_ids = bearing_ids.to(preds.device)  # ensure same device
            sorted_idx = torch.argsort(time_indices)
            sorted_preds = preds[sorted_idx]
            sorted_tidx = time_indices[sorted_idx]
            sorted_bids = bearing_ids[sorted_idx]

            diffs = sorted_preds[1:] - sorted_preds[:-1]
            # Consecutive windows only: same bearing AND time_idx increases by 1
            same_bearing = sorted_bids[1:] == sorted_bids[:-1]
            consecutive_time = (sorted_tidx[1:] - sorted_tidx[:-1]) == 1
            valid_pairs = same_bearing & consecutive_time

            if valid_pairs.any():
                pgmc_penalty = torch.relu(diffs[valid_pairs]).mean()
            else:
                pgmc_penalty = torch.tensor(0.0, device=preds.device)
        else:
            # Soft mode: sort by TARGET value (descending = high RUL first)
            # Penalize predictions that are higher-than-expected for lower-RUL targets
            # This is still a soft directional prior, not a within-sequence constraint
            sorted_idx   = torch.argsort(targets, descending=True)
            sorted_preds = preds[sorted_idx]
            diffs        = sorted_preds[1:] - sorted_preds[:-1]
            pgmc_penalty = torch.relu(diffs).mean()

        total_loss = base_loss + self._lambda_eff * pgmc_penalty
        return total_loss, base_loss, pgmc_penalty


# -----------------------------------------------------------------------------
# Gradient Reversal Layer (GRL)
# -----------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """
    Forward: identity (passes features unchanged)
    Backward: multiplies gradients by -lambda_
    This forces the preceding network to MAXIMIZE domain confusion.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        lambda_, = ctx.saved_tensors
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    GRL with adaptive lambda scheduling.

    Lambda schedule (standard DANN from Ganin et al.):
      lambda(p) = 2 / (1 + exp(-γ * p)) - 1
      where p ∈ [0, 1] is training progress and γ = 10

    This ramps lambda from 0 -> 1 smoothly over training,
    preventing early training instability from a too-aggressive adversary.
    """

    def __init__(self, gamma: float = 10.0):
        super().__init__()
        self.gamma = gamma
        self._lambda = 0.0

    def set_progress(self, progress: float) -> None:
        """
        Args:
            progress: float in [0.0, 1.0] — current_epoch / total_epochs
        """
        self._lambda = 2.0 / (1.0 + np.exp(-self.gamma * progress)) - 1.0

    def get_lambda(self) -> float:
        return self._lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lambda_ = torch.tensor(self._lambda, device=x.device, dtype=x.dtype)
        return GradientReversalFunction.apply(x, lambda_)


# -----------------------------------------------------------------------------
# Domain Classifier
# -----------------------------------------------------------------------------

class DomainClassifier(nn.Module):
    """
    2-layer MLP that classifies fused features as Condition 1 vs Condition 2.
    Intentionally small — we don't want a very powerful discriminator,
    just enough to provide adversarial signal.

    Binary output: logit (no sigmoid — use BCEWithLogitsLoss)
    """

    def __init__(self, feat_dim: int = 128, hidden_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
            # No sigmoid — use BCEWithLogitsLoss for numerical stability
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, feat_dim) features (after GRL)
        Returns:
            logit: (B,) domain classification logit
        """
        return self.classifier(x).squeeze(-1)


# -----------------------------------------------------------------------------
# DANNModule: The complete domain adaptation add-on
# -----------------------------------------------------------------------------

class DANNModule(nn.Module):
    """
    Complete DANN add-on for DSAFLite.

    Attaches to the fused features after the AdaptiveFusionGate.
    Does NOT modify stream_a, stream_b, fusion gate, or RUL head.

    Usage in training loop:
        dann = DANNModule(feat_dim=128)

        # Forward pass
        rul_pred, alpha, fused, fa, fb = model(x_1d)
        domain_logit = dann(fused)

        # Loss computation
        rul_loss    = criterion(rul_pred, rul_targets)
        domain_loss = dann.domain_loss(domain_logit, domain_labels)
        total_loss  = rul_loss + dann.current_lambda * domain_loss

        # Update GRL lambda at start of each epoch:
        dann.set_progress(epoch / total_epochs)
    """

    def __init__(self, feat_dim: int = 128, hidden_dim: int = 64,
                 gamma: float = 10.0, dropout: float = 0.3):
        super().__init__()
        self.grl        = GradientReversalLayer(gamma=gamma)
        self.classifier = DomainClassifier(feat_dim, hidden_dim, dropout)
        self.loss_fn    = nn.BCEWithLogitsLoss()

    @property
    def current_lambda(self) -> float:
        return self.grl.get_lambda()

    def set_progress(self, progress: float) -> None:
        """Call at start of each epoch: progress = epoch / total_epochs"""
        self.grl.set_progress(progress)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, feat_dim) — fused features from DSAFLite fusion gate
        Returns:
            domain_logit: (B,) — domain classification logit
        """
        reversed_features = self.grl(fused)
        return self.classifier(reversed_features)

    def domain_loss(self, domain_logits: torch.Tensor,
                    domain_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            domain_logits: (B,) from forward()
            domain_labels: (B,) float — 0.0 = source (Cond 1), 1.0 = target (Cond 2)
        Returns:
            loss: scalar
        """
        return self.loss_fn(domain_logits, domain_labels.float())

    def predict_domain(self, fused: torch.Tensor) -> torch.Tensor:
        """For evaluation: domain prediction WITHOUT gradient reversal."""
        with torch.no_grad():
            logit = self.classifier(fused)
            return torch.sigmoid(logit).squeeze(-1)
