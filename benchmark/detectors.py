#!/usr/bin/env python3
"""
V2X Misbehavior Detection Model Zoo
===================================
A collection of intrusion-detection models for V2X Basic Safety Message
(BSM) streams. Each model follows its source paper's architecture as
faithfully as possible while sharing a common data interface.

All models accept input shape (batch, seq_len, n_features) and
output scores of shape (batch,) for binary classification.

Detectors:
    1. GradientBoostingIDS         — Supervised gradient boosting baseline
    2. LSTMClassifier              — Supervised unidirectional LSTM
    3. IsolationForestIDS          — Classical unsupervised baseline (pyod)
    4. DIFIDS                      — Deep Isolation Forest (pyod)
    5. LOFIDS                      — Local Outlier Factor (density-based)
    6. OneClassSVMIDS              — One-Class SVM (boundary-based)
    7. DeepSVDDIDS                 — Deep SVDD (learned one-class)
    8. AnomalyTransformerClassifier— Anomaly Transformer (association discrepancy)
    9. GDNIDS                      — Graph Deviation Network
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# 1. Unidirectional LSTM (supervised baseline)
# ======================================================================

class LSTMClassifier(nn.Module):
    """Unidirectional LSTM binary classifier.

    Architecture: 1-layer LSTM → dropout → FC(1).
    Ref: Kamel et al., VTC-Fall 2019 (standard LSTM for VeReMi detection).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)  # no *2 (unidirectional)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(self.dropout(last_hidden)).squeeze(-1)


# ======================================================================
# 2. Anomaly Transformer (Xu et al., ICLR 2022 Spotlight)
# ======================================================================
# Ref: github.com/thuml/Anomaly-Transformer

class AnomalyAttention(nn.Module):
    """Anomaly-Attention with learnable Gaussian prior."""

    def __init__(self, d_model: int, n_heads: int, seq_len: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.seq_len = seq_len

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Learnable sigma for Gaussian prior (one per head)
        self.sigma = nn.Parameter(torch.ones(n_heads, 1, 1))

    def forward(self, x):
        B, L, _ = x.shape
        H, dk = self.n_heads, self.d_k

        Q = self.W_q(x).reshape(B, L, H, dk).transpose(1, 2)  # (B,H,L,dk)
        K = self.W_k(x).reshape(B, L, H, dk).transpose(1, 2)
        V = self.W_v(x).reshape(B, L, H, dk).transpose(1, 2)

        # Series-association: standard scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)
        series_assoc = torch.softmax(scores, dim=-1)  # (B,H,L,L)

        # Prior-association: Gaussian kernel based on distance
        dist = torch.arange(L, device=x.device).float().unsqueeze(0) - \
               torch.arange(L, device=x.device).float().unsqueeze(1)
        dist = dist.abs().unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
        sigma = torch.clamp(self.sigma.abs(), min=1e-4)
        prior_assoc = torch.softmax(-dist / (2 * sigma ** 2), dim=-1)  # (1,H,L,L)
        prior_assoc = prior_assoc.expand(B, -1, -1, -1)

        # Output from series attention
        out = torch.matmul(series_assoc, V)  # (B,H,L,dk)
        out = out.transpose(1, 2).reshape(B, L, H * dk)
        out = self.W_o(out)

        return out, series_assoc, prior_assoc


class AnomalyTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, seq_len, dropout=0.1):
        super().__init__()
        self.attention = AnomalyAttention(d_model, n_heads, seq_len)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, series, prior = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, series, prior


class AnomalyTransformerClassifier(nn.Module):
    """Anomaly Transformer with association discrepancy.

    Anomaly score = association_discrepancy × reconstruction_MSE.
    Training uses minimax with gradient detachment between phases.
    """

    is_unsupervised = True

    def __init__(self, input_dim: int, seq_len: int = 10,
                 d_model: int = 64, n_heads: int = 2, n_layers: int = 3,
                 d_ff: int = 128, dropout: float = 0.1, lambda_: float = 3.0):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.lambda_ = lambda_

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pe = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)
        self.layers = nn.ModuleList([
            AnomalyTransformerLayer(d_model, n_heads, d_ff, seq_len, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, input_dim)
        self.register_buffer("threshold", torch.tensor(0.0))

    def _forward_full(self, x):
        """Returns (reconstruction, list of (series, prior) per layer)."""
        h = self.pe(self.input_proj(x))
        associations = []
        for layer in self.layers:
            h, series, prior = layer(h)
            associations.append((series, prior))
        x_hat = self.output_proj(h)
        return x_hat, associations

    @staticmethod
    def _association_discrepancy(associations):
        """KL(prior || series) + KL(series || prior) summed over layers."""
        total = 0.0
        for series, prior in associations:
            # Symmetrized KL divergence
            kl_ps = (prior * (torch.log(prior + 1e-8) -
                              torch.log(series + 1e-8))).sum(dim=-1).mean()
            kl_sp = (series * (torch.log(series + 1e-8) -
                               torch.log(prior + 1e-8))).sum(dim=-1).mean()
            total = total + kl_ps + kl_sp
        return total

    def forward(self, x):
        """Anomaly score = assoc_discrepancy × reconstruction_MSE per sample."""
        x_hat, associations = self._forward_full(x)
        recon_mse = ((x - x_hat) ** 2).mean(dim=(1, 2))  # (batch,)
        # Per-sample symmetric KL association discrepancy
        disc_per_sample = torch.zeros(x.size(0), device=x.device)
        for series, prior in associations:
            kl_ps = (prior * (torch.log(prior + 1e-8) -
                              torch.log(series + 1e-8))).sum(dim=-1)
            kl_sp = (series * (torch.log(series + 1e-8) -
                               torch.log(prior + 1e-8))).sum(dim=-1)
            disc_per_sample += (kl_ps + kl_sp).mean(dim=(1, 2))
        return disc_per_sample * recon_mse

    def anomaly_scores(self, x):
        return self.forward(x)

    def train_unsupervised(self, benign_loader, device, epochs=30, lr=0.001,
                           print_every=10):
        self.train()
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = []

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            n_batches = 0

            for X_batch, _ in benign_loader:
                X_batch = X_batch.to(device)
                x_hat, associations = self._forward_full(X_batch)
                recon_loss = nn.functional.mse_loss(x_hat, X_batch)
                assoc_disc = self._association_discrepancy(associations)

                # Minimax Phase 1: minimize recon - λ·assoc (maximize assoc)
                loss_min = recon_loss - self.lambda_ * assoc_disc.detach()
                optimizer.zero_grad()
                loss_min.backward(retain_graph=True)

                # Minimax Phase 2: minimize recon + λ·assoc (minimize assoc)
                loss_max = recon_loss.detach() + self.lambda_ * assoc_disc
                loss_max.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += (recon_loss.item() + assoc_disc.item())
                n_batches += 1

            avg = total_loss / max(n_batches, 1)
            history.append({"epoch": epoch, "loss": avg})
            if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
                print(f"      Epoch {epoch:3d}/{epochs}  loss={avg:.4f}")
        return history

    def calibrate_threshold(self, benign_loader, device, percentile=95):
        self.eval()
        all_scores = []
        with torch.no_grad():
            for X_batch, _ in benign_loader:
                X_batch = X_batch.to(device)
                scores = self.forward(X_batch)
                all_scores.append(scores.cpu())
        all_scores = torch.cat(all_scores)
        thresh = torch.quantile(all_scores, percentile / 100.0)
        self.threshold.copy_(thresh)
        return thresh.item()


# ======================================================================
# Shared helper: sinusoidal positional encoding
# ======================================================================
# Used by the Anomaly Transformer above.

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (standard Transformer PE)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ======================================================================
# 5. Isolation Forest — Classical Unsupervised Baseline
# ======================================================================
#
# Classical anomaly-detection baseline (pyod IForest, contamination=0.01).
# Not a neural network — uses sklearn/pyod Isolation Forest.
# Wraps pyod as a nn.Module-like interface for pipeline compatibility.

class IsolationForestIDS(nn.Module):
    """Isolation Forest anomaly detector via pyod.

    Wraps pyod.models.iforest.IForest as a PyTorch Module for
    compatibility with the common evaluation pipeline.

    Training (unsupervised, benign-only):
        Flattens BSM windows (seq_len, features) → (seq_len * features)
        and fits IForest on benign-only data.

    Inference:
        Returns anomaly scores as logits (positive = attack).
        The pyod decision_function returns raw anomaly scores
        (higher = more anomalous), which maps directly to our
        logit convention.

    Attributes:
        is_unsupervised: True — trained on benign-only windows.
        is_sklearn: True — not a real nn.Module (no GPU, no gradients).
    """

    is_unsupervised = True
    is_sklearn = True

    def __init__(self, input_dim: int, seq_len: int = 10,
                 contamination: float = 0.01, n_estimators: int = 100,
                 random_state: int = 1, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.iforest = None  # Set during training

    def _flatten(self, x):
        """Flatten (batch, seq_len, features) → (batch, seq_len*features)."""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return x.reshape(x.shape[0], -1)

    def fit(self, X_benign):
        """Fit IForest on benign-only flattened windows."""
        from pyod.models.iforest import IForest
        self.iforest = IForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores for tree building
        )
        X_flat = self._flatten(X_benign)
        self.iforest.fit(X_flat)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return anomaly logits (positive = attack, negative = benign)."""
        X_flat = self._flatten(x)
        # decision_function: higher = more anomalous (matches our convention)
        scores = self.iforest.decision_function(X_flat)
        return torch.tensor(scores, dtype=torch.float32, device=x.device)

    def parameters(self, recurse=True):
        """No trainable parameters (sklearn model)."""
        return iter([])


# ======================================================================
# 6. Gradient Boosting (Supervised Classical Baseline)
# ======================================================================

class GradientBoostingIDS(nn.Module):
    """Gradient Boosting classifier via sklearn.

    Supervised classical baseline. Trains on labeled (benign + attack)
    data with flattened BSM windows.

    Attributes:
        is_sklearn: True — uses sklearn, no GPU/gradients.
    """

    is_sklearn = True
    is_unsupervised = False

    def __init__(self, input_dim: int, seq_len: int = 10,
                 n_estimators: int = 100, max_depth: int = 8,
                 learning_rate: float = 0.1, random_state: int = 42,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = learning_rate
        self.random_state = random_state
        self.model = None  # Set during training

    def _flatten(self, x):
        """Flatten (batch, seq_len, features) → (batch, seq_len*features)."""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return x.reshape(x.shape[0], -1)

    def fit(self, X, y):
        """Fit GB on all labeled data (supervised).

        Uses HistGradientBoostingClassifier for OpenMP multi-core
        parallelism (10-100x faster on datasets > 10K samples).
        """
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(
            max_iter=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.lr,
            random_state=self.random_state,
        )
        X_flat = self._flatten(X)
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        self.model.fit(X_flat, y)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return attack probabilities (0–1)."""
        X_flat = self._flatten(x)
        probs = self.model.predict_proba(X_flat)[:, 1]
        return torch.tensor(probs, dtype=torch.float32, device=x.device)

    def parameters(self, recurse=True):
        """No trainable parameters (sklearn model)."""
        return iter([])


class LOFIDS(nn.Module):
    """Local Outlier Factor anomaly detector (density-based).

    Detects anomalies as points in low-density regions.
    Uses novelty=True for one-class setting (fit on benign, predict on mixed).

    Ref: Breunig et al. (2000), "LOF: Identifying Density-Based Local Outliers"
    """

    is_unsupervised = True
    is_sklearn = True

    def __init__(self, input_dim: int, seq_len: int = 10,
                 n_neighbors: int = 20, contamination: float = 0.01,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.lof = None

    def _flatten(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return x.reshape(x.shape[0], -1)

    def fit(self, X_benign):
        """Fit LOF on benign-only flattened windows."""
        from sklearn.neighbors import LocalOutlierFactor
        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1,
        )
        X_flat = self._flatten(X_benign)
        self.lof.fit(X_flat)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return anomaly scores (positive = attack)."""
        X_flat = self._flatten(x)
        # Negate: LOF decision_function returns negative for outliers
        scores = -self.lof.decision_function(X_flat)
        return torch.tensor(scores, dtype=torch.float32, device=x.device)

    def parameters(self, recurse=True):
        return iter([])


class OneClassSVMIDS(nn.Module):
    """One-Class SVM anomaly detector (boundary-based negative control).

    Learns a decision boundary around benign data in kernel space.
    Subsamples to max_samples for tractability (SVM is O(n^2-n^3)).

    Ref: Scholkopf et al. (2001), "Estimating the Support of a
    High-Dimensional Distribution", Neural Computation.
    """

    is_unsupervised = True
    is_sklearn = True

    def __init__(self, input_dim: int, seq_len: int = 10,
                 kernel: str = "rbf", nu: float = 0.01,
                 max_samples: int = 50000, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.kernel = kernel
        self.nu = nu
        self.max_samples = max_samples
        self.ocsvm = None

    def _flatten(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        return x.reshape(x.shape[0], -1)

    def fit(self, X_benign):
        """Fit One-Class SVM on benign-only flattened windows."""
        import numpy as np
        from sklearn.svm import OneClassSVM
        X_flat = self._flatten(X_benign)
        if len(X_flat) > self.max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_flat), self.max_samples, replace=False)
            X_flat = X_flat[idx]
            print(f"    [OCSVM] Subsampled {self.max_samples:,}/{len(X_benign):,}")
        self.ocsvm = OneClassSVM(kernel=self.kernel, nu=self.nu)
        self.ocsvm.fit(X_flat)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return anomaly scores (positive = attack)."""
        X_flat = self._flatten(x)
        # Negate: decision_function returns negative for outliers
        scores = -self.ocsvm.decision_function(X_flat)
        return torch.tensor(scores, dtype=torch.float32, device=x.device)

    def parameters(self, recurse=True):
        return iter([])


class DeepSVDDIDS(nn.Module):
    """Deep SVDD anomaly detector (learned one-class, ICML 2018).

    MLP encoder maps features to a compact hypersphere.
    Anomaly score = distance from learned center.

    Ref: Ruff et al. (2018), "Deep One-Class Classification", ICML.
    """

    is_unsupervised = True
    is_sklearn = True  # DeepOD handles training internally

    def __init__(self, input_dim: int, seq_len: int = 10,
                 epochs: int = 50, hidden_dims: str = "64,32",
                 rep_dim: int = 16, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.epochs = epochs
        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.model = None

    def _flatten(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return x.astype(np.float32)

    def fit(self, X_benign):
        from deepod.models import DeepSVDD
        X_flat = self._flatten(X_benign)
        self.model = DeepSVDD(
            epochs=self.epochs,
            hidden_dims=self.hidden_dims,
            rep_dim=self.rep_dim,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
        self.model.fit(X_flat)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X_flat = self._flatten(x)
        scores = self.model.decision_function(X_flat)
        return torch.tensor(scores, dtype=torch.float32, device=x.device)

    def parameters(self, recurse=True):
        return iter([])


class DIFIDS(nn.Module):
    """Deep Isolation Forest (TKDE 2023, Xu & Pang et al.).

    Uses the official pyod DIF implementation. Neural network learns
    representations, then applies isolation forest in the learned space.

    Ref: Xu et al. (2023), "Deep Isolation Forest for Anomaly Detection", TKDE.
    Code: pyod.models.dif.DIF
    """

    is_unsupervised = True
    is_sklearn = True

    def __init__(self, input_dim: int, seq_len: int = 10,
                 n_ensemble: int = 50, contamination: float = 0.01,
                 hidden_neurons: list = None, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.n_ensemble = n_ensemble
        self.contamination = contamination
        self.hidden_neurons = hidden_neurons or [64, 32]
        self.model = None

    def _flatten(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        if x.ndim == 3:
            x = x.reshape(x.shape[0], -1)
        return x.astype(np.float32)

    def fit(self, X_benign):
        from pyod.models.dif import DIF
        X_flat = self._flatten(X_benign)
        self.model = DIF(
            hidden_neurons=self.hidden_neurons,
            n_ensemble=self.n_ensemble,
            contamination=self.contamination,
            random_state=42,
        )
        self.model.fit(X_flat)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X_flat = self._flatten(x)
        scores = self.model.decision_function(X_flat)
        return torch.tensor(scores, dtype=torch.float32, device=x.device)

    def parameters(self, recurse=True):
        return iter([])


class GDNIDS(nn.Module):
    """Graph Deviation Network (AAAI 2021, Deng & Hooi).

    Learns inter-feature graph via attention. Detects anomalies as
    deviations from learned feature-pair relationships.

    Simplified implementation: uses learned feature embeddings +
    attention-based prediction. Each feature is a "sensor node".

    Ref: Deng & Hooi (2021), "Graph Neural Network-Based Anomaly
    Detection in Multivariate Time Series", AAAI.
    """

    is_unsupervised = True
    is_sklearn = False  # needs GPU training

    def __init__(self, input_dim: int, seq_len: int = 10,
                 embed_dim: int = 64, topk: int = 5, **kwargs):
        super().__init__()
        self.n_features = input_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.topk = min(topk, input_dim - 1)

        # Feature embedding: each feature gets a learnable embedding
        self.feature_embed = nn.Embedding(input_dim, embed_dim)
        # Temporal encoder: compress seq_len timesteps per feature
        self.temporal_fc = nn.Linear(seq_len, embed_dim)
        # Attention for graph structure
        self.attn_query = nn.Linear(embed_dim, embed_dim)
        self.attn_key = nn.Linear(embed_dim, embed_dim)
        # Output: predict each feature value from graph context
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.threshold = 0.0

    def _compute_graph_features(self, x):
        """x: (batch, seq_len, n_features) -> per-feature context vectors."""
        B, S, F = x.shape
        # Temporal encoding per feature: (B, F, S) -> (B, F, embed_dim)
        x_t = x.transpose(1, 2)  # (B, F, S)
        h_temporal = torch.relu(self.temporal_fc(x_t))  # (B, F, embed_dim)

        # Feature embeddings
        feat_idx = torch.arange(F, device=x.device)
        h_feat = self.feature_embed(feat_idx)  # (F, embed_dim)

        # Combined: temporal + feature identity
        h = h_temporal + h_feat.unsqueeze(0)  # (B, F, embed_dim)

        # Attention-based graph
        Q = self.attn_query(h)  # (B, F, embed_dim)
        K = self.attn_key(h)    # (B, F, embed_dim)
        attn = torch.bmm(Q, K.transpose(1, 2)) / (self.embed_dim ** 0.5)  # (B, F, F)

        # Top-k sparsification
        if self.topk < F:
            topk_vals, topk_idx = attn.topk(self.topk, dim=-1)
            mask = torch.zeros_like(attn).scatter_(-1, topk_idx, 1.0)
            attn = attn * mask + (1 - mask) * (-1e9)

        attn = torch.softmax(attn, dim=-1)  # (B, F, F)

        # Graph-aggregated context
        h_context = torch.bmm(attn, h)  # (B, F, embed_dim)
        return h, h_context

    def forward(self, x):
        """Return anomaly scores (per-window max prediction error)."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if x.device != next(self.parameters()).device:
            x = x.to(next(self.parameters()).device)

        h, h_context = self._compute_graph_features(x)
        B, F, E = h.shape

        # Predict each feature's last timestep from graph context
        h_combined = torch.cat([h, h_context], dim=-1)  # (B, F, 2*embed)
        predictions = self.predictor(h_combined).squeeze(-1)  # (B, F)

        # Target: last timestep feature values
        targets = x[:, -1, :]  # (B, F)

        # Anomaly score: max absolute prediction error across features
        errors = (predictions - targets).abs()  # (B, F)
        scores = errors.max(dim=-1).values  # (B,)
        return scores

    def train_unsupervised(self, train_loader, device=None, epochs=50, lr=0.001,
                           print_every=10, **kwargs):
        """Train GDN on benign sequences."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        device = next(self.parameters()).device
        history = []

        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                # Filter benign only
                benign_mask = y_batch == 0
                if benign_mask.sum() == 0:
                    continue
                X_b = X_batch[benign_mask].to(device)

                h, h_context = self._compute_graph_features(X_b)
                h_combined = torch.cat([h, h_context], dim=-1)
                predictions = self.predictor(h_combined).squeeze(-1)
                targets = X_b[:, -1, :]
                loss = nn.functional.mse_loss(predictions, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            history.append(avg_loss)
            if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
                print(f"      Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}", flush=True)

        return {"train_loss": history}

    def calibrate_threshold(self, train_loader, device=None, percentile=95):
        """Set threshold at percentile of benign anomaly scores."""
        self.eval()
        device = next(self.parameters()).device
        all_scores = []
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                benign_mask = y_batch == 0
                if benign_mask.sum() == 0:
                    continue
                X_b = X_batch[benign_mask].to(device)
                scores = self.forward(X_b)
                all_scores.extend(scores.cpu().numpy())
        if all_scores:
            self.threshold = float(np.percentile(all_scores, percentile))
        print(f"    Anomaly threshold ({percentile}th percentile): {self.threshold:.4f}", flush=True)
        return self.threshold


# ======================================================================
# Model Factory
# ======================================================================

def build_model(model_cfg: dict, input_dim: int, seq_len: int = 10,
                n_domain_features: int = 0) -> nn.Module:
    """Instantiate a model from config.

    Args:
        model_cfg: Model-specific config dict from config.yaml
        input_dim: Number of input features
        seq_len: Sequence length (for models that need it)
        n_domain_features: Reserved for models that accept extra domain features

    Returns:
        nn.Module ready for training
    """
    model_type = model_cfg["type"]

    if model_type == "lstm":
        return LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_layers=model_cfg.get("num_layers", 1),
            dropout=model_cfg.get("dropout", 0.3),
        )

    elif model_type == "iforest":
        return IsolationForestIDS(
            input_dim=input_dim,
            seq_len=seq_len,
            contamination=model_cfg.get("contamination", 0.01),
            n_estimators=model_cfg.get("n_estimators", 100),
            random_state=model_cfg.get("random_state", 1),
        )

    elif model_type == "gb":
        return GradientBoostingIDS(
            input_dim=input_dim,
            seq_len=seq_len,
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", 8),
            learning_rate=model_cfg.get("learning_rate", 0.1),
            random_state=model_cfg.get("random_state", 42),
        )

    elif model_type == "deep_svdd":
        return DeepSVDDIDS(
            input_dim=input_dim,
            seq_len=seq_len,
            epochs=model_cfg.get("epochs", 50),
            hidden_dims=model_cfg.get("hidden_dims", "64,32"),
            rep_dim=model_cfg.get("rep_dim", 16),
        )

    elif model_type == "dif":
        return DIFIDS(
            input_dim=input_dim,
            seq_len=seq_len,
            n_ensemble=model_cfg.get("n_ensemble", 50),
            epochs=model_cfg.get("epochs", 50),
            hidden_dims=model_cfg.get("hidden_dims", "64,32"),
        )

    elif model_type == "gdn":
        return GDNIDS(
            input_dim=input_dim,
            seq_len=seq_len,
            embed_dim=model_cfg.get("embed_dim", 64),
            topk=model_cfg.get("topk", 5),
        )

    elif model_type == "lof":
        return LOFIDS(
            input_dim=input_dim,
            seq_len=seq_len,
            n_neighbors=model_cfg.get("n_neighbors", 20),
            contamination=model_cfg.get("contamination", 0.01),
        )

    elif model_type == "ocsvm":
        return OneClassSVMIDS(
            input_dim=input_dim,
            seq_len=seq_len,
            kernel=model_cfg.get("kernel", "rbf"),
            nu=model_cfg.get("nu", 0.01),
            max_samples=model_cfg.get("max_samples", 50000),
        )

    elif model_type == "anomaly_transformer":
        return AnomalyTransformerClassifier(
            input_dim=input_dim,
            seq_len=seq_len,
            d_model=model_cfg.get("d_model", 64),
            n_heads=model_cfg.get("n_heads", 2),
            n_layers=model_cfg.get("n_layers", 3),
            d_ff=model_cfg.get("d_ff", 128),
            dropout=model_cfg.get("dropout", 0.1),
            lambda_=model_cfg.get("lambda_", 3.0),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
