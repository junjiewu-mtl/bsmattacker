#!/usr/bin/env python3
"""
run_benchmark — V2X misbehavior-detection benchmark driver
==========================================================
Trains the nine-detector zoo on the synthetic corpus and evaluates it
against the injected attack suite on both synthetic and real-world corpora
in a single pipeline (sim-to-real transfer + detector comparison).

Usage:
    # Synthetic corpus (VeReMi), standard 11-feature set
    python -m benchmark.run_benchmark --corpus synthetic

    # Real corpus, standard features
    python -m benchmark.run_benchmark --corpus real

    # Filter detectors/attacks
    python -m benchmark.run_benchmark --models gb lstm --attacks constant_speed

    # Cross-eval only (requires synthetic results to exist)
    python -m benchmark.run_benchmark --cross-eval

    # Auto-parallel across GPUs (launches N subprocesses, 1 per GPU)
    python -m benchmark.run_benchmark --corpus synthetic --parallel
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import yaml

# ---------------------------------------------------------------------------
# GPU visibility: let --parallel handle CUDA_VISIBLE_DEVICES per subprocess.
# Only default to GPU 0 when running as a single (non-parallel) process
# and no explicit CUDA_VISIBLE_DEVICES is set by the caller.
# ---------------------------------------------------------------------------
# (Removed module-level CUDA_VISIBLE_DEVICES=0; set in main() for non-parallel)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[0]

sys.path.insert(0, str(PROJECT_ROOT))

# Shared injection / feature-engineering infrastructure
from benchmark.injection_engine import (  # noqa: E402
    load_corpus,
    label_scenarios,
    inject_single_attack_sweep,
    inject_single_attack_sweep_vectorized,
    add_derived_features,
    clean_real_site,
    BSMSequenceDataset,
    evaluate,
    evaluate_v62,
)
from benchmark.plausibility_check import apply_plausibility_check  # noqa: E402
from benchmark.detectors import (  # noqa: E402
    build_model, count_parameters,
)


# ======================================================================
# Configuration
# ======================================================================

_WARNED_VECTOR_DISABLED = False

def load_config(config_path: Path = None) -> dict:
    if config_path is None:
        config_path = SCRIPT_DIR / "configs" / "benchmark.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device(preferred: str = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ======================================================================
# Corpus Loading
# ======================================================================

def load_synthetic_corpus(cfg: dict) -> pd.DataFrame:
    """Load VeReMi Extension benign corpus."""
    syn_path = PROJECT_ROOT / cfg["corpus"]["synthetic_path"]
    print(f"  Loading synthetic: {syn_path}")
    df = pd.read_parquet(syn_path)

    bridge = cfg.get("column_bridge", {})
    rename_map = {old: new for old, new in bridge.items() if old in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  Column bridge applied: {rename_map}")

    df["_source_id"] = "SYN"
    print(f"  Loaded: {len(df):,} rows, {df['device_id'].nunique():,} devices")
    return df


# ======================================================================
# Data Preparation
# ======================================================================

def _stratified_subsample(labels: np.ndarray, max_n: int, rng: np.random.RandomState) -> np.ndarray:
    """Subsample indices preserving class ratio."""
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    pos_ratio = len(pos_idx) / len(labels) if len(labels) > 0 else 0.5
    n_pos = min(len(pos_idx), int(max_n * pos_ratio))
    n_neg = min(len(neg_idx), max_n - n_pos)
    chosen = np.concatenate([
        rng.choice(pos_idx, n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int),
        rng.choice(neg_idx, n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int),
    ])
    chosen.sort()
    return chosen


def _validate_feature_schema(df: pd.DataFrame, all_features: list[str], context: str) -> None:
    """Fail fast on feature leakage or config/schema drift."""
    forbidden = {
        "is_attack",
        "attack_type",
        "Scenario_Label",
        "scenario_label",
        "label",
        "target",
    }
    leaked = sorted(set(all_features).intersection(forbidden))
    if leaked:
        raise ValueError(
            f"[{context}] Feature leakage risk: forbidden columns present in feature set: {leaked}"
        )

    missing = sorted([c for c in all_features if c not in df.columns])
    if missing:
        raise ValueError(
            f"[{context}] Missing configured feature columns after derivation: {missing}"
        )

    if len(all_features) != len(set(all_features)):
        dupes = sorted({c for c in all_features if all_features.count(c) > 1})
        raise ValueError(
            f"[{context}] Duplicate feature columns in configuration: {dupes}"
        )


def prepare_data(df_labeled: pd.DataFrame, cfg: dict, cartesian: bool = False):
    """Prepare train/test datasets from labeled DataFrame.

    Returns: (train_ds, test_ds, scaler, n_features, heldout_source_ds)
    """
    tcfg = cfg["training"]
    feat_cfg = cfg["features"]["standard"]
    all_features = feat_cfg["raw"] + feat_cfg["derived"]

    # Subsample large corpora by device BEFORE derived features / windowing
    # Default is 0 (no subsample) — train on the full VeReMi corpus.
    max_rows = tcfg.get("max_rows_per_split", 0)
    if max_rows > 0 and len(df_labeled) > max_rows:
        rng = np.random.RandomState(42)
        devices = df_labeled["device_id"].unique()
        dev_sizes = df_labeled.groupby("device_id").size()
        # Randomly select devices until we reach max_rows
        shuffled = rng.permutation(devices)
        cumsum = 0
        keep_devs = []
        for d in shuffled:
            cumsum += dev_sizes[d]
            keep_devs.append(d)
            if cumsum >= max_rows:
                break
        orig_len = len(df_labeled)
        df_labeled = df_labeled[df_labeled["device_id"].isin(set(keep_devs))].reset_index(drop=True)
        print(f"    [subsample] {len(df_labeled):,}/{orig_len:,} rows ({len(keep_devs)}/{len(devices)} devices)")

    # Compute derived features BEFORE fillna — see injection_engine.py fix comment.
    df = add_derived_features(df_labeled, cartesian=cartesian)
    _validate_feature_schema(df, all_features, context="prepare_data")

    features = df[all_features].values.astype(np.float32)
    labels = df["is_attack"].values.astype(np.float32)
    device_ids = df["device_id"].values
    source_ids = df["_source_id"].values if "_source_id" in df.columns else None

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Vehicle-level label-shuffle negative control. Permutes which vehicles
    # are labelled as attackers, keeping each vehicle's rows consistent.
    # Applied AFTER labels are read but BEFORE the split, so the shuffled
    # labels flow through scaler fitting, windowing, train/eval. Used to
    # distinguish genuine feature-based leakage from pipeline-level label
    # contamination.
    if cfg.get("controls", {}).get("shuffle_vehicle_labels", False):
        rng = np.random.RandomState(42)
        unique_devs = np.unique(device_ids)
        dev_to_label = dict(
            zip(unique_devs, pd.Series(labels).groupby(device_ids).max().reindex(unique_devs).values)
        )
        shuffled_devs = rng.permutation(unique_devs)
        new_dev_label = dict(
            zip(shuffled_devs, [dev_to_label[d] for d in unique_devs])
        )
        labels = np.array(
            [new_dev_label[d] for d in device_ids], dtype=np.float32
        )
        print(f"    [control] shuffle_vehicle_labels=true — labels permuted at vehicle level "
              f"({len(unique_devs)} vehicles reassigned)")

    # split_strategy gate. "per_device_temporal" does a per-device temporal
    # 80/20 split; "vehicle_disjoint" avoids the trajectory-identity leak by
    # assigning whole vehicles to either train or test.
    split_strategy = cfg.get("injection", {}).get("split_strategy", "per_device_temporal")

    if split_strategy == "vehicle_disjoint":
        rng = np.random.RandomState(42)
        unique_devs = np.unique(device_ids)
        dev_label = (
            pd.Series(labels).groupby(device_ids).max().reindex(unique_devs).values
        )
        attacker_devs = unique_devs[dev_label == 1]
        benign_devs = unique_devs[dev_label == 0]
        rng.shuffle(attacker_devs); rng.shuffle(benign_devs)
        n_a_train = int(len(attacker_devs) * tcfg["train_ratio"])
        n_b_train = int(len(benign_devs) * tcfg["train_ratio"])
        train_devs = set(attacker_devs[:n_a_train]) | set(benign_devs[:n_b_train])
        train_mask = np.isin(device_ids, list(train_devs))
        assert set(device_ids[train_mask]).isdisjoint(
            set(device_ids[~train_mask])
        ), "vehicle-disjoint split failed"
    elif split_strategy == "per_device_temporal":
        # Per-device temporal split (80/20)
        train_mask = np.zeros(len(df), dtype=bool)
        for did in np.unique(device_ids):
            dev_idx = np.where(device_ids == did)[0]
            split_at = int(len(dev_idx) * tcfg["train_ratio"])
            train_mask[dev_idx[:split_at]] = True
    else:
        raise ValueError(f"Unknown split_strategy: {split_strategy}")

    X_train_raw = features[train_mask]
    X_test_raw = features[~train_mask]
    y_train = labels[train_mask]
    y_test = labels[~train_mask]
    dev_train = device_ids[train_mask]
    dev_test = device_ids[~train_mask]

    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Clip extreme values (±10σ)
    X_train = np.clip(X_train, -10, 10)
    X_test = np.clip(X_test, -10, 10)

    seq_len = tcfg["seq_len"]
    train_ds = BSMSequenceDataset(X_train, y_train, dev_train, seq_len)
    test_ds = BSMSequenceDataset(X_test, y_test, dev_test, seq_len)

    n_features = features.shape[1]

    # Optional held-out source breakout: if the corpus tags rows with a
    # specific source id, evaluate that source separately from the rest.
    heldout_source_ds = None
    if source_ids is not None:
        heldout_source = cfg["corpus"].get("heldout_source")
        if heldout_source is not None:
            heldout_mask = (~train_mask) & (source_ids == heldout_source)
            if heldout_mask.any():
                X_heldout = np.clip(scaler.transform(features[heldout_mask]), -10, 10)
                try:
                    heldout_source_ds = BSMSequenceDataset(
                        X_heldout, labels[heldout_mask],
                        device_ids[heldout_mask], seq_len,
                    )
                except ValueError:
                    pass

    return train_ds, test_ds, scaler, n_features, heldout_source_ds


def prepare_cross_eval_data(
    df_real_labeled: pd.DataFrame,
    scaler: StandardScaler,
    cfg: dict,
):
    """Prepare real test set using a scaler fitted on synthetic data.

    For cross-evaluation: train on synthetic, test on real.
    """
    tcfg = cfg["training"]
    feat_cfg = cfg["features"]["standard"]
    all_features = feat_cfg["raw"] + feat_cfg["derived"]

    # Compute derived features BEFORE fillna — see injection_engine.py fix comment.
    df = add_derived_features(df_real_labeled, cartesian=False)
    _validate_feature_schema(df, all_features, context="prepare_cross_eval_data")

    features = df[all_features].values.astype(np.float32)
    labels = df["is_attack"].values.astype(np.float32)
    device_ids = df["device_id"].values

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Use only the test split (20% tail per device)
    test_mask = np.zeros(len(df), dtype=bool)
    for did in np.unique(device_ids):
        dev_idx = np.where(device_ids == did)[0]
        split_at = int(len(dev_idx) * tcfg["train_ratio"])
        test_mask[dev_idx[split_at:]] = True

    X_test = np.clip(scaler.transform(features[test_mask]), -10, 10)
    y_test = labels[test_mask]
    dev_test = device_ids[test_mask]

    seq_len = tcfg["seq_len"]
    return BSMSequenceDataset(X_test, y_test, dev_test, seq_len)


# ======================================================================
# Training Dispatch (4-way)
# ======================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        n += len(y_batch)
    return total_loss / max(n, 1)


def train_supervised_deep(model, train_ds, cfg, device, model_cfg=None):
    """Train a supervised deep model (e.g. LSTM) with BCEWithLogitsLoss + val split."""
    tcfg = cfg["training"]
    model_cfg = model_cfg or {}

    # 80/20 train/val split
    n_total = len(train_ds)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(42)
    train_sub, val_sub = torch.utils.data.random_split(
        train_ds, [n_train, n_val], generator=g,
    )

    batch_size = model_cfg.get("batch_size", tcfg["batch_size"])
    train_loader = DataLoader(train_sub, batch_size=batch_size,
                              shuffle=True, generator=g)
    val_loader = DataLoader(val_sub, batch_size=batch_size)

    # pos_weight
    y_all = np.array([train_ds[i][1].item() for i in range(len(train_ds))])
    n_pos = y_all.sum()
    n_neg = len(y_all) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    lr = model_cfg.get("learning_rate", tcfg["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5,
    )

    best_val_loss = float("inf")
    best_state = None
    history = []
    patience_counter = 0
    early_stop_patience = 5

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, device)
        # Val loss
        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item() * len(y_b)
                val_n += len(y_b)
        val_loss /= max(val_n, 1)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss,
                        "lr": optimizer.param_groups[0]["lr"]})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}/{tcfg['epochs']}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                  f"({time.time() - t0:.1f}s)")

        if patience_counter >= early_stop_patience:
            print(f"      Early stop at epoch {epoch} (patience={early_stop_patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_gb(model, train_ds, cfg):
    """Train Gradient Boosting on ALL labeled data (supervised)."""
    X_all, y_all = [], []
    for i in range(len(train_ds)):
        x, y = train_ds[i]
        X_all.append(x.numpy())
        y_all.append(y.item())
    X = np.stack(X_all)
    y = np.array(y_all)
    print(f"    GB supervised: {len(X):,} windows "
          f"({int(y.sum()):,} attack, {int((1-y).sum()):,} benign)")
    model.fit(X, y)
    return model, []


def train_iforest(model, train_ds, cfg):
    """Train sklearn unsupervised models on benign-only data.

    For models with O(n^2+) complexity (LOF, OCSVM, DIF), subsample
    to max 200K windows to avoid multi-hour training on 1.5M rows.
    IForest and Deep SVDD are O(n log n) and handle full data.
    """
    benign_windows = []
    for i in range(len(train_ds)):
        x, y = train_ds[i]
        if y.item() == 0.0:
            benign_windows.append(x.numpy())
    X_benign = np.stack(benign_windows)

    # Subsample for O(n^2+) sklearn models
    model_name = model.__class__.__name__
    max_sklearn = 500_000  # cap for tractable density estimation
    if len(X_benign) > max_sklearn and model_name in (
        "LOFIDS", "OneClassSVMIDS", "DIFIDS", "DeepSVDDIDS",
    ):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_benign), max_sklearn, replace=False)
        print(f"    [sklearn subsample] {max_sklearn:,}/{len(X_benign):,} "
              f"for {model_name} (O(n^2) complexity)")
        X_benign = X_benign[idx]

    print(f"    {model_name} unsupervised: {len(X_benign):,} benign windows "
          f"(filtered from {len(train_ds):,} total)")
    model.fit(X_benign)
    return model, []


def train_unsupervised_deep(model, train_ds, cfg, device, model_cfg=None):
    """Train any unsupervised deep model on benign-only data.

    Works with the Anomaly Transformer, Deep SVDD, Deep Isolation Forest,
    GDN — any model exposing train_unsupervised() and calibrate_threshold().
    """
    tcfg = cfg["training"]
    model_cfg = model_cfg or {}
    benign_indices = [i for i in range(len(train_ds))
                      if train_ds[i][1].item() == 0.0]

    # Subsample benign windows to 200K for tractable DL training
    max_unsup = 500_000  # cap for tractable density estimation
    if len(benign_indices) > max_unsup:
        rng = np.random.RandomState(42)
        benign_indices = list(rng.choice(benign_indices, max_unsup, replace=False))

    benign_subset = torch.utils.data.Subset(train_ds, benign_indices)
    model_name = model.__class__.__name__
    print(f"    {model_name} unsupervised: {len(benign_subset):,} benign windows "
          f"(filtered from {len(train_ds):,} total)")

    # Per-model batch_size and learning_rate (falls back to global training config)
    batch_size = model_cfg.get("batch_size", tcfg["batch_size"])
    lr = model_cfg.get("learning_rate", tcfg.get("learning_rate", 0.001))

    g = torch.Generator().manual_seed(42)
    benign_loader = DataLoader(benign_subset, batch_size=batch_size,
                               shuffle=True, generator=g)

    model.to(device)
    n_epochs = model_cfg.get("epochs", tcfg["epochs"])
    history = model.train_unsupervised(
        benign_loader, device,
        epochs=n_epochs,
        lr=lr,
        print_every=10,
    )

    calib_loader = DataLoader(benign_subset, batch_size=batch_size)
    thresh = model.calibrate_threshold(calib_loader, device, percentile=95)
    print(f"    Anomaly threshold (95th percentile): {thresh:.4f}")
    return model, history


def dispatch_training(model, model_name, model_cfg, train_ds, cfg, device):
    """4-way training dispatch."""
    is_sklearn = getattr(model, "is_sklearn", False)
    is_unsup = getattr(model, "is_unsupervised", False)

    if is_sklearn and not is_unsup:
        # Gradient Boosting — supervised sklearn
        return train_gb(model, train_ds, cfg)
    elif is_sklearn and is_unsup:
        # IForest — unsupervised sklearn
        return train_iforest(model, train_ds, cfg)
    elif is_unsup:
        # Any unsupervised deep model
        return train_unsupervised_deep(model, train_ds, cfg, device, model_cfg)
    else:
        # LSTM — supervised deep
        return train_supervised_deep(model, train_ds, cfg, device, model_cfg)


# ======================================================================
# Injection
# ======================================================================

def inject_attack(df_scenario, attack_name, attack_params, cfg, cartesian=False,
                  dataset_class="synthetic"):
    """Inject attack with coordinate-aware dispatch.

    Args:
        dataset_class: "synthetic" (VeReMi) or "real_site". Real-site
            cross-eval must NOT receive the synthetic Kamel symmetric noise
            (real BSMs already carry natural GPS noise).
            This tag is surfaced to injection_engine's apply_sensor_noise blocks
            via inject_cfg["_dataset_class"], gated by
            cfg.injection.noise_apply_to_real_sites.
    """
    # Flatten injection params to top level (inject_single_attack_sweep expects
    # cfg["random_seed"], cfg["benign_fraction"], cfg["segment_size"] at root)
    inject_cfg = {**cfg}
    if "injection" in cfg:
        for k, v in cfg["injection"].items():
            inject_cfg.setdefault(k, v)
    inject_cfg.setdefault("corpus", {})
    inject_cfg["_dataset_class"] = dataset_class
    if cartesian:
        inject_cfg["corpus"]["coordinate_system"] = "cartesian"
    else:
        inject_cfg["corpus"]["coordinate_system"] = "gps"

    use_vec = os.environ.get("VECTORIZED_INJECTION", "1") == "1"
    strict_vec = bool(inject_cfg.get("strict_vectorized_injection", False))
    global _WARNED_VECTOR_DISABLED
    if not use_vec:
        msg = (
            "VECTORIZED_INJECTION=0 detected. Benchmark semantics typically "
            "expect vectorized segment-assignment injection."
        )
        if strict_vec:
            raise RuntimeError(f"[inject_attack] {msg} strict_vectorized_injection=True")
        if not _WARNED_VECTOR_DISABLED:
            print(f"    [WARN] {msg}")
            _WARNED_VECTOR_DISABLED = True

    if use_vec:
        try:
            return inject_single_attack_sweep_vectorized(
                df_scenario, attack_name, attack_params, inject_cfg,
            )
        except Exception as e:
            if strict_vec:
                raise RuntimeError(
                    "[inject_attack] Vectorized injection failed and strict mode is enabled: "
                    f"{e}"
                ) from e
            print(f"    [WARN] Vectorized injection failed: {e}, falling back")

    return inject_single_attack_sweep(
        df_scenario, attack_name, attack_params, inject_cfg,
    )


# ======================================================================
# Single Trial
# ======================================================================

def run_single_trial(
    model_name, model_cfg, attack_name, attack_params,
    df_scenario, cfg, device, results_dir,
    cartesian=False,
    df_real_scenario=None,
):
    """Run one (model, attack) trial. Optionally cross-eval on real."""
    print(f"\n  [{model_name} x {attack_name}] Injecting...")

    # Realism gate — filter short/incompatible devices
    if cfg.get("plausibility_check", {}).get("enabled", False):
        df_gated, gate_log = apply_plausibility_check(df_scenario, attack_name)
        print(f"    Realism gate: {gate_log['passed']} pass, "
              f"{gate_log['failed']} fail")
        for site, stats in sorted(gate_log["by_site"].items()):
            print(f"      {site}: {stats['pass']} pass, {stats['fail']} fail")
        if gate_log["passed"] == 0:
            print(f"    [SKIP] All devices filtered by realism gate")
            return _empty_result(model_name, model_cfg, attack_name, attack_params)
    else:
        df_gated = df_scenario

    # Inject on primary corpus
    df_labeled = inject_attack(
        df_gated, attack_name, attack_params, cfg, cartesian=cartesian,
    )

    n_attacked = int((df_labeled["is_attack"] == 1).sum())
    if n_attacked == 0:
        print(f"    [SKIP] No rows attacked.")
        return _empty_result(model_name, model_cfg, attack_name, attack_params)

    attack_ratio = n_attacked / len(df_labeled)

    # Prepare data
    print(f"    Preparing data (cartesian={cartesian})...")
    train_ds, test_ds, scaler, n_features, heldout_source_ds = prepare_data(
        df_labeled, cfg, cartesian=cartesian,
    )
    print(f"    Train: {len(train_ds):,}  |  Test: {len(test_ds):,}")

    # Build model
    seq_len = cfg["training"]["seq_len"]
    model = build_model(model_cfg, input_dim=n_features, seq_len=seq_len)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"    Model: {model_name} ({n_params:,} params)")

    # Train
    t0 = time.time()
    model, history = dispatch_training(
        model, model_name, model_cfg, train_ds, cfg, device,
    )
    train_time = time.time() - t0
    print(f"    Training time: {train_time:.1f}s")

    # Evaluate on primary test set
    tcfg = cfg["training"]
    eval_batch = model_cfg.get("batch_size", tcfg["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=eval_batch)

    # Detailed-eval mode: save predictions + balanced metrics + bootstrap CIs
    detailed_eval_mode = cfg.get("_detailed_eval", False)
    _eval_fn = evaluate_v62 if detailed_eval_mode else evaluate

    model_dir = results_dir / f"model_{model_name}_{attack_name}"
    if detailed_eval_mode:
        test_metrics = _eval_fn(model, test_loader, device, save_dir=model_dir)
    else:
        test_metrics = _eval_fn(model, test_loader, device)

    print(f"    Test: F1={test_metrics['f1']:.4f}  "
          f"P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  "
          f"AUROC={test_metrics['auc']:.4f}  "
          f"AUPRC={test_metrics['auprc']:.4f}"
          f"  n+={test_metrics['n_positive']}")
    if detailed_eval_mode and "balanced_auroc" in test_metrics:
        print(f"    Balanced AUROC={test_metrics['balanced_auroc']:.4f} "
              f"CI=[{test_metrics['auroc_ci_lower']:.4f}, {test_metrics['auroc_ci_upper']:.4f}]"
              f"{'  *** LOW-N' if test_metrics.get('low_n') else ''}"
              f"{'  *** WIDE-CI' if test_metrics.get('wide_ci') else ''}")

    # Cross-eval on real corpus (if synthetic training)
    cross_metrics = {}
    if df_real_scenario is not None:
        print(f"    Cross-eval on real corpus...")
        # Gate the real corpus before cross-eval injection
        _real_inject = df_real_scenario
        if cfg.get("plausibility_check", {}).get("enabled", False):
            _real_inject, _rg = apply_plausibility_check(
                df_real_scenario, attack_name)
            print(f"      Gate: {_rg['passed']} pass, {_rg['failed']} fail")
        if len(_real_inject) == 0:
            n_real_attacked = 0
        else:
            df_real_labeled = inject_attack(
                _real_inject, attack_name, attack_params, cfg, cartesian=False,
                dataset_class="real_site",  # real-site cross-eval: skip synthetic noise
            )
            n_real_attacked = int((df_real_labeled["is_attack"] == 1).sum())
        if n_real_attacked > 0:
            real_test_ds = prepare_cross_eval_data(df_real_labeled, scaler, cfg)
            real_loader = DataLoader(real_test_ds, batch_size=eval_batch)
            if detailed_eval_mode:
                cross_save = model_dir / "cross_predictions"
                cross_metrics = _eval_fn(model, real_loader, device, save_dir=cross_save)
            else:
                cross_metrics = _eval_fn(model, real_loader, device)
            print(f"    Cross: F1={cross_metrics['f1']:.4f}  "
                  f"AUROC={cross_metrics['auc']:.4f}  "
                  f"AUPRC={cross_metrics['auprc']:.4f}")
            del df_real_labeled, real_test_ds
        else:
            print(f"    Cross-eval SKIPPED: 0 attacked rows on real corpus "
                  f"(attack '{attack_name}' may require scenarios absent in real data)")

    # Held-out source breakout
    heldout_source_metrics = {}
    if heldout_source_ds is not None and len(heldout_source_ds) > 0:
        heldout_source_loader = DataLoader(heldout_source_ds, batch_size=eval_batch)
        heldout_source_metrics = evaluate(model, heldout_source_loader, device)

    # Save artifacts
    model_dir = results_dir / f"model_{model_name}_{attack_name}"
    model_dir.mkdir(parents=True, exist_ok=True)

    if getattr(model, "is_sklearn", False):
        joblib.dump(model, model_dir / "best_model.joblib")
    else:
        torch.save(model.state_dict(), model_dir / "best_model.pt")

    joblib.dump(scaler, model_dir / "scaler.joblib")
    with open(model_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(model_dir / "val_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    if cross_metrics:
        with open(model_dir / "cross_metrics.json", "w") as f:
            json.dump(cross_metrics, f, indent=2)
    if heldout_source_metrics:
        with open(model_dir / "heldout_source_metrics.json", "w") as f:
            json.dump(heldout_source_metrics, f, indent=2)

    result = {
        "model": model_name,
        "model_type": model_cfg["type"],
        "attack": attack_name,
        "family": attack_params.get("family", "unknown"),
        "auroc": test_metrics["auc"],
        "auprc": test_metrics["auprc"],
        "f1_optimal": test_metrics.get("f1_optimal", 0.0),
        "fpr_at_95recall": test_metrics.get("fpr_at_95recall", 1.0),
        "f1": test_metrics["f1"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "fpr": test_metrics["fpr"],
        "train_time_s": round(train_time, 1),
        "n_params": n_params,
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        # Cross-eval columns
        "cross_auroc": cross_metrics.get("auc"),
        "cross_auprc": cross_metrics.get("auprc"),
        "cross_f1": cross_metrics.get("f1"),
        "cross_f1_optimal": cross_metrics.get("f1_optimal"),
        "cross_fpr_at_95recall": cross_metrics.get("fpr_at_95recall"),
        # Held-out source breakout
        "heldout_source_f1": heldout_source_metrics.get("f1"),
    }

    # Detailed-eval extra columns
    if detailed_eval_mode:
        result.update({
            "n_positive": test_metrics.get("n_positive"),
            "n_negative": test_metrics.get("n_negative"),
            "balanced_auroc": test_metrics.get("balanced_auroc"),
            "balanced_auroc_std": test_metrics.get("balanced_auroc_std"),
            "auroc_ci_lower": test_metrics.get("auroc_ci_lower"),
            "auroc_ci_upper": test_metrics.get("auroc_ci_upper"),
            "low_n": test_metrics.get("low_n"),
            "wide_ci": test_metrics.get("wide_ci"),
            "cross_n_positive": cross_metrics.get("n_positive"),
            "cross_n_negative": cross_metrics.get("n_negative"),
            "cross_balanced_auroc": cross_metrics.get("balanced_auroc"),
            "cross_auroc_ci_lower": cross_metrics.get("auroc_ci_lower"),
            "cross_auroc_ci_upper": cross_metrics.get("auroc_ci_upper"),
        })

    del df_labeled, train_ds, test_ds, model, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def _empty_result(model_name, model_cfg, attack_name, attack_params):
    return {
        "model": model_name, "model_type": model_cfg["type"],
        "attack": attack_name,
        "family": attack_params.get("family", "unknown"),
        "auroc": 0.0, "auprc": 0.0, "f1_optimal": 0.0,
        "fpr_at_95recall": 1.0,
        "f1": 0.0, "precision": 0.0, "recall": 0.0, "fpr": 0.0,
        "train_time_s": 0.0, "n_params": 0, "n_train": 0, "n_test": 0,
        "cross_auroc": None, "cross_auprc": None, "cross_f1": None,
        "cross_f1_optimal": None, "cross_fpr_at_95recall": None,
        "heldout_source_f1": None,
    }


# ======================================================================
# CLI & Main
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="V2X misbehavior-detection benchmark driver")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--corpus", choices=["synthetic", "real"],
                   default="synthetic",
                   help="Training corpus (synthetic=VeReMi, real=OBU)")
    p.add_argument("--real-corpus", choices=["core", "full"],
                   default="full",
                   help="Which real corpus to load")
    p.add_argument("--models", nargs="+", default=None,
                   help="Run subset of models")
    p.add_argument("--attacks", nargs="+", default=None,
                   help="Run subset of attacks")
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    p.add_argument("--cross-eval", action="store_true",
                   help="Include cross-eval on real corpus (synthetic mode)")
    p.add_argument("--parallel", action="store_true",
                   help="Auto-parallelize across available GPUs")
    p.add_argument("--gpu-id", type=int, default=None,
                   help="(internal) GPU id for per-GPU CSV output in parallel mode")
    p.add_argument("--detailed-eval", action="store_true",
                   help="Detailed-eval mode: save predictions, balanced eval, bootstrap CIs")
    p.add_argument("--random-seed", type=int, default=None,
                   help="Override cfg['injection']['random_seed'] "
                        "(for seed-variance runs).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override cfg['output_dir'].")
    p.add_argument("--trial-list", type=Path, default=None,
                   help="(internal) File with explicit model,attack pairs "
                        "(one per line). Used by --parallel to avoid Cartesian-"
                        "product dispatch bug. When set, --models/--attacks are "
                        "ignored and only listed pairs are run.")
    return p.parse_args()


def _run_parallel(args, cfg):
    """Launch one subprocess per GPU, splitting trials across them."""
    import subprocess

    # Honor pre-set CUDA_VISIBLE_DEVICES to allow co-running with
    # other GPU jobs. If unset, fall back to nvidia-smi auto-detection.
    preset = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if preset:
        gpu_ids = [int(x.strip()) for x in preset.split(",") if x.strip()]
        n_gpus = len(gpu_ids)
        print(f"[parallel] honoring preset CUDA_VISIBLE_DEVICES={preset} "
              f"(using {n_gpus} GPU(s): {gpu_ids})")
    else:
        # Detect GPUs via nvidia-smi (more reliable than torch on WDDM multi-GPU)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            n_gpus = len([x for x in result.stdout.strip().split("\n") if x.strip()])
        except Exception:
            n_gpus = 0
        gpu_ids = list(range(n_gpus))
    if n_gpus == 0:
        print("[parallel] No GPUs found, falling back to sequential")
        return False

    # Build list of (model, attack) trials to run
    model_names = list(cfg["models"].keys())
    if args.models:
        model_names = [m for m in model_names if m in args.models]
    attack_names = list(cfg["attack_configs"].keys())
    if args.attacks:
        attack_names = [a for a in attack_names if a in args.attacks]

    # Check existing results to skip
    results_dir = Path(cfg["output_dir"])
    existing_keys = set()
    out_path = results_dir / f"benchmark_{args.corpus}_standard.csv"
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            existing_keys = {(r["model"], r["attack"])
                             for _, r in existing_df.iterrows()}
        except Exception:
            pass
    # Also check model dirs
    if results_dir.exists():
        for d in results_dir.glob("model_*"):
            parts = d.name[len("model_"):]
            for m in ["lstm", "gb", "iforest", "lof", "ocsvm",
                      "anomaly_transformer", "deep_svdd", "dif", "gdn"]:
                if parts.startswith(m + "_"):
                    existing_keys.add((m, parts[len(m) + 1:]))
                    break

    trials = [(m, a) for m in model_names for a in attack_names
              if (m, a) not in existing_keys]
    if not trials:
        print("[parallel] All trials already complete, nothing to run")
        return True

    print(f"[parallel] {len(trials)} trials across {n_gpus} GPUs")

    # Split trials across GPUs (cost-aware greedy load balancing)
    # Anomaly Transformer ~10min, LSTM ~15min, IForest/GB ~30s per attack
    MODEL_COST = {"anomaly_transformer": 10, "deep_svdd": 7, "gdn": 5,
                   "dif": 4, "lstm": 15,
                   "lof": 2, "ocsvm": 2, "iforest": 1, "gb": 1}
    # Sort heaviest first for better bin-packing
    trials.sort(key=lambda t: MODEL_COST.get(t[0], 1), reverse=True)
    gpu_trials = [[] for _ in range(n_gpus)]
    gpu_cost = [0.0] * n_gpus
    for m, a in trials:
        lightest = min(range(n_gpus), key=lambda g: gpu_cost[g])
        gpu_trials[lightest].append((m, a))
        gpu_cost[lightest] += MODEL_COST.get(m, 1)
    for g in range(n_gpus):
        if gpu_trials[g]:
            models_g = set(m for m, _ in gpu_trials[g])
            print(f"  GPU {gpu_ids[g]}: {len(gpu_trials[g])} trials "
                  f"(est. cost={gpu_cost[g]:.0f}) — {models_g}")

    # Launch subprocesses
    # Fix for Cartesian-product dispatch bug: write an explicit trial-list
    # file per GPU (one "model,attack" pair per line) and pass --trial-list
    # instead of --models <union> --attacks <union>. The union form caused
    # every subprocess to expand to the full Cartesian product (every GPU
    # redundantly ran all 250 trials with identical seed).
    script = str(Path(__file__).resolve())
    procs = []
    trial_list_dir = results_dir / "parallel_trial_lists"
    trial_list_dir.mkdir(parents=True, exist_ok=True)
    for slot_idx, gt in enumerate(gpu_trials):
        if not gt:
            continue
        # Map slot index to actual physical GPU id from gpu_ids
        # (allows running on a subset like CUDA_VISIBLE_DEVICES=0,1,3).
        physical_gpu_id = gpu_ids[slot_idx]

        # Write explicit (model, attack) list for this GPU
        trial_list_path = trial_list_dir / f"gpu{physical_gpu_id}_trials.txt"
        with open(trial_list_path, "w") as f:
            f.write("# explicit per-GPU trial list to avoid\n")
            f.write("# Cartesian-product dispatch bug in _run_parallel.\n")
            for m, a in gt:
                f.write(f"{m},{a}\n")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
        env["VECTORIZED_INJECTION"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        # Pass --trial-list so the child runs exactly the bin-packed pairs.
        # --models/--attacks are intentionally omitted; --trial-list overrides.
        cmd = [sys.executable, "-u", script,
               "--corpus", args.corpus,
               "--gpu-id", str(physical_gpu_id),
               "--trial-list", str(trial_list_path),
               "--device", "cuda"]
        if args.cross_eval:
            cmd.append("--cross-eval")
        if getattr(args, "detailed_eval", False):
            cmd.append("--detailed-eval")
        if args.config:
            cmd.extend(["--config", str(args.config)])
        # Propagate CLI overrides
        if getattr(args, "random_seed", None) is not None:
            cmd.extend(["--random-seed", str(args.random_seed)])
        if getattr(args, "output_dir", None) is not None:
            cmd.extend(["--output-dir", str(args.output_dir)])

        log_path = results_dir / f"parallel_gpu{physical_gpu_id}.log"
        log_file = open(log_path, "w")
        print(f"  GPU {physical_gpu_id}: {len(gt)} trials -> {log_path} "
              f"(trial-list: {trial_list_path.name})")
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
        procs.append((p, log_file, physical_gpu_id))

    print(f"  {len(procs)} subprocesses launched, waiting...")

    # Wait for all
    for p, lf, gid in procs:
        p.wait()
        lf.close()
        status = "OK" if p.returncode == 0 else f"FAILED (exit {p.returncode})"
        print(f"  GPU {gid}: {status}")

    failed = [gid for p, _, gid in procs if p.returncode != 0]
    if failed:
        print(f"[parallel] WARNING: GPUs {failed} had errors, check logs")
    else:
        print("[parallel] All subprocesses completed successfully")

    # Merge per-GPU CSVs into single results file
    merged_path = results_dir / f"benchmark_{args.corpus}_standard.csv"
    gpu_csvs = sorted(results_dir.glob(f"benchmark_{args.corpus}_standard_gpu*.csv"))
    if gpu_csvs:
        frames = []
        for csv_path in gpu_csvs:
            try:
                frames.append(pd.read_csv(csv_path))
            except Exception as e:
                print(f"  WARNING: failed to read {csv_path}: {e}")
        # Include existing merged CSV if present
        if merged_path.exists():
            try:
                frames.insert(0, pd.read_csv(merged_path))
            except Exception:
                pass
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            # Deduplicate: keep last (most recent) per (model, attack)
            merged = merged.drop_duplicates(
                subset=["model", "attack"], keep="last"
            )
            merged.to_csv(merged_path, index=False)
            print(f"[parallel] Merged {len(merged)} trials -> {merged_path}")
            # Clean up per-GPU CSVs
            for csv_path in gpu_csvs:
                csv_path.unlink()
    return True


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ── Global reproducibility seeds ──
    import random as _random
    _random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = "42"

    # Detailed-eval mode flag
    if getattr(args, "detailed_eval", False):
        cfg["_detailed_eval"] = True

    # CLI overrides
    if getattr(args, "random_seed", None) is not None:
        cfg.setdefault("injection", {})["random_seed"] = int(args.random_seed)
        print(f"  [--random-seed] Overrode cfg.injection.random_seed = {args.random_seed}")
    if getattr(args, "output_dir", None) is not None:
        cfg["output_dir"] = str(args.output_dir)
        print(f"  [--output-dir] Overrode cfg.output_dir = {args.output_dir}")

    # Auto-parallel mode: launch subprocesses and exit
    if args.parallel:
        _run_parallel(args, cfg)
        return

    # Non-parallel: default to GPU 0 if caller didn't set CUDA_VISIBLE_DEVICES
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    print("=" * 70)
    print("V2X misbehavior-detection benchmark")
    print(f"  Corpus: {args.corpus}")
    print(f"  Cross-eval: {args.cross_eval}")
    print("=" * 70)

    device = get_device(args.device)
    print(f"  Device: {device}")

    # Output paths
    results_dir = PROJECT_ROOT / cfg["output_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # --trial-list: explicit (model, attack) pairs (fixes _run_parallel
    # Cartesian-product dispatch bug). When set, overrides --models/--attacks.
    allowed_pairs = None
    if args.trial_list is not None:
        with open(args.trial_list) as f:
            allowed_pairs = set()
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    raise ValueError(
                        f"--trial-list line must be 'model,attack'; got: {line!r}"
                    )
                allowed_pairs.add((parts[0], parts[1]))
        print(f"  --trial-list: {len(allowed_pairs)} explicit pairs from "
              f"{args.trial_list.name}")

    # Filter models and attacks
    model_configs = cfg["models"]
    if args.models:
        model_configs = {k: v for k, v in model_configs.items()
                         if k in args.models}

    attack_configs = cfg["attack_configs"]
    if args.attacks:
        attack_configs = {k: v for k, v in attack_configs.items()
                          if k in args.attacks}

    # When --trial-list is set, further restrict the Cartesian of configs to
    # just the explicit pairs; n_trials then equals |allowed_pairs|.
    if allowed_pairs is not None:
        n_trials = len(allowed_pairs)
    else:
        n_trials = len(model_configs) * len(attack_configs)
    print(f"  Models: {len(model_configs)} ({', '.join(model_configs.keys())})")
    print(f"  Attacks: {len(attack_configs)}")
    print(f"  Total trials: {n_trials}")

    # Load corpus
    cartesian = False
    if args.corpus == "synthetic":
        print(f"\nLoading VeReMi Extension synthetic corpus...")
        df_pooled = load_synthetic_corpus(cfg)
        cartesian = True
    else:
        print(f"\nLoading {args.real_corpus} real corpus...")
        df_pooled = load_corpus(cfg, args.real_corpus)

    print("\nLabeling scenarios...")
    df_scenario = label_scenarios(df_pooled)
    del df_pooled
    gc.collect()

    # Optionally load real corpus for cross-eval
    df_real_scenario = None
    if args.cross_eval and args.corpus == "synthetic":
        print(f"\nLoading real corpus for cross-eval ({args.real_corpus})...")
        df_real = load_corpus(cfg, args.real_corpus)
        # Clean real-site data: parked-strip + 1 Hz downsample
        df_real = clean_real_site(df_real, site_name=args.real_corpus)
        df_real_scenario = label_scenarios(df_real)
        del df_real
        gc.collect()

    # Load existing results (incremental)
    # In parallel mode, each subprocess writes its own CSV to avoid race conditions
    csv_suffix = f"_gpu{args.gpu_id}" if args.gpu_id is not None else ""
    out_path = results_dir / f"benchmark_{args.corpus}_standard{csv_suffix}.csv"
    existing_results = []
    existing_keys = set()
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            existing_results = existing_df.to_dict("records")
            existing_keys = {(r["model"], r["attack"])
                             for r in existing_results}
            if existing_keys:
                print(f"  Loaded {len(existing_keys)} existing results")
        except Exception:
            pass
    # Also check model dirs (covers parallel runs where CSV may not exist yet)
    if results_dir.exists():
        for d in results_dir.glob("model_*"):
            parts = d.name[len("model_"):]
            for m in ["lstm", "gb", "iforest", "lof", "ocsvm",
                      "anomaly_transformer", "deep_svdd", "dif", "gdn"]:
                if parts.startswith(m + "_"):
                    existing_keys.add((m, parts[len(m) + 1:]))
                    break

    # Run sweep
    new_results = []
    trial_num = 0
    t_start = time.time()

    for model_name, model_cfg in model_configs.items():
        for attack_name, attack_params in attack_configs.items():
            # --trial-list filter: skip pairs not in the explicit allowlist
            if allowed_pairs is not None and (model_name, attack_name) not in allowed_pairs:
                continue

            trial_num += 1

            if (model_name, attack_name) in existing_keys:
                print(f"\n  [{trial_num}/{n_trials}] "
                      f"{model_name} x {attack_name} — SKIPPED")
                continue

            print(f"\n{'='*70}")
            print(f"  [{trial_num}/{n_trials}] {model_name} x {attack_name}")
            print(f"{'='*70}")

            try:
                result = run_single_trial(
                    model_name, model_cfg,
                    attack_name, attack_params,
                    df_scenario, cfg, device, results_dir,
                    cartesian=cartesian,
                    df_real_scenario=df_real_scenario,
                )
                result["corpus"] = args.corpus
                result["feature_set"] = "standard"
                new_results.append(result)
            except Exception as e:
                print(f"    [ERROR] {e}")
                import traceback
                traceback.print_exc()
                continue

            # Save after each trial
            all_results = existing_results + new_results
            pd.DataFrame(all_results).to_csv(out_path, index=False)

    elapsed = time.time() - t_start

    # Final save
    all_results = existing_results + new_results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_path, index=False)

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE — {len(all_results)} trials in {elapsed:.0f}s")
    print(f"  Results: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
