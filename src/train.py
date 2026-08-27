# -*- coding: utf-8 -*-
"""Train ViT multi-head regression with 5-fold CV."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

_V1_ROOT = Path(__file__).resolve().parents[1]
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

from src.data.dicom_dataset import (
    HEADS,
    InfectivityDataset,
    build_slice_samples,
    collate_batch,
    index_dicoms,
    load_labels_csv,
    study_balanced_weights,
)
from src.eval import aggregate_by_study, macro_average, per_head_metrics, save_metrics
from src.models.vit_multitask import ViTMultiTask, masked_mse_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def stratify_labels(labels: dict[int, np.ndarray], study_ids: list[int]) -> np.ndarray:
    """Binary stratify on D4 (liquid) >= 0.5 or any available head."""
    keys = []
    for s in study_ids:
        y = labels.get(s)
        if y is None or np.all(np.isnan(y)):
            keys.append(0)
        else:
            d4 = y[3]
            keys.append(1 if (not np.isnan(d4) and d4 >= 0.5) else 0)
    return np.array(keys, dtype=int)


def sample_strata(
    samples: list[tuple[int, Path]],
    labels: dict[int, np.ndarray],
    study_ids: list[int],
) -> np.ndarray:
    study_strat = dict(zip(study_ids, stratify_labels(labels, study_ids)))
    return np.array([study_strat[s] for s, _ in samples], dtype=int)


def split_train_val_samples(
    samples: list[tuple[int, Path]],
    *,
    seed: int,
    val_frac: float = 0.15,
) -> tuple[list[tuple[int, Path]], list[tuple[int, Path]]]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(samples))
    n_val = max(1, int(val_frac * len(samples)))
    val = [samples[i] for i in perm[:n_val]]
    train = [samples[i] for i in perm[n_val:]]
    return train, val


def count_unique_studies(samples: list[tuple[int, Path]]) -> int:
    return len({s for s, _ in samples})


@torch.no_grad()
def predict_loader(
    model: ViTMultiTask,
    loader: DataLoader,
    device: torch.device,
    *,
    study_aggregation: str | None = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, targets, study_nos = [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        pred = model(x).cpu().numpy()
        y = batch["y"].numpy()
        mask = batch["mask"].numpy()
        y_masked = y.copy()
        y_masked[mask == 0] = np.nan
        preds.append(pred)
        targets.append(y_masked)
        study_nos.extend(batch["study_no"])
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(targets)
    study_arr = np.array(study_nos, dtype=int)
    if study_aggregation and len(set(study_nos)) < len(study_nos):
        y_true, y_pred = aggregate_by_study(study_arr, y_pred, y_true, method=study_aggregation)
        study_arr = np.array(sorted(set(study_nos)), dtype=int)
    return y_pred, y_true, study_arr


def run_epoch(
    model: ViTMultiTask,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    amp: bool,
    scaler: torch.cuda.amp.GradScaler | None,
    desc: str = "epoch",
) -> float:
    train = optimizer is not None
    model.train(train)
    losses = []
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    pred = model(x)
                    loss = masked_mse_loss(pred, y, mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = masked_mse_loss(pred, y, mask)
                loss.backward()
                optimizer.step()
        else:
            pred = model(x)
            loss = masked_mse_loss(pred, y, mask)
        lv = float(loss.detach().cpu())
        losses.append(lv)
        pbar.set_postfix(loss=f"{lv:.4f}")
    return float(np.mean(losses)) if losses else float("nan")


def train_fold(
    fold: int,
    train_samples: list[tuple[int, Path]],
    val_samples: list[tuple[int, Path]],
    test_samples: list[tuple[int, Path]],
    cfg: dict,
    labels: dict[int, np.ndarray],
    out_dir: Path,
    device: torch.device,
    *,
    cv_unit: str,
) -> dict:
    pp = cfg["preprocess"]
    mcfg = cfg["model"]
    tcfg = cfg["train"]
    slice_mode = cfg.get("data", {}).get("slice_mode", "all")
    study_agg = cfg.get("eval", {}).get("study_aggregation")
    if study_agg in (None, "none", "null", ""):
        study_agg = None
    balance_studies = bool(tcfg.get("balance_study_sampling", True))

    def make_loader(samples: list[tuple[int, Path]], *, train: bool) -> DataLoader:
        ds = InfectivityDataset(
            samples, labels,
            image_size=pp["image_size"],
            clahe=pp.get("clahe", True),
            clahe_clip_limit=pp.get("clahe_clip_limit", 0.03),
            resize_mode=pp.get("resize_mode", "stretch"),
        )
        n_studies = count_unique_studies(samples)
        loader_kwargs: dict = dict(
            batch_size=tcfg["batch_size"],
            num_workers=tcfg.get("num_workers", 0),
            collate_fn=collate_batch,
        )
        if (
            train
            and slice_mode == "all"
            and balance_studies
            and len(samples) > n_studies
        ):
            weights = study_balanced_weights(samples)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(ds, shuffle=False, sampler=sampler, **loader_kwargs)
        return DataLoader(ds, shuffle=train, **loader_kwargs)

    train_loader = make_loader(train_samples, train=True)
    val_loader = make_loader(val_samples, train=False)
    test_loader = make_loader(test_samples, train=False)

    model = ViTMultiTask(
        mcfg["name"],
        pretrained=mcfg.get("pretrained", True),
        num_heads=mcfg.get("num_heads", 5),
        img_size=pp["image_size"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
    )
    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val = float("inf")
    patience = int(tcfg.get("patience", 5))
    stale = 0
    ckpt_path = out_dir / f"fold{fold}_best.pt"

    for epoch in range(int(tcfg["epochs"])):
        print(f"  fold{fold} epoch {epoch+1}/{tcfg['epochs']} train...", flush=True)
        tr_loss = run_epoch(model, train_loader, optimizer, device, amp=use_amp, scaler=scaler, desc="train")
        print(f"  fold{fold} epoch {epoch+1}/{tcfg['epochs']} val...", flush=True)
        va_loss = run_epoch(model, val_loader, None, device, amp=False, scaler=None, desc="val")
        print(f"  fold{fold} epoch {epoch+1}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f}", flush=True)
        if va_loss < best_val:
            best_val = va_loss
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": va_loss}, ckpt_path)
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch+1}")
                break

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])

    y_pred, y_true, _ = predict_loader(
        model, test_loader, device, study_aggregation=study_agg,
    )
    metrics = per_head_metrics(y_true, y_pred, binary_threshold=float(cfg["eval"].get("binary_threshold", 0.5)))
    fold_result = {
        "fold": fold,
        "cv_split_unit": cv_unit,
        "n_train_studies": count_unique_studies(train_samples),
        "n_val_studies": count_unique_studies(val_samples),
        "n_test_studies": count_unique_studies(test_samples),
        "n_train_images": len(train_samples),
        "n_val_images": len(val_samples),
        "n_test_images": len(test_samples),
        "slice_mode": slice_mode,
        "study_aggregation": study_agg,
        "best_val_loss": best_val,
        "per_head": metrics,
        "macro_mse": macro_average(metrics, "mse"),
        "macro_spearman": macro_average(metrics, "spearman"),
        "macro_auroc": macro_average(metrics, "auroc"),
    }
    save_metrics(out_dir / f"fold{fold}_metrics.json", fold_result)
    return fold_result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_V1_ROOT / "configs" / "default.yaml")
    ap.add_argument("--labels-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--fold", type=int, default=None, help="Run single fold only (0-based)")
    ap.add_argument("--tag", default="v1_default")
    ap.add_argument("--epochs", type=int, default=None, help="Override config train.epochs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    set_seed(int(cfg["train"]["seed"]))

    labels_csv = args.labels_csv or Path(cfg["data"]["labels_csv"])
    if not labels_csv.is_absolute():
        labels_csv = _V1_ROOT / labels_csv

    out_dir = args.out_dir or (_V1_ROOT / "artifacts" / f"v1_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    dicom_dir = Path(cfg["data"]["dicom_dir"])
    dicom_index = index_dicoms(dicom_dir)
    labels = load_labels_csv(labels_csv)

    study_ids = sorted(set(dicom_index.keys()) & set(labels.keys()))
    if not study_ids:
        raise RuntimeError("No overlapping studies between DICOM and labels CSV")

    max_studies = cfg.get("data", {}).get("max_studies")
    if max_studies is not None:
        study_ids = study_ids[: int(max_studies)]
        print(f"[smoke] limited to {len(study_ids)} studies (max_studies={max_studies})", flush=True)

    y_strat_study = stratify_labels(labels, study_ids)
    n_folds = int(cfg["train"]["n_folds"])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(cfg["train"]["seed"]))
    cv_unit = cfg.get("data", {}).get("cv_split_unit", "image")
    if cv_unit not in ("study", "image"):
        raise ValueError(f"cv_split_unit must be 'study' or 'image', got {cv_unit!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slice_mode = cfg.get("data", {}).get("slice_mode", "all")
    all_samples = build_slice_samples(study_ids, dicom_index, slice_mode=slice_mode)
    print(
        f"device={device} studies={len(study_ids)} images={len(all_samples)} "
        f"slice_mode={slice_mode} cv_split={cv_unit} out={out_dir}",
        flush=True,
    )
    if device.type == "cpu":
        print(
            "[warn] CUDA not available - ViT-base full training on CPU can take many hours. "
            "Use configs/smoke.yaml for pipeline check, or run on a GPU machine.",
            flush=True,
        )
    print(f"loading model {cfg['model']['name']} (first run may download weights)...", flush=True)

    fold_results = []
    fold_range = [args.fold] if args.fold is not None else range(n_folds)

    if cv_unit == "study":
        splits = list(skf.split(np.arange(len(study_ids)), y_strat_study))
        for fold in fold_range:
            train_val_idx, test_idx = splits[fold]
            train_val_ids = [study_ids[i] for i in train_val_idx]
            test_ids = [study_ids[i] for i in test_idx]
            train_val_samples = build_slice_samples(train_val_ids, dicom_index, slice_mode=slice_mode)
            test_samples = build_slice_samples(test_ids, dicom_index, slice_mode=slice_mode)
            train_samples, val_samples = split_train_val_samples(
                train_val_samples, seed=int(cfg["train"]["seed"]) + fold,
            )
            print(
                f"=== fold {fold}: train={len(train_samples)} val={len(val_samples)} "
                f"test={len(test_samples)} images (study CV) ==="
            )
            t0 = time.time()
            res = train_fold(
                fold, train_samples, val_samples, test_samples,
                cfg, labels, out_dir, device, cv_unit=cv_unit,
            )
            res["elapsed_sec"] = time.time() - t0
            fold_results.append(res)
    else:
        y_strat_img = sample_strata(all_samples, labels, study_ids)
        splits = list(skf.split(np.arange(len(all_samples)), y_strat_img))
        for fold in fold_range:
            train_val_idx, test_idx = splits[fold]
            train_val_samples = [all_samples[i] for i in train_val_idx]
            test_samples = [all_samples[i] for i in test_idx]
            train_samples, val_samples = split_train_val_samples(
                train_val_samples, seed=int(cfg["train"]["seed"]) + fold,
            )
            print(
                f"=== fold {fold}: train={len(train_samples)} val={len(val_samples)} "
                f"test={len(test_samples)} images (image CV, Phase III style) ==="
            )
            t0 = time.time()
            res = train_fold(
                fold, train_samples, val_samples, test_samples,
                cfg, labels, out_dir, device, cv_unit=cv_unit,
            )
            res["elapsed_sec"] = time.time() - t0
            fold_results.append(res)

    # aggregate
    agg_heads = {h: {"mse": [], "spearman": [], "auroc": []} for h in HEADS}
    for fr in fold_results:
        for h in HEADS:
            m = fr["per_head"].get(h, {})
            if m.get("n", 0) > 0:
                for k in ("mse", "spearman", "auroc"):
                    if k in m and not np.isnan(m[k]):
                        agg_heads[h][k].append(m[k])

    summary = {
        "tag": args.tag,
        "labels_csv": str(labels_csv),
        "config": cfg,
        "n_studies": len(study_ids),
        "folds": fold_results,
        "mean_per_head": {
            h: {k: float(np.mean(v)) if v else float("nan") for k, v in agg_heads[h].items()}
            for h in HEADS
        },
    }
    save_metrics(out_dir / "metrics.json", summary)
    print(f"WROTE {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
