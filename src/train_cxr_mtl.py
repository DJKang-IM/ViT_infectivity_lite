# -*- coding: utf-8 -*-
"""Train a CXR multi-task model (D1..D5) on a fixed patient-wise split.

Separate from src/train.py (which is the DICOM 5-fold CV pipeline). This uses
the JPEG manifest + study_split_70_15_15.json produced by the M1 pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

_V1_ROOT = Path(__file__).resolve().parents[1]
if str(_V1_ROOT) not in sys.path:
    sys.path.insert(0, str(_V1_ROOT))

from src.data.cxr_multitask_dataset import (
    HEADS,
    CXRMultiTaskDataset,
    collate_batch,
    load_cxr_labels,
    load_split,
)
from src.models.backbone_factory import create_mtl_model
from src.models.mtl_heads import masked_mtl_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _abs(root: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else root / q


def make_loader(cfg, ids, labels, *, train, batch_size):
    pp = cfg["preprocess"]
    clip = pp.get("clahe_clip_limit", 0.03)
    cache_dir = _V1_ROOT / "artifacts" / "clahe_cache" / f"clip{clip}"
    ds = CXRMultiTaskDataset.from_manifest(
        ids,
        _abs(_V1_ROOT, cfg["data"]["manifest_csv"]),
        labels,
        image_size=pp["image_size"],
        clahe=pp.get("clahe", True),
        clahe_clip_limit=clip,
        resize_mode=pp.get("resize_mode", "letterbox"),
        brightness_norm=pp.get("brightness_norm", True),
        cache_dir=cache_dir,
        cache_max_side=max(int(pp["image_size"]), 512),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=cfg["train"].get("num_workers", 0),
        collate_fn=collate_batch,
        drop_last=train,
        pin_memory=torch.cuda.is_available(),
    )


def cosine_warmup(step, total_steps, warmup_steps):
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(1, warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def train_one_epoch(model, loader, optimizer, scheduler_fn, device, *,
                    amp, scaler, accum_steps, head_types, epoch_state):
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc="train", leave=False)
    for i, batch in enumerate(pbar):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        if amp and scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                pred = model(x)
                loss, _ = masked_mtl_loss(pred, y, mask, head_types)
            scaler.scale(loss / accum_steps).backward()
        else:
            pred = model(x)
            loss, _ = masked_mtl_loss(pred, y, mask, head_types)
            (loss / accum_steps).backward()

        if (i + 1) % accum_steps == 0:
            lr_scale = scheduler_fn(epoch_state["step"])
            for pg in optimizer.param_groups:
                pg["lr"] = pg["base_lr"] * lr_scale
            if amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_state["step"] += 1
        lv = float(loss.detach().cpu())
        losses.append(lv)
        pbar.set_postfix(loss=f"{lv:.4f}")
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, head_types, *, study_agg="mean"):
    model.eval()
    rows_pred, rows_true, rows_mask, studies = [], [], [], []
    total_loss = []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        pred = model(x)
        loss, _ = masked_mtl_loss(pred, y, mask, head_types)
        total_loss.append(float(loss.detach().cpu()))
        # convert to interpretable predictions
        p = pred.detach().float().cpu().numpy()
        for j, ht in enumerate(head_types):
            if ht == "clf":
                p[:, j] = 1.0 / (1.0 + np.exp(-p[:, j]))
            else:
                p[:, j] = np.clip(p[:, j], 0.0, 1.0)
        rows_pred.append(p)
        rows_true.append(batch["y"].numpy())
        rows_mask.append(batch["mask"].numpy())
        studies.extend(batch["study_no"])

    y_pred = np.concatenate(rows_pred)
    y_true = np.concatenate(rows_true)
    y_mask = np.concatenate(rows_mask)
    study_arr = np.array(studies, dtype=int)

    # study-level aggregation (mean of image preds)
    if study_agg:
        uniq = sorted(set(studies))
        agg_pred, agg_true, agg_mask = [], [], []
        for s in uniq:
            idx = np.where(study_arr == s)[0]
            agg_pred.append(y_pred[idx].mean(axis=0))
            agg_true.append(y_true[idx[0]])
            agg_mask.append(y_mask[idx[0]])
        y_pred = np.stack(agg_pred)
        y_true = np.stack(agg_true)
        y_mask = np.stack(agg_mask)

    metrics = {}
    for j, h in enumerate(HEADS):
        m = y_mask[:, j] > 0
        n = int(m.sum())
        if n == 0:
            metrics[h] = {"n": 0}
            continue
        tv = y_true[m, j]
        pv = y_pred[m, j]
        entry = {"n": n, "type": head_types[j]}
        if head_types[j] == "clf":
            bin_t = (tv >= 0.5).astype(int)
            if len(np.unique(bin_t)) == 2:
                entry["auroc"] = float(roc_auc_score(bin_t, pv))
            else:
                entry["auroc"] = float("nan")
            entry["bce"] = float(np.mean(
                -(bin_t * np.log(np.clip(pv, 1e-7, 1)) +
                  (1 - bin_t) * np.log(np.clip(1 - pv, 1e-7, 1)))))
        else:
            entry["mse"] = float(np.mean((pv - tv) ** 2))
            entry["spearman"] = (
                float(spearmanr(tv, pv).statistic)
                if n >= 2 and np.std(tv) > 0 and np.std(pv) > 0 else float("nan")
            )
        metrics[h] = entry
    return float(np.mean(total_loss)) if total_loss else float("nan"), metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CXR MTL baseline")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="use smoke_epochs + subset")
    ap.add_argument("--limit-studies", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    set_seed(int(tcfg["seed"]))

    epochs = args.epochs or (tcfg.get("smoke_epochs", 1) if args.smoke else tcfg["epochs"])
    epochs = int(epochs)

    labels = load_cxr_labels(_abs(_V1_ROOT, cfg["data"]["labels_csv"]))
    split_json = _abs(_V1_ROOT, cfg["data"]["split_json"])
    train_ids = load_split(split_json, "train")
    val_ids = load_split(split_json, "val")
    test_ids = load_split(split_json, "test")

    if args.limit_studies:
        train_ids = train_ids[: args.limit_studies]
        val_ids = val_ids[: max(4, args.limit_studies // 4)]
        test_ids = test_ids[: max(4, args.limit_studies // 4)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = int(tcfg["batch_size"])
    accum = int(tcfg.get("accum_steps", 1))
    head_types = cfg["model"]["head_types"]

    out_dir = _V1_ROOT / "artifacts" / f"cxr_mtl_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(cfg, train_ids, labels, train=True, batch_size=bs)
    val_loader = make_loader(cfg, val_ids, labels, train=False, batch_size=bs)
    test_loader = make_loader(cfg, test_ids, labels, train=False, batch_size=bs)

    print(f"device={device} backbone={cfg['model']['backbone']} "
          f"img={cfg['preprocess']['image_size']} bs={bs} accum={accum} epochs={epochs}",
          flush=True)
    print(f"train imgs={len(train_loader.dataset)} val imgs={len(val_loader.dataset)} "
          f"test imgs={len(test_loader.dataset)}", flush=True)

    model = create_mtl_model(
        cfg["model"]["backbone"],
        pretrained=cfg["model"].get("pretrained", True),
        img_size=cfg["preprocess"]["image_size"],
        head_types=head_types,
    ).to(device)

    base_lr = float(tcfg["lr"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=float(tcfg.get("weight_decay", 0.05)),
    )
    for pg in optimizer.param_groups:
        pg["base_lr"] = base_lr

    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * int(tcfg.get("warmup_epochs", 0))
    sched = lambda s: cosine_warmup(s, total_steps, warmup_steps)

    use_amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    best_val = float("inf")
    best_metrics = None
    patience = int(tcfg.get("patience", 6))
    stale = 0
    epoch_state = {"step": 0}
    ckpt = out_dir / "best.pt"
    history = []

    for ep in range(epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, sched, device,
            amp=use_amp, scaler=scaler, accum_steps=accum,
            head_types=head_types, epoch_state=epoch_state,
        )
        va_loss, va_metrics = evaluate(
            model, val_loader, device, head_types,
            study_agg=cfg["eval"].get("study_aggregation", "mean"),
        )
        dt = time.time() - t0
        print(f"epoch {ep + 1}/{epochs}: train={tr_loss:.4f} val={va_loss:.4f} "
              f"({dt:.0f}s) " + json.dumps(va_metrics), flush=True)
        history.append({"epoch": ep + 1, "train_loss": tr_loss,
                        "val_loss": va_loss, "val_metrics": va_metrics})
        if va_loss < best_val:
            best_val = va_loss
            best_metrics = va_metrics
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "val_loss": va_loss, "config": cfg}, ckpt)
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {ep + 1}", flush=True)
                break

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    test_loss, test_metrics = evaluate(
        model, test_loader, device, head_types,
        study_agg=cfg["eval"].get("study_aggregation", "mean"),
    )
    summary = {
        "tag": args.tag,
        "backbone": cfg["model"]["backbone"],
        "image_size": cfg["preprocess"]["image_size"],
        "epochs_run": len(history),
        "best_val_loss": best_val,
        "best_val_metrics": best_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "n_train_images": len(train_loader.dataset),
        "history": history,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("TEST:", json.dumps(test_metrics, ensure_ascii=False), flush=True)
    print(f"WROTE {out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
