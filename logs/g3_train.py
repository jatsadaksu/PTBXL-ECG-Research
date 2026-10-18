"""PTB-XL G3 locked training; atomic epoch checkpoints and resumable 2x3 runs.

Install ONLY in the already-approved logs/ directory: logs/g3_train.py.
Never reads the test split. Smoke mode uses synthetic tensors only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.cnn1d import LightweightCNN1D
from src.models.resnet1d import ResNet1D

MODELS = {"cnn1d": (LightweightCNN1D, 38821), "resnet1d": (ResNet1D, 244261)}
SEEDS = (42, 123, 2026)
LABELS = ("NORM", "CD", "HYP", "MI", "STTC")
LR = 0.001
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 5
EPS = 1e-8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(obj, path: Path) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def fingerprint(model_name: str, seed: int, smoke: bool) -> tuple[str, dict]:
    files = {
        "cnn1d_source": ROOT / "src/models/cnn1d.py",
        "resnet1d_source": ROOT / "src/models/resnet1d.py",
        "training_source": Path(__file__).resolve(),
    }
    if not smoke:
        files.update({
            "dataset_source": ROOT / "src/dataset.py",
            "manifest": ROOT / "reports/dataset_manifest.json",
            "train_csv": ROOT / "data/processed/train.csv",
            "val_csv": ROOT / "data/processed/val.csv",
        })
    evidence = {name: sha256(path) for name, path in files.items()}
    config = {
        "schema": "g3-epoch-resume-v1", "model": model_name, "seed": seed,
        "smoke": smoke, "optimizer": "AdamW", "lr": LR,
        "loss": "BCEWithLogitsLoss", "batch_size": 4 if smoke else BATCH_SIZE,
        "max_epochs": 3 if smoke else EPOCHS,
        "patience": PATIENCE, "primary": "validation_macro_average_precision",
        "metric_tie_epsilon": EPS, "train_shuffle": True,
        "workers": 0, "amp": False, "labels": LABELS,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "sklearn_metric": "sklearn.metrics.average_precision_score",
        "files_sha256": evidence,
    }
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return digest, config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state(generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_generator": generator.get_state(),
    }


def restore_rng(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint was created with CUDA, but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader_generator"])


def load_local_checkpoint(path: Path, device: torch.device) -> dict:
    # weights_only=False is necessary for Python/NumPy RNG states. Load ONLY
    # checkpoints generated locally by this project; never untrusted files.
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_ckpt(obj: dict, digest: str, model: str, seed: int) -> None:
    if obj.get("fingerprint") != digest or obj.get("model_name") != model or obj.get("seed") != seed:
        raise RuntimeError("Checkpoint fingerprint/model/seed mismatch: REFUSING RESUME")
    for key in ("model", "optimizer", "rng", "epoch", "best_epoch", "best_score", "bad_epochs", "history", "completed"):
        if key not in obj:
            raise RuntimeError(f"Incomplete checkpoint: missing {key}")


def macro_ap(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, list[float]]:
    if y_true.shape[1] != 5 or probs.shape != y_true.shape:
        raise ValueError("Invalid multilabel validation arrays")
    counts = y_true.sum(axis=0)
    if np.any(counts == 0) or np.any(counts == len(y_true)):
        raise ValueError(f"Validation class degenerate: positives={counts.tolist()}")
    aps = [float(average_precision_score(y_true[:, c], probs[:, c])) for c in range(5)]
    if not np.isfinite(aps).all():
        raise ValueError("Non-finite validation AUPRC")
    return float(np.mean(aps)), aps


def make_loaders(smoke: bool, seed: int, generator: torch.Generator):
    if smoke:
        # Deterministic, balanced synthetic targets, independent of global RNG.
        local = torch.Generator().manual_seed(10101)
        xtr = torch.randn(12, 12, 1000, generator=local)
        xva = torch.randn(8, 12, 1000, generator=local)
        ytr = torch.tensor([[float((i + c) % 2) for c in range(5)] for i in range(12)])
        yva = torch.tensor([[float((i + c) % 2) for c in range(5)] for i in range(8)])
        train_ds, val_ds = TensorDataset(xtr, ytr), TensorDataset(xva, yva)
        batch = 4
    else:
        from src.dataset import PTBXLDataset
        train_ds = PTBXLDataset(split="train", root=ROOT)
        val_ds = PTBXLDataset(split="val", root=ROOT)
        if len(train_ds) != 17084 or len(val_ds) != 2146:
            raise RuntimeError("Dataset counts differ from G1 lock")
        batch = BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, generator=generator,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
    return train_loader, val_loader


def one_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, total_count = 0.0, 0
    all_y, all_p = [], []
    with torch.enable_grad() if train else torch.inference_mode():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss: stopping without marking completed")
            if train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(x)
            total_count += len(x)
            if not train:
                all_y.append(y.cpu().numpy())
                all_p.append(torch.sigmoid(logits).cpu().numpy())
    if train:
        return total_loss / total_count, None, None
    score, aps = macro_ap(np.concatenate(all_y), np.concatenate(all_p))
    return total_loss / total_count, score, aps


def run(model_name: str, seed: int, smoke: bool, stop_after_epoch: int | None):
    if model_name not in MODELS or seed not in SEEDS:
        raise ValueError("G2 permits only two models and seeds 42,123,2026")
    prefix = ("smoke_" if smoke else "") + f"{model_name}_seed{seed}"
    cp_dir, log_dir, res_dir = (ROOT / name for name in ("checkpoints", "logs", "results"))
    for directory in (cp_dir, log_dir, res_dir):
        if not directory.is_dir():
            raise RuntimeError(f"Missing approved directory: {directory}")
    last_path = cp_dir / f"{prefix}_last.pt"
    best_path = cp_dir / f"{prefix}_best.pt"
    status_path = log_dir / f"{prefix}_status.json"
    result_path = res_dir / f"{prefix}_validation.json"
    digest, config = fingerprint(model_name, seed, smoke)
    seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not smoke and device.type != "cuda":
        raise RuntimeError("Full training requires CUDA (RTX 4070 expected)")
    model_cls, expected = MODELS[model_name]
    model = model_cls().to(device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != expected:
        raise RuntimeError(f"G2 architecture parameter count mismatch: {count} != {expected}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    train_loader, val_loader = make_loaders(smoke, seed, generator)
    epochs = 3 if smoke else EPOCHS
    start, best_score, best_epoch, bad_epochs, history = 1, -1.0, 0, 0, []
    if last_path.exists():
        checkpoint = load_local_checkpoint(last_path, device)
        validate_ckpt(checkpoint, digest, model_name, seed)
        # Recover from a crash between committing LAST and committing BEST.
        # Only safe when LAST itself contains the newly selected best weights.
        if not best_path.exists():
            if int(checkpoint["epoch"]) == int(checkpoint["best_epoch"]):
                atomic_save(checkpoint, best_path)
            else:
                raise RuntimeError("Last checkpoint exists but best checkpoint is missing")
        best_ckpt = load_local_checkpoint(best_path, device)
        validate_ckpt(best_ckpt, digest, model_name, seed)
        if int(best_ckpt["epoch"]) != int(checkpoint["best_epoch"]):
            if int(checkpoint["epoch"]) == int(checkpoint["best_epoch"]):
                atomic_save(checkpoint, best_path)
            else:
                raise RuntimeError("Best checkpoint does not match last checkpoint; manual audit required")
        if checkpoint["completed"]:
            if not result_path.exists():
                atomic_json({"model": model_name, "seed": seed, "best_epoch": checkpoint["best_epoch"],
                             "best_macro_auprc": checkpoint["best_score"], "history": checkpoint["history"],
                             "fingerprint": digest, "status": "completed", "smoke": smoke}, result_path)
            print(f"SKIP COMPLETED: {prefix}, epoch={checkpoint['epoch']}, best={checkpoint['best_epoch']}", flush=True)
            return "SKIP"
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng(checkpoint["rng"], generator)
        start = int(checkpoint["epoch"]) + 1
        best_score, best_epoch = float(checkpoint["best_score"]), int(checkpoint["best_epoch"])
        bad_epochs, history = int(checkpoint["bad_epochs"]), list(checkpoint["history"])
        print(f"RESUME {prefix} from epoch {start}; last saved={start-1}", flush=True)
    elif best_path.exists():
        raise RuntimeError("Orphan best checkpoint found; manual audit required")
    else:
        print(f"NEW RUN {prefix} on {device}; params={count:,}", flush=True)
    if start > epochs:
        raise RuntimeError("Incomplete checkpoint past maximum epochs; manual audit required")
    for epoch in range(start, epochs + 1):
        tic = time.perf_counter()
        train_loss, _, _ = one_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, score, aps = one_epoch(model, val_loader, criterion, device)
        improved = score > best_score + EPS
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
        else:
            bad_epochs += 1
        entry = {"epoch": epoch, "train_bce": train_loss, "val_bce": val_loss,
                 "val_macro_auprc": score, "val_ap_per_label": dict(zip(LABELS, aps)),
                 "seconds": time.perf_counter() - tic, "best_epoch_so_far": best_epoch}
        history.append(entry)
        completed = bad_epochs >= PATIENCE or epoch == epochs
        state = {"schema": "g3-epoch-resume-v1", "fingerprint": digest, "config": config,
                 "model_name": model_name, "seed": seed, "epoch": epoch,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "rng": rng_state(generator), "best_score": best_score,
                 "best_epoch": best_epoch, "bad_epochs": bad_epochs,
                 "history": history, "completed": completed,
                 "stop_reason": "early_stopping" if bad_epochs >= PATIENCE else ("max_epochs" if epoch == epochs else None)}
        # Commit LAST first; if interrupted before BEST, recover BEST from
        # LAST when this epoch is the selected best.
        atomic_save(state, last_path)
        if improved:
            atomic_save(state, best_path)
        atomic_json({"model": model_name, "seed": seed, "epoch": epoch,
                     "best_epoch": best_epoch, "best_macro_auprc": best_score,
                     "fingerprint": digest, "completed": completed,
                     "stop_reason": state["stop_reason"], "smoke": smoke}, status_path)
        print(f"{prefix} epoch={epoch}/{epochs} train_bce={train_loss:.5f} "
              f"val_bce={val_loss:.5f} val_macro_auprc={score:.5f} "
              f"best_epoch={best_epoch} completed={completed}", flush=True)
        if completed:
            atomic_json({"model": model_name, "seed": seed, "best_epoch": best_epoch,
                         "best_macro_auprc": best_score, "history": history,
                         "fingerprint": digest, "status": "completed", "smoke": smoke}, result_path)
            print(f"COMPLETE {prefix}: {state['stop_reason']}", flush=True)
            return "COMPLETE"
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            print(f"INTENTIONAL STOP after saved epoch {epoch}; rerun without --stop-after-epoch", flush=True)
            return "STOPPED"
    raise AssertionError("Unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="cnn1d")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--smoke", action="store_true", help="Synthetic data only; never reads PTB-XL")
    parser.add_argument("--all", action="store_true", help="Full training of exactly 2 models x 3 seeds")
    parser.add_argument("--stop-after-epoch", type=int, default=None,
                        help="Safe interruption AFTER checkpoint; use for smoke test")
    args = parser.parse_args()
    if args.stop_after_epoch is not None and (not args.smoke or args.stop_after_epoch < 1):
        parser.error("--stop-after-epoch is permitted only in smoke mode")
    if args.all and (args.smoke or args.stop_after_epoch is not None):
        parser.error("--all is full training only")
    if args.all:
        for model in MODELS:
            for seed in SEEDS:
                run(model, seed, False, None)
    else:
        run(args.model, args.seed, args.smoke, args.stop_after_epoch)


if __name__ == "__main__":
    main()
