"""Lazy PTB-XL 100 Hz PyTorch dataset; normalization uses frozen train statistics."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset

CLASSES = ("NORM", "CD", "HYP", "MI", "STTC")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
ROOT = Path(__file__).resolve().parents[1]


class PTBXLDataset(Dataset):
    """Return (12, 1000) float32 signal, (5,) float32 multi-hot target."""

    def __init__(self, split: str, root: str | Path = ROOT, manifest_path: str | Path | None = None):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        self.root = Path(root).resolve()
        self.split = split
        path = self.root / "data" / "processed" / f"{split}.csv"
        self.frame = pd.read_csv(path, low_memory=False)
        needed = {"ecg_id", "patient_id", "strat_fold", "filename_lr", *CLASSES}
        if not needed.issubset(self.frame.columns):
            raise ValueError(f"Missing columns in {path}: {sorted(needed - set(self.frame.columns))}")
        if self.frame.ecg_id.isna().any() or self.frame.ecg_id.duplicated().any():
            raise ValueError("Missing or duplicate ecg_id")
        allowed = set(range(1, 9)) if split == "train" else ({9} if split == "val" else {10})
        if not self.frame.strat_fold.isin(allowed).all():
            raise ValueError("Fold mismatch")
        if not self.frame[list(CLASSES)].isin([0, 1]).all().all():
            raise ValueError("Non-binary targets")
        if not self.frame[list(CLASSES)].sum(axis=1).ge(1).all():
            raise ValueError("Unlabeled ECG in supervised split")
        self.targets = self.frame[list(CLASSES)].to_numpy(dtype=np.float32, copy=True)
        mp = Path(manifest_path) if manifest_path is not None else self.root / "reports" / "dataset_manifest.json"
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        norm = manifest["normalization"]
        if norm["fit_split"] != "train" or norm["method"] != "per_lead_zscore":
            raise ValueError("Manifest must contain train-only per-lead z-score normalization")
        self.mean = np.asarray(norm["mean"], dtype=np.float32)
        self.std = np.asarray(norm["std"], dtype=np.float32)
        if self.mean.shape != (12,) or self.std.shape != (12,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid normalization statistics")
        if manifest["split_records"][split] != len(self.frame):
            raise ValueError("Split count differs from frozen manifest")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        rel = str(row.filename_lr)
        if not rel.startswith("records100/") or ".." in Path(rel).parts:
            raise ValueError(f"Invalid filename_lr: {rel}")
        path = self.root / "data" / "raw" / "ptb-xl" / "1.0.3" / rel
        signal, fields = wfdb.rdsamp(str(path))
        if signal.shape != (1000, 12) or fields.get("fs") != 100 or tuple(str(name).upper() for name in fields.get("sig_name", [])) != tuple(name.upper() for name in LEADS):
            raise ValueError(f"Invalid signal metadata for ECG {row.ecg_id}")
        if not np.isfinite(signal).all():
            raise ValueError(f"Nonfinite ECG {row.ecg_id}")
        x = ((signal.astype(np.float32) - self.mean) / self.std).T.copy()
        return torch.from_numpy(x), torch.from_numpy(self.targets[index].copy())
