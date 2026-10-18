"""Compute train-only PTB-XL per-lead normalization and freeze dataset manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("NORM", "CD", "HYP", "MI", "STTC")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-only", action="store_true", help="Compute statistics without writing a manifest")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "config/dataset_config.yaml").read_text(encoding="utf-8"))["dataset"]
    if (str(cfg["version"]), cfg["sampling_rate"], cfg["samples_per_record"], cfg["leads"]) != ("1.0.3", 100, 1000, 12):
        raise ValueError("Configuration differs from locked protocol")
    if list(cfg["labels"]) != list(CLASSES) or list(cfg["train_folds"]) != list(range(1, 9)) or cfg["validation_fold"] != 9 or cfg["test_fold"] != 10:
        raise ValueError("Configuration differs from locked label/fold policy")
    processed = ROOT / "data" / "processed"
    paths = {name: processed / f"{name}.csv" for name in ("train", "val", "test")}
    frames = {name: pd.read_csv(path, low_memory=False) for name, path in paths.items()}
    for name, frame in frames.items():
        allowed = set(range(1, 9)) if name == "train" else ({9} if name == "val" else {10})
        required = {"ecg_id", "patient_id", "strat_fold", "filename_lr", *CLASSES}
        if not required.issubset(frame.columns) or frame[list(CLASSES)].isna().any().any():
            raise ValueError(f"Invalid split schema: {name}")
        if not frame.strat_fold.isin(allowed).all() or frame.ecg_id.duplicated().any() or frame.patient_id.isna().any():
            raise ValueError(f"Invalid folds or identifiers: {name}")
        if not frame[list(CLASSES)].isin([0, 1]).all().all() or not frame[list(CLASSES)].sum(axis=1).ge(1).all():
            raise ValueError(f"Invalid targets: {name}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(frames[a].patient_id) & set(frames[b].patient_id):
            raise ValueError(f"Patient leakage: {a}/{b}")
    if sum(map(len, frames.values())) != 21388 or [len(frames[x]) for x in ("train", "val", "test")] != [17084, 2146, 2158]:
        raise ValueError("Unexpected split sizes: compare with approved audit")
    target = ROOT / "reports" / "dataset_manifest.json"
    if not target.is_file() or target.stat().st_size == 0:
        raise FileNotFoundError("Existing verified dataset manifest required")
    original = json.loads(target.read_text(encoding="utf-8-sig"))
    if original.get("dataset") != "PTB-XL" or str(original.get("version")) != "1.0.3" or original.get("status") != "verified" or original.get("complete_download_claim") is not True:
        raise ValueError("Dataset manifest is not verified; refusing update")
    if "normalization" in original:
        raise FileExistsError("Normalization already present; refusing to overwrite frozen statistics")
    integrity = json.loads((ROOT / "reports/integrity_report.json").read_text(encoding="utf-8-sig"))
    if not (integrity.get("dataset") == "PTB-XL" and str(integrity.get("version")) == "1.0.3" and integrity.get("verification_scope") == "full" and integrity.get("verified_records") == 21799 and integrity.get("failed_records") == 0 and integrity.get("acceptance_pass") is True and not integrity.get("issues")):
        raise ValueError("Full integrity evidence is not acceptable")
    split_audit = json.loads((ROOT / "reports/split_audit.json").read_text(encoding="utf-8-sig"))
    if split_audit.get("acceptance_pass") is not True or split_audit.get("patient_overlap") != {"train_val": 0, "train_test": 0, "val_test": 0}:
        raise ValueError("Patient-level split audit is not acceptable")
    sums = np.zeros(12, dtype=np.float64)
    squares = np.zeros(12, dtype=np.float64)
    n_samples = 0
    raw = ROOT / cfg["raw_dir"]
    for i, row in enumerate(frames["train"].itertuples(index=False), 1):
        rel = str(row.filename_lr)
        if not rel.startswith("records100/") or ".." in Path(rel).parts:
            raise ValueError(f"Invalid record path for ecg_id={row.ecg_id}")
        signal, fields = wfdb.rdsamp(str(raw / rel))
        if signal.shape != (1000, 12) or fields.get("fs") != 100 or tuple(str(name).upper() for name in fields.get("sig_name", [])) != tuple(name.upper() for name in LEADS):
            raise ValueError(f"Invalid WFDB signal for ecg_id={row.ecg_id}")
        if not np.isfinite(signal).all():
            raise ValueError(f"Nonfinite WFDB signal for ecg_id={row.ecg_id}")
        sums += signal.sum(axis=0, dtype=np.float64)
        squares += np.square(signal, dtype=np.float64).sum(axis=0)
        n_samples += signal.shape[0]
        if i % 2000 == 0 or i == len(frames["train"]):
            print(f"Train ECG processed: {i}/{len(frames['train'])}", flush=True)
    mean = sums / n_samples
    variance = np.maximum(squares / n_samples - mean ** 2, 0)
    std = np.sqrt(variance)
    if (std <= 1e-12).any():
        raise ValueError("Zero-variance lead: cannot normalize")
    additions = {
        "source": "records100",
        "sampling_rate_hz": 100, "samples_per_record": 1000, "leads": list(LEADS),
        "labels": list(CLASSES), "policy": "exclude zero-diagnostic-superclass ECGs; preserve raw",
        "split_records": {k: len(v) for k, v in frames.items()},
        "split_csv_sha256": {k: digest(v) for k, v in paths.items()},
        "normalization": {
            "method": "per_lead_zscore", "fit_split": "train", "unit": "physical WFDB amplitude",
            "mean": mean.tolist(), "std": std.tolist(),
            "variance_definition": "population", "train_samples_per_lead": n_samples,
            "epsilon": 0.0,
        },
        "ecg_sha256_verification": "verified_full",
        "integrity_report_sha256": digest(ROOT / "reports/integrity_report.json"),
        "split_audit_sha256": digest(ROOT / "reports/split_audit.json"),
    }
    manifest = {**original, **additions}
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.audit_only:
        print("AUDIT ONLY: no files written")
    else:
        if not target.parent.is_dir():
            raise FileNotFoundError(target.parent)
        import os
        import tempfile
        fd, temporary = tempfile.mkstemp(prefix=".dataset_manifest_", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            # Existing manifest fields preserved, only approved normalization added.
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print("WROTE:", target)


if __name__ == "__main__":
    main()
