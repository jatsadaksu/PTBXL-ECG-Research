"""PTB-XL v1.0.3: patient-disjoint split validation and audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("NORM", "CD", "HYP", "MI", "STTC")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true", help="Read-only audit using raw metadata and diagnostic mapping")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "config/dataset_config.yaml").read_text(encoding="utf-8"))["dataset"]
    raw = pd.read_csv(ROOT / cfg["raw_dir"] / "ptbxl_database.csv")
    required = {"ecg_id", "patient_id", "strat_fold", "filename_lr"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing raw metadata columns: {sorted(required - set(raw.columns))}")
    if len(raw) != int(cfg["expected_records"]):
        raise ValueError("Unexpected raw metadata row count")
    if raw[list(required)].isna().any().any() or raw.ecg_id.duplicated().any() or raw.filename_lr.duplicated().any():
        raise ValueError("Missing or duplicate raw identifiers")
    if not raw.strat_fold.isin(range(1, 11)).all():
        raise ValueError("Invalid strat_fold")
    if args.audit_only:
        from importlib.util import module_from_spec, spec_from_file_location
        spec = spec_from_file_location("ptbxl_label_preparation", ROOT / "scripts/03_prepare_labels.py")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        _, labels, eligible, _ = module.prepare()
        meta = raw.loc[eligible].reset_index(drop=True)
        labels = labels.loc[eligible].reset_index(drop=True)
    else:
        meta = pd.read_csv(ROOT / "data/processed/metadata_clean.csv")
        labels = pd.read_csv(ROOT / "data/processed/labels_multihot.csv")
    if not {"ecg_id", "patient_id", "strat_fold", "filename_lr"}.issubset(meta.columns):
        raise ValueError("Missing processed metadata columns")
    if list(labels.columns) != ["ecg_id", *CLASSES]:
        raise ValueError("Unexpected labels_multihot columns")
    if meta.ecg_id.isna().any() or meta.ecg_id.duplicated().any() or labels.ecg_id.duplicated().any():
        raise ValueError("Duplicate or missing processed ecg_id")
    if not meta.ecg_id.equals(labels.ecg_id):
        raise ValueError("Metadata and label row order / ECG IDs differ")
    if meta.patient_id.isna().any() or meta.strat_fold.isna().any():
        raise ValueError("Missing patient_id or strat_fold in processed data")
    if not meta.strat_fold.isin(range(1, 11)).all():
        raise ValueError("Unexpected processed strat_fold")
    if not labels[list(CLASSES)].isin([0, 1]).all().all() or not labels[list(CLASSES)].sum(axis=1).ge(1).all():
        raise ValueError("Nonbinary or zero-superclass processed labels")
    if not set(meta.ecg_id).issubset(set(raw.ecg_id)):
        raise ValueError("Processed ECG IDs absent from raw metadata")
    source = raw.set_index("ecg_id").loc[meta.ecg_id]
    if not (source.patient_id.to_numpy() == meta.patient_id.to_numpy()).all():
        raise ValueError("patient_id changed from raw metadata")
    if not (source.strat_fold.to_numpy() == meta.strat_fold.to_numpy()).all():
        raise ValueError("strat_fold changed from raw metadata")
    if not (source.filename_lr.to_numpy() == meta.filename_lr.to_numpy()).all():
        raise ValueError("filename_lr changed from raw metadata")
    groups = {
        "train": meta.strat_fold.isin(range(1, 9)),
        "val": meta.strat_fold.eq(9),
        "test": meta.strat_fold.eq(10),
    }
    ids = {name: set(meta.loc[mask, "patient_id"]) for name, mask in groups.items()}
    overlaps = {
        "train_val": len(ids["train"] & ids["val"]),
        "train_test": len(ids["train"] & ids["test"]),
        "val_test": len(ids["val"] & ids["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Patient leakage: {overlaps}")
    counts = {name: int(mask.sum()) for name, mask in groups.items()}
    if sum(counts.values()) != len(meta):
        raise ValueError("Incomplete split coverage")
    audit = {
        "dataset": "PTB-XL", "version": cfg["version"],
        "mode": "audit_only" if args.audit_only else "processed",
        "policy": "exclude zero-diagnostic-superclass ECGs from supervised splits",
        "raw_records": len(raw), "eligible_records": len(meta),
        "excluded_no_diagnostic": len(raw) - len(meta),
        "split_records": counts,
        "split_unique_patients": {name: len(ids[name]) for name in groups},
        "patient_overlap": overlaps,
        "per_fold_records": {str(i): int(meta.strat_fold.eq(i).sum()) for i in range(1, 11)},
        "label_counts_by_split": {
            name: {label: int(labels.loc[mask, label].sum()) for label in CLASSES}
            for name, mask in groups.items()
        },
        "acceptance_pass": True,
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.audit_only:
        print("AUDIT ONLY: no files written")
        return
    processed = ROOT / "data/processed"
    reports = ROOT / "reports"
    if not processed.is_dir() or not reports.is_dir():
        raise FileNotFoundError("Expected data/processed and reports directories must already exist")
    targets = [processed / f"{name}.csv" for name in groups]
    report = reports / "split_audit.json"
    for target in [*targets, report]:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    for name, target in zip(groups, targets):
        mask = groups[name]
        output = meta.loc[mask].copy()
        output = output.merge(labels, on="ecg_id", how="left", validate="one_to_one", sort=False)
        output.to_csv(target, index=False)
    report.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("WROTE:", *(str(t) for t in [*targets, report]), sep="\n  ")


if __name__ == "__main__":
    main()
