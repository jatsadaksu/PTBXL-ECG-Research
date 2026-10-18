"""PTB-XL v1.0.3: deterministic five-superclass label preparation."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("NORM", "CD", "HYP", "MI", "STTC")
REQUIRED = {"ecg_id", "patient_id", "filename_lr", "scp_codes", "strat_fold"}


def load_config():
    cfg = yaml.safe_load((ROOT / "config/dataset_config.yaml").read_text(encoding="utf-8"))["dataset"]
    return cfg, ROOT / cfg["raw_dir"]


def prepare():
    cfg, raw = load_config()
    meta = pd.read_csv(raw / "ptbxl_database.csv")
    scp = pd.read_csv(raw / "scp_statements.csv", index_col=0)
    if not REQUIRED.issubset(meta.columns):
        raise ValueError(f"Missing metadata columns: {sorted(REQUIRED - set(meta.columns))}")
    if not {"diagnostic", "diagnostic_class"}.issubset(scp.columns):
        raise ValueError("Missing diagnostic or diagnostic_class in scp_statements.csv")
    if len(meta) != int(cfg["expected_records"]):
        raise ValueError(f"Unexpected metadata count: {len(meta)}")
    if meta.ecg_id.isna().any() or meta.ecg_id.duplicated().any():
        raise ValueError("Missing or duplicate ecg_id")
    if meta.patient_id.isna().any() or meta.strat_fold.isna().any():
        raise ValueError("Missing patient_id or strat_fold")
    if not meta.strat_fold.isin(range(1, 11)).all():
        raise ValueError("Unexpected strat_fold")
    if meta.filename_lr.isna().any() or meta.filename_lr.duplicated().any():
        raise ValueError("Missing or duplicate filename_lr")
    if scp.index.isna().any() or scp.index.duplicated().any():
        raise ValueError("Missing or duplicate SCP statement codes")
    flag = pd.to_numeric(scp.diagnostic, errors="coerce")
    mapping = {str(code): str(scp.at[code, "diagnostic_class"])
               for code in scp.index if flag.loc[code] == 1 and scp.at[code, "diagnostic_class"] in CLASSES}
    if not mapping:
        raise ValueError("Empty diagnostic mapping")
    vectors = []
    unknown_codes = set()
    for row in meta.itertuples(index=False):
        try:
            codes = ast.literal_eval(row.scp_codes)
        except (SyntaxError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid scp_codes for ecg_id={row.ecg_id}") from exc
        if not isinstance(codes, dict):
            raise ValueError(f"Non-dictionary scp_codes for ecg_id={row.ecg_id}")
        for code in codes:
            if not isinstance(code, str):
                raise ValueError(f"Non-string SCP code for ecg_id={row.ecg_id}")
            if code not in scp.index:
                unknown_codes.add(code)
        labels = {mapping[code] for code in codes if code in mapping}
        vectors.append([int(name in labels) for name in CLASSES])
    if unknown_codes:
        raise ValueError(f"Unknown SCP codes in metadata: {sorted(unknown_codes)}")
    labels = pd.DataFrame(vectors, columns=CLASSES)
    labels.insert(0, "ecg_id", meta.ecg_id.to_numpy())
    eligible = labels[list(CLASSES)].sum(axis=1).gt(0)
    if eligible.sum() == 0:
        raise ValueError("No diagnostic-labeled records")
    return meta, labels, eligible, mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true", help="Read-only; do not write output")
    args = parser.parse_args()
    meta, labels, eligible, mapping = prepare()
    selected_meta = meta.loc[eligible].reset_index(drop=True)
    selected_labels = labels.loc[eligible].reset_index(drop=True)
    summary = {
        "total_records": int(len(meta)),
        "diagnostic_mapping_codes": len(mapping),
        "eligible_records": int(eligible.sum()),
        "excluded_no_diagnostic": int((~eligible).sum()),
        "multi_label_records": int((labels[list(CLASSES)].sum(axis=1) > 1).sum()),
        "label_counts": {name: int(labels[name].sum()) for name in CLASSES},
        "excluded_by_fold": {str(i): int(((meta.strat_fold == i) & ~eligible).sum()) for i in range(1, 11)},
        "policy": "exclude zero-diagnostic-superclass ECGs from supervised splits; retain raw files",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.audit_only:
        print("AUDIT ONLY: no files written")
        return
    processed = ROOT / "data/processed"
    reports = ROOT / "reports"
    if not processed.is_dir() or not reports.is_dir():
        raise FileNotFoundError("Expected data/processed and reports directories must already exist")
    targets = [processed / "metadata_clean.csv", processed / "labels_multihot.csv", reports / "label_distribution.csv"]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {target}")
    distribution = pd.DataFrame([
        {"diagnostic_superclass": name, "positive_records": summary["label_counts"][name],
         "eligible_records": summary["eligible_records"], "total_raw_records": summary["total_records"]}
        for name in CLASSES
    ])
    selected_meta.to_csv(targets[0], index=False)
    selected_labels.to_csv(targets[1], index=False)
    distribution.to_csv(targets[2], index=False)
    print("WROTE:", *(str(t) for t in targets), sep="\n  ")


if __name__ == "__main__":
    main()
