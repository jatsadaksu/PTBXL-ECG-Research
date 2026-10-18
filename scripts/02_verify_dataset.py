"""Audit downloaded PTB-XL 100 Hz ECG records and official checksums."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import wfdb
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Pilot verification only, not full acceptance")
    args = ap.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be >= 1")
    cfg = yaml.safe_load((ROOT / "config/dataset_config.yaml").read_text(encoding="utf-8"))["dataset"]
    raw = ROOT / cfg["raw_dir"]
    base = cfg["base_url"].rstrip("/") + "/"
    metadata = pd.read_csv(raw / "ptbxl_database.csv")
    scp = pd.read_csv(raw / "scp_statements.csv")
    required = {"ecg_id", "patient_id", "filename_lr", "scp_codes", "strat_fold"}
    issues = []
    if not required.issubset(metadata.columns):
        raise RuntimeError(f"Missing metadata columns: {required - set(metadata.columns)}")
    if "diagnostic_class" not in scp.columns:
        issues.append("scp_statements.csv missing diagnostic_class")
    if len(metadata) != cfg["expected_records"]:
        issues.append(f"metadata rows={len(metadata)}, expected={cfg['expected_records']}")
    if metadata.ecg_id.duplicated().any():
        issues.append("duplicate ecg_id")
    if metadata.filename_lr.isna().any():
        issues.append("missing filename_lr")
    if not metadata.strat_fold.isin(range(1, 11)).all():
        issues.append("unexpected strat_fold values")
    # Upstream hashes, not locally invented reference hashes.
    response = requests.get(base + "SHA256SUMS.txt", timeout=(20, 120))
    response.raise_for_status()
    checksums = {}
    for line in response.text.splitlines():
        m = re.match(r"^([a-fA-F0-9]{64})\s+\*?(.+?)\s*$", line)
        if m:
            checksums[m.group(2).removeprefix("./").replace("\\", "/")] = m.group(1).lower()
    if not checksums:
        raise RuntimeError("Cannot parse official checksum manifest")
    for rel in ("ptbxl_database.csv", "scp_statements.csv"):
        if rel not in checksums or sha256(raw / rel) != checksums[rel]:
            issues.append(f"upstream checksum mismatch: {rel}")
    subset = metadata.sort_values("ecg_id")
    if args.limit:
        subset = subset.head(args.limit)
    checked = 0
    failures = []
    for row in tqdm(subset.itertuples(index=False), total=len(subset), unit="ECG"):
        rel = str(row.filename_lr)
        if not re.fullmatch(r"records100/\d{5}/\d+_lr", rel):
            failures.append({"ecg_id": int(row.ecg_id), "error": "invalid filename_lr"})
            continue
        try:
            for suffix in (".hea", ".dat"):
                name = rel + suffix
                path = raw / name
                if not path.is_file():
                    raise FileNotFoundError(name)
                if name not in checksums or sha256(path) != checksums[name]:
                    raise ValueError("upstream checksum mismatch: " + name)
            signal, fields = wfdb.rdsamp(str(raw / rel))
            if signal.shape != (cfg["samples_per_record"], cfg["leads"]):
                raise ValueError(f"unexpected shape {signal.shape}")
            if fields.get("fs") != cfg["sampling_rate"]:
                raise ValueError(f"unexpected fs {fields.get('fs')}")
            if not np.isfinite(signal).all():
                raise ValueError("NaN or Inf in signal")
            checked += 1
        except Exception as exc:
            failures.append({"ecg_id": int(row.ecg_id), "error": str(exc)})
    full = args.limit is None
    passed = not issues and not failures and checked == len(subset) and full
    report = {
        "dataset": "PTB-XL", "version": cfg["version"],
        "verification_scope": "full" if full else "pilot",
        "metadata_rows": len(metadata), "requested_records": len(subset),
        "verified_records": checked, "failed_records": len(failures),
        "sampling_rate_hz": cfg["sampling_rate"],
        "samples_per_record": cfg["samples_per_record"],
        "leads": cfg["leads"], "issues": issues,
        "failures": failures[:100], "acceptance_pass": passed,
        "checksum_provenance": base + "SHA256SUMS.txt"
    }
    out = ROOT / "reports/integrity_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    print("Report:", out)
    return 0 if (not issues and not failures and checked == len(subset)) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
