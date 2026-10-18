"""G4.2: frozen validation-only, per-label F1-optimal thresholds (pre-TS).

Install in the already-approved logs/ directory. No held-out test access.
Dry-run by default; --write exclusively creates results/g4_thresholds.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'logs'))
import g3_train as g3
from src.dataset import PTBXLDataset

LABELS = ('NORM', 'CD', 'HYP', 'MI', 'STTC')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def choose_threshold(y, p):
    """Exact search over distinct probabilities, predictions defined as p >= t.

    With ties, select the highest threshold. Include all-negative t=1.0
    (unless p=1.0, where t=1.0 still predicts that sample positive).
    """
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) != len(p) or not len(y) or not np.isfinite(p).all():
        raise ValueError('Invalid arrays')
    if not np.isin(y, (0, 1)).all() or y.sum() in (0, len(y)):
        raise ValueError('Degenerate labels')
    if (p < 0).any() or (p > 1).any():
        raise ValueError('Probabilities outside [0,1]')
    # Stable descending ordering and groups of equal probabilities.
    order = np.argsort(-p, kind='stable')
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(sorted_p[1:] != sorted_p[:-1]), len(p) - 1]
    tp = np.cumsum(sorted_y, dtype=np.int64)[ends]
    pred_pos = ends + 1
    positives = int(y.sum())
    fp = pred_pos - tp
    fn = positives - tp
    f1 = 2.0 * tp / (2.0 * tp + fp + fn)
    thresholds = sorted_p[ends]
    # t=1.0 is a valid candidate, including when sigmoid saturates at 1.0.
    # When max(p)<1, it predicts all negative and gives F1=0.
    if thresholds[0] < 1.0:
        thresholds = np.r_[1.0, thresholds]
        tp = np.r_[0, tp]
        fp = np.r_[0, fp]
        fn = np.r_[positives, fn]
        f1 = np.r_[0.0, f1]
    maximum = float(f1.max())
    # Floating tolerance for mathematical ties; prefer largest threshold.
    winner = int(np.flatnonzero(np.isclose(f1, maximum, atol=1e-12, rtol=0))[0])
    t = float(thresholds[winner])
    yp = p >= t
    TP = int(np.count_nonzero(yp & (y == 1)))
    FP = int(np.count_nonzero(yp & (y == 0)))
    FN = int(np.count_nonzero(~yp & (y == 1)))
    TN = int(np.count_nonzero(~yp & (y == 0)))
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0
    score = 2 * TP / (2 * TP + FP + FN) if 2 * TP + FP + FN else 0.0
    if abs(score - maximum) > 1e-12:
        raise AssertionError('Threshold audit mismatch')
    return dict(threshold=t, tp=TP, fp=FP, fn=FN, tn=TN,
                precision=precision, recall=recall, f1=score,
                positives=positives, predicted_positives=TP + FP,
                unique_probabilities=int(len(np.unique(p))))


def atomic_create_json(path, obj):
    if path.exists():
        raise FileExistsError(f'Refusing overwrite: {path}')
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        with tmp.open('x', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        # Hard link guarantees no overwrite even under concurrent execution.
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    audit_path = ROOT / 'results/g4_validation_audit.json'
    if not audit_path.is_file():
        raise FileNotFoundError('G4.1 audit missing: ' + str(audit_path))
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit.get('schema') != 'g4-validation-audit-v1' or audit.get('split') != 'val' or audit.get('test_accessed') is not False or audit.get('n_records') != 2146:
        raise RuntimeError('Invalid G4.1 audit')
    if audit.get('selected_architecture') != 'resnet1d' or audit.get('selected_seed') != 42:
        raise RuntimeError('Unexpected architecture/seed; do not silently change selection')
    selected = [r for r in audit['results'] if r['model'] == 'resnet1d' and r['seed'] == 42]
    if len(selected) != 1 or selected[0]['best_epoch'] != 12:
        raise RuntimeError('Invalid selected checkpoint record')
    ckpt_path = ROOT / 'checkpoints/resnet1d_seed42_best.pt'
    if sha256(ckpt_path) != selected[0]['checkpoint_sha256']:
        raise RuntimeError('Checkpoint SHA256 changed since G4.1')
    digest, _ = g3.fingerprint('resnet1d', 42, False)
    if digest != selected[0]['g3_fingerprint']:
        raise RuntimeError('G3 configuration fingerprint changed')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # trusted local G3 file only
    g3.validate_ckpt(ckpt, digest, 'resnet1d', 42)
    if int(ckpt['epoch']) != 12:
        raise RuntimeError('Unexpected checkpoint epoch')
    model = g3.ResNet1D().to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    dataset = PTBXLDataset(split='val', root=ROOT)
    if len(dataset) != 2146 or tuple(dataset.frame.columns.intersection(LABELS)) != LABELS:
        raise RuntimeError('Validation data mismatch')
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    ys, ps = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            if logits.shape != y.shape or not torch.isfinite(logits).all():
                raise RuntimeError('Non-finite or wrong-shape logits')
            ys.append(y.numpy().astype(np.int64))
            ps.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    if y.shape != (2146, 5) or p.shape != y.shape:
        raise RuntimeError('Invalid validation shape')
    report = {}
    for j, label in enumerate(LABELS):
        r = choose_threshold(y[:, j], p[:, j])
        if r['positives'] != selected[0]['metrics']['per_label'][label]['positives']:
            raise RuntimeError(f'Positive count mismatch: {label}')
        report[label] = r
        print(f"{label:5s} threshold={r['threshold']:.9f} F1={r['f1']:.6f} precision={r['precision']:.6f} recall={r['recall']:.6f} TP={r['tp']} FP={r['fp']} FN={r['fn']}", flush=True)
    output = {
        'schema': 'g4-per-label-threshold-v1', 'split': 'val', 'test_accessed': False,
        'model': 'resnet1d', 'seed': 42, 'best_epoch': 12,
        'checkpoint': ckpt_path.name, 'checkpoint_sha256': sha256(ckpt_path),
        'g3_fingerprint': digest, 'g4_validation_audit_sha256': sha256(audit_path),
        'n_records': 2146, 'labels': list(LABELS),
        'probability_transform': 'sigmoid(raw_logits); no temperature scaling',
        'prediction_rule': 'p >= threshold',
        'selection': 'maximize per-label validation F1 over unique probabilities and t=1; tie -> highest threshold',
        'results': report,
        'macro_f1_validation_pre_ts': float(np.mean([r['f1'] for r in report.values()])),
        'note': 'Validation is used for model, seed and threshold selection; metrics may be optimistic. Test remains locked.'
    }
    print(f"MACRO F1 (Validation, optimized): {output['macro_f1_validation_pre_ts']:.6f}")
    print('G4.2 THRESHOLD SELECTION: PASS')
    if args.write:
        out = ROOT / 'results/g4_thresholds.json'
        if not out.parent.is_dir():
            raise RuntimeError('Approved results/ missing')
        atomic_create_json(out, output)
        print('WROTE', out)
    else:
        print('DRY RUN: no output written. Use --write to save the locked thresholds.')

if __name__ == '__main__':
    main()
