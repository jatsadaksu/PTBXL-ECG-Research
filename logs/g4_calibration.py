"""G4.3: validation-only global sigmoid temperature scaling (selected ResNet1D).

Dry run by default. --write exclusively creates results/g4_calibration.json.
Never reads test split, changes thresholds, or overwrites checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'logs'))
import g3_train as g3
from src.dataset import PTBXLDataset

LABELS = ('NORM', 'CD', 'HYP', 'MI', 'STTC')
N = 2146
BINS = 15


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid in float64; avoids overflow and preserves monotonicity.
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def bce_from_logits(z: np.ndarray, y: np.ndarray, log_t: float) -> float:
    # Stable softplus loss; log(T) guarantees T > 0.
    scaled = z / math.exp(log_t)
    return float(np.mean(np.logaddexp(0.0, scaled) - y * scaled))


def fit_temperature(z: np.ndarray, y: np.ndarray) -> dict:
    """Deterministic bounded 1-D optimization on log(T), with failure checks.

    Proposed numerical rule: log(T) in [-8,8], bounded Brent search,
    xatol=1e-10, maxiter=1000; reject boundary or worse-than-baseline fits.
    """
    baseline = bce_from_logits(z, y, 0.0)
    bounds = (-8.0, 8.0)
    opt = minimize_scalar(lambda v: bce_from_logits(z, y, float(v)),
                          bounds=bounds, method='bounded',
                          options={'xatol': 1e-10, 'maxiter': 1000})
    if not opt.success or not np.isfinite(opt.x) or not np.isfinite(opt.fun):
        raise RuntimeError('Temperature optimization failed: ' + str(opt.message))
    if min(opt.x - bounds[0], bounds[1] - opt.x) < 1e-4:
        raise RuntimeError('Temperature optimum near search boundary; do not silently accept')
    if float(opt.fun) > baseline + 1e-10:
        raise RuntimeError('Fitted BCE worse than T=1 baseline')
    return {'temperature': float(math.exp(opt.x)), 'log_temperature': float(opt.x),
            'bce_before': baseline, 'bce_after': float(opt.fun),
            'optimizer': 'scipy.optimize.minimize_scalar(method=bounded)',
            'log_temperature_bounds': list(bounds), 'xatol': 1e-10,
            'maxiter': 1000, 'iterations': int(opt.nit),
            'function_evaluations': int(opt.nfev), 'success': bool(opt.success),
            'message': str(opt.message)}


def ece15(y: np.ndarray, p: np.ndarray) -> float:
    # Equal-width bins: [0,1/15),...,[14/15,1]; empty bins contribute 0.
    idx = np.minimum(np.floor(p * BINS).astype(np.int64), BINS - 1)
    total = len(y)
    value = 0.0
    for b in range(BINS):
        mask = idx == b
        if np.any(mask):
            value += float(mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(value)


def metrics(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> dict:
    per = {}
    for j, name in enumerate(LABELS):
        truth, prob = y[:, j], p[:, j]
        predicted = prob >= thresholds[j]
        tp = int(np.sum(predicted & (truth == 1)))
        fp = int(np.sum(predicted & (truth == 0)))
        fn = int(np.sum(~predicted & (truth == 1)))
        tn = int(np.sum(~predicted & (truth == 0)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per[name] = {
            'positives': int(truth.sum()), 'brier': float(np.mean((prob - truth) ** 2)),
            'ece15': ece15(truth, prob),
            'auprc': float(average_precision_score(truth, prob)),
            'auroc': float(roc_auc_score(truth, prob)),
            'threshold_frozen': float(thresholds[j]),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1,
        }
    return {
        'brier': float(np.mean((p - y) ** 2)),
        'macro_ece15': float(np.mean([per[k]['ece15'] for k in LABELS])),
        'macro_f1': float(np.mean([per[k]['f1'] for k in LABELS])),
        'macro_auprc': float(np.mean([per[k]['auprc'] for k in LABELS])),
        'macro_auroc': float(np.mean([per[k]['auroc'] for k in LABELS])),
        'per_label': per,
    }


def exclusive_json(path: Path, report: dict) -> None:
    if path.exists():
        raise FileExistsError('Refusing overwrite: ' + str(path))
    if not path.parent.is_dir():
        raise RuntimeError('Approved results/ directory missing')
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    try:
        with tmp.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)  # Exclusive publish: never replace an existing report.
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    device = torch.device('cuda:0' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')

    audit_path = ROOT / 'results/g4_validation_audit.json'
    threshold_path = ROOT / 'results/g4_thresholds.json'
    audit, frozen = load_json(audit_path), load_json(threshold_path)
    if (audit.get('schema') != 'g4-validation-audit-v1' or audit.get('split') != 'val'
            or audit.get('test_accessed') is not False or audit.get('n_records') != N
            or audit.get('selected_architecture') != 'resnet1d' or audit.get('selected_seed') != 42):
        raise RuntimeError('G4.1 selection/audit mismatch')
    if (frozen.get('schema') != 'g4-per-label-threshold-v1' or frozen.get('split') != 'val'
            or frozen.get('test_accessed') is not False or frozen.get('n_records') != N
            or frozen.get('model') != 'resnet1d' or frozen.get('seed') != 42
            or frozen.get('best_epoch') != 12 or tuple(frozen.get('labels', ())) != LABELS
            or frozen.get('g4_validation_audit_sha256') != sha256(audit_path)):
        raise RuntimeError('G4.2 frozen thresholds/audit mismatch')
    selected = [r for r in audit['results'] if r['model'] == 'resnet1d' and r['seed'] == 42]
    if len(selected) != 1 or selected[0]['best_epoch'] != 12:
        raise RuntimeError('Unexpected selected checkpoint')
    ckpt_path = ROOT / 'checkpoints/resnet1d_seed42_best.pt'
    ckpt_hash = sha256(ckpt_path)
    if (selected[0]['checkpoint_sha256'] != ckpt_hash
            or frozen.get('checkpoint_sha256') != ckpt_hash
            or frozen.get('checkpoint') != ckpt_path.name):
        raise RuntimeError('Checkpoint SHA256 mismatch')
    digest, _ = g3.fingerprint('resnet1d', 42, False)
    if digest != selected[0]['g3_fingerprint'] or digest != frozen.get('g3_fingerprint'):
        raise RuntimeError('G3 fingerprint mismatch')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)  # trusted local checkpoint only
    g3.validate_ckpt(ckpt, digest, 'resnet1d', 42)
    if int(ckpt['epoch']) != 12:
        raise RuntimeError('Best epoch mismatch')
    model = g3.ResNet1D().to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    ds = PTBXLDataset(split='val', root=ROOT)
    if len(ds) != N or tuple(ds.frame.columns.intersection(LABELS)) != LABELS:
        raise RuntimeError('Validation labels/order mismatch')
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    ys, zs = [], []
    with torch.inference_mode():
        for x, y in loader:
            z = model(x.to(device))
            if z.shape != y.shape or not torch.isfinite(z).all():
                raise RuntimeError('Invalid/non-finite validation logits')
            ys.append(y.numpy().astype(np.float64))
            zs.append(z.cpu().numpy().astype(np.float64))
    y, z = np.concatenate(ys), np.concatenate(zs)
    if y.shape != (N, 5) or z.shape != y.shape or not np.isfinite(z).all() or not np.isin(y, (0, 1)).all():
        raise RuntimeError('Validation array mismatch')
    thresholds = np.array([frozen['results'][k]['threshold'] for k in LABELS], dtype=np.float64)
    if not np.isfinite(thresholds).all() or (thresholds <= 0).any() or (thresholds >= 1).any():
        raise RuntimeError('Invalid frozen thresholds')

    # Reproduce the G4.2 pre-TS probabilities in original float32, not float64,
    # so threshold boundary comparisons reproduce the frozen TP/FP/FN/TN exactly.
    pre_p = torch.sigmoid(torch.from_numpy(z.astype(np.float32))).numpy().astype(np.float64)
    before = metrics(y, pre_p, thresholds)
    for k in LABELS:
        previous = frozen['results'][k]
        for field in ('tp', 'fp', 'fn', 'tn'):
            if before['per_label'][k][field] != previous[field]:
                raise RuntimeError(f'G4.2 frozen confusion matrix mismatch: {k}/{field}')
        if abs(before['per_label'][k]['f1'] - previous['f1']) > 1e-12:
            raise RuntimeError('G4.2 frozen F1 mismatch: ' + k)
    if abs(before['macro_f1'] - frozen['macro_f1_validation_pre_ts']) > 1e-12:
        raise RuntimeError('G4.2 macro F1 mismatch')
    if abs(before['macro_auprc'] - selected[0]['metrics']['macro_auprc']) > 1e-8:
        raise RuntimeError('G4.1 macro AUPRC mismatch')

    fit = fit_temperature(z, y)
    after_p = sigmoid(z / fit['temperature'])
    after = metrics(y, after_p, thresholds)
    # Strictly increasing transform: rank metrics must be preserved, except
    # numerical saturation/ties. Flag any material drift, do not optimize AP.
    for k in LABELS:
        for name in ('auprc', 'auroc'):
            if abs(before['per_label'][k][name] - after['per_label'][k][name]) > 1e-5:
                raise RuntimeError(f'Unexpected rank-metric change: {k}/{name}')
    print(f"PASS CHECKPOINT | resnet1d seed=42 best_epoch=12 | Validation={N}")
    print(f"T={fit['temperature']:.10f} | logT={fit['log_temperature']:.10f} | optimizer={fit['optimizer']} | iterations={fit['iterations']}")
    print(f"BCE before={fit['bce_before']:.8f} after={fit['bce_after']:.8f}")
    print(f"BRIER before={before['brier']:.8f} after={after['brier']:.8f}")
    print(f"MACRO ECE15 before={before['macro_ece15']:.8f} after={after['macro_ece15']:.8f}")
    print(f"MACRO F1 (frozen thresholds) before={before['macro_f1']:.8f} after={after['macro_f1']:.8f}")
    for k in LABELS:
        a, b = before['per_label'][k], after['per_label'][k]
        print(f"{k:5s} ECE15 {a['ece15']:.6f}->{b['ece15']:.6f} | Brier {a['brier']:.6f}->{b['brier']:.6f} | F1 {a['f1']:.6f}->{b['f1']:.6f}")
    report = {
        'schema': 'g4-global-sigmoid-temperature-v1', 'split': 'val', 'test_accessed': False,
        'model': 'resnet1d', 'seed': 42, 'best_epoch': 12, 'n_records': N,
        'labels': list(LABELS), 'checkpoint': ckpt_path.name,
        'checkpoint_sha256': ckpt_hash, 'g3_fingerprint': digest,
        'g4_validation_audit_sha256': sha256(audit_path),
        'g4_thresholds_sha256': sha256(threshold_path),
        'temperature_fit': fit, 'before': before, 'after': after,
        'bce_definition': 'mean BCEWithLogits over all validation records and five independent labels',
        'brier_definition': 'mean squared probability error over all records and five labels',
        'ece_definition': 'per-label positive-frequency ECE, 15 fixed-width bins in [0,1], left-closed/right-open except final bin includes 1; macro mean across 5 labels',
        'threshold_policy': 'G4.2 per-label thresholds frozen before TS; p >= threshold; never reoptimized',
        'note': 'Validation is reused for selection, thresholds and T; calibration results may be optimistic. Test remains locked.',
    }
    print('G4.3 VALIDATION CALIBRATION: PASS')
    if args.write:
        output = ROOT / 'results/g4_calibration.json'
        exclusive_json(output, report)
        print('WROTE', output)
    else:
        print('DRY RUN: no output written. Use --write only after reviewing results.')


if __name__ == '__main__':
    main()
