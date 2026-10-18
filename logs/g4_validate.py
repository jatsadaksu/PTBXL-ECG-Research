"""G4.1 read-only PTB-XL validation re-evaluation of all six frozen G3 best checkpoints.

Install as logs/g4_validate.py. Never instantiates or reads test dataset.
Writes only results/g4_validation_audit.json after all checks pass.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'logs'))
import g3_train as g3
from src.dataset import PTBXLDataset

LABELS = ('NORM', 'CD', 'HYP', 'MI', 'STTC')
EXPECTED = {
    ('cnn1d', 42): (15, 0.79910),
    ('cnn1d', 123): (26, 0.80676),
    ('cnn1d', 2026): (11, 0.80131),
    ('resnet1d', 42): (12, 0.82291),
    ('resnet1d', 123): (7, 0.81082),
    ('resnet1d', 2026): (7, 0.81510),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys, ps, total_loss, n = [], [], 0.0, 0
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if logits.shape != y.shape or not torch.isfinite(logits).all():
                raise RuntimeError('Invalid logits or non-finite predictions')
            total_loss += float(criterion(logits, y).item())
            n += len(x)
            ys.append(y.cpu().numpy())
            ps.append(torch.sigmoid(logits).cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    if y.shape != (2146, 5) or p.shape != y.shape:
        raise RuntimeError(f'Unexpected validation dimensions: {y.shape} {p.shape}')
    per_label = {}
    for j, label in enumerate(LABELS):
        positive = int(y[:, j].sum())
        if not 0 < positive < n:
            raise RuntimeError(f'Degenerate validation class {label}')
        per_label[label] = {
            'positives': positive,
            'prevalence': positive / n,
            'auprc': float(average_precision_score(y[:, j], p[:, j])),
            'auroc': float(roc_auc_score(y[:, j], p[:, j])),
        }
    return {
        'n_records': n,
        'macro_auprc': float(np.mean([v['auprc'] for v in per_label.values()])),
        'macro_auroc': float(np.mean([v['auroc'] for v in per_label.values()])),
        'validation_bce': total_loss / (n * 5),
        'per_label': per_label,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--write', action='store_true', help='Write one new results JSON after full validation')
    args = parser.parse_args()
    device = torch.device('cuda:0' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    output_path = ROOT / 'results' / 'g4_validation_audit.json'
    if args.write and output_path.exists():
        raise FileExistsError(f'Refusing to overwrite {output_path}')
    dataset = PTBXLDataset(split='val', root=ROOT)
    if len(dataset) != 2146:
        raise RuntimeError('Validation count changed')
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=device.type == 'cuda')
    all_results = []
    for (name, seed), (expected_epoch, expected_score) in EXPECTED.items():
        path = ROOT / 'checkpoints' / f'{name}_seed{seed}_best.pt'
        if not path.is_file():
            raise FileNotFoundError(path)
        digest, _ = g3.fingerprint(name, seed, False)
        # Checkpoints must be trusted local G3 artifacts (pickle enabled for RNG state).
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        g3.validate_ckpt(ckpt, digest, name, seed)
        if int(ckpt['epoch']) != expected_epoch or abs(float(ckpt['best_score']) - expected_score) >= 1e-5:
            raise RuntimeError(f'Best checkpoint metadata mismatch: {name} seed={seed}')
        model_cls, expected_params = g3.MODELS[name]
        model = model_cls().to(device)
        model.load_state_dict(ckpt['model'], strict=True)
        if sum(p.numel() for p in model.parameters() if p.requires_grad) != expected_params:
            raise RuntimeError('Parameter count mismatch')
        metrics = evaluate(model, loader, device)
        # G3 computes AP from sigmoid(float32 logits) via sklearn; tolerate last-bit backend drift.
        diff = abs(metrics['macro_auprc'] - float(ckpt['best_score']))
        if diff > 1e-4:
            raise RuntimeError(f'Recomputed AUPRC mismatch {name} seed={seed}: delta={diff:.8g}')
        print(f"PASS {name}_seed{seed} | epoch={expected_epoch} | AP={metrics['macro_auprc']:.5f} | AUROC={metrics['macro_auroc']:.5f} | BCE={metrics['validation_bce']:.5f} | delta={diff:.2g}", flush=True)
        all_results.append({
            'model': name, 'seed': seed, 'best_epoch': expected_epoch,
            'checkpoint': path.name, 'checkpoint_sha256': sha256(path),
            'g3_fingerprint': digest, 'g3_best_macro_auprc': float(ckpt['best_score']),
            'metrics': metrics,
        })
    summary = {}
    for name in g3.MODELS:
        vals = np.array([r['metrics']['macro_auprc'] for r in all_results if r['model'] == name])
        summary[name] = {'mean_macro_auprc': float(vals.mean()), 'sd_macro_auprc': float(vals.std(ddof=1))}
    selected = max(summary, key=lambda k: summary[k]['mean_macro_auprc'])
    seed_run = max((r for r in all_results if r['model'] == selected), key=lambda r: (r['metrics']['macro_auprc'], -r['metrics']['validation_bce'], -r['seed']))
    report = {
        'schema': 'g4-validation-audit-v1', 'split': 'val', 'test_accessed': False,
        'labels': LABELS, 'n_records': 2146, 'source': 'frozen_best_checkpoints',
        'results': all_results, 'summary': summary,
        'selected_architecture': selected, 'selected_seed': seed_run['seed'],
        'note': 'No threshold optimization, calibration, or test evaluation in G4.1',
    }
    print('G4.1 VALIDATION RE-EVALUATION: PASS (6/6)')
    print('Architecture summary:', json.dumps(summary, sort_keys=True))
    print('Selected:', selected, 'seed', seed_run['seed'])
    if args.write:
        if not output_path.parent.is_dir():
            raise RuntimeError('Approved results/ directory missing')
        # Exclusive creation avoids accidental overwrites; complete JSON prepared first.
        with output_path.open('x', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        print('WROTE', output_path)
    else:
        print('READ-ONLY MODE: no output file written; use --write after PASS')


if __name__ == '__main__':
    main()
