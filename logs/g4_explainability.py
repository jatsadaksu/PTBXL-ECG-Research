"""G4.4 validation-only XAI. First --freeze-selection; then dry run; then --write.
Run from project root. No Test access. Existing results are never overwritten.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'logs'))
import g3_train as g3
from src.dataset import PTBXLDataset
from src.explainability import integrated_gradients, best_two_windows, random_two_windows, masked

LABELS = ('NORM', 'CD', 'HYP', 'MI', 'STTC')
SEED = 42
N = 2146
SEL = ROOT / 'results/g4_xai_case_selection.json'
OUT = ROOT / 'results/g4_explainability.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def exclusive(path, data):
    if path.exists():
        raise FileExistsError(f'Refusing overwrite: {path}')
    if not path.parent.is_dir():
        raise RuntimeError('Approved results directory missing')
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    try:
        with tmp.open('x', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def prerequisites():
    audit_p = ROOT / 'results/g4_validation_audit.json'
    th_p = ROOT / 'results/g4_thresholds.json'
    cal_p = ROOT / 'results/g4_calibration.json'
    ckpt_p = ROOT / 'checkpoints/resnet1d_seed42_best.pt'
    a, t, c = map(read, (audit_p, th_p, cal_p))
    if (a.get('schema') != 'g4-validation-audit-v1' or a.get('split') != 'val'
            or a.get('n_records') != N or a.get('selected_architecture') != 'resnet1d'
            or a.get('selected_seed') != 42 or a.get('test_accessed') is not False):
        raise RuntimeError('G4.1 audit mismatch')
    if (t.get('schema') != 'g4-per-label-threshold-v1' or t.get('split') != 'val'
            or t.get('n_records') != N or t.get('model') != 'resnet1d'
            or t.get('seed') != 42 or t.get('best_epoch') != 12
            or t.get('test_accessed') is not False or t.get('g4_validation_audit_sha256') != digest(audit_p)):
        raise RuntimeError('G4.2 audit mismatch')
    if (c.get('schema') != 'g4-global-sigmoid-temperature-v1' or c.get('test_accessed') is not False or c.get('split') != 'val'):
        raise RuntimeError('G4.3 audit mismatch')
    selected = [r for r in a['results'] if r['model'] == 'resnet1d' and r['seed'] == 42]
    if len(selected) != 1 or selected[0]['best_epoch'] != 12:
        raise RuntimeError('Selected checkpoint mismatch')
    fp, _ = g3.fingerprint('resnet1d', 42, False)
    if (selected[0]['g3_fingerprint'] != fp or t['g3_fingerprint'] != fp
            or selected[0]['checkpoint_sha256'] != digest(ckpt_p)
            or t['checkpoint_sha256'] != digest(ckpt_p)):
        raise RuntimeError('Checkpoint fingerprint/hash mismatch')
    # Keep G4.3 as an immutable prerequisite; never re-fit T.
    return {'audit_sha256': digest(audit_p), 'thresholds_sha256': digest(th_p),
            'calibration_sha256': digest(cal_p), 'checkpoint_sha256': digest(ckpt_p),
            'g3_fingerprint': fp}, ckpt_p


def choose(ds, deps):
    if len(ds) != N:
        raise RuntimeError('Validation size mismatch')
    rng = np.random.default_rng(SEED)
    rows = []
    for label_index, label in enumerate(LABELS):
        eligible = ds.frame.loc[ds.frame[label].eq(1)].sort_values('ecg_id', kind='stable')
        if len(eligible) < 20:
            raise RuntimeError('Not enough eligible cases')
        picks = rng.choice(len(eligible), size=20, replace=False)
        for index in picks:
            row = eligible.iloc[int(index)]
            rows.append({'selection_order': len(rows) + 1, 'ecg_id': int(row.ecg_id),
                         'patient_id': int(row.patient_id), 'target_label': label,
                         'target_index': label_index})
    return {'schema': 'g4-xai-selection-v1', 'split': 'val', 'test_accessed': False,
            'n_validation': N, 'n_pairs': 100, 'n_unique_ecgs': len({r['ecg_id'] for r in rows}),
            'n_unique_patients': len({r['patient_id'] for r in rows}),
            'seed': SEED, 'rng': 'numpy.random.default_rng',
            'selection': 'sorted ecg_id; one shared RNG; 20 positive pairs per label',
            'labels': list(LABELS), 'dependencies': deps, 'cases': rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default='auto')
    ap.add_argument('--freeze-selection', action='store_true')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    if args.freeze_selection and args.write:
        ap.error('Use --freeze-selection and --write in separate steps')
    deps, ckpt_path = prerequisites()
    ds = PTBXLDataset(split='val', root=ROOT)
    proposal = choose(ds, deps)
    if args.freeze_selection:
        exclusive(SEL, proposal)
        print('G4.4.1 SELECTION FROZEN:', SEL, '| pairs=100 | unique ECGs=', proposal['n_unique_ecgs'])
        return
    if not SEL.exists():
        raise RuntimeError('Freeze selection first: --freeze-selection')
    if read(SEL) != proposal:
        raise RuntimeError('Selection manifest differs from frozen policy/data')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda' if (args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available())) else 'cpu')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    g3.validate_ckpt(ckpt, deps['g3_fingerprint'], 'resnet1d', 42)
    if int(ckpt['epoch']) != 12:
        raise RuntimeError('Checkpoint epoch mismatch')
    model = g3.ResNet1D().to(device).eval()
    model.load_state_dict(ckpt['model'], strict=True)
    idx = {int(row.ecg_id): i for i, row in ds.frame.iterrows()}
    rng = np.random.default_rng(SEED)
    results = []
    for case in proposal['cases']:
        x, y = ds[idx[case['ecg_id']]]
        target = case['target_index']
        if y[target].item() != 1:
            raise RuntimeError('Positive case mismatch')
        x = x.unsqueeze(0).to(device)
        ig, completeness = integrated_gradients(model, x, target, steps=32)
        scores = ig.abs().sum(dim=1).squeeze(0).cpu().numpy()
        starts = best_two_windows(scores)
        random_starts = [random_two_windows(rng) for _ in range(20)]
        # One original forward pass; batched perturbations for stable and efficient evaluation.
        batch = torch.cat([masked(x, starts)] + [masked(x, pair) for pair in random_starts], dim=0)
        with torch.no_grad():
            original = float(model(x)[0, target].item())
            altered = model(batch)[:, target].detach().cpu().numpy().astype(np.float64)
        drops = original - altered
        difference = float(drops[0] - np.mean(drops[1:]))
        if not np.isfinite(drops).all() or not np.isfinite(list(completeness.values())).all():
            raise RuntimeError('Nonfinite attribution/perturbation')
        results.append({**case, 'original_logit': original,
                        'ig_windows_start': list(starts), 'random_windows_start': [list(p) for p in random_starts],
                        'ig_logit_drop': float(drops[0]), 'random_logit_drops': drops[1:].tolist(),
                        'random_mean_drop': float(np.mean(drops[1:])),
                        'random_sd_drop': float(np.std(drops[1:], ddof=1)),
                        'difference': difference, 'ig_beats_random': bool(difference > 0),
                        'completeness': completeness})
        if len(results) % 10 == 0:
            print('PROGRESS', len(results), '/ 100')
    def summary(rows):
        d = np.asarray([r['difference'] for r in rows], dtype=np.float64)
        return {'n_pairs': len(rows), 'mean_difference': float(d.mean()),
                'sd_difference': float(d.std(ddof=1)), 'median_difference': float(np.median(d)),
                'ig_beats_random_count': int(np.sum(d > 0)),
                'ig_beats_random_fraction': float(np.mean(d > 0)),
                'mean_ig_drop': float(np.mean([r['ig_logit_drop'] for r in rows])),
                'mean_random_drop': float(np.mean([r['random_mean_drop'] for r in rows]))}
    report = {'schema': 'g4-xai-v1', 'split': 'val', 'test_accessed': False,
              'selection_sha256': digest(SEL), 'dependencies': deps,
              'model': 'resnet1d', 'seed': 42, 'best_epoch': 12,
              'ig_steps': 32, 'ig_rule': 'right-Riemann, target raw logit, zero normalized baseline',
              'mask_rule': 'two globally best non-overlapping 50-sample windows, all 12 leads zeroed',
              'random_rule': '20 uniformly sampled nonoverlapping window pairs per case; RNG seed42',
              'summary': summary(results),
              'per_label': {name: summary([r for r in results if r['target_label'] == name]) for name in LABELS},
              'cases': results}
    print('G4.4 VALIDATION XAI: PASS | mean difference:', report['summary']['mean_difference'])
    for name in LABELS:
        print(name, report['per_label'][name])
    if args.write:
        exclusive(OUT, report)
        print('WROTE', OUT)
    else:
        print('DRY RUN: no evaluation output written; use --write to persist.')

if __name__ == '__main__':
    main()
