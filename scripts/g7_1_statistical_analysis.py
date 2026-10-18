#!/usr/bin/env python3
"""G7.1 validation-only, read-only patient-cluster inference on frozen G7 JSON."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

LABELS = ('NORM','CD','HYP','MI','STTC')

def holm(p):
    idx=np.argsort(p); out=np.empty(len(p)); prev=0.
    for rank,i in enumerate(idx):
        prev=max(prev,min(1.,(len(p)-rank)*p[i]));out[i]=prev
    return out.tolist()

def cluster_test(rows, rng, B=10000, P=20000):
    ids=np.array([str(r['patient_id']) for r in rows]); d=np.array([float(r['difference']) for r in rows]);
    uniq, inverse=np.unique(ids,return_inverse=True); groups=[np.where(inverse==i)[0] for i in range(len(uniq))]
    if len(uniq)<2:raise ValueError('Insufficient independent patient clusters')
    boot=np.empty(B)
    for b in range(B):
        chosen=rng.integers(len(groups),size=len(groups)); indices=np.concatenate([groups[i] for i in chosen]);boot[b]=d[indices].mean()
    # cluster sign-flip test; preserves within-patient dependence
    sums=np.array([d[g].sum() for g in groups]); observed=d.mean();
    extreme=0; chunk=1000
    for start in range(0,P,chunk):
        n=min(chunk,P-start); signs=rng.choice([-1,1],size=(n,len(groups)))
        null=(signs@sums)/len(d);extreme+=int(np.count_nonzero(null>=observed-1e-12))
    p=(extreme+1)/(P+1)
    wins=int(np.count_nonzero(d>0)); ties=int(np.count_nonzero(d==0))
    return {'n_pairs':len(rows),'n_patients':len(uniq),'mean_difference':float(d.mean()),
            'median_difference':float(np.median(d)),'sd_difference':float(d.std(ddof=1)),
            'q05':float(np.quantile(d,.05)),'q25':float(np.quantile(d,.25)),
            'q75':float(np.quantile(d,.75)),'q95':float(np.quantile(d,.95)),
            'positive_fraction':float(np.mean(d>0)),'positive_count':wins,'ties':ties,
            'cluster_bootstrap_95ci':[float(v) for v in np.quantile(boot,[.025,.975])],
            'cluster_signflip_p_one_sided':float(p),
            'descriptive_pair_sign_p_one_sided':float(binomtest(wins,len(d)-ties,.5,alternative='greater').pvalue),
            'mean_ig_drop':float(np.mean([r['ig_logit_drop'] for r in rows])),
            'mean_random_drop':float(np.mean([r['random_mean_drop'] for r in rows])),
            'completeness_relative_error_mean':float(np.mean([r['completeness']['completeness_relative_error'] for r in rows])),
            'completeness_relative_error_median':float(np.median([r['completeness']['completeness_relative_error'] for r in rows])),
            'completeness_relative_error_q95':float(np.quantile([r['completeness']['completeness_relative_error'] for r in rows],.95))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--bootstrap',type=int,default=10000);ap.add_argument('--permutations',type=int,default=20000)
    args=ap.parse_args(); inp=Path(args.input).resolve();out=Path(args.output).resolve()
    if inp==out:raise ValueError('Output cannot overwrite original G7 evidence')
    raw=inp.read_bytes(); a=json.loads(raw)
    if a.get('schema')!='g7-validation-xai-v1' or a.get('split')!='val' or a.get('test_accessed') is not False:
        raise ValueError('STOP: Unexpected schema/split or test-access flag')
    rows=a['cases']
    if len(rows)!=100 or a.get('n_pairs')!=100:raise ValueError('STOP: Expected 100 frozen G7 pairs')
    for r in rows:
        if r['target_label'] not in LABELS:raise ValueError('Unexpected class')
        if not np.isclose(r['ig_logit_drop']-r['random_mean_drop'],r['difference'],rtol=0,atol=1e-7):raise ValueError('Difference mismatch')
        if len(r['random_windows_start'])!=20:raise ValueError('Random-window count mismatch')
    rng=np.random.default_rng(2026)
    result={k:cluster_test([r for r in rows if r['target_label']==k],rng,args.bootstrap,args.permutations) for k in LABELS}
    adj=holm([result[k]['cluster_signflip_p_one_sided'] for k in LABELS])
    for k,p in zip(LABELS,adj):
        result[k]['holm_adjusted_p_one_sided']=p
        result[k]['exploratory_significant_at_0_05']=bool(p<.05)
    overall=cluster_test(rows,rng,args.bootstrap,args.permutations)
    report={'schema':'g7.1-validation-statistics-v1','analysis_type':'posthoc_exploratory',
      'input_sha256':hashlib.sha256(raw).hexdigest(),'split':'val','test_accessed':False,
      'seed':2026,'bootstrap_replicates':args.bootstrap,'signflip_replicates':args.permutations,
      'hypothesis':'mean(IG logit drop - average random logit drop) > 0',
      'inference':'patient-cluster percentile bootstrap CI and patient-cluster sign-flip p; five per-label p-values Holm-adjusted',
      'limitations':['20 selected positive pairs per class; not representative of all validation ECGs',
      'selection may reuse patients across classes','IG completeness errors high for some classes',
      'random baseline is averaged over 20 windows, not an independent patient sample',
      'cluster sign-flip requires approximate cluster-level sign symmetry under the null',
      'exploratory significance is not evidence of clinical explanation validity'],
      'overall':overall,'per_label':result}
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError('STOP: Output already exists; do not overwrite')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('G7.1 PASS: read-only validation statistical analysis')
    print('Input SHA256:',report['input_sha256'])
    print('Overall mean difference:',round(overall['mean_difference'],6),'95% CI:',overall['cluster_bootstrap_95ci'])
    for k,v in result.items():print(k,'mean',round(v['mean_difference'],6),'CI', [round(x,6) for x in v['cluster_bootstrap_95ci']], 'p_Holm',round(v['holm_adjusted_p_one_sided'],6))
    print('Saved:',out)
if __name__=='__main__':main()
