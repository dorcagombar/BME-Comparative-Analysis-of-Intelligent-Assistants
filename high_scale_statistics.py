"""Scenario-clustered, repetition-aware companion analysis. Run: python high_scale_statistics.py results.csv --out statistics"""
import argparse
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

OUTCOMES = ['goal_achieved','safe_completion','critical_failure','checkpoint_completion','all_checkpoints_met','task_success']
MIN_SCENARIOS_FOR_INTERVAL = 20  # conservative reporting gate, not a power calculation

def valid_runs(df):
    if 'valid_evaluated_run' not in df: raise ValueError('Missing valid_evaluated_run')
    return df.loc[pd.to_numeric(df.valid_evaluated_run,errors='coerce').eq(1)].copy()

def checkpoint_long(df):
    rows=[]
    for _, row in df.iterrows():
        raw=row.get('checkpoint_results_json')
        if pd.isna(raw) or not str(raw).strip(): continue
        try: items=json.loads(raw)
        except (TypeError,ValueError): continue
        if not isinstance(items,list): continue
        for index,item in enumerate(items):
            if not isinstance(item,dict): continue
            # The current schema has no immutable checkpoint_id. Position is stable only
            # when scenario checkpoint definitions and their ordering are version-locked.
            met=pd.to_numeric(item.get('met'),errors='coerce')
            if met not in (0,1): continue
            rows.append(dict(model_name=row.model_name,scenario_id=row.scenario_id,
                repetition=int(row.repetition),checkpoint_id=f'cp_{index+1:03d}',
                checkpoint_turn=item.get('turn'),met=int(met),
                critical=bool(item.get('critical_checkpoint',False)),
                checkpoint_text=item.get('expected_assistant_action','')))
    return pd.DataFrame(rows,columns=['model_name','scenario_id','repetition','checkpoint_id','checkpoint_turn','met','critical','checkpoint_text'])

def wilson(successes,n,z=1.959963984540054):
    if not n:return (np.nan,np.nan)
    p=successes/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h),min(1,c+h)

def checkpoint_tables(cp):
    if cp.empty:return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    keys=['model_name','scenario_id','checkpoint_id']
    counts=cp.groupby(keys,as_index=False).agg(successes=('met','sum'),applicable_repetitions=('met','size'),critical=('critical','first'),checkpoint_text=('checkpoint_text','first'))
    counts['success_rate']=counts.successes/counts.applicable_repetitions
    counts['binary_variability']=counts.success_rate*(1-counts.success_rate)
    bounds=[wilson(int(r.successes),int(r.applicable_repetitions)) for r in counts.itertuples()]
    counts['wilson_low_fixed_scenario']=[x[0] for x in bounds]
    counts['wilson_high_fixed_scenario']=[x[1] for x in bounds]
    cp=cp.sort_values(keys+['repetition']).copy()
    g=cp.groupby(keys,sort=False)
    cp['cumulative_successes']=g.met.cumsum()
    cp['cumulative_applicable']=g.cumcount()+1
    cp['cumulative_checkpoint_success']=cp.cumulative_successes/cp.cumulative_applicable
    cp['previous_met']=g.met.shift(1)
    cp['repetition_change']=cp.met-cp.previous_met
    intervals=[wilson(int(s),int(n)) for s,n in zip(cp.cumulative_successes,cp.cumulative_applicable)]
    cp['cumulative_wilson_width_fixed_scenario']=[hi-lo for lo,hi in intervals]
    run=cp.groupby(['model_name','scenario_id','repetition'],as_index=False).agg(checkpoints_met=('met','sum'),applicable_checkpoints=('met','size'))
    run['run_checkpoint_accuracy']=run.checkpoints_met/run.applicable_checkpoints
    run=run.sort_values(['model_name','scenario_id','repetition'])
    rg=run.groupby(['model_name','scenario_id'],sort=False)
    run['cumulative_met']=rg.checkpoints_met.cumsum()
    run['cumulative_applicable']=rg.applicable_checkpoints.cumsum()
    run['cumulative_checkpoint_accuracy']=run.cumulative_met/run.cumulative_applicable
    run['repetition_accuracy_change_pp']=100*rg.run_checkpoint_accuracy.diff()
    return counts,cp,run

def scenario_outcomes(df):
    cols=[c for c in OUTCOMES if c in df]
    for c in cols:df[c]=pd.to_numeric(df[c],errors='coerce')
    result=df.groupby(['model_name','scenario_id'],as_index=False).agg(
        valid_repetitions=('repetition','size'),**{c:(c,'mean') for c in cols})
    return result

def paired_differences(scen,bootstraps=4000,seed=2026):
    rng=np.random.default_rng(seed);rows=[]
    cols=[c for c in OUTCOMES if c in scen]
    for a,b in combinations(sorted(scen.model_name.unique()),2):
        left=scen[scen.model_name.eq(a)].set_index('scenario_id')
        right=scen[scen.model_name.eq(b)].set_index('scenario_id')
        for metric in cols:
            paired=left[[metric]].join(right[[metric]],how='inner',lsuffix='_a',rsuffix='_b').dropna()
            if paired.empty:continue
            differences=(paired[f'{metric}_a']-paired[f'{metric}_b']).to_numpy()
            n=len(differences);effect=float(differences.mean())
            # Resample paired SCENARIOS, preserving each model's repetition aggregate.
            if n>=MIN_SCENARIOS_FOR_INTERVAL:
                sampled=rng.integers(0,n,size=(bootstraps,n))
                distribution=differences[sampled].mean(axis=1)
                low,high=np.quantile(distribution,[.025,.975])
                status='scenario-cluster bootstrap percentile CI; descriptive, not multiplicity adjusted'
            else:
                low=high=np.nan
                status='descriptive only: fewer than 20 paired scenarios; no inferential CI'
            rows.append(dict(model_a=a,model_b=b,measure=metric,paired_scenarios=n,
                mean_a=paired[f'{metric}_a'].mean(),mean_b=paired[f'{metric}_b'].mean(),
                mean_paired_difference_a_minus_b=effect,ci_95_low=low,ci_95_high=high,
                inference_status=status))
    return pd.DataFrame(rows)

def run(input_csv,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(input_csv)
    required={'model_name','scenario_id','repetition','valid_evaluated_run','checkpoint_results_json'}
    missing=required-set(df)
    if missing:raise ValueError(f'Missing required columns: {sorted(missing)}')
    if df.duplicated(['model_name','scenario_id','repetition']).any():
        raise ValueError('Duplicate model/scenario/repetition keys: resolve before analysis')
    validity=df.groupby('model_name',as_index=False).agg(attempts=('repetition','size'),distinct_scenarios=('scenario_id','nunique'),valid_runs=('valid_evaluated_run',lambda s: int(pd.to_numeric(s,errors='coerce').eq(1).sum())))
    validity['valid_run_fraction']=validity.valid_runs/validity.attempts
    scored=valid_runs(df)
    cp=checkpoint_long(scored)
    counts,curve,run_curve=checkpoint_tables(cp)
    scen=scenario_outcomes(scored)
    comparisons=paired_differences(scen)
    files={'validity.csv':validity,'checkpoint_success_by_scenario.csv':counts,
           'checkpoint_repetition_curve.csv':curve,'run_repetition_curve.csv':run_curve,
           'scenario_outcomes.csv':scen,'paired_model_differences.csv':comparisons}
    for name,table in files.items():table.to_csv(out/name,index=False)
    print('Wrote',', '.join(files),'to',out)
    print('Valid runs:',len(scored),'distinct scenarios:',scored.scenario_id.nunique(),
          'checkpoint observations:',len(cp))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results_csv');p.add_argument('--out',default='high_scale_statistics')
    a=p.parse_args();run(a.results_csv,a.out)
