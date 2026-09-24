"""Audited descriptive analysis for hierarchical model/scenario/repetition/checkpoint data.
No p-values or cross-scenario CIs are produced from a small or unplanned sample.
"""
import json
from itertools import combinations
import numpy as np
import pandas as pd
KEY=['model_name','scenario_id','repetition']
BINARY=['goal_achieved','safe_completion','critical_failure','terminal_state_valid']

def valid(df):
    if df.empty:return df.copy()
    flag=pd.to_numeric(df['valid_evaluated_run'],errors='coerce').eq(1) if 'valid_evaluated_run' in df else df['evaluation_status'].eq('scored')
    return df.loc[flag].copy()

def checkpoints(df):
    rows=[]
    for _,row in valid(df).iterrows():
        raw=row.get('checkpoint_results_json')
        try: items=json.loads(raw) if isinstance(raw,str) else raw
        except (ValueError,TypeError):continue
        if not isinstance(items,list):continue
        for k,item in enumerate(items,1):
            if not isinstance(item,dict):continue
            met=item.get('met')
            if met not in (0,1,True,False):continue
            rows.append({'model_name':row['model_name'],'scenario_id':row['scenario_id'],
                'repetition':int(row['repetition']),'checkpoint_id':f'CP{k:03d}',
                'checkpoint_text':str(item.get('expected_assistant_action','')),
                'met':int(met),'critical_checkpoint':bool(item.get('critical_checkpoint',False))})
    return pd.DataFrame(rows,columns=KEY+['checkpoint_id','checkpoint_text','met','critical_checkpoint'])

def audit_table(df):
    problems=[]
    if df.empty:return pd.DataFrame([{'Issue':'No runs','Detail':'No evaluation results available'}])
    for c in KEY+['valid_evaluated_run','checkpoint_results_json']:
        if c not in df:problems.append(('Missing field',c))
    if any(c not in df for c in KEY):return pd.DataFrame(problems,columns=['Issue','Detail'])
    dup=df.duplicated(KEY,keep=False)
    if dup.any():problems.append(('Duplicate run identifiers',f'{dup.sum()} rows; fix before analysis'))
    v=valid(df)
    for _,r in v.iterrows():
        key=f"{r['model_name']} / {r['scenario_id']} / rep {r['repetition']}"
        try: cp=json.loads(r['checkpoint_results_json'])
        except (TypeError,ValueError):cp=[]
        if not isinstance(cp,list) or not cp:problems.append(('Missing/malformed checkpoint details',key));continue
        vals=[x.get('met') for x in cp if isinstance(x,dict)]
        if len(vals)!=len(cp) or any(x not in (0,1,True,False) for x in vals):problems.append(('Invalid checkpoint scores',key));continue
        observed=sum(vals)/len(vals)
        claimed=pd.to_numeric(r.get('checkpoint_completion'),errors='coerce')
        if pd.notna(claimed) and abs(observed-claimed)>1e-7:problems.append(('Checkpoint aggregate mismatch',key))
        if pd.to_numeric(r.get('goal_achieved'),errors='coerce')==1 and pd.to_numeric(r.get('critical_failure'),errors='coerce')==1:
            problems.append(('Goal achieved with critical failure (distinct outcomes)',key))
    cp=checkpoints(df)
    if not cp.empty:
        for (scenario,cpid),g in cp.groupby(['scenario_id','checkpoint_id']):
            if g.checkpoint_text.nunique()>1 or g.critical_checkpoint.nunique()>1:
                problems.append(('Checkpoint definition differs across runs',f'{scenario} / {cpid}: verify version-locked scenario definitions'))
    if v.scenario_id.nunique()<20:problems.append(('Insufficient independent scenarios for general inference',f'{v.scenario_id.nunique()} distinct scenarios; descriptive comparisons only'))
    return pd.DataFrame(problems,columns=['Issue','Detail']) if problems else pd.DataFrame([{'Issue':'No detected data-quality warnings','Detail':'Automated checks passed; human judge validation remains necessary'}])

def checkpoint_repetition_table(df):
    """One row per checkpoint observation; group all runs of each scenario together."""
    cp=checkpoints(df)
    cols=['Model','Scenario','Repetition','Checkpoint','Requirement','Critical','Checkpoint outcome','Checkpoint success (valid reps)','Successful / valid reps']
    if cp.empty:return pd.DataFrame(columns=cols)
    cp=cp.sort_values(['model_name','scenario_id','repetition','checkpoint_id'],kind='stable')
    grp=cp.groupby(['model_name','scenario_id','checkpoint_id'],sort=False)['met']
    cp['rate']=grp.transform('mean')*100
    cp['count']=grp.transform('sum').astype(int).astype(str)+' / '+grp.transform('size').astype(int).astype(str)
    cp['outcome']=cp.met.map({1:'Met',0:'Not met'})
    return cp.rename(columns={'model_name':'Model','scenario_id':'Scenario','repetition':'Repetition','checkpoint_id':'Checkpoint','checkpoint_text':'Requirement','critical_checkpoint':'Critical','rate':'Checkpoint success (valid reps)','count':'Successful / valid reps','outcome':'Checkpoint outcome'})[cols].reset_index(drop=True)

def checkpoint_summary_table(df):
    """One row per model/scenario/checkpoint, with observed counts and variability."""
    cp=checkpoints(df)
    cols=['Model','Scenario','Checkpoint','Requirement','Critical requirement','Met / valid repetitions','Checkpoint success rate (%)','Outcome varies']
    if cp.empty:return pd.DataFrame(columns=cols)
    g=cp.groupby(['model_name','scenario_id','checkpoint_id','checkpoint_text','critical_checkpoint'],sort=True).met.agg(['sum','count','min','max']).reset_index()
    g['fraction']=g['sum'].astype(str)+' / '+g['count'].astype(str)
    g['rate']=(100*g['sum']/g['count']).round(2)
    g['varies']=np.where(g['min']!=g['max'],'Yes','No')
    return g.rename(columns={'model_name':'Model','scenario_id':'Scenario','checkpoint_id':'Checkpoint','checkpoint_text':'Requirement','critical_checkpoint':'Critical requirement','fraction':'Met / valid repetitions','rate':'Checkpoint success rate (%)','varies':'Outcome varies'})[cols]

def checkpoint_run_summary(df):
    """One row per attempted run; errors are visible but not scored."""
    cp=checkpoints(df)
    counts=cp.groupby(KEY,as_index=False).met.agg(['sum','count']).reset_index().rename(columns={'sum':'met_count','count':'total_count'}) if not cp.empty else pd.DataFrame(columns=KEY+['met_count','total_count'])
    out=df[KEY].copy() if all(c in df for c in KEY) else pd.DataFrame(columns=KEY)
    if out.empty:return pd.DataFrame(columns=['Model','Scenario','Repetition','Run status','Checkpoints met / applicable','Checkpoint completion (%)'])
    out['Run status']=np.where(out.index.isin(valid(df).index),'Scored','Not scored (excluded from checkpoint rates)')
    out=out.merge(counts,on=KEY,how='left')
    out['Checkpoints met / applicable']=out.apply(lambda r:f"{int(r.met_count)} / {int(r.total_count)}" if pd.notna(r.met_count) and pd.notna(r.total_count) else '—',axis=1)
    out['Checkpoint completion (%)']=out.apply(lambda r:round(100*r.met_count/r.total_count,2) if pd.notna(r.total_count) and r.total_count>0 else np.nan,axis=1)
    return out.rename(columns={'model_name':'Model','scenario_id':'Scenario','repetition':'Repetition'})[['Model','Scenario','Repetition','Run status','Checkpoints met / applicable','Checkpoint completion (%)']].sort_values(['Model','Scenario','Repetition'],kind='stable').reset_index(drop=True)

def run_scores(df):
    v=valid(df)
    if v.empty:return v.assign(verified_checkpoint_completion=pd.Series(dtype=float))
    cp=checkpoints(df)
    if cp.empty:return v.assign(verified_checkpoint_completion=np.nan)
    agg=cp.groupby(KEY,as_index=False).agg(passed=('met','sum'),applicable=('met','size'))
    agg['verified_checkpoint_completion']=agg.passed/agg.applicable
    return v.merge(agg[KEY+['verified_checkpoint_completion','passed','applicable']],on=KEY,how='left',validate='one_to_one')

def model_summary(df):
    columns=['Model','Attempted runs','Valid runs','Distinct scenarios','Checkpoint completion (equal-scenario mean)','Goal achievement','Safe completion','Critical failures','Complete task success (strict)']
    if df.empty:return pd.DataFrame(columns=columns)
    v=run_scores(df)
    rows=[]
    for model,allruns in df.groupby('model_name',sort=True):
        g=v[v.model_name.eq(model)].copy()
        for col in BINARY:g[col]=pd.to_numeric(g[col],errors='coerce')
        g['strict_success']=np.where(g[['goal_achieved','terminal_state_valid','critical_failure','verified_checkpoint_completion']].isna().any(axis=1),np.nan,((g.goal_achieved==1)&(g.terminal_state_valid==1)&(g.critical_failure==0)&(g.verified_checkpoint_completion==1)).astype(int))
        def rate(col):
            x=g[col].dropna();return f'{int(x.sum())}/{len(x)} ({100*x.mean():.1f}%)' if len(x) else '—'
        by=g.groupby('scenario_id').verified_checkpoint_completion.mean().dropna()
        rows.append([model,len(allruns),len(g),g.scenario_id.nunique(),f'{100*by.mean():.2f}%' if len(by) else '—',rate('goal_achieved'),rate('safe_completion'),rate('critical_failure'),rate('strict_success')])
    return pd.DataFrame(rows,columns=columns)

def repetition_summary(df):
    v=run_scores(df);rows=[]
    if v.empty:return pd.DataFrame()
    for (model,scenario),g in v.groupby(['model_name','scenario_id'],sort=True):
        x=g.verified_checkpoint_completion.dropna()
        rows.append({'Model':model,'Scenario':scenario,'Valid repetitions':len(g),'Attempted repetitions':len(df[(df.model_name==model)&(df.scenario_id==scenario)]),'Mean checkpoint completion':round(100*x.mean(),2) if len(x) else np.nan,'Within-scenario SD (pp)':round(100*x.std(ddof=1),2) if len(x)>1 else np.nan,'Range (pp)':round(100*(x.max()-x.min()),2) if len(x) else np.nan,'Goal successes':int(pd.to_numeric(g.goal_achieved,errors='coerce').eq(1).sum()),'Critical failures':int(pd.to_numeric(g.critical_failure,errors='coerce').eq(1).sum())})
    return pd.DataFrame(rows)

def paired_comparison_table(df):
    """Descriptive matched-scenario comparisons only; no inferential statistics."""
    v=run_scores(df)
    cols=['Model A','Model B','Measure','Scenarios evaluated by both models','Model A: average across shared scenarios (%)','Model B: average across shared scenarios (%)','Difference: A − B (percentage points)']
    if v.empty:return pd.DataFrame(columns=cols)
    metrics={'verified_checkpoint_completion':'Checkpoint completion','goal_achieved':'Goal achievement','safe_completion':'Safe completion','critical_failure':'Critical failure frequency'}
    rows=[]
    for a,b in combinations(sorted(v.model_name.unique()),2):
        left=v[v.model_name==a].groupby('scenario_id')[list(metrics)].mean()
        right=v[v.model_name==b].groupby('scenario_id')[list(metrics)].mean()
        for metric,label in metrics.items():
            pair=left[[metric]].join(right[[metric]],how='inner',lsuffix='_a',rsuffix='_b').dropna()
            if pair.empty:continue
            ma=100*pair.iloc[:,0].mean();mb=100*pair.iloc[:,1].mean()
            rows.append([a,b,label,len(pair),round(ma,2),round(mb,2),round(ma-mb,2)])
    return pd.DataFrame(rows,columns=cols)
