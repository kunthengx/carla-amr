from __future__ import annotations
import numpy as np
import pandas as pd
from .column_utils import find_col, phase_candidates

def clip_physical_bounds(df,cfg):
    x=df.copy(); p=cfg.get('physics',{})
    for ph in (1,2,3):
        v=find_col(x.columns,phase_candidates('V',ph)); pf=find_col(x.columns,phase_candidates('PF',ph))
        if v: x[v]=pd.to_numeric(x[v],errors='coerce').clip(p.get('voltage_min',100),p.get('voltage_max',300))
        if pf: x[pf]=pd.to_numeric(x[pf],errors='coerce').abs().clip(p.get('pf_min',0),p.get('pf_max',1))
    f=find_col(x.columns,['freq','frequency','frequency_hz','Hz'])
    if f: x[f]=pd.to_numeric(x[f],errors='coerce').clip(p.get('frequency_min',45),p.get('frequency_max',55))
    return x

def recompute_power_columns(df):
    x=df.copy(); Ps=[]; Qs=[]; Ss=[]
    for ph in (1,2,3):
        v=find_col(x.columns,phase_candidates('V',ph)); i=find_col(x.columns,phase_candidates('I',ph)); pf=find_col(x.columns,phase_candidates('PF',ph))
        pc=find_col(x.columns,phase_candidates('P',ph)); qc=find_col(x.columns,phase_candidates('Q',ph)); sc=find_col(x.columns,phase_candidates('S',ph))
        if v and i:
            vv=pd.to_numeric(x[v],errors='coerce'); ii=pd.to_numeric(x[i],errors='coerce'); s=vv*ii/1000.0
            if sc: x[sc]=s; Ss.append(sc)
            if pf:
                pfv=pd.to_numeric(x[pf],errors='coerce').abs().clip(0,1); pp=s*pfv
                if pc: x[pc]=pp; Ps.append(pc)
                if qc: x[qc]=np.sqrt(np.maximum(s**2-pp**2,0)); Qs.append(qc)
    for cols,names in [(Ps,['P_total','Ptot','ActivePowerTotal']),(Qs,['Q_total','Qtot','ReactivePowerTotal']),(Ss,['S_total','Stot','ApparentPowerTotal'])]:
        total=find_col(x.columns,names)
        if total and cols: x[total]=x[cols].sum(axis=1,min_count=1)
    return x

def enforce_physics(df,cfg):
    x=clip_physical_bounds(df,cfg)
    return recompute_power_columns(x) if cfg.get('physics',{}).get('recompute_power',True) else x
