from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .column_utils import find_col, phase_candidates
from .physics_rules import enforce_physics

@dataclass
class InjectionEvent:
    row_index:int; customer_id:str; timestamp:str; anomaly_type:str; anomaly_class:int; modified_columns:str; details:str

class AnomalyInjector:
    def __init__(self,config):
        self.cfg=config; self.rng=np.random.default_rng(int(config.get('seed',42))); cm=config['injection']['class_mapping']; self.normal=cm['normal']; self.defect=cm['defect']; self.theft=cm['theft']
    def _cols(self,df):
        d={k:{ph:find_col(df.columns,phase_candidates(k,ph)) for ph in (1,2,3)} for k in ('V','I','PF','P','Q','S')}
        d['IN']=find_col(df.columns,['I_N','IN','Current_N','NeutralCurrent']); d['PF_total']=find_col(df.columns,['PF_total','PFtot','PowerFactorTotal']); return d
    def _scale(self,x,cols,f):
        for c in cols:
            if c: x.at[self.idx,c]=pd.to_numeric(pd.Series([x.at[self.idx,c]]),errors='coerce').iloc[0]*f
    def _event(self,x,t,cls,mods,detail):
        cc=self.cfg['columns']['customer_id']; tc=self.cfg['columns']['timestamp']; mods=[m for m in mods if m]
        return InjectionEvent(int(self.idx),str(x.at[self.idx,cc]),str(x.at[self.idx,tc]),t,cls,','.join(mods),detail)
    def current_bypass_theft(self,x,c):
        p=self.cfg['parameters']['current_bypass_theft']; drop=self.rng.uniform(p['reduction_min'],p['reduction_max']); mods=[c['I'][1],c['I'][2],c['I'][3]]; self._scale(x,mods,1-drop); return self._event(x,'current_bypass_theft',self.theft,mods,f'drop={drop:.3f}')
    def intermittent_theft(self,x,c):
        p=self.cfg['parameters']['intermittent_theft']; ph=int(self.rng.integers(1,4)); drop=self.rng.uniform(p['reduction_min'],p['reduction_max']); mods=[c['I'][ph]]; self._scale(x,mods,1-drop); return self._event(x,'intermittent_theft',self.theft,mods,f'phase={ph}; drop={drop:.3f}')
    def phase_loss(self,x,c):
        p=self.cfg['parameters']['phase_loss']; ph=int(self.rng.integers(1,4)); r=self.rng.uniform(p['residual_min'],p['residual_max']); mods=[c[k][ph] for k in ('V','I','P','Q','S')]; self._scale(x,mods,r); return self._event(x,'phase_loss',self.defect,mods,f'phase={ph}; residual={r:.3f}')
    def voltage_imbalance(self,x,c):
        p=self.cfg['parameters']['voltage_imbalance']; ph=int(self.rng.integers(1,4)); drop=self.rng.uniform(p['reduction_min'],p['reduction_max']); mods=[c['V'][ph]]; self._scale(x,mods,1-drop); return self._event(x,'voltage_imbalance',self.defect,mods,f'phase={ph}; drop={drop:.3f}')
    def pf_manipulation(self,x,c):
        p=self.cfg['parameters']['pf_manipulation']; ph=int(self.rng.integers(1,4)); col=c['PF'][ph] or c['PF_total'];
        if col: x.at[self.idx,col]=self.rng.uniform(p['pf_min'],p['pf_max'])
        return self._event(x,'pf_manipulation',self.defect,[col],f'phase={ph}')
    def neutral_current_leakage(self,x,c):
        p=self.cfg['parameters']['neutral_current_leakage']; col=c['IN']; mult=self.rng.uniform(p['multiplier_min'],p['multiplier_max'])
        if col:
            v=pd.to_numeric(pd.Series([x.at[self.idx,col]]),errors='coerce').iloc[0]
            if pd.isna(v) or v==0:
                vals=pd.to_numeric(pd.Series([x.at[self.idx,q] for q in c['I'].values() if q]),errors='coerce').dropna(); v=float(vals.mean()) if len(vals) else 1.0
            x.at[self.idx,col]=v*mult
        return self._event(x,'neutral_current_leakage',self.defect,[col],f'multiplier={mult:.3f}')
    def meter_defect_drift(self,x,c):
        p=self.cfg['parameters']['meter_defect_drift']; drift=self.rng.uniform(p['drift_min'],p['drift_max']); factor=1+(-drift if self.rng.random()<.5 else drift); mods=[q for k in ('V','I','PF') for q in c[k].values() if q]; self._scale(x,mods,factor); return self._event(x,'meter_defect_drift',self.defect,mods,f'factor={factor:.3f}')
    def inject(self,df):
        x=df.copy(); c=self._cols(x); icfg=self.cfg['injection']; lc=self.cfg['columns']['label']
        if lc not in x.columns: x[lc]=self.normal
        eligible=x.index.to_numpy()
        if icfg.get('protect_existing_anomalies',True): eligible=x.index[x[lc].fillna(self.normal).eq(self.normal)].to_numpy()
        n=min(len(eligible),max(0,int(round(len(eligible)*float(icfg.get('anomaly_fraction',.2)))))); chosen=self.rng.choice(eligible,n,replace=False) if n else []
        names=list(icfg['type_weights']); probs=np.array([icfg['type_weights'][k] for k in names],float); probs/=probs.sum(); events=[]
        for idx in chosen:
            self.idx=int(idx); typ=str(self.rng.choice(names,p=probs)); ev=getattr(self,typ)(x,c); x.at[self.idx,lc]=ev.anomaly_class; events.append(ev)
        x=enforce_physics(x,self.cfg); return x,pd.DataFrame([e.__dict__ for e in events])
