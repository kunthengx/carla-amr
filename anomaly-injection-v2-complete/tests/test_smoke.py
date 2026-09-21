import pandas as pd, yaml
from pathlib import Path
from utils.anomaly_generator import AnomalyInjector

def test_smoke():
    cfg=yaml.safe_load(Path('anomaly_config.yaml').read_text()); rows=[]
    for d in range(1,21): rows.append({'location_code':'C001','reading_time':f'2026-01-{d:02d}','V_L1':220,'V_L2':221,'V_L3':219,'I_L1':10,'I_L2':11,'I_L3':9,'I_N':1,'PF_L1':.95,'PF_L2':.94,'PF_L3':.96,'P_L1':2.09,'P_L2':2.28,'P_L3':1.89,'label':0})
    inj,rep=AnomalyInjector(cfg).inject(pd.DataFrame(rows)); assert len(inj)==20; assert len(rep)>0; assert set(inj.label.unique()).issubset({0,1,2})
