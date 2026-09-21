from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
import yaml
from utils.daily_aggregation import aggregate_daily
from utils.anomaly_generator import AnomalyInjector
from utils.visualization import plot_label_distribution
from utils.carla_adapter import build_carla_windows

def read_table(path):
    return pd.read_excel(path) if path.suffix.lower() in {'.xlsx','.xls'} else pd.read_csv(path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--config',default='anomaly_config.yaml'); ap.add_argument('--output-dir',default='sample_output'); ap.add_argument('--window-size',type=int,default=None); args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8')); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); df=read_table(Path(args.input)); original=len(df); cc=cfg['columns']['customer_id']; tc=cfg['columns']['timestamp']; lc=cfg['columns']['label']
    for c in (cc,tc):
        if c not in df.columns: raise KeyError(f"Kolom wajib '{c}' tidak ditemukan. Sesuaikan anomaly_config.yaml.")
    if cfg.get('aggregation',{}).get('enabled',True): df=aggregate_daily(df,cc,tc,cfg['aggregation'].get('numeric_method','median'),lc if lc in df.columns else None)
    injected,report=AnomalyInjector(cfg).inject(df); data_path=out/'Data_AMR_Anomaly_v2.xlsx'; injected.to_excel(data_path,index=False); report.to_csv(out/'anomaly_report.csv',index=False)
    stats={'rows_original':original,'rows_after_daily_aggregation':len(df),'rows_output':len(injected),'anomalies_injected':len(report),'label_distribution':{str(k):int(v) for k,v in injected[lc].value_counts().sort_index().items()},'anomaly_type_distribution':{} if report.empty else {str(k):int(v) for k,v in report['anomaly_type'].value_counts().items()}}
    (out/'dataset_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8'); plot_label_distribution(injected,lc,str(out/'label_distribution.png'))
    ws=args.window_size or (cfg.get('windowing',{}).get('size') if cfg.get('windowing',{}).get('enabled') else None)
    if ws:
        import numpy as np
        excluded={cc,tc,lc}; feats=[c for c in injected.select_dtypes(include='number').columns if c not in excluded]; X,y,meta=build_carla_windows(injected,cc,tc,feats,lc,int(ws),int(cfg.get('windowing',{}).get('stride',1))); np.save(out/f'carla_X_w{ws}.npy',X); np.save(out/f'carla_y_w{ws}.npy',y); meta.to_csv(out/f'carla_meta_w{ws}.csv',index=False)
    print('Selesai:',data_path)
if __name__=='__main__': main()
