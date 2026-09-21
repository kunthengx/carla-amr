from __future__ import annotations
import numpy as np
import pandas as pd

def build_carla_windows(df,customer_col,time_col,feature_cols,label_col='label',window_size=7,stride=1):
    windows=[]; labels=[]; meta=[]; x=df.copy(); x[time_col]=pd.to_datetime(x[time_col],errors='coerce'); x=x.sort_values([customer_col,time_col])
    for cid,g in x.groupby(customer_col):
        g=g.dropna(subset=[time_col]).reset_index(drop=True)
        if len(g)<window_size: continue
        arr=g[feature_cols].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32); med=np.nanmedian(arr,axis=0); inds=np.where(np.isnan(arr))
        if len(inds[0]): arr[inds]=np.take(med,inds[1])
        arr=np.nan_to_num(arr); y=g[label_col].fillna(0).to_numpy(int) if label_col in g.columns else np.zeros(len(g),int)
        for st in range(0,len(g)-window_size+1,stride):
            en=st+window_size; windows.append(arr[st:en]); labels.append(int(y[st:en].max())); meta.append({'customer_id':cid,'start_time':str(g.loc[st,time_col]),'end_time':str(g.loc[en-1,time_col])})
    return np.asarray(windows,np.float32),np.asarray(labels,np.int64),pd.DataFrame(meta)
