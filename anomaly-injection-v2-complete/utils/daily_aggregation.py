from __future__ import annotations
import pandas as pd
import numpy as np

def aggregate_daily(df: pd.DataFrame, customer_col: str, time_col: str, numeric_method: str='median', label_col: str|None=None) -> pd.DataFrame:
    x=df.copy(); x[time_col]=pd.to_datetime(x[time_col],errors='coerce'); x=x.dropna(subset=[customer_col,time_col]); x['_date']=x[time_col].dt.floor('D')
    num=[c for c in x.select_dtypes(include=[np.number]).columns if c != label_col]
    meta=[c for c in x.columns if c not in set(num+[customer_col,time_col,'_date',label_col])]
    method = numeric_method if numeric_method in {'mean','min','max','median'} else 'median'
    agg={c:method for c in num}; agg.update({c:'first' for c in meta})
    if label_col and label_col in x.columns: agg[label_col]='max'
    out=x.groupby([customer_col,'_date'],as_index=False).agg(agg).rename(columns={'_date':time_col})
    return out.sort_values([customer_col,time_col]).reset_index(drop=True)
