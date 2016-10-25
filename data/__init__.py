from __future__ import division
import pandas as pd
import numpy as np
import pickle
import sys
import glob
import os
from os import path

__all__ = ["model_input"]
basedir = path.dirname(__file__) 

def none_or_int(x):
    if x == None:
        return None
    return int(x)

def none_or_div(x1, x2):
    if x1 == None:
        return None
    return x1 / x2

def fix_num_selected(x):
    if x == None:
        return 5e5
    return x

df=pd.read_csv(path.join(basedir, 'experiments.csv'))
df=df.dropna(how='all')
df=df.fillna('-')
df=df.where( df != '-', None)

alldata={}

for i in range(len(df)):
    if df['input'].values[i] not in alldata:
        alldata[df['input'].values[i]] = {}

    alldata[df['input'].values[i]][int(df['column'].values[i].replace('counts',''))] = dict(
        parent = df['parent'].values[i],
        selection_level=df['selection_strength'].values[i],
        num_selected=fix_num_selected(df['cells_collected'].values[i]),
        fraction_selected=none_or_div(df['fraction_collected'].values[i], df['parent_expression'].values[i]),
        conc_factor=df['conc_factor'].values[i]
    )

for exper in alldata:
    counts_df = pd.read_csv(path.join(basedir, exper),delim_whitespace=True)
    for i, col in enumerate(counts_df.columns[1:]):
        alldata[exper][i]['seq_counts'] = counts_df[col].astype(np.int)
    
    for k, v in alldata[exper].items():
        if v["seq_counts"].sum() > v["num_selected"]: 
            pfrac = v["seq_counts"].values.astype(float) / v['seq_counts'].sum()
            v["selected"] = np.floor(pfrac * v["num_selected"])
        else:
            v["num_selected"] = v["seq_counts"].sum()
            v["selected"] = np.array(v["seq_counts"].astype(np.float))
        v['min_fraction'] = None

model_input = { k.replace(".counts", "") : v for k, v in alldata.items() }
