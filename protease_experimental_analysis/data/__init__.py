from __future__ import division
import pandas as pd
import numpy as np
import pickle
import sys
import glob
import os
from os import path

import collections

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

summary=pd.read_csv(path.join(basedir, 'experiments.csv'))
summary=summary.dropna(how='all')
summary=summary.fillna('-')
summary=summary.where( summary != '-', None)

model_input = {}

for i, r in summary.iterrows():
    name = r["input"].replace(".counts", "")
    if name not in model_input:
        model_input[name] = {}

    rnd = int(r["column"].replace("counts", ""))
    assert rnd not in model_input[name]

    model_input[name][rnd] = dict(
        parent            = r['parent'],
        selection_level   = r['selection_strength'],
        num_selected      = fix_num_selected(r['cells_collected']),
        Frac_sel_pop = none_or_div(r['fraction_collected'], r['parent_expression']),
        conc_factor       = r['conc_factor']
    )
    if model_input[name][rnd]['Frac_sel_pop'] != None:
        model_input[name][rnd]['Frac_sel_pop'] *= r['matching_sequences']
    if r['matching_sequences'] != None:
        model_input[name][rnd]['num_selected']  *= r['matching_sequences']

counts = {
    exper : pd.read_csv(
        path.join(basedir, exper + ".counts"),
        delim_whitespace=True)
    for exper in model_input
}

for exper in model_input:
    counts_df = pd.read_csv(
            path.join(basedir, exper + ".counts"),
            delim_whitespace=True)

    for i, col in enumerate(counts_df.columns[1:]):
        model_input[exper][i]["name"] = counts_df["name"]
        model_input[exper][i]['seq_counts'] = counts_df[col].astype(np.int)
    
    for k, v in model_input[exper].items():
        if v["seq_counts"].sum() > v["num_selected"]: 
            pfrac = v["seq_counts"].values.astype(float) / v['seq_counts'].sum()
            v["P_sel"] = np.floor(pfrac * v["num_selected"])
        else:
            v["num_selected"] = v["seq_counts"].sum()
            v["P_sel"] = np.array(v["seq_counts"].astype(np.float))
        v['min_fraction'] = None
