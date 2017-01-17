import pandas
import numpy as np
from os import path
import glob

__all__ = ["scramble_data","large_scramble_data"]

full_df=pandas.DataFrame()
for protease in ['tryp','chymo']:
    files = glob.glob('rd*_%s.sel_k0.8.erf.5e-7.0.0001.3cycles.fulloutput' % protease)
    for fn in files:
        my_round = int(fn[2])
        my_df = pandas.read_csv(fn,delim_whitespace=True)
        my_df['protease'] = protease
        my_df['round'] = int(my_round)

        if my_round == 4:
            my_df['traindata'] = ['_PG_hp' in x and '_prottest' not in x and ('_rd4_' in x or '_rd' not in x) for x in my_df['name']]
        else:
            my_df['traindata'] = ['_PG_hp' in x or '_hp' in x or '_random' in x for x in my_df['name']]
        print fn, protease, my_round, len(my_df[my_df.traindata == True]),
        
        if my_round == 4:
            my_df['traindata_large'] = [('_PG_hp' in x or '_prottest' in x or 'tryp_test' in x or 'chymo_test' in x) and ('_rd4_' in x or '_rd' not in x) for x in my_df['name']]
        else:
            my_df['traindata_large'] = my_df['traindata']
        print len(my_df[my_df.traindata_large == True])
        

        full_df=full_df.append(my_df)

print len(full_df)
sequence_df=pandas.read_csv('rd1234_allnames',delim_whitespace=True)
sequence_df['C'] = ['C' in x for x in sequence_df['sequence']]
sequence_df=sequence_df.drop_duplicates('name')
sequence_df=sequence_df[sequence_df['C'] == False]
full_df=pandas.merge(left=full_df, right=sequence_df[['name','sequence']], on='name', how='inner')
print len(full_df)


scramble_data = (full_df
    .query("ec50_95ci < 1.0")
    .query("traindata == True")
    [["name", "round", "sequence", "protease", "ec50", "ec50_95ci_lbound", "ec50_95ci_ubound", "ec50_95ci"]])

scramble_data["full_sequence"] = "GGGSASHM" + scramble_data["sequence"] + "LEGGGSEQ"
max_len=max([len(x) for x in scramble_data["full_sequence"].values])
scramble_data["full_sequence"] = [old_seq + ('Z' * (max_len - len(old_seq))) for old_seq in scramble_data["full_sequence"].values]

large_scramble_data = (full_df
    .query("ec50_95ci < 1.0")
    .query("traindata_large == True")
    [["name", "round", "sequence", "protease", "ec50", "ec50_95ci_lbound", "ec50_95ci_ubound", "ec50_95ci"]])

large_scramble_data["full_sequence"] = "GGGSASHM" + large_scramble_data["sequence"] + "LEGGGSEQ"
max_len=max([len(x) for x in large_scramble_data["full_sequence"].values])
large_scramble_data["full_sequence"] = [old_seq + ('Z' * (max_len - len(old_seq))) for old_seq in large_scramble_data["full_sequence"].values]