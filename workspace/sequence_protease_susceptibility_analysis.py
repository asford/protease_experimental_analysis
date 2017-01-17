from __future__ import print_function, division
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs).03d %(name)s %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S'
)

import pandas

import numpy

import itertools

import jug

from itertools import product, izip, chain
def dict_product(iterable_dict):
    return (dict(izip(iterable_dict, x)) for x in product(*iterable_dict.itervalues()))

@jug.TaskGenerator
def model_cv_predictions(protease, parameters):
    import sklearn.cross_validation as cross_validation
    from sklearn.metrics import mean_squared_error
    import protease_experimental_analysis.sequence_protease_susceptibility as sequence_protease_susceptibility

    import data_constk_161218_rd1234
    import pandas as pd
    import numpy as np

    ssm_data = pandas.DataFrame.from_dict({
        "sequence" : map(str.strip, open('160924_grocklin_ssm2_myseqs').readlines())
    })

    ssm_data["pred_sequence"] = ["GGGSASHM" + x + "LEGGGSEQ" + ('Z' * (46 - len(x))) for x in ssm_data.sequence.values]
    #newbg=pd.read_csv('ssm_mut_pred_logistic_upweight10_161224',delim_whitespace=True)
    bigtable_prot=pd.read_csv('big_ssm2_nextseq_table_known_frac_named',delim_whitespace=True,header=None)
    bigtable_prot=bigtable_prot[[0,1]]
    bigtable_prot.columns=['name','sequence']
    ssm=pd.merge(left=bigtable_prot, right=ssm_data,on='sequence',how='inner')

    ssm_lines="""EEHEE_rd3_0037.pdb   TTIKVNGQEYTVPLSPEQAAKAAKKRWPDYEVQIHGNTVKVTR
EEHEE_rd3_1498.pdb  GTLHLNGVTVKVPSLEKAIKAAKKFAKKYNLEVQVHGNTVHVH
EEHEE_rd3_1702.pdb  TTIHVGDLTLKYDNPKKAYEIAKKLAKKYNLTVTIKNGKITVT
EEHEE_rd3_1716.pdb  TEVHLGDIKLKYPNPEQAKKAAEKLAQKYNLTWTVIGDYVKIE
EHEE_0882.pdb       QETIEVEDEEEARRVAKELRKKGYEVKIERRGNKWHVHRT
EHEE_rd2_0005.pdb   TTRYRFTDEEEARRAAKEWARRGYQVHVTQNGTYWEVEVR
EHEE_rd3_0015.pdb   KTQYEYDTKEEAQKAYEKFKKQGIPVTITQKNGKWFVQVE
HEEH_rd2_0779.pdb   TLDEARELVERAKKEGTGVDVNGQRFEDWREAERWVREQEKNK
HEEH_rd3_0223.pdb   TIDEIIKALEQAVKDNKPIQVGNYTVTSADEAEKLAKKLKKEY
HEEH_rd3_0726.pdb   TELKKKLEEALKKGEEVRVKFNGIEIRITSEDAARKAVELLEK
HEEH_rd3_0872.pdb   TWQDLVKIAEKALEKGEPITINGITVTTKEQAKQAIEYLKKAY
HHH_0142.pdb        RKWEEIAERLREEFNINPEEAREAVEKAGGNEEEARRIVKKRL
HHH_rd2_0134.pdb    SKDEAQREAERAIRSGNKEEARRILEEAGYSPEQAERIIRKLG
HHH_rd3_0138.pdb    ERRKIEEIAKKLYQSGNPEAARRFLRKAGISEEEIERILQKAG
Pin1                MADEEKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSG
hYAP65              FEIPDDVPLPAGWEMAKTSSGQRYFKNHIDQTTTWQDPRKAMLSQM
villin              LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF""".split('\n')
    wt_seqs={}
    for line in ssm_lines:
        wt_seqs[line.split()[0]] = line.split()[1]

    def wt(x):
        if '_wt' in x or x.split('_')[-1][-4:] in ['.pdb','AP65','llin','Pin1']:
            return x
        else:
            return '_'.join(x.split('_')[0:-1])
    ssm['my_wt'] = ssm['name'].map(wt)

    def delta_pred_vs_wt(wts, preds):
        wt_preds=dict(zip(wts, preds))
        return [wt_preds[wt] - pred for wt, pred in zip(wts, preds)]






    scramble_data = data_constk_161218_rd1234.scramble_data
    
    full_data = data_constk_161218_rd1234.full_df
    full_data["full_sequence"] = "GGGSASHM" + full_data["sequence"] + "LEGGGSEQ"
    max_len=max([len(x) for x in full_data["full_sequence"].values])
    full_data["full_sequence"] = [old_seq + ('Z' * (max_len - len(old_seq))) for old_seq in full_data["full_sequence"].values]



    full_data_byprot = {
        t : full_data[full_data["protease"] == t]
        for t in (protease,)
    }

    prottest={}

    full_data_byprot[protease]['prottest'] = ['prottest' in x for x in full_data_byprot[protease]['name']]
    prottest[protease] = full_data_byprot[protease].query('prottest == True')
    prottest[protease]['parent'] = [x.split('_prottest')[0] for x in prottest[protease]['name']]
    prottest[protease] = pd.merge(left=prottest[protease][['name','parent','ec50','full_sequence']],
                                  right=full_data_byprot[protease][['name','ec50','full_sequence']],
                                 left_on='parent',right_on='name',how='inner')
    prottest[protease]['delta_ec50'] = prottest[protease]['ec50_x'] - prottest[protease]['ec50_y']


    data = scramble_data.query("round != 'pdb' and protease == '%s'" % protease).copy()
    model = sequence_protease_susceptibility.CenterLimitedPSSMModel(**dict(parameters))

    
    model.fit(list(data["full_sequence"].values), data["ec50"].values)
    
    ssm_preds = model.predict(list(ssm['pred_sequence']))
    delta_pred = delta_pred_vs_wt(ssm['my_wt'].values, ssm_preds)
    
    extras = {}
    extras['min_delta_vs_pred'] = min(delta_pred)
    extras['max_delta_vs_pred'] = max(delta_pred)
    extras['delta_pred'] = delta_pred
    extras['ssm_pred'] = ssm_preds

    prottest_pred_delta = (model.predict(list(prottest[protease]['full_sequence_x'])) - 
                           model.predict(list(prottest[protease]['full_sequence_y'])))

    extras['prottest_mse']= mean_squared_error(prottest_pred_delta, prottest[protease]['delta_ec50'])
    extras['prottest_R2'] = np.corrcoef(prottest_pred_delta, prottest[protease]['delta_ec50'])[0][1]**2.0
    extras['prottest_slope'] = np.polyfit(prottest_pred_delta, prottest[protease]['delta_ec50'], 1)[0]
    extras['fit_coeffs_'] = model.fit_coeffs_
    extras['cen_to_flank'] = max(abs(model.fit_coeffs_['seq_weights'][0:-1])) / max(abs(np.ravel(model.fit_coeffs_['weights'][0:-1,0:-1])))

    data["fit_pred_ec50"] = model.predict(list(data["full_sequence"].values))
    data["pred_ec50"] = cross_validation.cross_val_predict(
        model, data["full_sequence"], data["ec50"], )

    return (data, extras)

param_space = {
    #alpha_center" : numpy.exp(numpy.array([-6,-5.5,-5,-4.5,-4,-3.5])),
    "alpha_center" : numpy.exp(numpy.array([-15])),
    #"alpha_flank" : numpy.exp(numpy.array([-8,-7,-6.5,-6,-5.5,-5,-4.5,-4,-3.5,-3])),
    "alpha_flank" : numpy.exp(numpy.array([-15])),
    "max_data_upweight" : (1.0,),
    "init_tot_l" : (20,),
    "init_max_sumweight" : (10,),
    "error_upper_lim": (10,),
    "flanking_window": (1,2,3,4),
}

proteases = ["tryp","chymo"]
#proteases=["chymo",]
parameter_sets = [frozenset(d.items()) for d in dict_product(param_space)]
model_results = { (d, p) : model_cv_predictions(d, p) for d, p in product(proteases, parameter_sets) }
