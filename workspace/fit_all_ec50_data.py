from __future__ import print_function, division
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs).03d %(name)s %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S'
)

import pandas
import numpy as np
import scipy.stats

import itertools
from itertools import izip, product

import protease_experimental_analysis
import protease_experimental_analysis.data as data
import protease_experimental_analysis.protease_sequencing_model as protease_sequencing_model

def dict_product(iterable_dict):
    return (dict(izip(iterable_dict, x)) for x in itertools.product(*iterable_dict.itervalues()))

def fit_model(dataset, parameters):
    model = (
        protease_sequencing_model.FractionalSelectionModel(**dict(parameters))
        .build_model(data.model_input[dataset])
    )

    return model.find_MAP()

def report_model_ec50(dataset, model_parameters, fit_parameters):
    model = (
        protease_sequencing_model.FractionalSelectionModel(**dict(model_parameters))
        .build_model(data.model_input[dataset])
    )

    counts_df = data.counts[dataset]
    counts_df['ec50'] = fit_parameters['sel_ec50']

    cis = np.array([
        model.estimate_ec50_cred(fit_parameters, i, cred_spans=[.95])["cred_intervals"][.95]
        for i in range(len(counts_df))
    ])

    predictions = {
        p : {
            k : ps[k](fit_parameters)
            for k in ("Frac_sel_pop", "P_cleave")
        }
        for p, ps in model.model_populations.items()
    }

    sum_llh=np.zeros(len(counts_df))
    sum_signed_llh=np.zeros(len(counts_df))

    for p, ps in predictions.items():
        counts_df['downsamp_counts%s' % p] = model.population_data[p]["P_sel"]
        counts_df['pred_counts%s' % p] = np.round(model.population_data[p]["P_sel"].sum() * predictions[p]['P_cleave'])
        bn=scipy.stats.binom(n=model.population_data[p]["P_sel"].sum(),p=predictions[p]['P_cleave'])
        my_llh = bn.logpmf(counts_df['downsamp_counts%s' % p])
        best_llh = bn.logpmf(counts_df['pred_counts%s' % p])
        counts_df['delta_llh%s' % p] = my_llh - best_llh
        counts_df['signed_delta_llh%s' % p] = counts_df['delta_llh%s' % p] * np.sign( counts_df['downsamp_counts%s' % p] - counts_df['pred_counts%s' % p]  )
        sum_llh +=  counts_df['delta_llh%s' % p]
        sum_signed_llh += counts_df['signed_delta_llh%s' % p]
    counts_df['sum_delta_llh'] = sum_llh
    counts_df['sum_signed_delta_llh'] = sum_signed_llh

    counts_df["ec50_95ci_lbound"] = cis[:,0]
    counts_df["ec50_95ci_ubound"] = cis[:,1]
    counts_df["ec50_95ci"] = cis[:,1] - cis[:,0]
    counts_df['sel_k'] = dict(model_parameters)['sel_k'] # dict(model_parameters)['sel_k']


    counts_df=counts_df[['name'] +
                        ['counts%s' % p for p in sorted(model.population_data)] +
                        ['downsamp_counts%s' % p for p in sorted(model.model_populations)] + 
                        ['pred_counts%s' % p for p in sorted(model.model_populations)] +
                        ['delta_llh%s' % p for p in sorted(model.model_populations)] + 
                        ['signed_delta_llh%s' % p for p in sorted(model.model_populations)] +
                        ['sel_k','sum_delta_llh','sum_signed_delta_llh','ec50_95ci_lbound','ec50_95ci_ubound',
                         'ec50_95ci','ec50']]

    counts_df.to_csv('%s.sel_k%s.erf.5e-7.%s.3cycles.fulloutput' % (dataset,dict(model_parameters)['sel_k'],dict(model_parameters)['min_selection_rate']),index=False,sep='\t')



    return counts_df


param_space = dict(
    response_fn = ("NormalSpaceErfResponse",),
    min_selection_mass = [5e-7],
    min_selection_rate = [0.0001],
    outlier_detection_opt_cycles=[3],
    sel_k=[0.8]
)


datasets=['rd1_tryp','rd2_merge_tryp','rd3_merge_tryp','rd4_tryp','rd1_chymo','rd2_merge_chymo','rd3_merge_chymo','rd4_chymo',
'ssm2_tryp','ssm2_chymo']
print (datasets)

parameter_sets = [frozenset(d.items()) for d in dict_product(param_space)]

for d, p in product(datasets, parameter_sets):
    print (d)
    my_fit_model = (d, p, fit_model(d, p))
    report_model_ec50(*my_fit_model)
