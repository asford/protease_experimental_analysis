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
from itertools import izip, product

import jug

import protease_experimental_analysis
import protease_experimental_analysis.data as data
import protease_experimental_analysis.protease_sequencing_model as protease_sequencing_model

def dict_product(iterable_dict):
    return (dict(izip(iterable_dict, x)) for x in itertools.product(*iterable_dict.itervalues()))

@jug.TaskGenerator
def fit_model(dataset, parameters):
    model = (
        protease_sequencing_model.FractionalSelectionModel(**dict(parameters))
        .build_model(data.model_input[dataset])
    )

    return model.find_MAP()

@jug.TaskGenerator
def report_model_ec50(dataset, model_parameters, fit_parameters):
    model = (
        protease_sequencing_model.FractionalSelectionModel(**dict(parameters))
        .build_model(data.model_input[dataset])
    )

    counts_df = data.counts[dataset]
    counts_df['ec50'] = fit_parameters['sel_ec50']

    cis = numpy.array([
        model.estimate_ec50_cred(fit_parameters, i, cred_spans=[.95])["cred_intervals"][.95]
        for i in range(len(counts_df))
    ])

    counts_df["ec50_95ci_lbound"] = cis[:,0]
    counts_df["ec50_95ci_ubound"] = cis[:,1]
    counts_df["ec50_95ci"] = cis[:,1] - cis[:,0]

    return counts_df


param_space = dict(
    response_fn = ("NormalSpaceErfResponse",),
    min_selection_mass = [5e-7],
)

datasets = data.model_input.keys()

parameter_sets = [frozenset(d.items()) for d in dict_product(param_space)]

fit_params = [ (d, p, fit_model(d, p)) for d, p in product(datasets, parameter_sets) ]

model_ec50s = [report_model_ec50(*f) for f in fit_params]
