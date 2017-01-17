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

import protease_experimental_analysis
import protease_experimental_analysis.data as data
import protease_experimental_analysis.protease_sequencing_model as protease_sequencing_model

from itertools import product, izip, chain
def dict_product(iterable_dict):
    return (dict(izip(iterable_dict, x)) for x in product(*iterable_dict.itervalues()))

@jug.TaskGenerator
def build_and_fit_model(dataset, parameters):
    model =(
        protease_sequencing_model.FractionalSelectionModel(**dict(parameters))
        .build_model(data.model_input[dataset])
    )

    params = model.find_MAP()

    predictions = {
        p : {
            k : ps[k](params)
            for k in ("fraction_selected", "selection_dist")
        }
        for p, ps in model.model_populations.items()
    }

    predictions['logp'] = model.logp(params)

    cis = numpy.array([
        model.estimate_ec50_cred(params, i, cred_spans=[.95])["cred_intervals"][.95]
        for i in range(len(params['sel_ec50']))
    ])

    return {
        "params" : params,
        "predictions" : predictions,
        "cis": cis,
        "ec50_95ci" : cis[:,1] - cis[:,0], 
        "logp" : model.logp(params),
    }

replicate_pairs = {
    "%s_%s" % s : ("%s_%s" % s, "%s_redo_%s" % s)
    for s in itertools.product(("rd2", "rd3"), ("chymo", "tryp"))
}


param_space = dict(
    response_fn = ("NormalSpaceErfResponse",),
    #min_selection_mass = ["global", "per_selection", False] + [1 * 10 ** -a for a in range(5, 8)] + [5e-7],
    min_selection_mass=[5e-7],
    alex_opt=[False],
    sel_k_dict=['mid'],#,'low','lowest'],
    min_selection_rate=[3e-5,1e-4,False],
    outlier_detection_opt_cycles=[2],
    sel_k=[0.8,2.0]
)

param_space2 = dict(
    response_fn = ("NormalSpaceLogisticResponse",),
    #min_selection_mass = ["global", "per_selection", False] + [1 * 10 ** -a for a in range(5, 8)] + [5e-7],
    min_selection_mass=[5e-7],
    alex_opt=[False],
    sel_k_dict=['mid'],
    min_selection_rate=[False],
    outlier_detection_opt_cycles=[2]
)

datasets = chain(*replicate_pairs.values())

parameter_sets = [frozenset(d.items()) for d in dict_product(param_space)]
#print (len(parameter_sets))
#for k in parameter_sets:
#    print (k)
model_results = { (d, p) : build_and_fit_model(d, p) for d, p in product(datasets, parameter_sets) }
