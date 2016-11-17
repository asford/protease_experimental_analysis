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

    return {
        "params" : params,
        "predictions" : predictions
    }

replicate_pairs = {
    "%s_%s" % s : ("%s_%s" % s, "%s_redo_%s" % s)
    for s in itertools.product(("rd2", "rd3"), ("chymo", "tryp"))
}

param_space = dict(
    response_fn = ("LogisticResponse", "NormalSpaceLogisticResponse"),
    min_selection_rate = (True, False),
    min_selection_mass = (True, False),
)

datasets = chain(*replicate_pairs.values())
parameter_sets = [frozenset(d.items()) for d in dict_product(param_space)]

model_results = { (d, p) : build_and_fit_model(d, p) for d, p in product(datasets, parameter_sets) }
