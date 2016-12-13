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

    import protease_experimental_analysis.sequence_protease_susceptibility as sequence_protease_susceptibility

    import data_160912_grocklin_ec50_summary
    scramble_data = data_160912_grocklin_ec50_summary.scramble_data

    tdata = scramble_data.query("round != 'pdb' and protease == '%s'" % protease).copy()
    model = sequence_protease_susceptibility.CenterLimitedPSSMModel(**dict(parameters))


    tdata["pred_ec50"] = cross_validation.cross_val_predict(
        model, tdata["full_sequence"], tdata["ec50"] )

    return tdata

param_space = {
    "weights_C" : numpy.exp(numpy.arange(-9, 0, .5))
}

proteases = ["chymo", "tryp"]
parameter_sets = [frozenset(d.items()) for d in dict_product(param_space)]
model_results = { (d, p) : model_cv_predictions(d, p) for d, p in product(proteases, parameter_sets) }
