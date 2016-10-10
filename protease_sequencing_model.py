# coding: utf-8

from __future__ import division

import numpy

import theano.tensor as T
import theano

import pymc3
theano.config.compute_test_value = "ignore"

from scipy.special import expit
import scipy.stats.distributions

import scipy.stats
import scipy.optimize

class FractionalSelectionModel(object):
        
    @staticmethod
    def parent_depth(start_key, pop_specs):
        seen_keys = set()
        cur_depth = 0
        key = start_key

        while pop_specs[key]["parent"] is not None:
            seen_keys.add(key)
            parent = pop_specs[key]["parent"]

            if parent in seen_keys:
                raise ValueError(
                    "Cycle in population parents. start_key: %s cycle_key: %s" %
                    (start_key, pop_specs[key]["parent"]))
            if parent not in pop_specs:
                raise ValueError(
                    "Invalid parent specified: %s pop_specs: %s" %
                    (parent, pop_specs.keys()))
            cur_depth += 1
            key = parent

        return cur_depth

    def generate_data(self, pop_specs, sel_k, sel_ec50, init_pop):
        populations = {}
        
        for pkey in sorted(pop_specs.keys(), key=lambda pkey: self.parent_depth(pkey, pop_specs)):
            p = pop_specs[pkey]
            if p["parent"] is None:
                start_pop = init_pop
            else:
                start_pop = populations[p["parent"]]["selected"]
                
            start_dist = start_pop / start_pop.sum()
            if p["selection_level"] is not None:
                src_dist = start_dist / (1 + numpy.exp(sel_k * (p["selection_level"] - sel_ec50)))
                fraction_selected = src_dist.sum() / start_dist.sum()
            else:
                src_dist = start_dist
                fraction_selected = 1
                
            selected = numpy.random.multinomial(p["selected"], src_dist / src_dist.sum()).astype(float)
            
            populations[pkey] = {}
            populations[pkey].update(p)
            populations[pkey].update({
                "selected" : selected,
                "fraction_selected" : fraction_selected
            })
            
        return populations
    
    def build_model(self, pop_data):
        
        num_members = set( len(p["selected"]) for p in pop_data.values() )
        assert len(num_members) == 1, "Different observed population memberships: %s" % num_members
        num_members = num_members.pop()
        
        self.model = pymc3.Model()
        with self.model:
            populations = {}
            
            sel_k = pymc3.Uniform("sel_k", lower=.1, upper=20, testval=1)
            sel_ec50 = pymc3.Uniform("sel_ec50", lower=-3.0, upper=9.0, shape=num_members, testval=0)
            populations = populations
            sel_k = sel_k
            sel_ec50 = sel_ec50
        
            for pkey in sorted(pop_data.keys(), key=lambda pkey: self.parent_depth(pkey, pop_data)):
                p = pop_data[pkey]
                if p["parent"] is None:
                    continue
                else:
                    start_pop = pop_data[p["parent"]]["selected"].astype(float)

                start_dist = start_pop / start_pop.sum()
                if p["selection_level"] is not None:
                    selection_dist = start_dist / (1 + T.exp(sel_k * (T.constant(p["selection_level"]) - sel_ec50)))
                    fraction_selected = selection_dist.sum() / start_dist.sum()
                else:
                    selection_dist = start_dist
                    fraction_selected = 1.0
                
                pop_mask = numpy.flatnonzero(start_pop > 0)  #multinomial formula returns nan if any p == 0, file bug?
                selected = pymc3.distributions.Multinomial(
                    name = "selected_%s" % pkey,
                    n=p["selected"][pop_mask].sum(),
                    p=(selection_dist / selection_dist.sum())[pop_mask],
                    observed=p["selected"][pop_mask]
                )
                
                if p.get("fraction_selected", None) is not None:
                    selected_count = p["selected"].sum()
                    source_count = numpy.floor(float(selected_count) / p["fraction_selected"])
                    total_selected = pymc3.distributions.Binomial(
                        name = "total_selected_%s" % pkey,
                        n = source_count,
                        p = fraction_selected,
                        observed = selected_count)
                else:
                    total_selected = p["selected"].sum()
                    
                populations[pkey] = {
                    "selection_dist" : self._function(selection_dist),
                    "fraction_selected" : self._function(fraction_selected),
                    "selected" : self._function(selected),
                    "total_selected" : self._function(total_selected),
                }
                
        self.populations = populations
        self.sel_k = self._function(sel_k)
        self.sel_ec50 = self._function(sel_ec50)
        
        return self
    
    def find_MAP(self):
        MAP = pymc3.find_MAP(model=self.model, fmin=scipy.optimize.fmin_l_bfgs_b)
        
        return {
            "sel_k" : self.sel_k(MAP),
            "sel_ec50" : self.sel_ec50(MAP),
        }
    
    def _function(self, f):
        if isinstance(f, theano.tensor.TensorVariable):
            fn = theano.function(self.model.free_RVs, f, on_unused_input="ignore")

            def call_fn(val_dict):
                return fn(*[val_dict[str(n)] for n in self.model.free_RVs])

            return call_fn
        else:
            return lambda _: f

import unittest

class TestFractionalSelectionModel(unittest.TestCase):
    def test_basic_model(self):
        numpy.random.seed(1663)
        
        test_members = 10
        num_sampled = 1e3
        pop_specs= {
            g : dict(parent = g - 1, selection_level = g, selected = num_sampled)
            for g in range(1, 7)
        }
        pop_specs[0] = dict(parent = None, selection_level = None, selected = num_sampled)

        self.test_vars = {
            "sel_k" : 1.5,
            "sel_ec50" : numpy.concatenate((
                numpy.random.normal(loc=.5, scale=.75, size=int(test_members * .9)),
                numpy.random.normal(loc=3, scale=1.5, size=int(test_members * .1)),
            )),
            "init_pop" : numpy.random.lognormal(size=test_members)
        }

        self.test_model = FractionalSelectionModel()
        self.test_data = self.test_model.generate_data(pop_specs, **self.test_vars)

        pops = self.test_model.build_model( self.test_data )

        test_map = self.test_model.find_MAP()
        
        numpy.testing.assert_allclose( test_map["sel_k"], self.test_vars["sel_k"], rtol=1e-2)
        numpy.testing.assert_allclose( test_map["sel_ec50"], self.test_vars["sel_ec50"], rtol=.3)