# coding: utf-8

from __future__ import division

import logging
logger = logging.getLogger(__name__)

import copy
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
    def __init__(self, homogenous_k=True):
        self.homogenous_k = homogenous_k
        
    @staticmethod
    def lognorm_params(sd, mode):
        return dict(
            tau = sd**-2.,
            mu = numpy.log(mode) + sd ** 2,
        )
    
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
    
    def build_model(self, population_data):
        for k, p in population_data.items():
            unused_keys = set(p.keys()).difference( {"selected", "fraction_selected", "selection_level", "parent"} )
            if unused_keys:
                logger.warning("Unused keys in population_data[%r] : %s", k, unused_keys)
                
        num_members = set( len(p["selected"]) for p in population_data.values() )
        assert len(num_members) == 1, "Different observed population memberships: %s" % num_members
        self.num_members = num_members.pop()
        selected_observations = {
            v["selection_level"] : v["selected"]
            for v in population_data.values() if v["selection_level"] is not None 
        }

        start_ec50 = numpy.full_like(selected_observations.values()[0], min(selected_observations) - 1)
        for sl in selected_observations:
            start_ec50[ ((sl - 1) > start_ec50) & (selected_observations[sl] > 0) ] = sl - 1
        
        self.model = pymc3.Model()
        
        self.to_trans = {}
        
        self.model_populations = {}
        self.population_data = population_data

        with self.model:
            sel_k = pymc3.Lognormal( "sel_k",
                shape=1 if self.homogenous_k else self.num_members,
                tau=self.lognorm_params(sd=.2, mode=1.5)["tau"],
                mu=self.lognorm_params(sd=.2, mode=1.5)["mu"],
                testval=1.5)
            sel_k_transform = pymc3.distributions.transforms.log.forward(sel_k)
            self.to_trans["sel_k"] = ("sel_k_log_"), lambda v: sel_k_transform.eval({sel_k : v})
            
            sel_ec50 = pymc3.Uniform("sel_ec50", lower=-3.0, upper=9.0, shape=self.num_members, testval=start_ec50)
            sel_ec50_transform = pymc3.distributions.transforms.Interval(-3.0, 9).forward(sel_ec50)
            self.to_trans["sel_ec50"] = ("sel_ec50_interval_"), lambda v: sel_ec50_transform.eval({sel_ec50 : v})

        
            pops_by_depth = sorted(
                population_data.keys(),
                key=lambda pkey: self.parent_depth(pkey, population_data)) 
            
            for pkey in pops_by_depth:
                pdat = population_data[pkey]
                if pdat["parent"] is None:
                    continue
                else:
                    start_pop = population_data[pdat["parent"]]["selected"].astype(float)

                start_dist = start_pop / start_pop.sum()
                if pdat["selection_level"] is not None:
                    selection_dist = start_dist / (1 + T.exp(sel_k * (T.constant(pdat["selection_level"]) - sel_ec50)))
                    fraction_selected = selection_dist.sum() / start_dist.sum()
                else:
                    selection_dist = start_dist
                    fraction_selected = 1.0
                
                pop_mask = numpy.flatnonzero(start_pop > 0)  #multinomial formula returns nan if any p == 0, file bug?
                selected = pymc3.distributions.Multinomial(
                    name = "selected_%s" % pkey,
                    n=pdat["selected"][pop_mask].sum(),
                    p=(selection_dist / selection_dist.sum())[pop_mask],
                    observed=pdat["selected"][pop_mask]
                )
                
                if pdat.get("fraction_selected", None) is not None:
                    selected_count = pdat["selected"].sum()
                    source_count = numpy.floor(float(selected_count) / pdat["fraction_selected"])
                    total_selected = pymc3.distributions.Binomial(
                        name = "total_selected_%s" % pkey,
                        n = source_count,
                        p = fraction_selected,
                        observed = selected_count)
                else:
                    total_selected = pdat["selected"].sum()
                    
                self.model_populations[pkey] = {
                    "selection_dist" : self._function(selection_dist),
                    "fraction_selected" : self._function(fraction_selected),
                    "selected" : self._function(selected),
                    "total_selected" : self._function(total_selected),
                }
                
        self.sel_k = self._function(sel_k)
        self.sel_ec50 = self._function(sel_ec50)
        self.logp = self._function(self.model.logpt)
        
        return self
    
    def optimize_params(self, start = None):
        logger.info("optimize_params: %i members", self.num_members)
        if start is not None:
            start = self.to_transformed(start)
            for k in self.model.test_point:
                if k not in start:
                    start[k] = self.model.test_point[k]
        MAP = pymc3.find_MAP(start=start, model=self.model, fmin=scipy.optimize.fmin_l_bfgs_b)
        
        return {
            "sel_k" : self.sel_k(MAP),
            "sel_ec50" : self.sel_ec50(MAP),
        }
    
    def opt_ec50_cred_outliers(self, src_params):
        logger.info("scan_ec50_outliers: %i members", self.num_members)
        params = copy.deepcopy(src_params)

        num_outlier = 0
        for i in range(self.num_members):
            if i % 1000 == 0:
                logger.info("scan_ec50_outliers: %i / %i", i, self.num_members)
                
            cred_summary = self.estimate_ec50_cred(params, i)
            lb = numpy.searchsorted(cred_summary["xs"], cred_summary["sel_ec50"], "left")
            rb = numpy.searchsorted(cred_summary["xs"], cred_summary["sel_ec50"], "right")

            m_pmf = cred_summary["pmf"].argmax()

            if m_pmf < lb or m_pmf > rb:
                num_outlier += 1
                params["sel_ec50"][i] = cred_summary["xs"][m_pmf]

        logger.info(
            "Modified %.3f outliers. (%i/%i)",
             num_outlier / self.num_members, num_outlier, self.num_members)
    
        return params
    
    def find_MAP(self, start = None):
        init = self.optimize_params(start)
        resampled = self.opt_ec50_cred_outliers(init)
        
        return self.optimize_params(resampled)
    
    def ec50_logp_trace(self, base_params, ec50_i, ec50_range, logp_min = numpy.log(1e-5)):
        work_params = { k : v.copy() for k, v in base_params.items() }

        results = numpy.full_like(ec50_range, -numpy.inf)

        # Optimization to improve evaluation times.
        # Model probability monotonically decreases w/ difference from
        # map estimate, so scan outward from the map estimate until
        # lower logp threashold is reached. Fill remaining values w/ -inf.

        map_i = numpy.searchsorted(ec50_range, base_params["sel_ec50"][ec50_i])
        b_logp = self.logp(work_params)

        for i in range(map_i, len(ec50_range)):
            v = ec50_range[i]
            work_params["sel_ec50"][ec50_i] = v
            vlogp = self.logp(work_params)
            if not numpy.isfinite(vlogp):
                vlogp = -numpy.inf
            results[i] = vlogp

            delta_log = vlogp - b_logp
            if delta_log > 0:
                b_logp = vlogp
            elif delta_log < logp_min:
                break

        for i in range(map_i - 1, -1, -1):
            v = ec50_range[i]
            work_params["sel_ec50"][ec50_i] = v
            vlogp = self.logp(work_params)
            if not numpy.isfinite(vlogp):
                vlogp = -numpy.inf
            results[i] = vlogp

            delta_log = vlogp - b_logp
            if delta_log > 0:
                b_logp = vlogp 
            elif delta_log < logp_min:
                break

        return results - b_logp
    
    def estimate_ec50_cred(self, base_params, ec50_i, cred_spans = [.68, .95]):
        """Estimate EC50 credible interval for a single ec50 parameter via model probability."""
        xs = numpy.arange(-3.1, 9, .1)
        logp = numpy.nan_to_num(self.ec50_logp_trace(base_params, ec50_i, xs))
        pmf = numpy.exp(logp) / numpy.sum(numpy.exp(logp))
        cdf = numpy.cumsum(pmf)

        cred_intervals = {}
        for cred_i in cred_spans:
            cdf_b = (1 - cred_i) / 2
            l_b = xs[numpy.searchsorted(cdf, cdf_b, side="left")]
            u_b = xs[numpy.searchsorted(cdf, 1 - cdf_b, side="right")]
            cred_intervals[cred_i] = (l_b, u_b)

        return dict(
            xs = xs,
            pmf = pmf,
            cdf = cdf,
            sel_ec50 = base_params["sel_ec50"][ec50_i],
            cred_intervals = cred_intervals
        )

    @staticmethod
    def plot_cred_summary(ec50_cred, ax=None):
        if ax is None:
            from matplotlib import pylab
            ax = pylab.gca()

        ax.plot( ec50_cred["xs"], ec50_cred["pmf"], label="pmf" )
        ax.plot( ec50_cred["xs"], ec50_cred["cdf"], label="cdf" )
        ax.axvline(ec50_cred["sel_ec50"], alpha=.5, label="sel_e50: %.2f" % ec50_cred["sel_ec50"])

        for ci, (cl, cu) in ec50_cred["cred_intervals"].items():
            ax.axvspan(cl, cu, color="red", alpha=.2, label="%.2f cred" % ci)
            
    def model_selection_summary(self, params):
        def normed_pop(v):
            return v / v.sum()

        return {
            pkey : {
                "selected"  : self.population_data[pkey]["selected"],
                "selected_fraction" : normed_pop(self.population_data[pkey]["selected"].astype(float)),
                "pop_fraction" : normed_pop(self.model_populations[pkey]["selection_dist"](params))
            }
            for pkey in self.model_populations
        }

    def model_outlier_summary(self, params):
        selection_summary = self.model_selection_summary(params)

        for v in selection_summary.values():
            logpmf = scipy.stats.binom.logpmf(
                v["selected"],
                n=v["selected"].sum(),
                p=v["pop_fraction"])
            
            max_logpmf = scipy.stats.binom.logpmf(
                numpy.round(v["selected"].sum() * v["pop_fraction"]),
                n=v["selected"].sum(),
                p=v["pop_fraction"])

            sel_llh = logpmf - max_logpmf
            v["sel_log_likelihood"] = numpy.where(sel_llh != -numpy.inf, sel_llh, numpy.nan)

            sel_error = (v["selected"] / v["selected"].sum()) - v["pop_fraction"]
            v["sel_log_likelihood_signed"] = -v["sel_log_likelihood"] * numpy.sign(sel_error) 
    
    def to_transformed(self, val_dict):
        r = {}
        for n, val in val_dict.items():
            if n in self.to_trans:
                k, f = self.to_trans[n]
                r[k] = f(val)
            else:
                r[n] = val

        return r

    def _function(self, f):
        if isinstance(f, theano.tensor.TensorVariable):
            fn = theano.function(self.model.free_RVs, f, on_unused_input="ignore")

            def call_fn(val_dict):
                val_dict = self.to_transformed(val_dict)
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

        test_map = self.test_model.build_model( self.test_data ).find_MAP()
        
        numpy.testing.assert_allclose( test_map["sel_k"], self.test_vars["sel_k"], rtol=1e-2)
        numpy.testing.assert_allclose( test_map["sel_ec50"], self.test_vars["sel_ec50"], rtol=.3)
