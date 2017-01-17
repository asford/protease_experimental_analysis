# coding: utf-8

from __future__ import division

import logging
logger = logging.getLogger(__name__)

import copy

import traitlets

import numpy

import theano.tensor as T
import theano

import pymc3
theano.config.compute_test_value = "ignore"

from scipy.special import expit
import scipy.stats.distributions

import scipy.stats
import scipy.optimize

from pymc3.distributions import Continuous
from pymc3.distributions.continuous import get_tau_sd
from pymc3.distributions.continuous import bound

from .utility import resolve_subclass, SubclassName

default_selk_dict={'mid': dict(mu=1.63, sd=0.00002, w=1.02),
           'low': dict(mu=1.53, sd=0.00002, w=1.15),
           'high': dict(mu=1.75, sd=0.00002, w=0.9),
           'lowest': dict(mu=1.48, sd=0.00002, w=1.20)}

class FlatNormal(Continuous):

    def __init__(self, mu=0.0, w=0.0, tau=None, sd=None, *args, **kwargs):
        super(FlatNormal, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.tau, self.sd = get_tau_sd(tau=tau, sd=sd)
        self.variance = 1. / self.tau
        self.w = w

    def logp(self, value):
        tau = self.tau
        sd = self.sd
        mu = self.mu
        w = self.w
        return bound(
            (-tau * ( T.maximum(
                T.minimum(value - mu+w, 0)**2.0,
                T.maximum(value - mu-w, 0)**2.0
            )) + T.log(tau / numpy.pi / 2.)) / 2.,
            tau > 0,
            sd > 0
        )

def unorm(v):
    return v / v.sum()

class SelectionResponseFn(object):
    @property
    def population_params(self):
        return []

    def selection_mass(self, **kwargs):
        if any(isinstance(v, (T.TensorVariable, T.TensorConstant)) for v in kwargs.values()):
            kwargs = { k : T.as_tensor_variable(v) for k, v in kwargs.items() }
            return self.selection_mass_impl(num=T, **kwargs)
        else:
            return self.selection_mass_impl(num=numpy, **kwargs)

class LogisticResponse(SelectionResponseFn):
    def selection_mass_impl(self, num, sel_level, sel_k, sel_ec50):
        sel_xs = sel_k * (sel_level - sel_ec50)
        return 1 - 1 / (1 + num.exp(-sel_xs))
    
class NormalSpaceErfResponse(SelectionResponseFn):
    @property
    def population_params(self):
        return ["conc_factor"]

    def selection_mass_impl(self, num, sel_level, sel_k, sel_ec50, conc_factor):
        sel_xs = sel_k * (conc_factor ** (sel_level - sel_ec50) - 1.0)
        
        if num == numpy:
            erf = scipy.special.erf
        else:
            erf = T.erf
            
        return (1.0 - erf(sel_xs)) / 2.0

class NormalSpaceLogisticResponse(SelectionResponseFn):
    @property
    def population_params(self):
        return ["conc_factor"]

    def selection_mass_impl(self, num, sel_level, sel_k, sel_ec50, conc_factor):
        sel_xs = sel_k * (conc_factor ** (sel_level - sel_ec50) - 1.0)
        return 1 - 1 / (1 + num.exp(-sel_xs * 2.45))

class FractionalSelectionModel(traitlets.HasTraits):
    response_fn = SubclassName(SelectionResponseFn)

    @property
    def response_impl(self):
       return resolve_subclass(SelectionResponseFn, self.response_fn)()

    #sel_k = traitlets.Dict(
    #    traits = dict(
    #        __class__ = SubclassName(Continuous)
    #    ),
    #    default_value = dict(
    #        __class__="FlatNormal",
    #        mu=1.63, #4.0, #1.63,
    #        sd=0.00002, #0.0001, #0.00002,
    #        w=1.02 #2.5 #1.02,
    #    )
    #)
    #
    #@property
    #def sel_k_class(self):
    #    return resolve_subclass(Continuous, self.sel_k["__class__"])
    #
    #@property
    #def sel_k_kwargs(self):
    #    kwargs = dict(self.sel_k)
    #    kwargs.pop("__class__")
    #    return kwargs

    sel_k_dict = traitlets.Enum(['mid','low','high','lowest'], default_value='mid')

    sel_k = traitlets.Float()#min_selection_rate

    min_selection_rate = traitlets.Union([
        traitlets.Bool(),
        traitlets.Float()
    ])

    min_selection_mass = traitlets.Union([
        traitlets.Enum(["global", "per_selection"]),
        traitlets.Float()
    ])

    homogenous_k = traitlets.Bool(default_value=True)

    alex_opt = traitlets.Bool(default_value=False)

    outlier_detection_opt_cycles = traitlets.Integer(default_value=1)

    def __init__(self, **kwargs):
        # Override 'super' error-handling logic in HasTraits base __init__
        # __init__ swallows errors from unused kwargs until v4.3
        for key in kwargs:
            if not self.has_trait(key):
                raise TypeError("__init__() got an unexpected keyword argument '%s'" % key )
        traitlets.HasTraits.__init__(self, **kwargs)

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
                
            start_dist = unorm(start_pop)
            if p["selection_level"] is not None:
                src_dist = start_dist * self.response_impl.selection_mass(
                    sel_level = p["selection_level"], sel_k = sel_k, sel_ec50 = sel_ec50,
                    **{param : p[param] for param in self.response_impl.population_params}
                )
                fraction_selected = src_dist.sum() / start_dist.sum()
            else:
                src_dist = start_dist
                fraction_selected = 1
                
            selected = numpy.random.multinomial(p["selected"], unorm(src_dist)).astype(float)
            
            populations[pkey] = {}
            populations[pkey].update(p)
            populations[pkey].update({
                "selected" : selected,
                "fraction_selected" : fraction_selected
            })
            
        return populations
    
    def add_fit_param(self, name, dist):
        var = self.model.Var(name, dist, data=None)

        if dist.transform:
            forward_trans = dist.transform.forward(var)
            self.to_trans[name] = (
                "%s_%s_" % (name, dist.transform.name),
                lambda v: forward_trans.eval({var : v})
            ) 


        self.fit_params[name] = var

        return var

    def build_model(self, population_data):
        for k, p in population_data.items():
            unused_keys = set(p.keys()).difference(
                ["selected", "fraction_selected", "selection_level", "parent"] +
                list(self.response_impl.population_params)
            )
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
        self.fit_params = {}
        
        self.model_populations = {}
        self.population_data = population_data

        pops_by_depth = sorted(
            population_data.keys(),
            key=lambda pkey: self.parent_depth(pkey, population_data)) 
        self.modeled_populations = [ p for p in pops_by_depth if population_data[p]["parent"] is not None ]

        with self.model:
            #sel_k = self.add_fit_param(
            #    "sel_k",
            #    self.sel_k_class.dist(**self.sel_k_kwargs))

            #sel_k = self.add_fit_param(
            #            "sel_k",
            #            FlatNormal.dist(**default_selk_dict[self.sel_k_dict]))

            sel_k=self.sel_k

            sel_values = set(
                float(p["selection_level"])
                for p in self.population_data.values()
                if p["selection_level"] is not None
            )
            sel_mag = max(sel_values) - min(sel_values)
            self.sel_range = dict(lower=min(sel_values) - sel_mag * .5, upper=max(sel_values)+sel_mag*.5)
            logger.info("Inferred sel_ec50 range: %s", self.sel_range)
            
            sel_ec50 = self.add_fit_param(
                "sel_ec50",
                pymc3.Uniform.dist(
                    shape=self.num_members,
                    testval=start_ec50,
                    **self.sel_range)
                )

            if self.min_selection_rate:
                if isinstance(self.min_selection_rate, bool):
                    logger.info("Adding adaptive min_selection_rate.")
                    min_selection_rate = self.add_fit_param(
                        "min_selection_rate",
                        pymc3.HalfNormal.dist(sd=.0002, testval=.0001))
                else:
                    logger.info("Adding const min_selection_rate: %.03f" % self.min_selection_rate)
                    min_selection_rate = float(self.min_selection_rate)
            else:
                min_selection_rate = 0.0

            if self.min_selection_mass:
                if self.min_selection_mass == "global":
                    logger.info("Adding global adaptive min_selection_mass.")
                    min_selection_mass = self.add_fit_param(
                        "min_selection_mass",
                        pymc3.HalfNormal.dist(sd=1e-3, testval=1e-12))
                elif self.min_selection_mass == "per_selection":
                    logger.info("Adding per-selection adaptive min_selection_mass.")
                    min_selection_mass = self.add_fit_param(
                        "min_selection_mass",
                        pymc3.HalfNormal.dist(shape=len(self.modeled_populations), sd=1e-3, testval=1e-12))
                else:
                    logger.info("Adding const min_selection_mass: %.03f" % self.min_selection_mass)
                    min_selection_mass = float(self.min_selection_mass)
            else:
                min_selection_mass = 0.0
        
            for pidx, pkey in enumerate(self.modeled_populations):
                pdat = population_data[pkey]
                p_min_selection_mass = (
                    min_selection_mass[pidx]
                        if self.min_selection_mass == "per_selection" else
                    min_selection_mass
                )

                start_pop = population_data[pdat["parent"]]["selected"].astype(float)
                start_dist = unorm(start_pop)

                if pdat["selection_level"] is not None:
                    selection_mass_frac = self.response_impl.selection_mass(
                        sel_level = pdat["selection_level"], sel_k = sel_k, sel_ec50 = sel_ec50,
                        **{param : pdat[param] for param in self.response_impl.population_params}
                    )

                    selection_mass_frac = min_selection_rate + (selection_mass_frac * (1 - min_selection_rate))

                    selection_mass = start_dist * selection_mass_frac

                    fraction_selected = selection_mass.sum() / start_dist.sum()
                else:
                    selection_mass_dist = start_dist
                    fraction_selected = 1.0

                #multinomial formula returns nan if any p == 0, file bug?
                # Add epsilon to selection prob to avoid nan-results when p==0
                selection_dist = unorm(
                    T.clip(selection_mass, (p_min_selection_mass + 1e-9) * fraction_selected, 1))
                pop_mask = numpy.flatnonzero(start_pop > 0)

                selected = pymc3.distributions.Multinomial(
                    name = "selected_%s" % pkey,
                    n=pdat["selected"][pop_mask].sum(),
                    p=selection_dist[pop_mask],
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
                    "selection_mass" : self._function(selection_mass),
                    "selection_dist" : self._function(selection_dist),
                    "fraction_selected" : self._function(fraction_selected),
                    "selected" : self._function(selected),
                    "total_selected" : self._function(total_selected),
                }
                
        self.fit_params = { k : self._function(v) for k, v in self.fit_params.items() }
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
        
        return { k : v(MAP) for k, v in self.fit_params.items() }
    
    def opt_ec50_cred_outliers_alex(self, src_params):
        logger.info("scan_ec50_outliers: %i members", self.num_members)
        params = copy.deepcopy(src_params)

        num_outlier = 0
        for i in range(self.num_members):
            if i % 1000 == 0:
                logger.info("scan_ec50_outliers: %i / %i", i, self.num_members)
                
            cred_summary = self.estimate_ec50_cred(params, i)

            xs = cred_summary["xs"]
            cdf = cred_summary["cdf"]

            cur_i = numpy.searchsorted(xs, params["sel_ec50"][i])
            l_b = numpy.searchsorted(cdf, .25, side="left")
            u_b = numpy.searchsorted(cdf, .75, side="right")

            if cur_i < l_b or cur_i > u_b:
                num_outlier += 1
                #Clip back into self.sel_range
                params["sel_ec50"][i] = numpy.clip(
                    cred_summary["xs"][int((l_b + u_b) / 2)],
                    self.sel_range["lower"] + .1,
                    self.sel_range["upper"] - .1)

    def opt_ec50_cred_outliers(self, src_params):
        logger.info("scan_ec50_outliers: %i members", self.num_members)
        params = copy.deepcopy(src_params)

        num_outlier = 0
        
        for i in range(self.num_members):
            if i % 1000 == 0:
                logger.info("scan_ec50_outliers: %i / %i  outlier count: %s", i, self.num_members, num_outlier)
                
            cred_summary = self.estimate_ec50_cred(params, i)
            current = numpy.searchsorted(cred_summary["xs"], cred_summary["sel_ec50"], "left")
            #rb = numpy.searchsorted(cred_summary["xs"], cred_summary["sel_ec50"], "right")

            m_pmf = cred_summary["pmf"].argmax()

            if m_pmf < current - 1 or m_pmf > current:
                num_outlier += 1
                params["sel_ec50"][i] = cred_summary["xs"][m_pmf]

        logger.info(
            "Modified %.3f outliers. (%i/%i)",
             num_outlier / self.num_members, num_outlier, self.num_members)

        return params
    
    def find_MAP(self, start = None):
        params = self.optimize_params(start)

        for _ in range(self.outlier_detection_opt_cycles):
            if self.alex_opt:
                resampled = self.opt_ec50_cred_outliers_alex(params)
            else:
                resampled = self.opt_ec50_cred_outliers(params)
            params = self.optimize_params(resampled)

        return params 
    
    def ec50_logp_trace(self, base_params, sample_i, ec50_range, include_global_terms=True):
        llh_by_ec50_gen = numpy.zeros((len(ec50_range), len(self.model_populations)))

        if self.min_selection_rate:
            if self.min_selection_rate == True:
                min_selection_rate = base_params["min_selection_rate"]
            else:
                min_selection_rate = self.min_selection_rate
        else:
            min_selection_rate = 0

        if self.min_selection_mass:
            if isinstance(self.min_selection_mass, basestring):
                min_selection_mass = base_params["min_selection_mass"]
            else:
                min_selection_mass = self.min_selection_mass
        else:
            min_selection_mass = 0
        
        for pidx, pkey in enumerate(self.modeled_populations):
            pdat = self.population_data[pkey]

            p_min_selection_mass = (
                min_selection_mass[pidx]
                    if self.min_selection_mass == "per_selection" else
                min_selection_mass
            )
            
            parent_pop_fraction = unorm(self.population_data[pdat['parent']]['selected'])[sample_i]
            
            if parent_pop_fraction == 0:
                continue

            # calculate selection results for full ec50 range
            # base_params['sel_k'] * (pdat['conc_factor'] ** (pdat['selection_level'] - ec50_range) - 1.0 ))
            if "sel_k" in base_params:
                sel_k = base_params["sel_k"]
            else:
                sel_k = self.sel_k

            selected_fraction = self.response_impl.selection_mass(
                sel_level = pdat["selection_level"], sel_k = sel_k, sel_ec50 = ec50_range,
                **{param : pdat[param] for param in self.response_impl.population_params}
            )

            selected_fraction = min_selection_rate + (selected_fraction * (1 - min_selection_rate))

            sel_pop_fraction = parent_pop_fraction * selected_fraction / self.model_populations[pkey]['fraction_selected'](base_params)
            
            sample_llhs = scipy.stats.binom.logpmf(
                pdat['selected'][sample_i],
                n=pdat['selected'].sum(),
                p=numpy.clip(sel_pop_fraction, p_min_selection_mass + 1e-9, 1.0)
            )

            if include_global_terms and pdat.get("fraction_selected") is not None:
                prev_selected_fraction = self.response_impl.selection_mass(
                    sel_level = pdat["selection_level"], sel_k = sel_k, sel_ec50 = base_params['sel_ec50'][sample_i],
                    **{param : pdat[param] for param in self.response_impl.population_params}
                )

                prev_selected_mass = parent_pop_fraction * prev_selected_fraction

                selected_mass = parent_pop_fraction * selected_fraction

                selected_count = pdat["selected"].sum()
                source_count = numpy.floor(float(selected_count) / pdat["fraction_selected"])

                modified_global_selection_fractions = (
                    self.model_populations[pkey]['fraction_selected'](base_params)
                    + selected_mass - prev_selected_mass
                )
            
                sample_llhs += scipy.stats.binom.logpmf(
                    selected_count,
                    n=source_count,
                    p=modified_global_selection_fractions
                )

            
            llh_by_ec50_gen[:,pidx] = sample_llhs

        llh_by_ec50 = llh_by_ec50_gen.sum(axis=1)
                
        return llh_by_ec50 - numpy.nanmax(llh_by_ec50)
    
    def estimate_ec50_cred(self, base_params, ec50_i, cred_spans = [.68, .95]):
        """Estimate EC50 credible interval for a single ec50 parameter via model probability."""
        #xs = numpy.arange(self.sel_range["lower"]+0.1, self.sel_range["upper"]-0.1, .1)
        xs=numpy.linspace(self.sel_range['lower']+1,self.sel_range['upper']-1, (self.sel_range['upper'] - self.sel_range['lower'] - 2)*10 + 1)
        
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
            logp = logp,
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


    def plot_fit_summary(model, i, fit):
        import scipy.stats
        from matplotlib import pylab

        sel_sum = model.model_selection_summary(fit)
        
        sel_levels = {
            k : p["selection_level"] if p["selection_level"] else 0
            for k, p in model.population_data.items()}
        
        sel_fracs = {
            k : p["selected"][i] / p["selected"].sum()
            for k, p in model.population_data.items()}
        
        pylab.xticks(
            sel_levels.values(), sel_levels.keys())
        pylab.xlim((-1, 7))
        
        porder = [
            k for k, p in
            sorted(model.population_data.items(), key=lambda (k, p): p["selection_level"])]
        
        pylab.plot(
            [sel_levels[k] for k in porder],
            [sel_fracs[k] for k in porder],
            "-o",
            color="black", label="observed")
        
        lbl = False
        for k in sel_sum:
            n = sel_sum[k]["selected"].sum()
            p = sel_sum[k]["pop_fraction"][i]
            sel_level = model.population_data[k]["selection_level"]
            counts=sel_sum[k]["selected"][i]
            plt.text(sel_levels[k] + 0.2, sel_fracs[k], '%.0f' % counts)
            
            if p<=0:
                continue
            
            bn = scipy.stats.binom(n=n, p=p)
            
            parkey = model.population_data[k]["parent"]
            pylab.plot(
                [sel_levels[parkey], sel_levels[k]],
                [sel_fracs[parkey], float(bn.ppf(.5)) / n],
                "--", color="red", alpha=.25
            )
            
            
            
            
            for ci in (.68, .95, .99):
                pylab.plot(
                    [sel_level] * 2, bn.ppf([ci, 1-ci]) / n,
                    linewidth=10, color="red", alpha=.25,
                    label="predicted" if not lbl else None
                )
                lbl=True
                
        pylab.legend(fontsize="large", loc="best")
        
        pylab.twinx()
        xs = numpy.linspace(-2, 8)
        sel_ec50 = fit["sel_ec50"][i]
        sel_k = fit["sel_k"][i] if len(fit["sel_k"]) > 1 else fit["sel_k"]
        pylab.plot(xs, scipy.special.expit(-sel_k * (xs - sel_ec50)), alpha=.75)
        pylab.yticks([], [])
        
        pylab.title("%s - ec50: %.2f - k: %.2f" % (i, sel_ec50, sel_k))

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

        return selection_summary
    
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
    def setUp(self):
        numpy.random.seed(1663)
        
        test_members = 10
        num_sampled = 1e4
        self.pop_specs = dict(
            [(0, dict(parent = None, selection_level = None, selected = num_sampled))] +
            [(g, dict(parent = g - 1, selection_level = g, selected = num_sampled))
                for g in range(1, 7)]
        )
        self.test_vars = {
            "sel_k" : 1.5,
            "sel_ec50" : numpy.concatenate((
                numpy.random.normal(loc=.5, scale=.75, size=int(test_members * .9)),
                numpy.random.normal(loc=3, scale=1.5, size=int(test_members * .1)),
            )),
            "init_pop" : numpy.random.lognormal(size=test_members)
        }


    def test_basic_model(self):
        self.test_model = FractionalSelectionModel(homogenous_k=True)
        self.test_data = self.test_model.generate_data(self.pop_specs, **self.test_vars)

        test_map = self.test_model.build_model( self.test_data ).find_MAP()
        
        numpy.testing.assert_allclose( test_map["sel_k"], self.test_vars["sel_k"], rtol=1e-2, atol=.025)
        numpy.testing.assert_allclose( test_map["sel_ec50"], self.test_vars["sel_ec50"], rtol=.3)
