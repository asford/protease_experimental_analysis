"""Sequence-based protease susceptibility modeling."""

from __future__ import division

import logging
import random
import copy
import traitlets

import numpy
import Bio
import Bio.Alphabet.IUPAC as IUPAC

import scipy.optimize
import scipy.special

import theano
import theano.tensor as T
theano.config.mode = "FAST_RUN"

from sklearn.base import BaseEstimator, RegressorMixin

class CenterLimitedPSSMModel(BaseEstimator, RegressorMixin, traitlets.HasTraits):
    
    alpha_center = traitlets.Float(default_value=0, allow_none=False, min=0)
    alpha_flank = traitlets.Float(default_value=0, allow_none=False, min=0)

    error_upper_lim = traitlets.Float(default_value=10, allow_none=False)
    error_lower_lim = traitlets.Float(default_value=-10, allow_none=False)

    max_data_upweight = traitlets.Float(default_value=1.0, allow_none=False, min=1.0)

    init_EC50_max = traitlets.Float(default_value=150.0, allow_none=False)
    init_k_max = traitlets.Float(default_value=80.0, allow_none=False)
    init_c0 = traitlets.Float(default_value=4.0, allow_none=False)

    flanking_window = traitlets.Integer(default_value=4, allow_none=False, min=0)

    init_aas = traitlets.Set(
        trait=traitlets.Enum( IUPAC.IUPACProtein.letters ),
        default_value = None, allow_none = True,
    )

    def _get_param_names(self):
        return self.trait_names()

    @classmethod
    def from_state(cls, params):
        instance = cls.__new__(cls)
        instance.__setstate__(params)
        return instance
        
    def get_state(self):
        return self.__getstate__()
    
    def __setstate__(self, state):
        HasTraits.__setstate__(self, copy.deepcopy(state))

        if state.get("fit_coeffs_", None) is not None:
            self.setup()
            self.fit_coeffs_ = state["fit_coeffs_"]
        
    def __getstate__(self):
        state = HasTrait.__getstate__(self)
        statekeys = (
            "_trait_values", "_fit_coeffs",
            "_trait_validators", "_trait_notifiers", "_cross_validation_lock",
        )

        for k in set(state) - set(statekeys):
            del state[k]

        return state
        
    def setup(self):
        """Validate parameters and setup estimator."""

        self.dat = {
            "seq" : T.lmatrix("seq"),
            "targ" : T.dvector("targ"),
            "data_weights" : T.dvector("data_weights"),
        }

        self.dat["seq"].tag.test_value = numpy.random.randint(21, size=(5, 30))
        self.dat["targ"].tag.test_value = numpy.random.random(5)
        self.dat["data_weights"].tag.test_value = numpy.random.random(5)
        
        self.vs = {
            "outer_PSSM" : T.dmatrix("outer_PSSM"),             
            "P1_PSSM" : T.dvector("P1_PSSM"),
            "c0" : T.scalar("c0"),
            "EC50_max" : T.scalar("EC50_max"),
            "k_max" : T.scalar("k_max"),
        }
        
        self.v_dtypes = {
            "outer_PSSM" : (float, (2 * self.flanking_window + 1, 21)),
            "P1_PSSM" : (float, 21),
            "c0" : (float, ()),
            "EC50_max" : (float, ()),
            "k_max" : (float, ()),
        }

        for v in self.vs:
            self.vs[v].tag.test_value = numpy.random.random( self.v_dtypes[v][1] )
        
        self.coeff_dtype = numpy.dtype([(n,) + t for n, t in self.v_dtypes.items()])
        
        self.targets = {}
        self._build_model()
        
        self.functions = {
            k : theano.function(self.dat.values() + self.vs.values(), outputs = f, name = k, on_unused_input="ignore")
            for k, f in self.targets.items()
        }
        self.predict_fn = theano.function([self.dat["seq"]] + self.vs.values(), outputs=self.targets["score"], name="predict")
        
    
    def encode_seqs(self, seqs):
        if isinstance(seqs, list) or seqs.dtype == 'O':
            assert len(set(map(len, seqs))) == 1, "Sequences of unequal length."
            seqs = numpy.array([numpy.fromstring(s, dtype="S1") for s in seqs])
        
        if seqs.dtype == numpy.dtype("S1"):
            for aa in numpy.unique(seqs):
                assert aa in IUPAC.IUPACProtein.letters + 'Z', "Non-IUPAC/non-Z letter code in input sequences: %r" % aa
                    
            seqs = numpy.searchsorted(
                numpy.fromstring(IUPAC.IUPACProtein.letters + 'Z', dtype="S1"),
                seqs
            )

        assert seqs.dtype == numpy.int, "Invalid sequence dtype: %r" % seqs.dtype
        assert seqs.min() >= 0 and seqs.max() < 21, "Invalid encoded sequence range: (%s, %s)" % (seqs.min(), seqs.max())
        
        return seqs
    
    def _build_model(self):
        
        outer_PSSM = self.vs["outer_PSSM"]
        P1_PSSM = self.vs["P1_PSSM"]


        window_size = self.flanking_window * 2 + 1
        window_pos = range(self.flanking_window) + range(self.flanking_window+1, window_size)

        seq = self.dat["seq"]
            
        point_scores = [
            outer_PSSM[
                i,
                seq[:, i:-(window_size - 1 - i) if i < window_size - 1 else None]
            ]
          for i in window_pos
        ]

        point_scores.append(
            P1_PSSM[
                seq[:, self.flanking_window:-(self.flanking_window)]
            ]

        )

        k_max = self.vs["k_max"]


        pssm_score=sum(point_scores)

        c0 = self.vs["c0"]

        
        ind_score = k_max / (1.0 + T.exp(c0 - pssm_score)) # Eq. 19
        
        EC50_max = self.vs["EC50_max"]
        

        score = T.log(EC50_max / (ind_score.sum(axis=-1) + 1)) / T.log(3) # Eq. 18, converted to a log EC50 instead of a raw EC50
        
        targ = self.dat["targ"]
        data_weights = self.dat["data_weights"]

        error_upper_lim = self.error_upper_lim
        error_lower_lim = self.error_lower_lim

        error=T.clip(targ - score, error_lower_lim, error_upper_lim)

        mse = T.mean(T.square(error) + (0.25 * T.abs_(targ - score))) # Eq. 20
        weighted_mse = T.sum((T.square(error) + (0.25 * T.abs_(targ - score)))* data_weights)  / T.sum(data_weights) #Eq. 20, including weights on the data (not used)
        regularization = (self.alpha_flank * T.square(outer_PSSM[:,0:-1])).sum() + (self.alpha_center * T.abs_(P1_PSSM[0:-1])).sum()

        loss = weighted_mse + regularization
        loss_jacobian = dict(zip(self.vs, T.jacobian(loss, self.vs.values())))
        
        self.targets = {
            "score" : score,
            "pssm_score" : pssm_score,
            "ind_score" : ind_score,

            "mse" : mse,
            "weighted_mse" : weighted_mse,
            "regularization" : regularization,

            "loss" : loss,
            "loss_jacobian" : loss_jacobian,
        }
    
    def opt_cycle(self, target_vars, **opt_options):
        packed_dtype = numpy.dtype(
            [(n,) + self.v_dtypes[n] for n in target_vars])

        def eval_mse(packed_vars):
            vpack = packed_vars.copy().view(packed_dtype).reshape(())
            vs = self.fit_coeffs_.copy()
            for n in vpack.dtype.names:
                vs[n] = vpack[n]
                
            return self.eval_f("mse", coeffs=vs)
        
        def eval_loss(packed_vars):
            vpack = packed_vars.copy().view(packed_dtype).reshape(())
            vs = self.fit_coeffs_.copy()
            for n in vpack.dtype.names:
                vs[n] = vpack[n]
                
            return self.eval_f("loss", coeffs=vs)
        
        def eval_loss_jac(packed_vars):
            vpack = packed_vars.copy().view(packed_dtype).reshape(())
            vs = self.fit_coeffs_.copy()
            for n in vpack.dtype.names:
                vs[n] = vpack[n]
                
            jacobian = self.eval_f("loss_jacobian", coeffs=vs)
            jpack = numpy.zeros((), dtype=packed_dtype)
            for n in jpack.dtype.names:
                jpack[n] = jacobian[n]
            return jpack.reshape(1).view(float)
        
        def iter_callback(packed_vars):
            if ic[0] % 10 == 0:
                logging.debug("iter: %03i mse: %.3f loss: %.3f", ic[0], eval_mse(packed_vars), eval_loss(packed_vars))
            ic[0] += 1
            
        ic = [0]
        start_coeffs = numpy.empty((), packed_dtype)
        for n in start_coeffs.dtype.names:
            start_coeffs[n] = self.fit_coeffs_[n]
            
        opt_result = scipy.optimize.minimize(
            fun=eval_loss,
            jac=eval_loss_jac,
            x0=start_coeffs.reshape(1).view(float),
            callback=iter_callback,
            **opt_options
        )
        
        opt_result.packed_x = opt_result.x.copy().view(packed_dtype).reshape(())
        for n in opt_result.packed_x.dtype.names:
            self.fit_coeffs_[n] = opt_result.packed_x[n]


        logging.info("last_opt iter: %03i fun: %.3f mse: %.3f", ic[0], opt_result.fun, eval_mse(opt_result.packed_x))
        
        return opt_result
    
    def fit(self, X, y):
        from scipy.stats import gaussian_kde
        self.setup()
        
        self.fit_X_ = self.encode_seqs(X)
        self.fit_y_ = y

        data_density=gaussian_kde(y)
        data_y_range=numpy.linspace(min(y), max(y), 1000)
        max_data_density=max(data_density(data_y_range))
        
        self.data_weights = numpy.clip( max_data_density / data_density(y), 1.0, self.max_data_upweight )
        
        self.fit_coeffs_ = numpy.zeros((), self.coeff_dtype)
        
        self.fit_coeffs_["outer_PSSM"][self.flanking_window] = 1
        if self.init_aas:
            self.fit_coeffs_["P1_PSSM"] = [1 if aa in self.init_aas else 0 for aa in (IUPAC.IUPACProtein.letters+'Z')]
        else:
            self.fit_coeffs_["P1_PSSM"] = 0

        self.fit_coeffs_["c0"] = self.init_c0
        self.fit_coeffs_["k_max"] = self.init_k_max
        
        self.fit_coeffs_["EC50_max"] = self.init_EC50_max 

        opt_cycles = [
            ("P1_PSSM",),
            ("c0","k_max", "EC50_max"),
            ("P1_PSSM",),
            ("outer_PSSM", "P1_PSSM"),
            ("c0","k_max","EC50_max"),
            ("outer_PSSM","P1_PSSM"),
            ("c0", "EC50_max", "k_max"),
            ("outer_PSSM", "P1_PSSM", "k_max"),
            ("c0", "EC50_max", "outer_PSSM", "P1_PSSM", "k_max"),

        ]
        
        for i, vset in enumerate(opt_cycles):
            logging.info("opt_cycle: %i vars: %s", i, vset)

            self._last_opt_result = self.opt_cycle(
                vset, tol=1e-3 if i < len(opt_cycles) - 1 else 2e-4 #2e-5
            )
            
    def predict(self, X):
        return self.predict_fn(
            self.encode_seqs(X),
            **{n : self.fit_coeffs_[n] for n in self.fit_coeffs_.dtype.names}
        )
            
    def eval_f(self, fname, X=None, y=None, data_weights=None, coeffs = None):
        if coeffs is None:
            coeffs = self.fit_coeffs_
        if X is None:
            X = self.fit_X_
        else:
            X = self.encode_seqs(X)
        
        if y is None:
            y = self.fit_y_
        if data_weights is None:
            data_weights = self.data_weights
        
        return self.functions[fname](seq = X, targ = y, data_weights = data_weights, **{n : coeffs[n] for n in coeffs.dtype.names})

class TestDataGenerator(object):
    
    center_weights = { "A" : 1, "C" : 1.5 }

    flanking_weights = {}
    flanking_weights.update({("D", i) : .1 for i in range(1, 5)})
    flanking_weights.update({("D", -1) : -3})
    flanking_weights.update({("E", i) : .2 for i in range(1, 3)})

    base_alphabet = numpy.array(list(set(IUPAC.IUPACProtein.letters) - set("ACDE")))
    
    @classmethod
    def score_seq(cls, seq):
        assert isinstance(seq, numpy.ndarray) and seq.dtype == "S1"

        site_scores = []
        for i in range(len(seq)):
            site_score = 1

            if not seq[i] in cls.center_weights:
                continue
            for fdi in range(-1, 5):
                fi = i + fdi
                if fi == i or fi < 0 or fi >= len(seq):
                    continue
                if (seq[fi], fdi) in cls.flanking_weights:
                    site_score += cls.flanking_weights[(seq[fi], fdi)]
            site_scores.append(cls.pssm_to_hit(site_score) * cls.center_weights[seq[i]])

        return cls.hit_scores_to_ec50(sum(site_scores))

    @classmethod
    def hit_scores_to_ec50(cls, s):
        return 6 * (1-scipy.special.expit((s - 2) * 1.2))

    @classmethod
    def pssm_to_hit(cls, s):
        return scipy.special.expit((s * 4.9) - 4)

    @classmethod
    def make_decoy(cls):
        seq_length = 50
        max_sites = 8
        prob_flanking = .6
        max_flanking = 3

        seq = cls.base_alphabet[numpy.random.randint(len(cls.base_alphabet) - 1, size=seq_length)]

        for _ in range(numpy.random.randint(max_sites + 1)):
            sc = random.choice( cls.center_weights.keys() )
            si = numpy.random.randint(seq_length)
            seq[si] = sc

            for _ in range(numpy.random.randint(max_flanking + 1)):
                if random.random() < prob_flanking:
                    fc, fdi = random.choice(cls.flanking_weights.keys())
                    fi = si + fdi
                    if fi >= seq_length or fi < 0:
                        continue

                    seq[fi] = fc

        return {
            "seq" : "".join(seq),
            "score" : cls.score_seq(seq) 
        }
