from __future__ import division, print_function
from CompyledFunc import CompyledFunc
from copy import deepcopy
from frozendict import frozendict
from HelpyFuncs.SymPy import is_non_atomic_sympy_expr, sympy_xreplace
from HelpyFuncs.Dicts import combine_dict_and_kwargs, merge_dicts_ignoring_dup_keys_and_none_values
from HelpyFuncs.zzz import shift_time_subscripts
from itertools import product
from MathDict import exp as exp_math_dict, MathDict
from MathFunc import MathFunc
from numpy.linalg import inv, slogdet
from pprint import pprint
from scipy.stats import uniform, multivariate_normal
from sympy import exp, log, pi, pprint as sympy_print
from sympy.matrices import BlockMatrix, det, Matrix


DO_NOTHING_FUNC = lambda *args, **kwargs: None
SELF_FUNC = lambda x: x
ZERO_FUNC = lambda *args, **kwargs: .0


class PDF(MathFunc):
    def __init__(self, family='', var_names_and_syms={}, param={}, cond={}, scope={},
                 neg_log_dens_func=DO_NOTHING_FUNC, norm_func=DO_NOTHING_FUNC, max_func=DO_NOTHING_FUNC,
                 marg_func=DO_NOTHING_FUNC, cond_func=DO_NOTHING_FUNC, sample_func=DO_NOTHING_FUNC, compile=False):
        self.Family = family
        self.Param = param
        if hasattr(self, 'Mapping'):
            dens = self.Mapping
        else:
            self.NegLogDensFunc = neg_log_dens_func
            neg_log_dens = neg_log_dens_func(self, var_names_and_syms)
            if self.is_discrete_finite():
                dens = exp_math_dict(-neg_log_dens)
            else:
                dens = exp(-neg_log_dens)
        MathFunc.__init__(self, var_names_and_syms=var_names_and_syms, mapping=dens, param=param,
                          cond=cond, scope=scope, compile=compile)
        self.NormFunc = norm_func
        self.MaxFunc = max_func
        self.MargFunc = marg_func
        self.CondFunc = cond_func
        self.SampleFunc = sample_func

    def is_discrete_finite(self):
        return self.Family == 'DiscreteFinite'

    def is_one(self):
        return self.Family == 'One'

    def is_uniform(self):
        return self.Family == 'Uniform'

    def is_gaussian(self):
        return self.Family == 'Gaussian'

    def at(self, var_and_param_names_and_values={}, **kw_var_and_param_names_and_values):
        var_and_param_names_and_values =\
            combine_dict_and_kwargs(var_and_param_names_and_values, kw_var_and_param_names_and_values)

        cond = deepcopy(self.Cond)   # just to be careful
        scope = deepcopy(self.Scope)   # just to be careful
        param = self.Param.copy()
        syms_and_values = {}
        for var, value in var_and_param_names_and_values.items():
            if var in self.Vars:
                if var in cond:
                    cond[var] = value
                if var in scope:
                    scope[var] = value
                syms_and_values[self.Vars[var]] = value
            if var in param:
                try:
                    syms_and_values[param[var]] = value
                except:
                    pass
                param[var] = value

        cond = sympy_xreplace(cond, syms_and_values)
        scope = sympy_xreplace(scope, syms_and_values)

        self.CompyledFunc = None   # remove compiled version because many things can be changing

        if self.is_discrete_finite():
            neg_log_p = {}
            s = set(var_and_param_names_and_values)
            s_items = set(var_and_param_names_and_values.items())
            for var_values___frozen_dict, mapping_value in param['NegLogP'].items():
                other_items___dict = dict(set(var_values___frozen_dict.items()) - s_items)
                if not (set(other_items___dict) & s):
                    neg_log_p[frozendict(set(var_values___frozen_dict.items()) - set(cond.items()))] =\
                        sympy_xreplace(mapping_value, syms_and_values)
            return DiscreteFinitePMF(var_names_and_syms=self.Vars.copy(), p_or_neg_log_p=neg_log_p, p=False,
                                     cond=cond, scope=scope)
        else:
            pdf = self.copy()
            pdf.Cond = cond
            pdf.Scope = scope
            pdf.Param = sympy_xreplace(param, syms_and_values)
            #print(pdf.Mapping)
            #print(syms_and_values)
            #print(type(syms_and_values.keys().pop()))
            pdf.Mapping = sympy_xreplace(pdf.Mapping, syms_and_values)
            return pdf

    def norm(self):
        return self.NormFunc(self)

    def max(self, **kwargs):
        return self.MaxFunc(self, **kwargs)

    def marg(self, *marginalized_vars):
        return self.MargFunc(self, *marginalized_vars)

    def cond(self, cond={}, **kw_cond):
        cond = combine_dict_and_kwargs(cond, kw_cond)
        return self.CondFunc(self, cond)

    def sample(self, num_samples=1):
        return self.SampleFunc(self, num_samples)

    def __mul__(self, probability_density_function_to_multiply):
        return product_of_2_PDFs(self, probability_density_function_to_multiply)

    def __rmul__(self, probability_density_function_to_multiply):
        return product_of_2_PDFs(probability_density_function_to_multiply, self)

    def multiply(self, *probability_density_functions_to_multiply):
        pdf = self.copy()
        for pdf_to_multiply in probability_density_functions_to_multiply:
            pdf = pdf.__mul__(pdf_to_multiply)
        return pdf

    def pprint(self):
        discrete = self.is_discrete_finite()
        if discrete:
            print('DISCRETE FINITE MASS FUNCTION')
            print('_____________________________')
        else:
            print('CONTINUOUS DENSITY FUNCTION')
            print('___________________________')
            print('FAMILY:', self.Family)
        print("VARIABLES' SYMBOLS:")
        pprint(self.Vars)
        print('CONDITION:')
        pprint(self.Cond)
        print('SCOPE:')
        pprint(self.Scope)
        if not discrete:
            print('PARAMETERS:')
            pprint(self.Param)
            print('DENSITY:')
        else:
            print('MASS:')
        d = self()
        sympy_print(d)
        if discrete:
            print('   sum =', sum(d.values()))

    def shift_time_subscripts(self, t):
        pdf = self.copy()
        pdf.Vars = shift_time_subscripts(pdf.Vars, t)
        pdf.Cond = shift_time_subscripts(pdf.Cond, t)
        pdf.Scope = shift_time_subscripts(pdf.Scope, t)
        pdf.Param = shift_time_subscripts(pdf.Param, t)
        return pdf


def p_from_neg_log_p(expr_or_dict):
    if hasattr(expr_or_dict, 'keys'):
        probs___math_dict = MathDict()
        for k, v in expr_or_dict.items():
            probs___math_dict[k] = exp(-v)
        return probs___math_dict
    else:
        return exp(-expr_or_dict)


def product_of_2_PDFs(pdf0, pdf1):
    families = (pdf0.Family, pdf1.Family)
    if families == ('DiscreteFinite', 'DiscreteFinite'):
        return product_of_2_DiscreteFinitePMFs(pdf0, pdf1)
    elif pdf0.is_discrete_finite():
        return product_of_DiscreteFinitePMF_and_continuousPDF(pdf0, pdf1)
    elif pdf1.is_discrete_finite():
        return product_of_DiscreteFinitePMF_and_continuousPDF(pdf1, pdf0)
    elif families == ('One', 'Gaussian'):
        return product_of_OnePDF_and_GaussPDF(pdf0, pdf1)
    elif families == ('Gaussian', 'One'):
        return product_of_OnePDF_and_GaussPDF(pdf1, pdf0)
    elif families == ('Gaussian', 'Gaussian'):
        return product_of_2_GaussPDFs(pdf0, pdf1)


class DiscreteFinitePMF(PDF):
    def __init__(self, var_names_and_syms={}, p_or_neg_log_p={}, p=True, cond={}, scope={}):
        non_none_scope = {var: value for var, value in scope.items() if value is not None}
        if p:
            f = lambda x: -log(x)
        else:
            f = lambda x: x
        p_or_neg_log_p = MathDict({var_values___frozen_dict: f(func_value)
                                   for var_values___frozen_dict, func_value in p_or_neg_log_p.items()
                                   if set(var_values___frozen_dict.items()) >= set(non_none_scope.items())})
        PDF.__init__(self, family='DiscreteFinite', var_names_and_syms=var_names_and_syms,
                     param=dict(NegLogP=p_or_neg_log_p), cond=cond, scope=non_none_scope,
                     neg_log_dens_func=discrete_finite_neg_log_mass, norm_func=discrete_finite_norm,
                     max_func=discrete_finite_max, marg_func=discrete_finite_marg, cond_func=discrete_finite_cond,
                     sample_func=DO_NOTHING_FUNC)

    def allclose(self, *PMFs, **kwargs):
        for pmf in PMFs:
            if not ((self.Vars == pmf.Vars) and (self.Cond == pmf.Cond) and (self.Scope == pmf.Scope) and
                    self.Param['NegLogP'].allclose(pmf.Param['NegLogP'], **kwargs)):
                return False
        return True


def discrete_finite_neg_log_mass(pmf, var_names_and_values={}):
    v = var_names_and_values.copy()
    for var, value in var_names_and_values.items():
        if (value is None) or is_non_atomic_sympy_expr(value):
            del v[var]
    s0 = set(v.items())
    d = MathDict(())
    for var_names_and_values___frozen_dict, func_value in pmf.Param['NegLogP'].items():
        spare_var_values = dict(s0 - set(var_names_and_values___frozen_dict.items()))
        s = set(spare_var_values.keys())
        if not(s) or (s and not(s & set(var_names_and_values___frozen_dict))):
            d[var_names_and_values___frozen_dict] = sympy_xreplace(func_value, var_names_and_values)
    return d


def discrete_finite_norm(pmf):
    pmf = pmf.copy()
    pmf.Param['NegLogP'] = pmf.Param['NegLogP'].copy()
    neg_log_p = pmf.Param['NegLogP']
    condition_sums = {}
    for var_values___frozen_dict, function_value in neg_log_p.items():
        condition_instance = pmf.CondInstances[var_values___frozen_dict]
        if condition_instance in condition_sums:
            condition_sums[condition_instance] += exp(-function_value)
        else:
            condition_sums[condition_instance] = exp(-function_value)
    for var_values___frozen_dict in neg_log_p:
        pmf.Param['NegLogP'][var_values___frozen_dict] +=\
            log(condition_sums[pmf.CondInstances[var_values___frozen_dict]])
    return pmf


def discrete_finite_max(pmf, leave_unoptimized=None):
    neg_log_p = pmf.Param['NegLogP']
    if leave_unoptimized:
        comparison_bases = {}
        conditioned_and_unoptimized_vars = set(pmf.Cond) | set(leave_unoptimized)
        for var_names_and_values___frozen_dict in neg_log_p:
            comparison_basis = {}
            for var in (set(var_names_and_values___frozen_dict) & conditioned_and_unoptimized_vars):
                comparison_basis[var] = var_names_and_values___frozen_dict[var]
            comparison_bases[var_names_and_values___frozen_dict] = frozendict(comparison_basis)
    else:
        comparison_bases = pmf.CondInstances
    neg_log_mins = {}
    for var_names_and_values___frozen_dict, func_value in neg_log_p.items():
        comparison_basis = comparison_bases[var_names_and_values___frozen_dict]
        if comparison_basis in neg_log_mins:
            neg_log_mins[comparison_basis] = min(neg_log_mins[comparison_basis], func_value)
        else:
            neg_log_mins[comparison_basis] = func_value
    optims = {}
    for var_names_and_values___frozen_dict, func_value in neg_log_p.items():
        if func_value <= neg_log_mins[comparison_bases[var_names_and_values___frozen_dict]]:
            optims[var_names_and_values___frozen_dict] = func_value
    return DiscreteFinitePMF(var_names_and_syms=pmf.Vars.copy(), p_or_neg_log_p=optims, p=False,
                             cond=pmf.Cond.copy(), scope=pmf.Scope.copy())


def discrete_finite_marg(pmf, *marginalized_vars):
    var_symbols = pmf.Vars.copy()
    mappings = pmf.Param['NegLogP'].copy()
    for marginalized_var in marginalized_vars:
        del var_symbols[marginalized_var]
        d = {}
        for var_values___frozen_dict, mapping_value in mappings.items():
            marginalized_var_value = var_values___frozen_dict[marginalized_var]
            fd = frozendict(set(var_values___frozen_dict.items()) - {(marginalized_var, marginalized_var_value)})
            if fd in d:
                d[fd] += exp(-mapping_value)
            else:
                d[fd] = exp(-mapping_value)
        mappings = {k: -log(v) for k, v in d.items()}
    return DiscreteFinitePMF(var_symbols, mappings,
                             cond=deepcopy(pmf.Cond),
                             scope=deepcopy(pmf.Scope), p=False)


def discrete_finite_cond(pmf, cond={}, **kw_cond):
    cond = combine_dict_and_kwargs(cond, kw_cond)
    mappings = pmf.Param['NegLogP'].copy()
    d = {}
    s0 = set(cond.items())
    for var_values___frozen_dict, mapping_value in mappings.items():
        s = set(var_values___frozen_dict.items())
        if s >= s0:
            d[frozendict(s - s0)] = mapping_value
    new_cond = deepcopy(pmf.Cond)
    new_cond.update(cond)
    scope = deepcopy(pmf.Scope)
    for var in cond:
        del scope[var]
    return DiscreteFinitePMF(pmf.Vars.copy(), d, cond=new_cond, scope=scope, p=False)


class OnePMF(DiscreteFinitePMF):
    def __init__(self, var_names_and_syms={}, var_names_and_values=set(), cond={}):
        DiscreteFinitePMF.__init__(self, var_names_and_syms=var_names_and_syms,
                                   p_or_neg_log_p={item: 1. for item in var_names_and_values}, p=True, cond=cond)


class OnePDF(PDF):
    def __init__(self, cond={}):
        PDF.__init__(self, family='One', var_names_and_syms={}, param={}, cond=cond, scope={},
                     neg_log_dens_func=ZERO_FUNC, norm_func=SELF_FUNC, max_func=SELF_FUNC,
                     marg_func=SELF_FUNC, cond_func=DO_NOTHING_FUNC, sample_func=DO_NOTHING_FUNC)


def uniform_density_function(var_symbols, parameters, cond={}, scope={}):
    return PDF('Uniform', deepcopy(var_symbols), deepcopy(parameters),
                                      uniform_density, uniform_normalization, lambda *args, **kwargs: None,
                                      uniform_marginalization, uniform_conditioning, uniform_sampling,
                                      deepcopy(cond), deepcopy(scope))


def uniform_density(var_symbols, parameters):
    d = 1.
    return d


def uniform_normalization():
    return 0


def uniform_marginalization():
    return 0


def uniform_conditioning():
    return 0


def uniform_sampling():
    return 0


class GaussPDF(PDF):
    def __init__(self, var_names_and_syms={}, param={}, cond={}, scope={}, compile=False):
        self.Vars = var_names_and_syms
        self.Param = param

        self.PreProcessed = False
        self.VarList = None
        self.NumVars = None
        self.VarVector = None
        self.NumDims = None
        self.Mean = None
        self.DemeanedVarVector = None
        self.Cov = None
        self.LogDetCov = None
        self.InvCov = None

        if compile:
            self.preprocess()

        PDF.__init__(self, family='Gaussian', var_names_and_syms=var_names_and_syms, param=param,
                     cond=cond, scope=scope, neg_log_dens_func=gauss_neg_log_dens, norm_func=DO_NOTHING_FUNC,
                     max_func=gauss_max, marg_func=gauss_marg, cond_func=gauss_cond, sample_func=gauss_sample,
                     compile=compile)

    def preprocess(self):
        self.VarList = tuple(self.Vars)
        self.NumVars = len(self.VarList)
        self.VarVector = BlockMatrix((tuple(self.Vars[var] for var in self.VarList),))
        self.NumDims = self.VarVector.shape[1]
        self.Mean = BlockMatrix((tuple(self.Param[('Mean', var)] for var in self.VarList),))
        self.DemeanedVarVector = self.VarVector - self.Mean
        cov = [self.NumVars * [None] for _ in range(self.NumVars)]   # careful not to create same mutable object
        for i in range(self.NumVars):
            for j in range(i):
                if ('Cov', self.VarList[i], self.VarList[j]) in self.Param:
                    cov[i][j] = self.Param[('Cov', self.VarList[i], self.VarList[j])]
                    cov[j][i] = cov[i][j].T
                else:
                    cov[j][i] = self.Param[('Cov', self.VarList[j], self.VarList[i])]
                    cov[i][j] = cov[j][i].T
            cov[i][i] = self.Param[('Cov', self.VarList[i])]
        self.Cov = BlockMatrix(cov)
        try:
            cov = CompyledFunc(var_names_and_syms={}, dict_or_expr=self.Cov)()
            sign, self.LogDetCov = slogdet(cov)
            self.LogDetCov *= sign
            self.InvCov = inv(cov)
        except:
            pass
        self.PreProcessed = True


def gauss_neg_log_dens(pdf, var_and_param_names_and_values={}, **kw_var_and_param_names_and_values):
    var_and_param_names_and_values = combine_dict_and_kwargs(var_and_param_names_and_values,
                                                             kw_var_and_param_names_and_values)
    if not pdf.PreProcessed:
        pdf.preprocess()
    if pdf.LogDetCov is None:
        neg_log_dens = (pdf.NumDims * log(2 * pi) + log(det(pdf.Cov)) +
                        det(pdf.DemeanedVarVector * pdf.Cov.inverse() * pdf.DemeanedVarVector.T)) / 2
    else:
        neg_log_dens = (pdf.NumDims * log(2 * pi) + pdf.LogDetCov +
                        det(pdf.DemeanedVarVector * Matrix(pdf.InvCov) * pdf.DemeanedVarVector.T)) / 2
    return sympy_xreplace(neg_log_dens, var_and_param_names_and_values)


def gauss_max(pdf):
    pdf = pdf.copy()
    for var, value in pdf.Scope.items():
        if value is None:
            pdf.Scope[var] = pdf.Param[('Mean', var)]
    return pdf


def gauss_marg(pdf, *marginalized_vars):
    var_names_and_syms = pdf.Vars.copy()
    scope = pdf.Scope.copy()
    param = pdf.Param.copy()
    for marginalized_var in marginalized_vars:
        del var_names_and_syms[marginalized_var]
        del scope[marginalized_var]
        p = param.copy()
        for k in p:
            if marginalized_var in k:
                del param[k]
    if scope:
        return GaussPDF(var_names_and_syms=var_names_and_syms, param=param, cond=pdf.Cond.copy(), scope=scope)
    else:
        return OnePDF(cond=pdf.Cond.copy())


def gauss_cond(pdf, cond={}, **kw_cond):
    cond = combine_dict_and_kwargs(cond, kw_cond)
    new_cond = pdf.Cond.copy()
    new_cond.update(cond)
    scope = pdf.Scope.copy()
    for var in cond:
        del scope[var]
    point_cond = {}
    for var, value in cond.items():
        if value is not None:
            point_cond[pdf.Vars[var]] = value
    cond_vars = tuple(cond)
    num_cond_vars = len(cond_vars)
    scope_vars = tuple(set(pdf.VarsList) - set(cond))
    num_scope_vars = len(scope_vars)
    x_c = BlockMatrix((tuple(pdf.Vars[cond_var] for cond_var in cond_vars),))
    m_c = BlockMatrix((tuple(pdf.Param[('Mean', cond_var)] for cond_var in cond_vars),))
    m_s = BlockMatrix((tuple(pdf.Param[('Mean', scope_var)] for scope_var in scope_vars),))

    S_c = [num_cond_vars * [None] for _ in range(num_cond_vars)]   # careful not to create same mutable object
    for i in range(num_cond_vars):
        for j in range(i):
            if ('Cov', cond_vars[i], cond_vars[j]) in pdf.Param:
                S_c[i][j] = pdf.Param[('Cov', cond_vars[i], cond_vars[j])]
                S_c[j][i] = S_c[i][j].T
            else:
                S_c[j][i] = pdf.Param[('Cov', cond_vars[j], cond_vars[i])]
                S_c[i][j] = S_c[j][i].T
        S_c[i][i] = pdf.Param[('Cov', cond_vars[i])]
    S_c = BlockMatrix(S_c)

    S_s = [num_scope_vars * [None] for _ in range(num_scope_vars)]   # careful not to create same mutable object
    for i in range(num_scope_vars):
        for j in range(i):
            if ('Cov', scope_vars[i], scope_vars[j]) in pdf.Param:
                S_s[i][j] = pdf.Param[('Cov', scope_vars[i], scope_vars[j])]
                S_s[j][i] = S_s[i][j].T
            else:
                S_s[j][i] = pdf.Param[('Cov', scope_vars[j], scope_vars[i])]
                S_s[i][j] = S_s[j][i].T
        S_s[i][i] = pdf.Param[('Cov', scope_vars[i])]
    S_s = BlockMatrix(S_s)

    S_cs = [num_scope_vars * [None] for _ in range(num_cond_vars)]   # careful not to create same mutable object
    for i, j in product(range(num_cond_vars), range(num_scope_vars)):
        if ('Cov', cond_vars[i], scope_vars[j]) in pdf.Param:
            S_cs[i][j] = pdf.Param[('Cov', cond_vars[i], scope_vars[j])]
        else:
            S_cs[i][j] = pdf.Param[('Cov', scope_vars[j], cond_vars[i])].T
    S_cs = BlockMatrix(S_cs)
    S_sc = S_cs.T

    m = (m_s + (x_c - m_c) * S_c.inverse() * S_cs).xreplace(point_cond)
    S = S_s - S_sc * S_c.inverse() * S_cs

    param = {}
    index_ranges_from = []
    index_ranges_to = []
    k = 0
    for i in range(num_scope_vars):
        l = k + pdf.Vars[scope_vars[i]].shape[1]
        index_ranges_from += [k]
        index_ranges_to += [l]
        param[('Mean', scope_vars[i])] = m[0, index_ranges_from[i]:index_ranges_to[i]]
        for j in range(i):
            param[('Cov', scope_vars[j], scope_vars[i])] =\
                S[index_ranges_from[j]:index_ranges_to[j], index_ranges_from[i]:index_ranges_to[i]]
        param[('Cov', scope_vars[i])] =\
            S[index_ranges_from[i]:index_ranges_to[i], index_ranges_from[i]:index_ranges_to[i]]
        k = l

    return GaussPDF(var_names_and_syms=pdf.Vars.copy(), param=param, cond=new_cond, scope=scope)


def gauss_sample(gaussian_pdf, num_samples):
#    scope_vars
#    for scope
#
#    scope_vars = tuple(gaussian_pdf.Scope)
#
#    num_scope_vars = len(scope_vars)
#    m = []
#    S = [num_scope_vars * [None] for _ in range(num_scope_vars)]   # careful not to create same mutable object
#    for i in range(num_scope_vars):
#        m += [gaussian_pdf.Param[('mean', scope_vars[i])]]
#        for j in range(i):
#            if ('cov', scope_vars[i], scope_vars[j]) in gaussian_pdf.Param:
#                S[i][j] = gaussian_pdf.Param[('cov', scope_vars[i], scope_vars[j])]
#                S[j][i] = S[i][j].T
#            else:
#                S[j][i] = gaussian_pdf.Param[('cov', scope_vars[j], scope_vars[i])]
#                S[i][j] = S[j][i].T
#        S[i][i] = gaussian_pdf.Param[('cov', scope_vars[i])]
#    m = BlockMatrix([m]).as_explicit().tolist()[0]
#    S = BlockMatrix(S).as_explicit().tolist()
#    X = multivariate_normal(m, S)
#    samples = X.rvs(num_samples)
#    densities = X.pdf(samples)
#    mappings = {}
#    for i in range(num_samples):
#        fd = {}
#        k = 0
#        for j in range(num_scope_vars):
#            scope_var = scope_vars[j]
#            l = k + gaussian_pdf.Vars[scope_var].shape[1]
#            fd[scope_var] = samples[i, k:l]
#        mappings[FrozenDict(fd)] = densities[i]
    return 0 #discrete_finite_mass_function(deepcopy(gaussian_pdf.Vars), dict(NegLogP=mappings),
#                                         deepcopy(gaussian_pdf.Cond))


def product_of_2_DiscreteFinitePMFs(pmf0, pmf1):
    cond = merge_dicts_ignoring_dup_keys_and_none_values(pmf0.Cond, pmf1.Cond)
    scope = merge_dicts_ignoring_dup_keys_and_none_values(pmf0.Scope, pmf1.Scope)
    for var in (set(cond) & set(scope)):
        del cond[var]
    var_names_and_syms = merge_dicts_ignoring_dup_keys_and_none_values(pmf0.Vars, pmf1.Vars)
    neg_log_p0 = pmf0.Param['NegLogP'].copy()
    neg_log_p1 = pmf1.Param['NegLogP'].copy()
    neg_log_p = {}
    for item_0, item_1 in product(neg_log_p0.items(), neg_log_p1.items()):
        var_names_and_values_0___frozen_dict, func_value_0 = item_0
        var_names_and_values_1___frozen_dict, func_value_1 = item_1
        same_vars_same_values = True
        for var in (set(var_names_and_values_0___frozen_dict) & set(var_names_and_values_1___frozen_dict)):
            if not (var_names_and_values_0___frozen_dict[var] == var_names_and_values_1___frozen_dict[var]):
                same_vars_same_values = False
                break
        if same_vars_same_values:
            neg_log_p[frozendict(set(var_names_and_values_0___frozen_dict.items()) |
                                 set(var_names_and_values_1___frozen_dict.items()))] = func_value_0 + func_value_1
    return DiscreteFinitePMF(var_names_and_syms=var_names_and_syms, p_or_neg_log_p=neg_log_p, p=False,
                             cond=cond, scope=scope)


def product_of_DiscreteFinitePMF_and_continuousPDF(pmf, pdf):
    cond = merge_dicts_ignoring_dup_keys_and_none_values(pmf.Cond, pdf.Cond)
    scope = merge_dicts_ignoring_dup_keys_and_none_values(pmf.Scope, pdf.Scope)
    for var in (set(cond) & set(scope)):
        del cond[var]
    var_names_and_symbols = merge_dicts_ignoring_dup_keys_and_none_values(pmf.Vars, pdf.Vars)
    neg_log_p = {}
    for var_names_and_values___frozen_dict, func_value in pmf.Param['NegLogP'].items():
        neg_log_p[var_names_and_values___frozen_dict] = func_value - log(pdf.Mapping)
    return DiscreteFinitePMF(var_names_and_syms=var_names_and_symbols, p_or_neg_log_p=neg_log_p, p=False,
                             cond=cond, scope=scope)


def product_of_OnePDF_and_GaussPDF(one_pdf, gauss_pdf):
    cond = merge_dicts_ignoring_dup_keys_and_none_values(gauss_pdf.Cond, one_pdf.Cond)
    scope = merge_dicts_ignoring_dup_keys_and_none_values(gauss_pdf.Scope, one_pdf.Scope)
    for var in (set(cond) & set(scope)):
        del cond[var]
    var_names_and_symbols = merge_dicts_ignoring_dup_keys_and_none_values(gauss_pdf.Vars, one_pdf.Vars)
    return GaussPDF(var_names_and_syms=var_names_and_symbols, param=gauss_pdf.Param.copy(), cond=cond, scope=scope)


def product_of_2_GaussPDFs(pdf0, pdf1):
    cond = merge_dicts_ignoring_dup_keys_and_none_values(pdf0.Cond, pdf1.Cond)
    scope = merge_dicts_ignoring_dup_keys_and_none_values(pdf0.Scope, pdf1.Scope)
    for var in (set(cond) & set(scope)):
        del cond[var]
    var_names_and_symbols = merge_dicts_ignoring_dup_keys_and_none_values(pdf0.Vars, pdf1.Vars)
    param = {}
    return GaussPDF(var_names_and_syms=var_names_and_symbols, param=param, cond=cond, scope=scope)
