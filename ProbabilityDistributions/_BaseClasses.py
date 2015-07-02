from __future__ import print_function
from copy import deepcopy
from frozendict import frozendict
from pprint import pprint
from sympy import exp, log, sympify
from _MultiplyPDFs import multiply_2_DiscreteFinitePMFs, multiply_DiscreteFinitePMF_and_continuousPDF,\
    multiply_OnePDF_and_GaussianPDF, multiply_2_GaussianPDFs
from zzzUtils import combine_dict_and_kwargs, dicts_all_close, prob_from_neg_log_prob,\
    is_non_atomic_sympy_expr, sympy_xreplace, sympy_xreplace_doit_explicit, sympy_xreplace_doit_explicit_eval,\
    shift_time_subscripts


class PDF:
    def __init__(self, family_name, var_names_and_symbols___dict, params___dict, density_func, normalization_func,
                 max_func, marginalization_func, conditioning_func, sampling_func, conditions={}, scope={}):
        self.Family = family_name
        self.Vars = var_names_and_symbols___dict
        self.Conditions = conditions
        self.Scope = dict.fromkeys(set(var_names_and_symbols___dict) - set(conditions))
        for var, value in scope.items():
            if (var in self.Scope) and (value is not None):
                self.Scope[var] = value
        self.Params = params___dict
        self.DensityFunc = density_func
        self.NormalizationFunc = normalization_func
        self.MaxFunc = max_func
        self.MarginalizationFunc = marginalization_func
        self.ConditioningFunc = conditioning_func
        self.SamplingFunc = sampling_func

    def copy(self, deep=True):
        if deep:
            return PDF(self.Family, deepcopy(self.Vars), deepcopy(self.Params),
                       self.DensityFunc, self.NormalizationFunc, self.MaxFunc,
                       self.MarginalizationFunc, self.ConditioningFunc, self.SamplingFunc,
                       deepcopy(self.Conditions), deepcopy(self.Scope))
        else:
            return PDF(self.Family, self.Vars.copy(), self.Params.copy(),
                       self.DensityFunc, self.NormalizationFunc, self.MaxFunc,
                       self.MarginalizationFunc, self.ConditioningFunc, self.SamplingFunc,
                       self.Conditions.copy(), self.Scope.copy())

    def is_one(self):
        return self.Family == 'One'

    def is_discrete_finite(self):
        return self.Family == 'DiscreteFinite'

    def is_uniform(self):
        return self.Family == 'Uniform'

    def is_gaussian(self):
        return self.Family == 'Gaussian'

    def at(self, var_and_param_values___dict={}, **kw_var_and_param_values___dict):
        var_and_param_values___dict = combine_dict_and_kwargs(var_and_param_values___dict,
                                                              kw_var_and_param_values___dict)
        for var in (set(self.Vars) & set(var_and_param_values___dict)):
            var_and_param_values___dict[self.Vars[var]] = var_and_param_values___dict[var]
        pdf = self.copy()
        for var, value in var_and_param_values___dict.items():
            if var in pdf.Conditions:
                pdf.Conditions.update({var: value})
            if var in pdf.Scope:
                pdf.Scope.update({var: value})
        pdf.Conditions = sympy_xreplace(pdf.Conditions, var_and_param_values___dict)
        pdf.Scope = sympy_xreplace(pdf.Scope, var_and_param_values___dict)
        if pdf.is_discrete_finite():
            neg_log_probs = {}
            for vars_and_values___frozen_dict, neg_log_prob in pdf.Params['NegLogProb'].items():
                other_items___dict = dict(set(vars_and_values___frozen_dict.items()) -
                                          set(var_and_param_values___dict.items()))
                if not (set(other_items___dict) & set(var_and_param_values___dict)):
                    neg_log_probs[frozendict(set(vars_and_values___frozen_dict.items()) -
                                             set(pdf.Conditions.items()))] =\
                        sympy_xreplace(neg_log_prob, var_and_param_values___dict)
            return DiscreteFinitePMF(pdf.Vars, neg_log_probs, conditions=pdf.Conditions, scope=pdf.Scope, prob=False)
        else:
            pdf.Params = sympy_xreplace(pdf.Params, var_and_param_values___dict)
            return pdf

    def __call__(self, var_and_param_values___dict={}, prob=True):
        scope_vars = deepcopy(self.Vars)
        for var in self.Vars:
            if var not in self.Scope:
                del scope_vars[var]
            elif self.Scope[var] is not None:
                scope_vars[var] = self.Scope[var]
        if var_and_param_values___dict:
            symbols_and_values___dict = {}
            for var_or_param, value in var_and_param_values___dict.items():
                if var_or_param in self.Vars:
                    symbols_and_values___dict[self.Vars[var_or_param]] = value
                elif var_or_param in self.Params:
                    symbols_and_values___dict[self.Params[var_or_param]] = value
            if prob:
                return sympy_xreplace_doit_explicit_eval(
                    prob_from_neg_log_prob(self.DensityFunc(scope_vars, self.Params)), symbols_and_values___dict)
            else:
                return sympy_xreplace_doit_explicit_eval(self.DensityFunc(scope_vars, self.Params),
                                                         symbols_and_values___dict)
        elif prob:
            return prob_from_neg_log_prob(self.DensityFunc(scope_vars, self.Params))
        else:
            return self.DensityFunc(scope_vars, self.Params)

    def normalize(self):
        return self.NormalizationFunc(self)

    def max(self, **kwargs):
        return self.MaxFunc(self, **kwargs)

    def marginalize(self, *marginalized_vars):
        return self.MarginalizationFunc(self, *marginalized_vars)

    def condition(self, conditions={}, **kw_conditions):
        conditions = combine_dict_and_kwargs(conditions, kw_conditions)
        return self.ConditioningFunc(self, conditions)

    def sample(self, num_samples=1):
        return self.SamplingFunc(self, num_samples)

    def __mul__(self, pdf):
        return product_of_2_PDFs(self, pdf)

    def __rmul__(self, pdf):
        return product_of_2_PDFs(pdf, self)

    def multiply(self, *PDFs):
        pdf = self.copy()
        for pdf_to_multiply in PDFs:
            pdf *= pdf_to_multiply
        return pdf

    def pprint(self):
        discrete_finite = self.is_discrete_finite()
        print('\n')
        if discrete_finite:
            print('MASS FUNCTION')
            print('_____________')
        else:
            print('DENSITY FUNCTION')
            print('________________')
            print('FAMILY:', self.Family)
        print("VARIABLES' SYMBOLS:")
        pprint(self.Vars)
        print('CONDITIONS:')
        pprint(self.Conditions)
        print('SCOPE:')
        pprint(self.Scope)
        if not discrete_finite:
            print('PARAMETERS:')
            pprint(self.Params)
            print('DENSITY:')
        else:
            print('MASS:')
        d = self()
        pprint(d)
        if discrete_finite:
            print('   sum =', sum(d.values()))
        print('\n')

    def shift_time_subscripts(self, t):
        pdf = self.copy()
        pdf.Vars = shift_time_subscripts(pdf.Vars, t)
        pdf.Conditions = shift_time_subscripts(pdf.Conditions, t)
        pdf.Scope = shift_time_subscripts(pdf.Scope, t)
        pdf.Params = shift_time_subscripts(pdf.Params, t)
        return pdf


class DiscreteFinitePMF(PDF):
    def __init__(var_names_and_symbols___dict, probs_or_neg_log_probs___dict, conditions={}, scope={}, prob=True):
        if prob:
            neg_log_probs___dict = {k: log(v) for k, v in probs_or_neg_log_probs___dict.items()}
        else:
            neg_log_probs___dict = probs_or_neg_log_probs___dict
        non_none_scope = {var: value for var, value in scope.items() if value is not None}
        mappings = {var_values___frozen_dict: mapping_value
                    for var_values___frozen_dict, mapping_value in neg_log_probs___dict['mappings'].items()
                    if set(var_values___frozen_dict.items()) >= set(non_none_scope.items())}
        condition_instances = {}
        for var_values___frozen_dict in mappings:
            condition_instance = {}
            for var in (set(var_values___frozen_dict) & set(conditions)):
                condition_instance[var] = var_values___frozen_dict[var]
            condition_instances[var_values___frozen_dict] = frozendict(condition_instance)
        PDF.__init__('DiscreteFinite',
                     var_names_and_symbols___dict.copy(),
                                          dict(NegLogProbs=mappings, ConditionInstances=condition_instances),
                                          discrete_finite_mass, discrete_finite_normalization, discrete_finite_max,
                                          discrete_finite_marginalization, discrete_finite_conditioning,
                                          lambda *args, **kwargs: None, deepcopy(conditions), deepcopy(non_none_scope))


def discrete_finite_mass(var_values___dict, parameters):
    v = deepcopy(var_values___dict)
    for var, value in var_values___dict.items():
        if (value is None) or is_non_atomic_sympy_expr(value):
            del v[var]
    s0 = set(v.items())
    d = {}
    mappings = parameters['mappings']
    for var_values___frozen_dict, mapping_value in mappings.items():
        spare_var_values = dict(s0 - set(var_values___frozen_dict.items()))
        s = set(spare_var_values.keys())
        if not(s) or (s and not(s & set(var_values___frozen_dict))):
            d[var_values___frozen_dict] = sympy_xreplace_doit_explicit(mapping_value, var_values___dict)
    return d


def discrete_finite_normalization(discrete_finite_pmf):
    pmf = discrete_finite_pmf.copy()
    pmf.parameters['mappings'] = pmf.parameters['mappings'].copy()
    mappings = pmf.parameters['mappings']
    condition_instances = pmf.parameters['condition_instances']
    condition_sums = {}
    for var_values___frozen_dict, function_value in mappings.items():
        condition_instance = condition_instances[var_values___frozen_dict]
        if condition_instance in condition_sums:
            condition_sums[condition_instance] += exp(-function_value)
        else:
            condition_sums[condition_instance] = exp(-function_value)
    for var_values___frozen_dict in mappings:
        pmf.parameters['mappings'][var_values___frozen_dict] +=\
            log(condition_sums[condition_instances[var_values___frozen_dict]])
    return pmf


def discrete_finite_max(discrete_finite_pmf, leave_unoptimized=None):
    mappings = discrete_finite_pmf.parameters['mappings']
    if leave_unoptimized:
        comparison_bases = {}
        conditioned_and_unoptimized_vars = set(discrete_finite_pmf.conditions) | set(leave_unoptimized)
        for var_values___frozen_dict in mappings:
            comparison_basis = {}
            for var in (set(var_values___frozen_dict) & conditioned_and_unoptimized_vars):
                comparison_basis[var] = var_values___frozen_dict[var]
            comparison_bases[var_values___frozen_dict] = frozendict(comparison_basis)
    else:
        comparison_bases = discrete_finite_pmf.parameters['condition_instances']
    minus_log_mins = {}
    for var_values___frozen_dict, mapping_value in mappings.items():
        comparison_basis = comparison_bases[var_values___frozen_dict]
        if comparison_basis in minus_log_mins:
            minus_log_mins[comparison_basis] = min(minus_log_mins[comparison_basis], mapping_value)
        else:
            minus_log_mins[comparison_basis] = mapping_value
    max_mappings = {}
    for var_values___frozen_dict, mapping_value in mappings.items():
        if mapping_value <= minus_log_mins[comparison_bases[var_values___frozen_dict]]:
            max_mappings[var_values___frozen_dict] = mapping_value
    return DiscreteFinitePMF(discrete_finite_pmf.vars.copy(), dict(mappings=max_mappings),
                                         deepcopy(discrete_finite_pmf.conditions), deepcopy(discrete_finite_pmf.scope))


def discrete_finite_marginalization(discrete_finite_pmf, *marginalized_vars):
    var_symbols = discrete_finite_pmf.vars.copy()
    mappings = discrete_finite_pmf.parameters['mappings'].copy()
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
    return DiscreteFinitePMF(var_symbols, dict(mappings=mappings),
                                         deepcopy(discrete_finite_pmf.conditions), deepcopy(discrete_finite_pmf.scope))


def discrete_finite_conditioning(discrete_finite_pmf, conditions={}, **kw_conditions):
    conditions = combine_dict_and_kwargs(conditions, kw_conditions)
    mappings = discrete_finite_pmf.parameters['mappings'].copy()
    d = {}
    s0 = set(conditions.items())
    for var_values___frozen_dict, mapping_value in mappings.items():
        s = set(var_values___frozen_dict.items())
        if s >= s0:
            d[frozendict(s - s0)] = mapping_value
    new_conditions = deepcopy(discrete_finite_pmf.conditions)
    new_conditions.update(conditions)
    scope = deepcopy(discrete_finite_pmf.scope)
    for var in conditions:
        del scope[var]
    return DiscreteFinitePMF(discrete_finite_pmf.vars.copy(), dict(mappings=d), new_conditions, scope)


def discrete_finite_mass_functions_all_close(*pmfs, **kwargs):
    if len(pmfs) == 2:
        pmf_0, pmf_1 = pmfs
        return (set(pmf_0.vars.items()) == set(pmf_1.vars.items())) &\
            (set(pmf_0.conditions.items()) == set(pmf_1.conditions.items())) &\
            (set(pmf_0.scope.items()) == set(pmf_1.scope.items())) &\
            dicts_all_close(pmf_0.parameters['mappings'], pmf_1.parameters['mappings'], **kwargs)
    else:
        for i in range(1, len(pmfs)):
            if not discrete_finite_mass_functions_all_close(pmfs[0], pmfs[i], **kwargs):
                return False
        return True


def one_density_function(var_symbols={}, conditions={}):
    return PDF('One', var_symbols.copy(), {}, one, one, one, one, one,
                                      lambda *args, **kwargs: None, deepcopy(conditions))


def one(*args, **kwargs):
    return sympify(0.)







def one_mass_function(var_symbols, frozen_dicts___set=set(), conditions={}):
    mappings = {item: sympify(0.) for item in frozen_dicts___set}
    return DiscreteFinitePMF(var_symbols, dict(mappings=mappings), conditions, scope={})


def product_of_2_PDFs(pdf_1, pdf_2):
    families = (pdf_1.family, pdf_2.family)
    if families == ('DiscreteFinite', 'DiscreteFinite'):
        return multiply_2_DiscreteFinitePMFs(pdf_1, pdf_2)
    elif pdf_1.is_discrete_finite():
        return multiply_DiscreteFinitePMF_and_continuousPDF(
            pdf_1, pdf_2)
    elif pdf_2.is_discrete_finite():
        return multiply_DiscreteFinitePMF_and_continuousPDF(
            pdf_2, pdf_1)
    elif families == ('One', 'Gaussian'):
        return multiply_OnePDF_and_GaussianPDF(
            pdf_1, pdf_2)
    elif families == ('Gaussian', 'One'):
        return multiply_OnePDF_and_GaussianPDF(
            pdf_2, pdf_1)
    elif families == ('Gaussian', 'Gaussian'):
        return multiply_2_GaussianPDFs(pdf_1, pdf_2)