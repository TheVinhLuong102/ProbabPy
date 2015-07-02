from __future__ import division
from copy import deepcopy
from itertools import product
from scipy.stats import multivariate_normal
from sympy import log, pi
from sympy.matrices import BlockMatrix, det
from _BaseClasses import PDF, one_density_function
from zzzUtils import combine_dict_and_kwargs


class GaussianPDF:
    def __init__(var_symbols, params___dict, conditions={}, scope={}):
        return PDF('Gaussian', var_symbols.copy(), deepcopy(params___dict),
                                          gaussian_density, lambda *args, **kwargs: None, gaussian_max,
                                          gaussian_marginalize, gaussian_condition, gaussian_sample,
                                          deepcopy(conditions), deepcopy(scope))


def gaussian_density(var_row_vectors___dict, params___dict):
    var_names = tuple(var_row_vectors___dict)
    num_vars = len(var_names)
    x = []
    m = []
    S = [num_vars * [None] for _ in range(num_vars)]   # careful not to create same mutable object
    d = 0
    for i in range(num_vars):
        x += [var_row_vectors___dict[var_names[i]]]
        d += var_row_vectors___dict[var_names[i]].shape[1]
        m += [params___dict[('mean', var_names[i])]]
        for j in range(i):
            if (var_names[i], var_names[j]) in params___dict.cov:
                S[i][j] = params___dict[('cov', var_names[i], var_names[j])]
                S[j][i] = S[i][j].T
            else:
                S[j][i] = params___dict[('cov', var_names[j], var_names[i])]
                S[i][j] = S[j][i].T
        S[i][i] = params___dict.cov[var_names[i]]
    x = BlockMatrix([x])
    m = BlockMatrix([m])
    S = BlockMatrix(S)
    return (d * log(2 * pi) + log(det(S)) + det((x - m) * S.inverse() * (x - m).T)) / 2


def gaussian_max(gaussian_pdf):
    pdf = gaussian_pdf.copy()
    for var, value in gaussian_pdf.scope.items():
        if value is None:
            pdf.scope[var] = pdf.parameters.mean[var]
    return pdf


def gaussian_marginalize(gaussian_pdf, *marginalized_vars):
    vars_and_symbols = gaussian_pdf.vars.copy()
    var_scope = deepcopy(gaussian_pdf.scope)
    parameters = deepcopy(gaussian_pdf.parameters)
    for marginalized_var in marginalized_vars:
        del vars_and_symbols[marginalized_var]
        del var_scope[marginalized_var]
        p = deepcopy(parameters)
        for key in p:
            if marginalized_var in key:
                del parameters[key]
    if var_scope:
        return GaussianPDF(vars_and_symbols, parameters, deepcopy(gaussian_pdf.conditions), var_scope)
    else:
        return one_density_function(vars_and_symbols, deepcopy(gaussian_pdf.conditions))


def gaussian_condition(gaussian_pdf, conditions={}, **kw_conditions):
    conditions = combine_dict_and_kwargs(conditions, kw_conditions)
    new_conditions = deepcopy(gaussian_pdf.conditions)
    new_conditions.update(conditions)
    scope = deepcopy(gaussian_pdf.scope)
    for var in conditions:
        del scope[var]
    point_conditions = {}
    for var, value in conditions.items():
        if value is not None:
            point_conditions[gaussian_pdf.vars[var]] = value
    condition_var_names = list(conditions)
    num_condition_vars = len(condition_var_names)
    scope_var_names = list(set(gaussian_pdf.scope) - set(conditions))
    num_scope_vars = len(scope_var_names)
    x_c = []
    m_c = []
    m_s = []
    S_c = [num_condition_vars * [None] for _ in range(num_condition_vars)]   # careful not to create same mutable object
    S_s = [num_scope_vars * [None] for _ in range(num_scope_vars)]   # careful not to create same mutable object
    S_cs = [num_scope_vars * [None] for _ in range(num_condition_vars)]   # careful not to create same mutable object
    for i in range(num_condition_vars):
        x_c += [gaussian_pdf.vars[condition_var_names[i]]]
        m_c += [gaussian_pdf.parameters[('mean', condition_var_names[i])]]
        for j in range(i):
            if ('cov', condition_var_names[i], condition_var_names[j]) in gaussian_pdf.parameters:
                S_c[i][j] = gaussian_pdf.parameters[('cov', condition_var_names[i], condition_var_names[j])]
                S_c[j][i] = S_c[i][j].T
            else:
                S_c[j][i] = gaussian_pdf.parameters[('cov', condition_var_names[j], condition_var_names[i])]
                S_c[i][j] = S_c[j][i].T
        S_c[i][i] = gaussian_pdf.parameters[('cov', condition_var_names[i])]
    for i in range(num_scope_vars):
        m_s += [gaussian_pdf.parameters[('mean', scope_var_names[i])]]
        for j in range(i):
            if ('cov', scope_var_names[i], scope_var_names[j]) in gaussian_pdf.parameters:
                S_s[i][j] = gaussian_pdf.parameters[('cov', scope_var_names[i], scope_var_names[j])]
                S_s[j][i] = S_s[i][j].T
            else:
                S_s[j][i] = gaussian_pdf.parameters[('cov', scope_var_names[j], scope_var_names[i])]
                S_s[i][j] = S_s[j][i].T
        S_s[i][i] = gaussian_pdf.parameters[('cov', scope_var_names[i])]
    for i, j in product(range(num_condition_vars), range(num_scope_vars)):
        if ('cov', condition_var_names[i], scope_var_names[j]) in gaussian_pdf.parameters:
            S_cs[i][j] = gaussian_pdf.parameters[('cov', condition_var_names[i], scope_var_names[j])]
        else:
            S_cs[i][j] = gaussian_pdf.parameters[('cov', scope_var_names[j], condition_var_names[i])].T
    x_c = BlockMatrix([x_c])
    m_c = BlockMatrix([m_c])
    m_s = BlockMatrix([m_s])
    S_c = BlockMatrix(S_c)
    S_s = BlockMatrix(S_s)
    S_cs = BlockMatrix(S_cs)
    S_sc = S_cs.T
    m = (m_s + (x_c - m_c) * S_c.inverse() * S_cs).xreplace(point_conditions)
    S = S_s - S_sc * S_c.inverse() * S_cs
    parameters = {}
    index_ranges_from = []
    index_ranges_to = []
    k = 0
    for i in range(num_scope_vars):
        l = k + gaussian_pdf.vars[scope_var_names[i]].shape[1]
        index_ranges_from += [k]
        index_ranges_to += [l]
        parameters[('mean', scope_var_names[i])] = m[0, index_ranges_from[i]:index_ranges_to[i]]
        for j in range(i):
            parameters[('cov', scope_var_names[j], scope_var_names[i])] =\
                S[index_ranges_from[j]:index_ranges_to[j], index_ranges_from[i]:index_ranges_to[i]]
        parameters[('cov', scope_var_names[i])] =\
            S[index_ranges_from[i]:index_ranges_to[i], index_ranges_from[i]:index_ranges_to[i]]
        k = l
    return GaussianPDF(deepcopy(gaussian_pdf.vars), parameters,
                                     new_conditions, scope)


def gaussian_sample(gaussian_pdf, num_samples):
#    scope_vars
#    for scope
#
#    scope_vars = tuple(gaussian_pdf.scope)
#
#    num_scope_vars = len(scope_vars)
#    m = []
#    S = [num_scope_vars * [None] for _ in range(num_scope_vars)]   # careful not to create same mutable object
#    for i in range(num_scope_vars):
#        m += [gaussian_pdf.parameters[('mean', scope_vars[i])]]
#        for j in range(i):
#            if ('cov', scope_vars[i], scope_vars[j]) in gaussian_pdf.parameters:
#                S[i][j] = gaussian_pdf.parameters[('cov', scope_vars[i], scope_vars[j])]
#                S[j][i] = S[i][j].T
#            else:
#                S[j][i] = gaussian_pdf.parameters[('cov', scope_vars[j], scope_vars[i])]
#                S[i][j] = S[j][i].T
#        S[i][i] = gaussian_pdf.parameters[('cov', scope_vars[i])]
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
#            l = k + gaussian_pdf.vars[scope_var].shape[1]
#            fd[scope_var] = samples[i, k:l]
#        mappings[FrozenDict(fd)] = densities[i]
    return 0 #discrete_finite_mass_function(deepcopy(gaussian_pdf.vars), dict(mappings=mappings),
#                                         deepcopy(gaussian_pdf.conditions))
