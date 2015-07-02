from copy import copy, deepcopy
from frozendict import frozendict
from numpy import allclose, array
from sympy import Atom, exp, Float, Integer
from sympy.matrices import MatrixSymbol
from sympy.printing.theanocode import theano_function


def combine_dict_and_kwargs(dict_obj, kwargs):
    d = kwargs
    if dict_obj:
        d.update(dict_obj)
    return d


def merge_dicts(dict_0, dict_1):
    d = deepcopy(dict_0)
    for key, value in dict_1.items():
        if (key not in dict_0) or (value is not None):
            d[key] = value
    return d


def sympy_to_float(sympy_number_or_matrix):
    if isinstance(sympy_number_or_matrix, (int, float, Integer, Float)):
        return float(sympy_number_or_matrix)
    else:
        return array(sympy_number_or_matrix.tolist(), dtype=float)


def sympy_allclose(*sympy_matrices, **kwargs):
    if len(sympy_matrices) == 2:
        return allclose(sympy_to_float(sympy_matrices[0]), sympy_to_float(sympy_matrices[1]), **kwargs)
    else:
        for i in range(1, len(sympy_matrices)):
            if not sympy_allclose(sympy_matrices[0], sympy_matrices[i], **kwargs):
                return False
        return True


def dicts_all_close(*dicts, **kwargs):
    if len(dicts) == 2:
        if set(dicts[0]) == set(dicts[1]):
            for key in dicts[0]:
                if not sympy_allclose(dicts[0][key], dicts[1][key], **kwargs):
                    return False
            return True
        else:
            return False
    else:
        for i in range(1, len(dicts)):
            if not dicts_all_close(dicts[0], dicts[i]):
                return False
        return True


def prob_from_neg_log_prob(sympy_expr):
    if isinstance(sympy_expr, dict):
        return {k: exp(-v) for k, v in sympy_expr.items()}
    else:
        return exp(-sympy_expr)


def is_non_atomic_sympy_expr(obj):
    return hasattr(obj, 'doit') and not isinstance(obj, Atom)


def sympy_xreplace(obj, xreplace_dict={}):
    if isinstance(obj, tuple):
        return tuple(sympy_xreplace(item, xreplace_dict) for item in obj)
    elif isinstance(obj, list):
        return [sympy_xreplace(item, xreplace_dict) for item in obj]
    elif isinstance(obj, set):
        return set(sympy_xreplace(item, xreplace_dict) for item in obj)
    elif isinstance(obj, frozenset):
        return frozenset(sympy_xreplace(item, xreplace_dict) for item in obj)
    elif isinstance(obj, dict):
        return {sympy_xreplace(key, xreplace_dict): sympy_xreplace(value, xreplace_dict)
                for key, value in obj.items()}
    elif isinstance(obj, frozendict):
        return frozendict({sympy_xreplace(key, xreplace_dict): sympy_xreplace(value, xreplace_dict)
                           for key, value in obj.items()})
    elif hasattr(obj, 'xreplace'):
        return obj.xreplace(xreplace_dict)
    else:
        return deepcopy(obj)


def sympy_xreplace_doit_explicit(obj, xreplace_dict={}):
    obj = copy(obj)
    if isinstance(obj, dict):
        obj = {key: sympy_xreplace_doit_explicit(value, xreplace_dict) for key, value in obj.items()}
    else:
        # xreplace into all nodes of the expression tree first
        if xreplace_dict:
            obj = obj.xreplace(xreplace_dict)
        # traverse the tree to compute
        if is_non_atomic_sympy_expr(obj):
            args = []
            for arg in obj.args:
                # compute each argument
                args += [sympy_xreplace_doit_explicit(arg)]
            # reconstruct function
            obj = obj.func(*args)
            # try to do it if expression is complete
            try:
                obj = obj.doit()
            except:
                pass
            # try to make it explicit if possible
            try:
                obj = obj.as_explicit()
            except:
                pass
    return obj


def sympy_xreplace_doit_explicit_eval(obj, xreplace_dict={}):
    obj = sympy_xreplace_doit_explicit(obj, xreplace_dict)
    # try evaluating out to get numerical value
    if isinstance(obj, dict):
        for key, value in obj.items():
            try:
                obj[key] = value.evalf()
            except:
                pass
    else:
        try:
            obj = obj.evalf()
        except:
            pass
    return obj


def shift_time_subscripts(obj, t, *matrix_symbols_to_shift):
    if isinstance(obj, frozendict):
        return frozendict({shift_time_subscripts(key, t): shift_time_subscripts(value, t)
                           for key, value in obj.items()})
    elif isinstance(obj, tuple):
        if len(obj) == 2 and not(isinstance(obj[0], (int, float))) and isinstance(obj[1], int):
            return shift_time_subscripts(obj[0], t), obj[1] + t
        else:
            return tuple(shift_time_subscripts(item, t) for item in obj)
    elif isinstance(obj, list):
        return [shift_time_subscripts(item, t) for item in obj]
    elif isinstance(obj, set):
        return {shift_time_subscripts(item, t) for item in obj}
    elif isinstance(obj, dict):
        return {shift_time_subscripts(key, t): shift_time_subscripts(value, t) for key, value in obj.items()}
    elif isinstance(obj, MatrixSymbol):
        args = obj.args
        if isinstance(args[0], tuple):
            return MatrixSymbol(shift_time_subscripts(args[0], t), args[1], args[2])
        else:
            return obj
    elif is_non_atomic_sympy_expr(obj):
        return obj.xreplace({matrix_symbol: shift_time_subscripts(matrix_symbol, t)
                             for matrix_symbol in matrix_symbols_to_shift})
    else:
        return obj
