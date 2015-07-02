from scipy.stats import uniform


def uniform_density_function(var_symbols, parameters, conditions={}, scope={}):
    return PDF('Uniform', deepcopy(var_symbols), deepcopy(parameters),
                                      uniform_density, uniform_normalization, lambda *args, **kwargs: None,
                                      uniform_marginalization, uniform_conditioning, uniform_sampling,
                                      deepcopy(conditions), deepcopy(scope))


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