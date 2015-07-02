from zzzUtils import merge_dicts


def multiply_2_DiscreteFinitePMFs(pmf_1, pmf_2):
    conditions = merge_dicts(pmf_1.conditions, pmf_2.conditions)
    scope = merge_dicts(pmf_1.scope, pmf_2.scope)
    for var in (set(conditions) & set(scope)):
        del conditions[var]
    var_symbols = merge_dicts(pmf_1.vars, pmf_2.vars)
    mappings_1 = pmf_1.parameters['mappings'].copy()
    mappings_2 = pmf_2.parameters['mappings'].copy()
    mappings = {}
    for item_1, item_2 in itertools.product(mappings_1.items(), mappings_2.items()):
        var_values_1___frozen_dict, mapping_value_1 = item_1
        var_values_2___frozen_dict, mapping_value_2 = item_2
        same_vars_same_values = True
        for var in (set(var_values_1___frozen_dict) & set(var_values_2___frozen_dict)):
            same_vars_same_values &= (var_values_1___frozen_dict[var] == var_values_2___frozen_dict[var])
        if same_vars_same_values:
            mappings[fdict(set(var_values_1___frozen_dict.items()) | set(var_values_2___frozen_dict.items()))] =\
                mapping_value_1 + mapping_value_2
    return DiscreteFinitePMF(var_symbols, dict(mappings=mappings), conditions, scope)


def multiply_DiscreteFinitePMF_and_continuousPDF(pmf, pdf):
    conditions = merge_dicts(pmf.conditions, pdf.conditions)
    scope = merge_dicts(pmf.scope, pdf.scope)
    for var in (set(conditions) & set(scope)):
        del conditions[var]
    var_symbols = merge_dicts(pmf.vars, pdf.vars)
    mappings = {}
    for var_values___frozen_dict, mapping_value in pmf.parameters['mappings'].items():
        mappings[var_values___frozen_dict] = mapping_value + pdf.density_lambda(pdf.vars)
    return DiscreteFinitePMF(var_symbols, dict(mappings=mappings), conditions, scope)


def multiply_OnePDF_and_GaussianPDF(one_pdf, gaussian_pdf):
    conditions = merge_dicts(gaussian_pdf.conditions, one_pdf.conditions)
    scope = deepcopy(gaussian_pdf.scope, one_pdf.scope)
    for var in (set(conditions) & set(scope)):
        del conditions[var]
    var_symbols = merge_dicts(gaussian_pdf.vars, one_pdf.vars)
    return gaussian_density_function(var_symbols, deepcopy(gaussian_pdf.parameters), conditions, scope)


def multiply_2_GaussianPDFs(gaussian_pdf_1, gaussian_pdf_2):
    conditions = merge_dicts(gaussian_pdf_1.conditions, gaussian_pdf_2.conditions)
    scope = merge_dicts(gaussian_pdf_1.scope, gaussian_pdf_2.scope)
    for var in (set(conditions) & set(scope)):
        del conditions[var]
    var_symbols = merge_dicts(gaussian_pdf_1.vars, gaussian_pdf_2.vars)
    parameters = {}
    return gaussian_density_function(var_symbols, parameters, conditions, scope)