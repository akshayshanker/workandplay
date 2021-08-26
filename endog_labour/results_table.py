
import numpy as np
from tabulate import tabulate
from gini import gini
import yaml
import dill as pickle
from gini import gini
import csv


# Load the model parameter values from saved file
results_path = "/scratch/kq62/endoglabour/results_09_Aug_2021/"
result_file_name = 'pjmas_high_gammac.pickle'
settings_file = 'Settings/pjmas_high_gammac.yml'

with open(settings_file) as fp:
    model_in = yaml.load(fp)

name = 'pjmas_low_gammac'

# initialise the consumer and firm class, load operators

cp = ConsumerProblem(Pi=model_in["Pi"],
                     z_vals=model_in["z_vals"],
                     gamma_c=model_in["gamma_c"],
                     gamma_l=model_in["gamma_l"],
                     A_L=model_in["A_L"],
                     grid_max=model_in["grid_max"],
                     grid_size=model_in["grid_size"],
                     beta=model_in['beta'])


fp = FirmProblem(delta=model_in["delta"])
mc = MarkovChain(cp.Pi)
P_stat = mc.stationary_distributions[0]


Results = pickle.load(
    open(
        '{}_results_{}'.format(
            results_path,
            result_file_name),
        "rb"))

Results["CP"]["CP_gini_a"] = gini(Results["CP"]["CP_a"])
Results["CP"]["CP_gini_i"] = gini(Results["CP"]["CP_z_rlz"])

Results["IM"]["im_gini_a"] = gini(Results["IM"]["IM_a"])
Results["IM"]["im_gini_i"] = gini(Results["IM"]["IM_z_rlz"])

Results["CF"]["CF_gini_a"] = gini(Results["CF"]["CF_a"])
Results["CF"]["CF_gini_i"] = gini(Results["CF"]["CF_z_rlz"])


results_IM = Results['IM']
results_CP = Results['CP']
results_FB = Results['FB']
results_CF = Results['CF']
results_CM = Results['CM']


results_CF["CF_w"] = results_CP["CP_w"]
results_CF["CF_r"] = results_CP["CP_r"]


for economy in ['IM', 'CP', 'CF']:
    Results[economy][economy +
                     '_policy_h'] = Results[economy][economy +
                                                     '_policy'][0]
    Results[economy][economy +
                     '_policy_l'] = Results[economy][economy +
                                                     '_policy'][1]
    Results[economy][economy +
                     '_policy_v'] = Results[economy][economy +
                                                     '_policy'][2]

# Generate cross-sectional and time-series moments

for economy in ['IM', 'CP', 'CF']:
    Results[economy][economy + '_cval'] = Results[economy][economy + '_a']\
        * (1 + Results[economy]["{}_r".format(economy)])\
        + Results[economy]["{}_w".format(economy)]\
        * Results[economy][economy + '_z_rlz']\
        - Results[economy][economy + '_h_val']
    Results[economy][economy + '_sval'] = Results[economy][economy + '_h_val']
    Results[economy][economy + '_dcval'] = np.diff(
        Results[economy][economy + '_cval'], n=1) / Results[economy][economy + '_cval'][:-1]
    Results[economy][economy + '_dsval'] = np.diff(
        Results[economy][economy + '_sval'], n=1) / Results[economy][economy + '_sval'][:-1]
    Results[economy][economy + '_zval'] = Results[economy][economy + '_z_rlz']\
        / (1 - Results[economy][economy + '_l_val'])
    Results[economy][economy + '_dhval'] = np.diff(1 - Results[economy][economy + '_l_val'])\
        / (1 - Results[economy][economy + '_l_val'][:-1])
    Results[economy][economy + '_huval'] = 1 - \
        Results[economy][economy + '_l_val']
    Results[economy][economy +
                     '_dzval'] = np.diff(Results[economy][economy + '_zval'], n=1)
    Results[economy][economy +
                     '_dzrlz'] = np.diff(Results[economy][economy + '_z_rlz'], n=1)
    Results[economy][economy + '_icoef_disp'] = 1 - np.cov(Results[economy][economy + '_cval'],
                                                           Results[economy][economy + '_zval'])[0][1] / np.var(Results[economy][economy + '_zval'])
    Results[economy][economy + '_icoef'] = 1 - np.cov(Results[economy][economy + '_dcval'],
                                                      Results[economy][economy + '_dzval'])[0][1] / np.var(Results[economy][economy + '_dzval'])
    Results[economy][economy + '_icoef_a'] = 1 - Results[economy][economy + "_H"]\
        * np.cov(Results[economy][economy + '_dcval'],
                 Results[economy][economy + '_dzrlz'])[0][1]\
        / np.var(Results[economy][economy + '_dzrlz'])
    Results[economy][economy + 'work_coef'] = np.cov(Results[economy][economy + '_dhval'],
                                                     Results[economy][economy + '_dzval'])[0][1] / np.var(Results[economy][economy + '_dzval'])
    Results[economy][economy + 's_coef'] = np.cov(Results[economy][economy + '_dsval'],
                                                  Results[economy][economy + '_dzval'])[0][1]\
        / np.var(Results[economy][economy + '_dzval'])
    Results[economy][economy + 'ca_coef'] = 1 - np.cov(Results[economy][economy + '_cval'],
                                                       Results[economy][economy + '_a'])[0][1] / np.var(Results[economy][economy + '_a'])
    Results[economy][economy + '_corr_h_z'] = np.cov(Results[economy][economy + '_zval'],
                                                     Results[economy][economy + '_huval'])[0][1]\
        / (np.std(Results[economy][economy + '_huval'])
           * np.std(Results[economy][economy + '_zval']))
    Results[economy][economy + '_corr_L_a'] = np.cov((1 - Results[economy][economy + '_l_val']),
                                                     Results[economy][economy + '_a'])[0][1]\
        / (np.std((1 - Results[economy][economy + '_l_val']))
           * np.std(Results[economy][economy + '_a']))
    Results[economy][economy + '_corr_z_a'] = np.cov((Results[economy][economy + '_zval']),
                                                     Results[economy][economy + '_a'])[0][1]\
        / (np.std((Results[economy][economy + '_zval']))
           * np.std(Results[economy][economy + '_a']))


# Make aggregate results table

mainres_CP = ["CP", results_CP["CP_r"] * 100, results_CP["CP_w"],
              results_CP["CP_K"], results_CP["CP_Y"][0],
              results_CP["CP_H"] * 100, results_CP["CP_L"] * 100]

mainres_IM = ["CE", results_IM["IM_r"] * 100,
              results_IM["IM_w"], results_IM["IM_K"], results_IM["IM_Y"],
              results_IM["IM_H"] * 100, results_IM["IM_L"] * 100]

mainres_FB = ["RA", results_FB["fb_r"] * 100,
              fp.AA * (1 - fp.alpha)
              * (np.power(results_FB["fb_K"] / results_FB["fb_H"], fp.alpha)),
              results_FB["fb_K"], results_FB["fb_Y"], results_FB["fb_H"] * 100,
              results_FB["fb_H"] * 100]


mainres_CM = ["CM", results_CM["CM_r"] * 100,
              fp.AA * (1 - fp.alpha)
              * (np.power(results_CM["CM_K"] / results_CM["CM_H"], fp.alpha)),
              results_CM["CM_K"], results_CM["CM_Y"], results_CM["CM_H"] * 100,
              results_CM["CM_L"] * 100]

mainres_CF = [
    "CF",
    results_CP["CP_r"] * 100,
    results_CP["CP_w"],
    results_CF["CF_K"],
    results_CF["CF_Y"][0],
    results_CF["CF_H"] * 100,
    results_CF["CF_L"] * 100]


header = ["Interest rate", "Wage", "Capital", "Output", "Hours", "Labour"]

table = [mainres_FB, mainres_CM, mainres_IM, mainres_CP, mainres_CF]

with open("Results/aggregates_{}.csv".format(Results["Name of Model"]), "w") as f:
    writer = csv.writer(f)
    writer.writerows(table)

print(
    tabulate(
        table,
        headers=header,
        tablefmt="latex_booktabs",
        floatfmt=".2f"))

restab = open("{}_tab.tex".format(name), 'w')
restab.write(
    tabulate(
        table,
        headers=header,
        tablefmt="latex_booktabs",
        floatfmt=".2f"))
restab.close()
caption = "Model parameters are $\beta$ = {}, $\alpha$ = {}, $\delta$ = {}, $\gamma_c$ = {}, $\gamma_l$ = {}, $A_L$ = {}, $A$ = {} and size of grid is {}".\
    format(cp.beta, fp.alpha, fp.delta, cp.gamma_c, cp.gamma_l, cp.A_L, fp.AA, cp.grid_size)
restab = open("Results/{}_caption.tex".format(name), 'w')
restab.write(caption)
restab.close()


# Make insurance coefficients table

Results["CP"]["CP_gini_c"] = gini(Results["CP"]["CP_cval"])

Results["IM"]["im_gini_c"] = gini(Results["IM"]["IM_cval"])

Results["CF"]["CF_gini_c"] = gini(Results["CF"]["CF_cval"])


mainres_CP = ["CP", Results['CP']['CP' + '_icoef'],
              Results['CP']['CP' + '_icoef_a'],
              Results['CP']['CP' + 'work_coef']]

mainres_IM = ["IM", Results['IM']['IM' + '_icoef'],
              Results['IM']['IM' + '_icoef_a'],
              Results['IM']['IM' + 'work_coef']]
mainres_CF = ["IM with CP prices",
              Results['CF']['CF' + '_icoef'],
              Results['CF']['CF' + '_icoef_a'],
              Results['CF']['CF' + 'work_coef']]

header = ["Overall I coef.", "Fixed H I coef.", "Hours I coef."]

table = [mainres_IM, mainres_CP, mainres_CF]
with open("aggregates_icoef_{}.csv".format(Results["Name of Model"]), "w") as f:
    writer = csv.writer(f)
    writer.writerows(table)


restab = open("{}_icoef_tab.tex".format(name), 'w')
print(
    tabulate(
        table,
        headers=header,
        tablefmt="latex_booktabs",
        floatfmt=".2f"))


restab.write(
    tabulate(
        table,
        headers=header,
        tablefmt="latex_booktabs",
        floatfmt=".2f"))

restab.close()

# Make gini coefficients table

mainres_CP = ["CP", Results['CP']["CP_gini_i"],
              Results['CP']["CP_gini_a"],
              Results['CP']["CP_gini_c"]]

mainres_IM = ["IM", Results['IM']["im_gini_i"],
              Results['IM']["im_gini_a"],
              Results['IM']["im_gini_c"]]
mainres_CF = ["IM with CP prices",
              Results['CF']["CF_gini_i"],
              Results['CF']["CF_gini_a"],
              Results['CF']["CF_gini_c"]]

header = ["Income gini", "Wealth gini", "Consumption gini"]

table = [mainres_IM, mainres_CP, mainres_CF]
with open("aggregates_icoef_{}.csv".format(Results["Name of Model"]), "w") as f:
    writer = csv.writer(f)
    writer.writerows(table)


restab = open("{}_gini_tab.tex".format(name), 'w')
print(
    tabulate(
        table,
        headers=header,
        tablefmt="latex_booktabs",
        floatfmt=".2f"))


restab.write(
    tabulate(
        table,
        headers=header,
        tablefmt="latex_booktabs",
        floatfmt=".2f"))

restab.close()

# Correlation tables

header = ["Hours and productivity", "Hours and assets",
          "Productivity and assets"]
mainres_CP = ["CP", Results['CP']['CP' + '_corr_h_z'],
              Results['CP']['CP' + '_corr_L_a'],
              Results['CP']['CP' + '_corr_z_a']]
mainres_IM = ["IM", Results['IM']['IM' + '_corr_h_z'],
              Results['IM']['IM' + '_corr_L_a'],
              Results['IM']['IM' + '_corr_z_a']]
mainres_CF = ["CF", Results['CF']['CF' + '_corr_h_z'],
              Results['CF']['CF' + '_corr_L_a'],
              Results['CF']['CF' + '_corr_z_a']]
restab = open("{}_corr_tab.tex".format(name), 'w')
table = [mainres_IM, mainres_CP, mainres_CF]
restab.write(tabulate(table, headers=header,
                      tablefmt="latex_booktabs", floatfmt=".2f"))
restab.close()


L_val_im = 1 - results_IM["IM_l_val"]
a_im = results_IM["IM_a"]
z_im = results_IM['IM_z_rlz'] / (1 - results_IM['IM_l_val'])
a_cp = results_CP["CP_a"]
L_val_cp = 1 - results_CP["CP_l_val"]
z_cp = results_CP['CP_z_rlz'] / (1 - results_CP['CP_l_val'])
