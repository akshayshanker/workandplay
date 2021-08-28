"""
This script instanizes a parameterised consumer and firm problem based on 

The script then instantizes the equilibrium operators and 
solves complete market, incomplete market and constrained planner
problem for Aiyagari-Huggett model with endogenous
labour choice (Shanker and Wolfe, 2021) from module `endog_labour_mpi`

Model parameters loaded from yml file in /Settings

Solution method based on parrallelised cross-entropy method. Number of cores
will equal number of draws at each iteration of X-entropy. N_elite parameter
sets the number of elite draws 

Suggested execution:

mpiexec -n 1920  python3 -m mpi4py main_mpi.py modelname /myresultspath

Results saved as dictionaries in  /myresultspath


Author: Akshay Shanker, University of New South Wales, Sydney
		a.shanker@unsw.edu.au 

"""

import numpy as np
import dill as pickle
from pathlib import Path
from scipy.optimize import brentq, fsolve
from quantecon import MarkovChain
import yaml
import sys
from mpi4py import MPI as MPI4py


from endog_labour_mpi import ConsumerProblem, FirmProblem, Operator_Factory
from solve_complete_markets import endog_labour_model as complete_market_model

if __name__ == "__main__":
    world = MPI4py.COMM_WORLD

    # Define the path for settings and results for saving  and model name
    name = sys.argv[1]
    results_path = sys.argv[2]
    #name = 'pjmas'
    #results_path = '/scratch/kq62/endoglabour/results_26_Aug_2021/'

    # Read model settings
    settings_file = 'Settings/{}.yml'.format(name)
    result_file_name = '{}.pickle'.format(name)

    # Load the model parameter values from saved file

    with open(settings_file) as fp:
        model_in = yaml.load(fp)

    if world.rank == 0:
        # Initialise the consumer and firm class
        cp = ConsumerProblem(Pi=model_in["Pi"],
                             z_vals=model_in["z_vals"],
                             gamma_c=model_in["gamma_c"],
                             gamma_l=model_in["gamma_l"],
                             A_L=model_in["A_L"],
                             grid_max=model_in["grid_max"],
                             grid_size=model_in["grid_size"],
                             beta=model_in["beta"])

        fp = FirmProblem(delta=model_in["delta"],
                         AA=model_in["AA"],
                         alpha=model_in["alpha"])
    else:
        cp = None
        fp = None

    cp = world.bcast(cp, root =0 )
    fp = world.bcast(fp, root =0 )

    # Normalize mean of Labour distributuons to 1
    mc = MarkovChain(cp.Pi)
    stationary_distributions = mc.stationary_distributions
    mean = np.dot(stationary_distributions, cp.z_vals)
    cp.z_vals = cp.z_vals / mean

    compute_CEE, firstbest = Operator_Factory(cp, fp)

    # Create dictionary for saving results

    Results = {}

    Results["Name of Model"] = model_in["name"]

    # Calcualte first best

    fb_r, fb_K, fb_H, fb_L, fb_Y, fb_w = firstbest()

    if world.rank == 0:
        print(
            'Running model {}, with grid_zize {}, max assets {}, T length of {},\
				 prob matrix {}, z_vals {}, gamma_c {}, gamma_l{}' .format(
                name, len(
                    cp.asset_grid), np.max(
                    cp.asset_grid), cp.T, cp.Pi, cp.z_vals, cp.gamma_c, cp.gamma_l))

        print('First best output {}, capital {}, interest rate {},\
				 hours {} and labour supply {}'.format(fb_Y, fb_K, fb_r * 100, fb_H, fb_L))

    results_FB = dict((name, eval(name)) for name in ['fb_Y', 'fb_K', 'fb_r',
                                                      'fb_w', 'fb_H', 'fb_L'])

    # Calulate complete markets

    mc = MarkovChain(cp.Pi)
    P_stat = mc.stationary_distributions[0]
    model = complete_market_model(A=fp.AA,
                                  delta=fp.delta,
                                  nu=cp.gamma_l,
                                  sigma=cp.gamma_c,
                                  beta=cp.beta,
                                  P=cp.Pi,
                                  P_stat=P_stat,
                                  e_shocks=cp.z_vals)

    c_init = 2
    b_init = np.ones(len(cp.z_vals))
    init = np.append(b_init, c_init)

    sol = fsolve(model.system, init)

    H = np.inner(P_stat, 1 - model.labour(sol[-1]))
    L = model.L_supply(sol[-1])
    Y = model.output(sol)
    K = model.K_supply(np.array(sol[0:-1]))
    r, w, fk = fp.K_to_rw(K, L)

    results_CM = {}
    results_CM['CM_K'] = K
    results_CM['CM_Y'] = Y
    results_CM['CM_L'] = L
    results_CM['CM_H'] = H
    results_CM['CM_r'] = r
    Results['CM'] = results_CM
    if world.rank == 0:
        print(results_CM)

    # Calcualte IM, CP and CF
    results_IM, results_CP, results_CF = compute_CEE(
        world, model_in["N_elite"])

    # Collect results in results dictionary
    Results['IM'] = results_IM
    Results['CP'] = results_CP
    Results['FB'] = results_FB
    Results['CF'] = results_CF

    # Save to results file
    if world.rank == 0:
        Path(results_path).mkdir(parents=True, exist_ok=True)
        pickle.dump(
            Results,
            open(
                '{}results_{}' .format(
                    results_path,
                    result_file_name),
                "wb"))
        print("Saved results")
