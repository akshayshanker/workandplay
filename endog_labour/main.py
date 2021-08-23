"""
Script solves incomplete market and constrained planner 
problem for Aiyagari-Huggett model with endogenous 
labour choice (Shanker and Wolfe, 2018)

Model parameters loaded from pickle files
saved in /Settings. 

Script runs on single core python3/3.7.4

Author: Akshay Shanker, University of New South Wales, Sydney
		a.shanker@unsw.edu.au


"""

import numpy as np
import  dill as pickle
from pathlib import Path
from scipy.optimize import brentq, fsolve
from quantecon import MarkovChain
import yaml



from endog_labour import ConsumerProblem, FirmProblem, Operator_Factory
from solve_complete_markets import endog_labour_model as complete_market_model

if __name__ == "__main__":

	# Define the path for settings and results for saving  and model name
	name = 'pjmas'
	results_path = "/scratch/kq62/endoglabour/results_09_Aug_2021/"

	# set paths open model
	settings_file = 'Settings/{}.yml'.format(name)
	result_file_name = '{}.pickle'.format(name)

	# Load the model parameter values from saved file 


	with open(settings_file) as fp:
		model_in = yaml.load(fp)


	# Initialise the consumer and firm class, load solver 

	cp = ConsumerProblem(Pi = model_in["Pi"],
						z_vals =  model_in["z_vals"],
						gamma_c =  model_in["gamma_c"],
						gamma_l = model_in["gamma_l"],
						A_L = model_in["A_L"],
						grid_max = 150,
						grid_size = 600,
						beta = model_in["beta"])

	fp = FirmProblem(delta = model_in["delta"],
						AA =  model_in["AA"],
						alpha =  model_in["alpha"])

	#====Normalize mean of Labour distributuons===#

	mc = MarkovChain(cp.Pi)
	stationary_distributions = mc.stationary_distributions
	mean = np.dot(stationary_distributions, cp.z_vals)
	cp.z_vals = cp.z_vals/ mean   ## Standardise mean of avaible labour supply to 1

	compute_CEE, firstbest = Operator_Factory(cp, fp)


	# Create dictionary for saving results 

	Results = {}

	Results["Name of Model"] = model_in["name"]

	# Calcualte first best 

	fb_r, fb_K, fb_H, fb_L, fb_Y, fb_w  = firstbest()

	print('Running model {}, with grid_zize {}, max assets {}, T length of {},\
			 prob matrix {}, z_vals {}, gamma_c {}, gamma_l{}'\
			 .format(name, len(cp.asset_grid), np.max(cp.asset_grid), cp.T, cp.Pi,\
			  cp.z_vals, cp.gamma_c, cp.gamma_l))

	print('First best output {}, capital {}, interest rate {},\
			 hours {} and labour supply {}'.format(fb_Y, fb_K, fb_r*100, fb_H, fb_L))

	results_FB = dict( (name, eval(name)) for name in ['fb_Y', 'fb_K', 'fb_r',\
														 'fb_w', 'fb_H', 'fb_L'])

	# Calulate complete markets
	mc = MarkovChain(cp.Pi)
	P_stat = mc.stationary_distributions[0]
	model = complete_market_model(A = fp.AA, delta = fp.delta, nu = cp.gamma_l, sigma = cp.gamma_c, beta = cp.beta, P = cp.Pi, P_stat = P_stat, e_shocks = cp.z_vals)

	c_init = 2
	b_init = np.ones(len(cp.z_vals))
	init = np.append(b_init,c_init)

	sol = fsolve(model.system, init) 

	H = np.inner(P_stat,1-model.labour(sol[-1]))      
	L = model.L_supply(sol[-1])
	Y = model.output(sol)
	K = model.K_supply(np.array(sol[0:-1]))  
	r,w,fk = fp.K_to_rw(K,L)

	results_CM = {}
	results_CM['CM_K'] = K
	results_CM['CM_Y'] = Y
	results_CM['CM_L'] = L
	results_CM['CM_H'] = H
	results_CM['CM_r'] = r
	Results['CM'] = results_CM
	print(results_CM)


	# Calcualte incomplete markets (IM), constrained planner (CP) 
	# and counter-factual (CF)
	results_IM, results_CP, results_CF = compute_CEE()

	# Collect results in results dictionary 
	Results['IM'] = results_IM
	Results['CP'] = results_CP
	Results['FB'] =	results_FB
	Results['CF'] = results_CF

	# Save the results file
	Path(results_path).mkdir(parents=True, exist_ok=True)
	pickle.dump(Results, open('{}_results_{}'.format(results_path,result_file_name), "wb" ) )
