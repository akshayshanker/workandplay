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

from endog_labour import ConsumerProblem, FirmProblem, Operator_Factory


if __name__ == "__main__":

	# Define the path for settings and results for saving 

	settings_path = 'Settings/pjmas.pickle'
	results_path = "/scratch/kq62/endoglabour/results_09_Aug_2021/pjmas_results.pickle"

	# Load the model parameter values from saved file 

	model = open(settings_path, 'rb')
	model_in = pickle.load(model)
	name = model_in["filename"]
	model.close()

	# Initialise the consumer and firm class, load solver 

	cp = ConsumerProblem(Pi = model_in["Pi"],
						z_vals = model_in["z_vals"],
						gamma_c = model_in["gamma_c"],
						gamma_l = model_in["gamma_l"],
						A_L = model_in["A_L"],
						grid_max = 150,
						grid_size = 300,
						beta = model_in["beta"])

	fp = FirmProblem(delta = model_in["delta"],
						AA= model_in["AA"],
						alpha = model_in["alpha"])

	compute_CEE, firstbest = Operator_Factory(cp, fp)


	# Create dictionary for saving results 

	Results = {}

	Results["Name of Model"] = name

	# Calcualte first best 

	fb_r, fb_K, fb_H, fb_L, fb_Y, fb_w  = firstbest()

	print('Running model {}, with grid_zize {}, max assets {}, T length of {},\
			 prob matrix {}, z_vals {}, gamma_c {}, gamma_l{}'\
			 .format(name, len(cp.asset_grid), np.max(cp.asset_grid), cp.T, cp.Pi,\
			  cp.z_vals, cp.gamma_c, cp.gamma_l))

	print('First best output {}, capital {}, interest rate {},\
			 hours {} and labour supply {}'.format(fb_Y, fb_K, fb_r, fb_H, fb_L))

	results_FB = dict( (name, eval(name)) for name in ['fb_Y', 'fb_K', 'fb_r',\
														 'fb_w', 'fb_H', 'fb_L'])



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
	pickle.dump(Results, open(results_path, "wb" ) )
