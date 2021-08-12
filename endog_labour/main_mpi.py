
import numpy as np
from scipy.optimize import minimize, brentq, root, fsolve
import scipy.optimize as optimize
from quantecon import MarkovChain
import quantecon.markov as Markov
import quantecon as qe
from numba import jit, vectorize
from pathos.pools import ProcessPool 
import time
import  dill as pickle
from gini import gini
import pdb 
from sklearn.utils.extmath import cartesian 
from numba import njit, prange
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from interpolation import interp 
from endog_labour import ConsumerProblem, FirmProblem, Operator_Factory

import csv
from numpy import genfromtxt




# Input markov matrix from Pijoan-mas replication material 

Gamma  =	np.array([[0.746464, 0.252884, 0.000652, 0.000000, 0.000000, 0.000000,0.000000],
	     	[0.046088,  0.761085,  0.192512,  0.000314,  0.000000,  0.000000,  0.000000],
	     	[0.000028,  0.069422,  0.788612,  0.141793,  0.000145,  0.000000,  0.000000],
	     	[0.000000,  0.000065,  0.100953,  0.797965,  0.100953,  0.000065,  0.000000],
	     	[0.000000,  0.000000,  0.000145,  0.141793,  0.788612,  0.069422,  0.000028],
	     	[0.000000,  0.000000,  0.000000,  0.000314,  0.192512,  0.761085,  0.046088],
	     	[0.000000,  0.000000,  0.000000,  0.000000,  0.000652,  0.252884,  0.746464]])

Gamma_bar 	=	np.array([0.0154,   0.0847,    0.2349,    0.3300,    0.2349,    0.0847,    0.0154])

e_shocks = np.array([np.exp(-1.385493),np.exp(-0.923662),np.exp(-0.461831),
						np.exp(0.000000),
						np.exp(0.461831),
							np.exp(0.923662),
					np.exp(1.385493)])

e_shocks = e_shocks/np.dot(e_shocks, Gamma_bar)


# Load the model parameter values from saved file 

model = open('Settings/pjmas2.mod', 'rb')
model_in = pickle.load(model)
name = model_in["filename"]
model.close()


# Adjusted model parameters

from mpi4py import MPI as MPI4py
world = MPI4py.COMM_WORLD

index = int(world.Get_rank())
array = genfromtxt('Settings/array.csv', delimiter=',')[1:]      
parameters_array = np.array(array)
parameters_array = cartesian([parameters_array[:,0], parameters_array[:,1]])
parameters = parameters_array[index]

model_in["gamma_c"] = parameters[0]
model_in["gamma_l"] = parameters[1]

name = name + '_array_' + str(index)

# Initialise the consumer and firm class, load operators 

cp = ConsumerProblem(Pi = Gamma,
					z_vals = e_shocks,
					gamma_c= model_in["gamma_c"],
					gamma_l = model_in["gamma_l"],
					A_L = model_in["A_L"],
					grid_max = 65,
					grid_size = 200,
					beta = .945)


fp = FirmProblem(delta = .083)

compute_CEE, firstbest = Operator_Factory(cp, fp)

#mc = MarkovChain(cp.Pi)
#stationary_distributions = mc.stationary_distributions
#mean = np.dot(stationary_distributions, cp.z_vals)
#cp.z_vals = cp.z_vals/ mean   ## Standardise mean of avaible labour supply to 1



# Create dictionary for saving results 

Results = {}

Results["Name of Model"] = "Pijoan_mas" + '_array_' + str(index)

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



# Calcualte incomplete markets (IM), constrained planner (CP) and counter-factual (CF)
results_IM, results_CP, results_CF = compute_CEE()

Results['IM'] = results_IM
Results['CP'] = results_CP
Results['FB'] =	results_FB
Results['CF'] = results_CF

# Calculate Complete Markets

from solve_complete_markets import endog_labour_model as complete_market_model


model = complete_market_model(A=1.5, P =Gamma, P_stat = Gamma_bar, e_shocks= e_shocks)


c_init = 2
b_init= np.ones(len(e_shocks))

init = np.append(b_init,c_init)

sol = fsolve(model.system, init) 

H = np.inner(Gamma_bar,1-model.labour(sol[-1]))      
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

# Save the results file
pickle.dump(Results, open("/scratch/kq62/endoglabour/results_062021/results_PJmas_array_{}.mod".format(str(index)), "wb" ) )
