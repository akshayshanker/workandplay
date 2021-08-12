"""
Module contains operators to solve
incomplete market and constrained planner 
problem for Aiyagari-Huggett model with endogenous 
labour choice (Shanker and Wolfe, 2018)

Operators run via main.py 

Author: Akshay Shanker, University of New South Wales, Sydney
		a.shanker@unsw.edu.au

"""
import numpy as np
from scipy.optimize import minimize, root, fsolve
import scipy.optimize as optimize
from quantecon.optimize import brentq
from quantecon import MarkovChain
import quantecon.markov as Markov

from mc_sample_path import mc_sample_path
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
 

class ConsumerProblem:

	"""
	A class that stores primitives for the income fluctuation problem.  The
	income process is assumed to be a finite state Markov chain.

	Parameters
	----------
	r : scalar(float), optional(default=0.01)
		A strictly positive scalar giving the interest rate
	Lambda: scalar(float), optional(default = 0.1)
		The shadow social value of accumulation 
	beta : scalar(float), optional(default=0.96)
		The discount factor, must satisfy (1 + r) * beta < 1
	Pi : array_like(float), optional(default=((0.60, 0.40),(0.05, 0.95))
		A 2D NumPy array giving the Markov matrix for {z_t}
	z_vals : array_like(float), optional(default=(0.5, 0.95))
		The state space of {z_t}
	b : scalar(float), optional(default=0)
		The borrowing constraint
	grid_max : scalar(float), optional(default=16)
		Max of the grid used to solve the problem
	grid_size : scalar(int), optional(default=50)
		Number of grid points to solve problem, a grid on [-b, grid_max]
	u : callable, optional(default=np.log)
		The utility function
	du : callable, optional(default=lambda x: 1/x)
		The derivative of u

	Attributes
	----------
	r, beta, Pi, z_vals, b, u, du : see Parameters
	asset_grid : np.ndarray
		One dimensional grid for assets

	"""

	def __init__(self, 
				 r=0.074, 
				 w =.4,
				 Lambda_H = 0,
				 Lambda_E = 0,
				 beta=.945, 
				 Pi=((0.09, 0.91), (0.06, 0.94)),
				 z_vals=(0.1, 1.0), 
				 b= 1e-2, 
				 grid_max= 50, 
				 grid_size= 100,
				 gamma_c = 1.458,
				 gamma_l = 2.833,
				 A_L  = .856, 
				 T = int(1E7)):
		
		# Declare parameters
		self.r, self.R = r, 1 + r
		self.w = w
		self.Lambda_H = Lambda_H
		self.Lambda_E= Lambda_E
		self.beta, self.b = beta, b
		self.Pi, self.z_vals = np.array(Pi), np.asarray(z_vals)
		self.asset_grid = np.linspace(b, grid_max, grid_size)
		self.k = self.asset_grid[1]-self.asset_grid[0] # used for explicit point finding
		self.gamma_c, self.gamma_l = gamma_c, gamma_l
		self.A_L = A_L
		self.T = T

		# Declare grids
		self.grid_max =grid_max
		self.grid_size = grid_size
		self.X_all = cartesian([np.arange(len(z_vals)), np.arange(len(self.asset_grid))])

		# Define functions 
		@njit
		def du(x):
			return  np.power(x, -gamma_c) 
		
		@njit
		def u(x,l):

			cons_u = (x**(1 - gamma_c)-1)/(1 - gamma_c)
			lab_u  = A_L*(l**(1 - gamma_l) - 1)/(1 - gamma_l)

			return cons_u + lab_u
		
		self.u, self.du = u, du
		self.z_seq = mc_sample_path(Pi, sample_size=T)
		self.dul = lambda l: A_L*np.power(l, - gamma_l) 

class FirmProblem:
	"""
	A class that stores primitives for the firm problem

	Parameters
	----------
	alpha : scalar(float)
		A strictly positive scalar giving the capital share
	delta: scalar(float)
		Depreciation rate of capital  
	AA : Productivity paramter

	Attributes
	----------
	r, beta, Pi, z_vals, b, u, du : see Parameters
	asset_grid : np.ndarray
		One dimensional grid for assets

	"""

	def __init__(self, 
				 alpha = 1-.64, 
				 delta = .083,
				 AA = 1.5):


		self.alpha, self.delta, self.AA = alpha, delta, AA

		@njit
		def K_to_rw(K, L):
			#ststkfp.AA=alpha/((1/beta-1)+delta)
			#fp.AA= 1.0/(ststkfp.AA**alpha *  L**(1.0-alpha)  )
			r = AA*alpha*np.power(K,alpha-1)*np.power(L,1-alpha)- delta
			w = AA*(1-alpha)*np.power(K,alpha)*np.power(L,-alpha)
			fkk = AA*(alpha-1)*alpha*np.power(K,alpha-2)*np.power(L,1-alpha)
			return r,w, fkk

		@njit
		def r_to_w(r):
			"""
			Equilibrium wages associated with a given interest rate r.
			"""

			#ststkfp.AA=fp.alpha/((1/cp.beta-1)+fp.delta)
			#fp.AA= 1.0/(ststkfp.AA**fp.alpha *  L**(1.0-fp.alpha)  )
			#A =1
			return AA * (1 - alpha) * (AA * alpha / (r + delta))**(alpha/ (1 - alpha))

		@njit
		def r_to_K(r, L):
			"""
			Returns capital stock associated with interest rate and labor supply 
			"""
			#ststkfp.AA=alpha/((1/beta-1)+delta)
			#fp.AA= 1.0/(ststkfp.AA**alpha *  L**(1.0-alpha)  )
			K = L*(((r +delta)/(AA*alpha)))**(1/(alpha-1))
			return K
		@njit
		def KL_to_Y(K,L):
			"""
			Returns output associated with capital and labor
			"""
			return AA*np.power(K,alpha)*np.power(L,1-alpha)

		self.K_to_rw, self.r_to_w, self.r_to_K, self.KL_to_Y = K_to_rw, r_to_w, r_to_K, KL_to_Y


def Operator_Factory(cp, fp):

	# tolerances

	tol_brent = 10e-5
	tol_bell = 10e-4
	tol_cp = 1e-3
	eta = 1


	# Create the variables that remain constant through iteration 

	beta, b                 = cp.beta, cp.b
	asset_grid, z_vals, Pi  = cp.asset_grid, cp.z_vals, cp.Pi
	k                       = cp.k
	A_L, gamma_c, gamma_l   = cp.A_L, cp.gamma_c, cp.gamma_l 

	grid_max, grid_size     =cp.grid_max, cp.grid_size
	u, du,dul               = cp.u, cp.du, cp.dul


	X_all                   = cp.X_all
	X_all = np.ascontiguousarray(X_all)
	Pi = np.ascontiguousarray(Pi)
	asset_grid = np.ascontiguousarray(asset_grid)
	z_vals = np.ascontiguousarray(z_vals)
	z_idx                   = np.arange(len(z_vals))


	shape                   = len(z_vals), len(asset_grid)
	
	V_init, h_init, c_init  = np.empty(shape), np.empty(shape), np.empty(shape)

	K_to_rw, r_to_w, r_to_K, KL_to_Y =fp.K_to_rw, fp.r_to_w, fp.r_to_K, fp.KL_to_Y

	z_seq = cp.z_seq
	T = cp.T

	alpha, delta, AA = fp.alpha, fp.delta, fp.AA 
	@njit
	def initialize(R):
		"""
		Creates a suitable initial conditions V and c for value function and time
		iteration respectively.

		Parameters
		----------
		cp : ConsumerProblem
			An instance of ConsumerProblem that stores primitives

		Returns
		-------
		V : array_like(float)
			Initial condition for value function iteration
		h : array_like(float)
			Initial condition for Coleman operator iteration

		"""
		shape =  len(z_vals),len(asset_grid)
		V, h, c= np.empty(shape), np.empty(shape), np.empty(shape)

		# === Populate V and c === #
		for i_a, a in enumerate(asset_grid):
			for i_z, z in enumerate(z_vals):
				h_max = R * a + z + b
				h[i_z, i_a] = 0
				V[i_z, i_a] = u(h_max,.438) / (1 - beta)
				c[i_z, i_a] = h_max

		return V, h,c

	@njit
	def obj(x,a,z, i_z,V,R, w, Lambda_E, Lambda_H, asset_grid):  # objective function to be *minimized*
		if R * a + w*z*(1-x[1])-b -x[0]>0:
			#y = np.sum(np.array([interp(asset_grid, V[j],x[0])*Pi[i_z, j] for j in z_idx]))
			 
			return u(R*a +w*z*(1-x[1]) - x[0],x[1]) + x[0]*Lambda_H + z*x[1]*Lambda_E + beta*interp(asset_grid, V,x[0])
		else:
			return - np.inf

	@njit
	def bellman_operator(V,R, w, Lambda_E, Lambda_H, Pi, X_all,asset_grid,z_vals,  return_policy=False):
		"""
		The approximate Bellman operator, which computes and returns the
		updated value function TV (or the V-greedy policy c if
		return_policy is True).

		Parameters
		----------
		V : array_like(float)
			A NumPy array of dim len(cp.asset_grid) times len(cp.z_vals)
		cp : ConsumerProblem
			An instance of ConsumerProblem that stores primitives
		return_policy : bool, optional(default=False)
			Indicates whether to return the greed policy given V or the
			updated value function TV.  Default is TV.

		Returns
		-------
		array_like(float)
			Returns either the greed policy given V or the updated value
			function TV.

		"""

		# === Linear interpolation of V along the asset grid === #
		#vf = lambda a, i_z: np.interp(a, asset_grid, V[:, i_z])

		VC  = np.zeros(V.shape)
		#Pi = np.ascontiguousarray(Pi)
		
		#Pi = np.ascontiguousarray(Pi)
		# numpy dot sum product over last axis of matrix_A (t+1 continuation value unconditioned)
		# see nunpy dot docs
		for state in range(len(X_all)):
			i_a = X_all[state][1]
			i_z = X_all[state][0]
			VC[i_z, i_a] = np.dot(Pi[i_z,:], V[:, i_a])
		

		# === Solve r.h.s. of Bellman equation === #
		new_V = np.ascontiguousarray(np.zeros(V.shape))
		new_h = np.ascontiguousarray(np.zeros(V.shape)) # next period capital 
		new_l = np.ascontiguousarray(np.zeros(V.shape)) #lisure
		
		for state in range(len(X_all)):
			a = asset_grid[X_all[state][1]]
			i_a = X_all[state][1]

			i_z = X_all[state][0]
			z   = z_vals[i_z]
			
			"""
			bnds = ((b, cp.grid_max ),(0+1e-4,1- 1e-4))
			cons = ({'type': 'ineq', 'fun': lambda x:  R * a + w*z*(1-x[1])-b -x[0]}, {'type': 'ineq', 'fun': lambda x: x[0]})
			h0 = [b, .438]
			#print(h0)
			h_star = optimize.minimize(lambda x: -obj(x,  a, z, i_z, V\
																	,R, w, Lambda_E, Lambda_H), h0, bounds = bnds,constraints=cons).x

			"""

			
			bnds = np.array([[b, grid_max],[0+1e-4,1- 1e-4]])
			h0 = np.array([b, .438])
			args = (a, z, i_z, VC[i_z],R, w, Lambda_E, Lambda_H, asset_grid)
			
			h_star  = qe.optimize.nelder_mead(obj, h0, bounds = bnds, args = args)[0]
			#h_star3= fminbound(obj, b, R * a + w*z + b)
			#print(obj(h_star.x[0]), obj(h_star3))
			
			
			if return_policy==True:
				new_h[ i_z,i_a],new_l[i_z, i_a], new_V[i_z, i_a] = h_star[0],h_star[1], obj(h_star, a, z, i_z, VC[i_z]\
																	,R, w, Lambda_E, Lambda_H, asset_grid)
			#Pool.clear
				
			if return_policy==False:
				new_V[i_z, i_a] =obj(h_star, *args)
				#Pool.clear


	   
		return [new_h, new_l, new_V]




	@njit
	def series(T, a, h_val, l_val,z_rlz, z_seq, z_vals, hf,lf):
		for t in range(T-1):
			a[t+1] = interp(asset_grid, hf[z_seq[t]], a[t])
			h_val[t] = a[t+1] #this can probably be vectorized 
			l_val[t] = max([0,min([.9999999,interp(asset_grid, lf[z_seq[t]], a[t])])])
			z_rlz[t]= z_vals[z_seq[t]]*(1-l_val[t]) #this can probably be vectorized 
	   
		
		l_val[T-1] = max([0,min([.9999999,interp(asset_grid, lf[z_seq[T-1]],  a[T-1])])])
		h_val[T-1] = interp(asset_grid, hf[z_seq[T-1]],  a[T-1])
		z_rlz[T-1] = z_vals[z_seq[T-1]]*(1-l_val[T-1]) 
		return a, h_val, l_val, z_rlz

	@njit
	def compute_asset_series_bell(R, w, Lambda_E, Lambda_H, T=T, verbose=False):
		"""
		Simulates a time series of length T for assets, given optimal savings
		behavior.  Parameter cp is an instance of consumerProblem
		"""

		# === Simplify names, set up arrays === #

		v, h_init, c_init = initialize(R)
		
		i=0
		error = 1
		while error> tol_bell and i<200:
			#time_stat = time.time()
			
			Tv = bellman_operator(v, R, w, Lambda_E, Lambda_H, Pi, X_all,asset_grid, z_vals, return_policy=False)[2]
			error = np.max(np.abs(Tv - v))
			v = Tv
			#print(error)
			i+=1
			#print(time.time()-time_stat)

		policy = bellman_operator(v, R, w, Lambda_E, Lambda_H, Pi, X_all,asset_grid, z_vals, return_policy=True)
		a = np.zeros(T)
		a[0] = b*2

		z_rlz = np.zeros(T) #total labour supply after endogenous decisions. That is, e*(1-l)
		h_val = np.zeros(T)
		l_val = np.zeros(T) #liesure choice l! do NOT confuse with labour 
		a, h_val, l_val, z_rlz = series(T, a, h_val,l_val, z_rlz, z_seq, z_vals, policy[0], policy[1])
		return a, z_rlz, h_val, l_val, policy


	@njit
	def coef_var(a):
		return np.sqrt(np.mean((a-np.mean(a))**2))/np.mean(a)



	@njit
	def compute_agg_prices(R, w, Lambda_E, Lambda_H, social= 0):
		a, z_rlz, h_val, l_val, policy = compute_asset_series_bell(R, w, Lambda_E, Lambda_H, T=T)

		agg_K = np.mean(a)

		L = np.mean(z_rlz)
		H = np.mean(1-l_val)
		coefvar = coef_var(a)
		r_f,w_f, fkk = K_to_rw(agg_K, L)
		if social == 1:
			Lambda = beta*agg_K*fkk*np.mean(\
						du(a*(1+r_f) + w_f*z_rlz - h_val)*((a/agg_K) - (z_rlz/L))\
						)
		else:
			Lambda = 0
		return r_f, w_f, Lambda, agg_K, L, H, coefvar, a, z_rlz, h_val, l_val, policy 

	@njit
	def Gamma_IM(r,Lambda_E,Lambda_H):
		"""
		Function whose zero is the incomplete markets allocation. 
		"""
		print(r)
		R = 1+ r
		w = r_to_w(r)
		r_nil, w_nil, Lambda_supply, K_supply, L_supply, Hours, coefvar, a_nil, z_rlz_nil, h_val_nil, l_val_nil, policy_nil= compute_agg_prices(R, w, Lambda_E, Lambda_H, social= 0)
		K_demand = r_to_K(r, L_supply)
		excesssupply_K = K_supply- K_demand

		return excesssupply_K

	def firstbest():
		#ststkfp.AA=alpha/((1/beta-1)+delta)
		#fp.AA= 1.0/(ststkfp.AA**alpha *  L**(1.0-alpha)  )
		##Get average productivity
		mc = MarkovChain(Pi)
		stationary_distributions = mc.stationary_distributions[0]
		E = np.dot(stationary_distributions, z_vals)
		def fbfoc(l):
			#l = np.min([np.max([x,0.001]),.999])
			L = E*(1-l)
			K = L*(((1-beta +delta*beta)/(AA*alpha*beta))**(1/(alpha-1)))
			Y = AA*(K**alpha)*(L**(1-alpha))
			Fl = -E*AA*(1-alpha)*(K**alpha)*(L**(-alpha))
			diff = du(Y - delta*K)*Fl + dul(l)
			#print(cp.du(Y - fp.delta*K)*Fl )
			return diff
		
		l = fsolve(fbfoc, .5)[0]
		Labour_Supply = E*(1-l)
		L = E*(1-l)
		Hours = 1-l
		K = L*(((1-beta +delta*beta)/(AA*alpha*beta))**(1/(alpha-1)))
		Y = AA*np.power(K,alpha)*np.power(L,1-alpha)
		#r = fp.AA*fp.alpha*np.power(K,fp.alpha-1)*np.power(L,1-fp.alpha)- fp.delta
		r,w,fkk = K_to_rw(K, L)
		return r, K, Hours,Labour_Supply, Y, w 

	#@njit
	def Omega_CE(Lambda_H,r):
		cap_lab_ratio = ((r+delta)/alpha)**(1/(alpha-1))
		Lambda_E = -(Lambda_H*cap_lab_ratio/beta)

		eqm_r_IM = brentq(Gamma_IM, -delta*.99, (1-beta)/beta, args = (Lambda_E,Lambda_H), xtol = tol_brent)[0]

		R = 1+ eqm_r_IM
		w = r_to_w(eqm_r_IM)
		CP_r,CP_w, CP_Lambda, CP_K, CP_L, CP_H, CP_coefvar, CP_a, CP_z_rlz, CP_h_val, CP_l_val, CP_policy = compute_agg_prices(R, w, Lambda_E, Lambda_H, social= 1)

		return [CP_r,CP_w, CP_Lambda, CP_K, CP_L, CP_H, CP_coefvar, CP_a, CP_z_rlz, CP_h_val, CP_l_val, CP_policy]


	#@njit
	def compute_CEE():
		eqm_r_IM = brentq(Gamma_IM, -delta*.99, (1-beta)/beta, args = (0,0), xtol = tol_brent)[0]
		R = 1+ eqm_r_IM
		w = r_to_w(eqm_r_IM)
		im_r, im_w, im_Lambda, im_K, im_L, im_H, im_coefvar, im_a, im_z_rlz, im_h_val, im_l_val, im_policy  = compute_agg_prices(R, w, 0, 0, social =1)
		print('IM calculated interest rate {}, hours are {}, labour supply is {}, k_supply is {}'.format(im_r*100, im_H,im_L, im_K))

		im_Y 	  = KL_to_Y(im_K, im_L)
		im_gini_a = gini(im_a)
		im_gini_i = gini(im_z_rlz)

		IM_out   = [im_r, im_w, im_Lambda, im_K, im_L, im_H, im_coefvar, im_a, im_z_rlz, im_h_val, im_l_val, im_policy, im_Y, im_gini_a,im_gini_i]

		IM_list = 										 	['IM_r', 'IM_w', 'IM_Lambda',\
															'IM_K', 'IM_L', 'IM_H',\
															'IM_coefvar', 'IM_a', 'IM_z_rlz',\
															'IM_h_val', 'IM_l_val', 'IM_policy',\
															 'IM_Y', 'IM_gini_a', 'IM_gini_i']

		results_IM  = {}

		for var, name in zip(IM_out,IM_list):
			results_IM[name] = var


		# set initial CP interest rate
		r = 0.006612680218336248
		#r = eqm_r_IM
		R = 1+r
		w = r_to_w(r)
		cap_lab_ratio = ((r+delta)/alpha)**(1/(alpha-1))
		Lambda_H =im_Lambda

		error =1
		i = 0
		while error> tol_cp:
			out = Omega_CE(Lambda_H,r)
			OLambda_H, Or = out[2], out[0]
			error = np.max(np.abs([Lambda_H- OLambda_H, r- Or]))
			print('Iteration {} of Omega, interest rate {}, Lambda_H {} and error1 {}'.format(i, Or, OLambda_H, error))
			print('Iteration {} calculated interest rate {}, hours are {}, labour supply is {}, k_supply is {}'.format(i,out[0], out[5],out[4], out[3]))
			Lambda_H = OLambda_H
			r = Or
			i +=1

		print('CE calculated interest rate {}, hours are {}, labour supply is {}, k_supply is {}'.format(out[0], out[5],out[4], out[3]))

		CP_Y 		= KL_to_Y(out[3], out[4])
		CP_gini_a 	= gini(out[7])
		CP_gini_i 	= gini(out[8])
		out.append([CP_Y,CP_gini_a,CP_gini_i])

		results_CP = {}

		CP_list = [		'CP_r','CP_w',\
						'CP_Lambda',\
						'CP_K', 'CP_L',\
						'CP_H', 'CP_coefvar',\
						'CP_a', 'CP_z_rlz',\
						'CP_h_val', 'CP_l_val',\
						'CP_policy', 'CP_Y', \
						'CP_gini_a', \
						'CP_gini_i']
		
		for var, name in zip(out,CP_list) :
			results_CP[name] = var

		# calculate counter-factual 1
		R = 1+ results_CP['CP_r']
		w = r_to_w(r)


		CF_r, CF_w, CF_Lambda, CF_K, CF_L, CF_H, CF_coefvar, CF_a, CF_z_rlz, CF_h_val, CF_l_val, CF_policy  = compute_agg_prices(R, w, 0, 0, social =1)
		CF_out = [CF_r, CF_w, CF_Lambda, CF_K, CF_L, CF_H, CF_coefvar, CF_a, CF_z_rlz, CF_h_val, CF_l_val, CF_policy]
		print('CF calculated interest rate {}, hours are {}, labour supply is {}, k_supply is {}'.format(CF_r*100, CF_H,CF_L, CF_K))


		CF_Y 		= KL_to_Y(CF_out[3], CF_out[4])
		CF_gini_a 	= gini(CF_out[7])
		CF_gini_i 	= gini(CF_out[8])
		CF_out.append([CF_Y,CF_gini_a,CF_gini_i])

		results_CF = {}

		CF_list = [		'CF_r','CF_w',\
						'CF_Lambda',\
						'CF_K', 'CF_L',\
						'CF_H', 'CF_coefvar',\
						'CF_a', 'CF_z_rlz',\
						'CF_h_val', 'CF_l_val',\
						'CF_policy', 'CF_Y', \
						'CF_gini_a', \
						'CF_gini_i']
		
		for var, name in zip(CF_out,CF_list) :
			results_CF[name] = var


		



		return results_IM, results_CP, results_CF


	return compute_CEE, firstbest


if __name__ == "__main__":


	# Shock matrix rom Pijoan-Mas
	Gamma  = np.array([[0.746464, 0.252884, 0.000652, 0.000000, 0.000000, 0.000000,0.000000],
		  [0.046088,  0.761085,  0.192512,  0.000314,  0.000000,  0.000000,  0.000000],
		  [0.000028,  0.069422,  0.788612,  0.141793,  0.000145,  0.000000,  0.000000],
		  [0.000000,  0.000065,  0.100953,  0.797965,  0.100953,  0.000065,  0.000000],
		  [0.000000,  0.000000,  0.000145,  0.141793,  0.788612,  0.069422,  0.000028],
		  [0.000000,  0.000000,  0.000000,  0.000314,  0.192512,  0.761085,  0.046088],
		 [0.000000,  0.000000,  0.000000,  0.000000,  0.000652,  0.252884,  0.746464]])

	# Stationary distribution
	Gamma_bar = np.array([0.0154,   0.0847,    0.2349,    0.3300,    0.2349,    0.0847,    0.0154])


	e_shocks = np.array([np.exp(-1.385493),np.exp(-0.923662),np.exp(-0.461831),
						np.exp(0.000000),
						np.exp(0.461831),
						np.exp(0.923662),
						np.exp(1.385493)])
	
	e_shocks = e_shocks/np.dot(e_shocks, Gamma_bar)


	# Load the model file with parameter settings

	model = open('pjmas2.mod', 'rb')
	model_in = pickle.load(model)
	name = model_in["filename"]
	model.close()


	cp = ConsumerProblem(Pi = Gamma,
							z_vals = e_shocks,
							gamma_c= model_in["gamma_c"],
							gamma_l = model_in["gamma_l"],
							A_L = model_in["A_L"],
							grid_max = 60,
							grid_size= 1000,
							beta = .945)

	fp = FirmProblem(delta = .083)

	tol_brent = 10e-7
	tol_bell = 10e-4
	tol_cp = 4e-6
	eta = 1
 
	compute_CEE, firstbest= Operator_Factory(cp, fp)

	import time
	R = 1.03
	v, h_init, c_init = initialize(R)
	w = r_to_w(R)
	start = time.time()
	bellman_operator(v, R, w, 0, 0, return_policy=False)
	bellman_operator.parallel_diagnostics(level=4)
	print(time.time()-start)

