import numpy as np 

from numba import njit

from scipy.optimize import fsolve, brentq

from quantecon import tauchen

import math
from sklearn.utils.extmath import cartesian 

from pathos.pools import ProcessPool 


class endog_labour_model:
	def __init__(self,
			 P,     # Dictionary containing functions of the model 
			 P_stat,    # Dictionary containing parameters
			 e_shocks,
			 A=1.5,
			 beta=0.945,
			 delta=0.083,
			 sigma=1.458,
			 nu=2.833,
			 Lambda=0.856,
			 theta=0.640):

		self.P = P
		self.P_stat = P_stat
		self.e_shocks = e_shocks
		self.A=A
		self.beta= beta
		self.delta=delta
		self.sigma=sigma
		self.nu=nu
		self.Lambda=Lambda
		self.theta=theta

		K_over_L = ((1/self.beta-1+self.delta)/(self.A*(1-self.theta)))**(-1/self.theta)

		@njit
		def L(K):
			return   K/K_over_L

		self.L = L


		@njit
		def labour(c):
			hours =  ((Lambda/(e_shocks*A*theta*K_over_L**(1-theta)))\
									*(c**sigma))\
									**(1/nu)

			for e in range(len(e_shocks)):
				hours[e] = max([0,min([1,hours[e] ])])

			return hours


		self.labour = labour

		@njit
		def budget_cons(c, b):
			RHS = b+ (A*theta*K_over_L**(1-theta))*e_shocks*(1-labour(c))


			LHS = c+ beta*np.dot(P,b)

			return RHS-LHS

		@njit
		def L_supply(c):
			return np.dot(e_shocks*(1-labour(c)), P_stat)



		@njit
		def market_clearing(c,b):
			K_supply = beta*np.dot(b, P_stat)

			return L_supply(c) - L(K_supply)



		@njit
		def system(array):
			return np.append(budget_cons(array[-1], array[0: len(e_shocks)]),\
								market_clearing(array[-1], array[0: len(e_shocks)]))

		


		self.budget_cons, self.market_clearing, self.labour, self.L =budget_cons,   market_clearing, labour, L
		self.system = system
		self.L_supply = L_supply
		self.K_over_L = K_over_L

		@njit
		def K_supply(array):
			return beta*np.dot(array, P_stat)

		@njit
		def output(array):
			b = array[0:-1]
			c = array[-1]
			K_supply = beta*np.dot(b, P_stat)
			return A*(K_supply**(1-theta))*(L_supply(c)**theta)

		self.output = output
		self.K_supply = K_supply



if __name__ == '__main__':


	# input markov matrix 
	Gamma  =np.array([[0.746464, 0.252884, 0.000652, 0.000000, 0.000000, 0.000000,0.000000],
			  [0.046088,  0.761085,  0.192512,  0.000314,  0.000000,  0.000000,  0.000000],
			  [0.000028,  0.069422,  0.788612,  0.141793,  0.000145,  0.000000,  0.000000],
			  [0.000000,  0.000065,  0.100953,  0.797965,  0.100953,  0.000065,  0.000000],
			  [0.000000,  0.000000,  0.000145,  0.141793,  0.788612,  0.069422,  0.000028],
			  [0.000000,  0.000000,  0.000000,  0.000314,  0.192512,  0.761085,  0.046088],
			 [0.000000,  0.000000,  0.000000,  0.000000,  0.000652,  0.252884,  0.746464]])

	Gamma_bar =np.array([0.0154,   0.0847,    0.2349,    0.3300,    0.2349,    0.0847,    0.0154])


	e_shocks = np.array([np.exp(-1.385493),np.exp(-0.923662),np.exp(-0.461831),
				np.exp(0.000000),
				np.exp(0.461831),
				np.exp(0.923662),
				np.exp(1.385493)])
	e_shocks = e_shocks/np.dot(e_shocks, Gamma_bar)

	#A 		 = np.inner(e_shocks,Gamma_bar)**(-.640)

	def A_zero(a):


		model = endog_labour_model(A=a, P =Gamma, P_stat = Gamma_bar, e_shocks= e_shocks)
		# solving the model 
		c_init = 2
		b_init= np.ones(len(e_shocks))

		init = np.append(b_init,c_init)

		sol = fsolve(model.system, init) 
		return model.L_supply(sol[-1])-.34
		


	A = brentq(A_zero, .1, 3)

	model = endog_labour_model(A=1.5, P =Gamma, P_stat = Gamma_bar, e_shocks= e_shocks)


	c_init = 2
	b_init= np.ones(len(e_shocks))

	init = np.append(b_init,c_init)

	sol = fsolve(model.system, init) 

	H = np.inner(Gamma_bar,1-model.labour(sol[-1]))      
	L = model.L_supply(sol[-1])
	Y = model.output(sol)
	K = model.K_supply(np.array(sol[0:-1]))  

	
	"""

	def Ptest(X):

		#print("testing {}".format(X))
		rho = X[0]
		sigma = X[1]
		A = X[2]

		if rho >.6 and rho<.999 and sigma >.03 and sigma< .5 and A>.01 and A<3:
			P_markov = tauchen(rho, sigma, n=7)

			model = endog_labour_model(A=X[2], P =P_markov.P, P_stat = P_markov.stationary_distributions[0], e_shocks= np.exp(P_markov.state_values))


			c_init = 2
			b_init= np.ones(len(P_markov.state_values))

			init = np.append(b_init,c_init)


			sol = fsolve(model.system, init) 
			H = np.inner(P_markov.stationary_distributions[0],1-model.labour(sol[-1]))      
			L = model.L_supply(sol[-1])
			Y = model.output(sol)

			print("H= {}".format(H))
			print("L = {}".format(L))
			print("Y= {}".format(Y))
			print("consumption ={}".format(sol[-1]))
			
			out =  np.max([(sol[-1]-.86)**2, (Y-1.09)**2,(H-.27)**2,(L-.34)**2])

			if math.isnan(out):
				return 100

			else:
				return out
		else:
			return np.inf

  
	rhos = np.linspace(.6, .9999, 10)
	sigma = np.linspace(.05, .2, 10)
	A = np.array([1, 1.5])

	grid = cartesian([rhos, sigma, A])


	p                           = ProcessPool()

	errors= np.array(p.map(Ptest, grid))

	x0 = grid[np.where(errors==np.min(errors))[0][0]]

	minimize(Ptest, x0 ,tol = 1e-8, method = "Nelder-Mead") """
