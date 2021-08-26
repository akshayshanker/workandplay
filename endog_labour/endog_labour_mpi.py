"""Module contains operators to solve incomplete market and constrained planner
problem for Aiyagari-Huggett model with endogenous labour choice (Shanker and
Wolfe, 2018)

This file should imported as a module and contains the following
classes:

    * ConsumerProblem - parameterised model of consumers
    * FirmProblem - parameterised model of firms

And the following functions:
    * Operator_Factory - returns paramterised operators to solve the eqm 


Author: Akshay Shanker, University of New South Wales, Sydney
                a.shanker@unsw.edu.au

Todo
----

- Deprecicate Bellman operator
- Improve efficiency of computing stationary distribution
- Adjust X-entropy so possible candidates for elite members from previous
    iterations are retained 
- Tidy Docstrings 
- Does error in monte carlo draw of stationary distribution affect X-entropy
    convergence?

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
import dill as pickle
from gini import gini
import pdb
from sklearn.utils.extmath import cartesian
from numba import njit, prange
from interpolation.splines import UCGrid, CGrid, nodes, eval_linear
from interpolation import interp


class ConsumerProblem:

    """A class that stores primitives for the income fluctuation problem.  The
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
                 w=.4,
                 Lambda_H=0,
                 Lambda_E=0,
                 beta=.945,
                 Pi=((0.09, 0.91), (0.06, 0.94)),
                 z_vals=(0.1, 1.0),
                 b=1e-10,
                 grid_max=50,
                 grid_size=100,
                 gamma_c=1.458,
                 gamma_l=2.833,
                 A_L=.856,
                 T=int(1E7)):

        # Declare parameters
        self.r, self.R = r, 1 + r
        self.w = w
        self.Lambda_H = Lambda_H
        self.Lambda_E = Lambda_E
        self.beta, self.b = beta, b
        self.Pi, self.z_vals = np.array(Pi), np.asarray(z_vals)
        self.asset_grid = np.linspace(b, grid_max, grid_size)
        # used for explicit point finding
        self.k = self.asset_grid[1] - self.asset_grid[0]
        self.gamma_c, self.gamma_l = gamma_c, gamma_l
        self.A_L = A_L
        self.T = T

        # Declare grids
        self.grid_max = grid_max
        self.grid_size = grid_size
        self.X_all = cartesian(
            [np.arange(len(z_vals)), np.arange(len(self.asset_grid))])

        # Define functions
        @njit
        def du(x):
            return np.power(x, -gamma_c)

        @njit
        def du_inv(x):
            return np.power(x, -1 / gamma_c)

        @njit
        def dul(x):
            return A_L * np.power(x, - gamma_l)

        @njit
        def dul_inv(x):
            return np.power((1 / A_L) * x, - 1 / gamma_l)

        @njit
        def u(x, l):

            cons_u = (x**(1 - gamma_c) - 1) / (1 - gamma_c)
            lab_u = A_L * (l**(1 - gamma_l) - 1) / (1 - gamma_l)

            return cons_u + lab_u

        self.u, self.du = u, du
        self.dul, self.dul_inv, self.du_inv = dul, dul_inv, du_inv
        self.z_seq = mc_sample_path(Pi, sample_size=T)


class FirmProblem:
    """A class that stores primitives for the firm problem.

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
                 alpha=1 - .64,
                 delta=.083,
                 AA=1.51):

        self.alpha, self.delta, self.AA = alpha, delta, AA

        @njit
        def K_to_rw(K, L):

            r = AA * alpha * np.power(K, alpha - 1) * \
                np.power(L, 1 - alpha) - delta
            w = AA * (1 - alpha) * np.power(K, alpha) * np.power(L, -alpha)
            fkk = AA * (alpha - 1) * alpha * np.power(K,
                                                      alpha - 2) * np.power(L, 1 - alpha)

            return r, w, fkk

        @njit
        def r_to_w(r):

            return AA * (1 - alpha) * (AA * alpha /
                                       (r + delta))**(alpha / (1 - alpha))

        @njit
        def r_to_K(r, L):
            """Returns capital stock associated with interest rate and labor
            supply."""

            K = L * (((r + delta) / (AA * alpha)))**(1 / (alpha - 1))

            return K

        @njit
        def KL_to_Y(K, L):
            """Returns output associated with capital and labor."""
            return AA * np.power(K, alpha) * np.power(L, 1 - alpha)

        self.K_to_rw, self.r_to_w, self.r_to_K, self.KL_to_Y = K_to_rw, r_to_w, r_to_K, KL_to_Y


def Operator_Factory(cp, fp):

    # tolerances
    tol_brent = 10e-9
    tol_bell = 10e-6
    tol_cp = 1e-3
    eta = 1
    max_iter_bell = 500
    max_iter_xe = 50
    eta_b = .8
    tol_contract = 1e-12

    # Create the variables that remain constant through iteration

    beta, b = cp.beta, cp.b
    asset_grid, z_vals, Pi = cp.asset_grid, cp.z_vals, cp.Pi
    k = cp.k
    A_L, gamma_c, gamma_l = cp.A_L, cp.gamma_c, cp.gamma_l

    grid_max, grid_size = cp.grid_max, cp.grid_size
    u, du, dul = cp.u, cp.du, cp.dul

    X_all = cp.X_all
    K_to_rw, r_to_w, r_to_K, KL_to_Y = fp.K_to_rw, fp.r_to_w, fp.r_to_K, fp.KL_to_Y
    z_seq = cp.z_seq
    T = cp.T

    dul, dul_inv, du_inv = cp.dul, cp.dul_inv, cp.du_inv

    alpha, delta, AA = fp.alpha, fp.delta, fp.AA

    @njit
    def interp_as(xp, yp, x, extrap=False):
        """Function  interpolates 1D with linear extraplolation.

        Parameters
        ----------
        xp : 1D array
              points of x values
        yp : 1D array
              points of y values
        x  : 1D array
              points to interpolate

        Returns
        -------
        evals: 1D array
                y values at x
        """

        evals = np.zeros(len(x))
        if extrap and len(xp) > 1:
            for i in range(len(x)):
                if x[i] < xp[0]:
                    if (xp[1] - xp[0]) != 0:
                        evals[i] = yp[0] + (x[i] - xp[0]) * (yp[1] - yp[0])\
                            / (xp[1] - xp[0])
                    else:
                        evals[i] = yp[0]

                elif x[i] > xp[-1]:
                    if (xp[-1] - xp[-2]) != 0:
                        evals[i] = yp[-1] + (x[i] - xp[-1]) * (yp[-1] - yp[-2])\
                            / (xp[-1] - xp[-2])
                    else:
                        evals[i] = yp[-1]
                else:
                    evals[i] = np.interp(x[i], xp, yp)
        else:
            evals = np.interp(x, xp, yp)
        return evals

    @njit
    def initialize(R):
        """Creates a suitable initial conditions V and c for value function and
        time iteration respectively.

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
        shape = len(z_vals), len(asset_grid)
        V, h, c = np.empty(shape), np.empty(shape), np.empty(shape)

        # Populate V and c 
        for i_a, a in enumerate(asset_grid):
            for i_z, z in enumerate(z_vals):
                h_max = R * a + z + b
                h[i_z, i_a] = 0
                V[i_z, i_a] = du(h_max / 100)
                c[i_z, i_a] = h_max

        return V, h, c

    @njit
    # objective function to be *minimized*
    def obj(x, a, z, i_z, V, R, w, Lambda_E, Lambda_H, asset_grid):
        if R * a + w * z * (1 - x[1]) - b - x[0] > 0:

            return u(R * a + w * z * (1 - x[1]) - x[0], x[1]) + x[0] * \
                Lambda_H + z * x[1] * Lambda_E + \
                beta * interp(asset_grid, V, x[0])
        else:
            return - np.inf

    @njit
    def bellman_operator(V, R, w, Lambda_E, Lambda_H, Pi,
                         X_all, asset_grid, z_vals, return_policy=False):
        """The approximate Bellman operator, which computes and returns the
        updated value function TV (or the V-greedy policy c if return_policy is
        True).

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

        VC = np.zeros(V.shape)


        for state in range(len(X_all)):
            i_a = X_all[state][1]
            i_z = X_all[state][0]
            VC[i_z, i_a] = np.dot(Pi[i_z, :], V[:, i_a])

        # Solve r.h.s. of Bellman equation
        new_V = np.zeros(V.shape)
        new_h = np.zeros(V.shape)
        new_l = np.zeros(V.shape)

        for state in range(len(X_all)):
            a = asset_grid[X_all[state][1]]
            i_a = X_all[state][1]

            i_z = X_all[state][0]
            z = z_vals[i_z]

            bnds = np.array([[b, grid_max], [0 + 1e-4, 1 - 1e-4]])
            h0 = np.array([b, .438])
            args = (a, z, i_z, VC[i_z], R, w, Lambda_E, Lambda_H, asset_grid)

            h_star = qe.optimize.nelder_mead(
                obj, h0, bounds=bnds, args=args)[0]

            if return_policy:
                new_h[i_z, i_a], new_l[i_z, i_a], new_V[i_z, i_a] = h_star[0], h_star[1], obj(
                    h_star, a, z, i_z, VC[i_z], R, w, Lambda_E, Lambda_H, asset_grid)

            if not return_policy:
                new_V[i_z, i_a] = obj(h_star, *args)

        return [new_h, new_l, new_V]

    @njit
    def coleman_operator(V,
                         R,
                         w,
                         Lambda_E,
                         Lambda_H,
                         Pi,
                         X_all,
                         asset_grid,
                         z_vals,
                         return_policy=False):
        """The EGM Coleman operator, computes and returns the updated policy
        functions and next period MUC.

        Parameters
        ----------
        V : array_like(float)
                A NumPy array of dim len(cp.z_vals) tines len(cp.asset_grid)
                Next period MUC
        R : float64
                Rate of return on assets
        w : float64
                Wage rate
        Lambda_E: float64
                Constrained planner's F_ll*Delta multipler
        Lambda_H: float64
                Constrained planner's - F_kk*Delta multipler
        Pi: 2D array
                 Transition matrix for exogenous shocks
        X_all: 2D array
                 Cartesian product of shock and asset grid indices
        asset_grid: 1D array
                    asset grid
        z_vals : 1D array
                    shock values

        Returns
        -------
        new_h: 2D array
                Asset policy function
        new_l: 2D array
                Leisure policy function
        new_V: 2D array
                MUC

        Note: The endogenous grid method is used

        """

        # Condition the next period MUC to today's shock state
        VC = np.zeros(V.shape)

        for state in range(len(X_all)):
            i_a = X_all[state][1]
            i_z = X_all[state][0]
            VC[i_z, i_a] = np.dot(Pi[i_z, :], V[:, i_a])

        # Create empty arrays to fill for new policies
        new_V = np.zeros(V.shape)
        new_h = np.zeros(V.shape)
        new_l = np.zeros(V.shape)

        # Grids for policiesc conditioned on t+1 assets
        curr_A = np.zeros(V.shape)
        curr_l = np.zeros(V.shape)
        curr_c = np.zeros(V.shape)

        for state in range(len(X_all)):

            # Unpack state space values
            # a in the loop is t + 1 assets
            a = asset_grid[X_all[state][1]]
            i_a = X_all[state][1]
            i_z = X_all[state][0]
            z = z_vals[i_z]

            # Consumption and leisure at t via inverting FOC
            c = du_inv(max(1e-200, (beta * R * VC[i_z, i_a] + Lambda_H)))
            l = min(1, dul_inv(max(1e-200, du(c) * w * z - Lambda_E * z)))
            curr_A[i_z, i_a] = min(asset_grid[-1],
                                   max(b, (c - w * (1 - l) * z + a) / (R)))
            curr_l[i_z, i_a] = l

        # Replace any out of bound values with bound
        curr_A = curr_A.ravel()
        curr_l = curr_l.ravel()
        curr_A[np.isnan(curr_A)] = b
        curr_l[np.isnan(curr_l)] = b
        curr_A = curr_A.reshape(len(z_vals), len(asset_grid))
        curr_l = curr_l.reshape(len(z_vals), len(asset_grid))

        # Interpolate polcies on time t assets
        for i_z in range(len(z_vals)):

            curr_A[i_z, :] = np.sort(curr_A[i_z, :])
            curr_l[i_z, :] = np.take(curr_l[i_z, :],
                                     np.argsort(curr_A[i_z, :]))
            asset_grid_sorted = np.take(asset_grid,
                                        np.argsort(curr_A[i_z, :]))
            new_h[i_z, :] = interp_as(curr_A[i_z, :],
                                      asset_grid_sorted, asset_grid)
            new_l[i_z, :] = interp_as(curr_A[i_z, :],
                                      curr_l[i_z, :], asset_grid)

            cons_array = R * asset_grid + w * \
                z_vals[i_z] * (1 - new_l[i_z, :]) - new_h[i_z, :]

            cons_array[cons_array <= 0] = b
            new_V[i_z, :] = du(cons_array)

        return new_h, new_l, new_V

    @njit
    def series(T, a, h_val, l_val, z_rlz, z_seq, z_vals, hf, lf):
        for t in range(T - 1):
            a[t + 1] = np.interp(a[t], asset_grid, hf[z_seq[t]])
            h_val[t] = a[t + 1]  # this can probably be vectorized
            l_val[t] = max(
                [0, min([.9999999, np.interp(a[t], asset_grid, lf[z_seq[t]])])])
            # this can probably be vectorized
            z_rlz[t] = z_vals[z_seq[t]] * (1 - l_val[t])

        l_val[T - 1] = max([0,
                            min([.9999999,
                                 interp(asset_grid,
                                        lf[z_seq[T - 1]],
                                        a[T - 1])])])
        h_val[T - 1] = interp(asset_grid, hf[z_seq[T - 1]], a[T - 1])
        z_rlz[T - 1] = z_vals[z_seq[T - 1]] * (1 - l_val[T - 1])
        return a, h_val, l_val, z_rlz

    @njit
    def compute_asset_series_bell(
            R, w, Lambda_E, Lambda_H, T=T, verbose=False):
        """Simulates a time series of length T for assets, given optimal
        savings behavior.

        Parameter cp is an instance of consumerProblem
        """

        v, h_init, c_init = initialize(R)
        h = h_init
        l = c_init
        i = 0
        error = 1
        while error > tol_bell and i < max_iter_bell:

            new_h, new_l, new_V = coleman_operator(
                v, R, w, Lambda_E, Lambda_H, Pi, X_all, asset_grid, z_vals,
                return_policy=False)
            Th = new_h
            Tl = new_l
            Tv = new_V
            error = np.max(
                np.array(
                    [np.max(np.abs(Th - h)),
                     np.max(np.abs(Tl - l)),
                     np.max(np.abs(Tv - v))]))
            v = new_V
            h = Th
            l = Tl
            i += 1

        a = np.zeros(T)
        a[0] = b * 2

        # total labour supply after endogenous decisions: e*(1-l)
        # liesure choice l! do NOT confuse with labour
        z_rlz = np.zeros(T)
        h_val = np.zeros(T)
        l_val = np.zeros(T)  
        a, h_val, l_val, z_rlz = series(
            T, a, h_val, l_val, z_rlz, z_seq, z_vals, new_h, new_l)
        return a, z_rlz, h_val, l_val, [new_h, new_l, new_V]

    @njit
    def coef_var(a):
        return np.sqrt(np.mean((a - np.mean(a))**2)) / np.mean(a)

    @njit
    def compute_agg_prices(R, w, Lambda_E, Lambda_H, social=0):
        a, z_rlz, h_val, l_val, policy = compute_asset_series_bell(
            R, w, Lambda_E, Lambda_H, T=T)

        agg_K = np.mean(a)

        L = np.mean(z_rlz)
        H = np.mean(1 - l_val)
        coefvar = coef_var(a)
        r_f, w_f, fkk = K_to_rw(agg_K, L)

        if social == 1:
            c = a * (1 + r_f) + w_f * z_rlz - h_val
            c[c <= b] = b
            Lambda = beta * agg_K * fkk * np.mean(
                du(c) * ((a / agg_K) - (z_rlz / L))
            )
        else:
            Lambda = 0
        return r_f, w_f, Lambda, agg_K, L, H, coefvar, a, z_rlz, h_val, l_val, policy

    @njit
    def Gamma_IM(r, Lambda_E, Lambda_H):
        """Returns excess supply given r, and multipliers.

        Note
        ----
        Function whose zero is the incomplete markets allocation.
        """
        R = 1 + r
        w = r_to_w(r)
        r_nil, w_nil, Lambda_supply, K_supply, L_supply, Hours, coefvar, \
            a_nil, z_rlz_nil, h_val_nil, l_val_nil, policy_nil \
            = compute_agg_prices(R, w, Lambda_E, Lambda_H, social=0)
        K_demand = r_to_K(r, L_supply)
        excesssupply_K = K_supply - K_demand

        return excesssupply_K

    def firstbest():

        mc = MarkovChain(Pi)
        stationary_distributions = mc.stationary_distributions[0]
        E = np.dot(stationary_distributions, z_vals)

        def fbfoc(l):
            L = E * (1 - l)
            K = L * (((1 - beta + delta * beta) / (AA * alpha * beta))
                     ** (1 / (alpha - 1)))
            Y = AA * (K**alpha) * (L**(1 - alpha))
            Fl = -E * AA * (1 - alpha) * (K**alpha) * (L**(-alpha))
            diff = du(Y - delta * K) * Fl + dul(l)
            return diff

        l = fsolve(fbfoc, .6)[0]
        Labour_Supply = E * (1 - l)
        L = E * (1 - l)
        Hours = 1 - l
        K = L * (((1 - beta + delta * beta) / (AA * alpha * beta))
                 ** (1 / (alpha - 1)))
        Y = AA * np.power(K, alpha) * np.power(L, 1 - alpha)
        r, w, fkk = K_to_rw(K, L)
        return r, K, Hours, Labour_Supply, Y, w

    def Omega_CE(Lambda_H, r):
        """Computes IM for given Lambda_H and r.

        Parameters
        ----------
        Lambda_H: float64
        r: float64

        Returns
        -------
        results: list

        Note
        ----

        Results list containts:

        CP_r: eqm. interest rate given Lambda_H and Lambda_E (given r)
        CP_w: eqm. wage rate given Lambda_H and Lambda_E (given r)
        CP_Lambda: eqm. Lambda_H calculated using eqm dist.
        CP_K: eqm. capital given Lambda_H and Lambda_E (given r)
        CP_L: eqm. effective labour given Lambda_H and Lambda_E (given r)
        CP_H: eqm. hours of work labour given Lambda_H and Lambda_E (given r)
        CP_coefvar
        CP_a:
        CP_z:
        CP_z_rlz:
        CP_h_val:
        CP_l_val:
        CP_policy
        """
        cap_lab_ratio = ((r + delta) / alpha)**(1 / (alpha - 1))
        Lambda_E = -(Lambda_H * cap_lab_ratio / beta)

        if Gamma_IM(- delta * .99, Lambda_E, Lambda_H)\
                * Gamma_IM((1 - beta) / beta, Lambda_E, Lambda_H) < 0:
            eqm_r_IM = brentq(
                Gamma_IM, -delta * .99, (1 - beta) / beta,
                args=(Lambda_E, Lambda_H),
                xtol=tol_brent, disp=False)[0]
        else:
            eqm_r_IM = r

        R = 1 + eqm_r_IM
        w = r_to_w(eqm_r_IM)
        CP_r, CP_w, CP_Lambda, CP_K, CP_L, CP_H, CP_coefvar, CP_a, CP_z_rlz,\
            CP_h_val, CP_l_val, CP_policy = compute_agg_prices(R, w, Lambda_E,
                                                               Lambda_H, social=1)

        return [CP_r, CP_w, CP_Lambda, CP_K, CP_L, CP_H, CP_coefvar,
                CP_a, CP_z_rlz, CP_h_val, CP_l_val, CP_policy]

    def compute_CEE(world, N_elite):
        """Calculate IM and CP allocations.

        Parameters
        ----------
        world: MPI Communicator class
                        Class with MPI nodes to distribute cross entropy
        N_elite: int
                        Number of elite draws

        Returns
        -------
        Results_IM: dict
        Results_CP: dict
        Results_CE: dict

        Note
        ----
        world.rank>=N_elite
        """

        if world.rank == 0:
            print("Solving IM on all ranks")

        else:
            pass

        # Calcuate IM on all ranks (probably inefficient)
        eqm_r_IM = brentq(Gamma_IM, - delta * .99, (1 - beta) / beta,
                          args=(0, 0), xtol=tol_brent)[0]
        R = 1 + eqm_r_IM
        w = r_to_w(eqm_r_IM)
        im_r, im_w, im_Lambda, im_K, im_L, im_H, im_coefvar, im_a,\
            im_z_rlz, im_h_val, im_l_val, im_policy \
            = compute_agg_prices(R, w, 0, 0, social=1)

        if world.rank == 0:
            print('IM calculated interest rate {}, hours are {},\
					labour supply is {}, k_supply is {}'.format(im_r * 100,
                                                 im_H, im_L, im_K))

        im_Y = KL_to_Y(im_K, im_L)
        im_gini_a = gini(im_a)
        im_gini_i = gini(im_z_rlz)

        IM_out = [im_r, im_w, im_Lambda, im_K, im_L, im_H, im_coefvar,
                  im_a, im_z_rlz, im_h_val, im_l_val, im_policy, im_Y,
                  im_gini_a, im_gini_i]

        IM_list = ['IM_r', 'IM_w', 'IM_Lambda',
                   'IM_K', 'IM_L', 'IM_H',
                   'IM_coefvar', 'IM_a', 'IM_z_rlz',
                   'IM_h_val', 'IM_l_val', 'IM_policy',
                   'IM_Y', 'IM_gini_a', 'IM_gini_i']

        results_IM = {}

        for var, name in zip(IM_out, IM_list):
            results_IM[name] = var

        world.Barrier()

        # set bounds on each cpu
        r_bounds = [- delta * .99, (1 - beta) / beta]
        l_bounds = [-.04, .04]

        # Initialise empty variables (do we need this?)
        i = 0
        mean_errors = 1
        mean_errors_all = 1

        # initial uniform draw
        r = np.random.uniform(r_bounds[0], r_bounds[1])
        Lambda_H = np.random.uniform(l_bounds[0], l_bounds[1])

        # Cross entropy
        while mean_errors > 1e-04 and i < max_iter_xe:

            # Empty array to fill with next iter vals
            rstats = np.empty(3, dtype=np.float64)
            lstats = np.empty(3, dtype=np.float64)
            cov_matrix = np.empty((2, 2), dtype=np.float64)

            # Calculate IM with r and Lambda_H draw
            R = 1 + r
            w = r_to_w(r)
            CP_r, CP_w, CP_Lambda, CP_K, CP_L, CP_H, CP_coefvar, CP_a,\
                CP_z_rlz, CP_h_val, CP_l_val, CP_policy = Omega_CE(Lambda_H, r)

            # Evaluate mean error
            error = np.mean(
                np.abs([(Lambda_H - CP_Lambda) / Lambda_H, (r - CP_r) / r]))

            if np.isnan(error):
                error = 1e100
                CP_Lambda = im_Lambda
                CP_r = im_r

            world.Barrier()

            # If error is small, then iterate on new market values
            if mean_errors < tol_contract:
                indexed_errors = world.gather(error, root=0)
                parameter_r = world.gather(CP_r, root=0)
                parameter_l = world.gather(CP_Lambda, root=0)
                parameter_K = world.gather(CP_K, root=0)
                parameter_H = world.gather(CP_H, root=0)
                parameter_L = world.gather(CP_L, root=0)

            else:
                indexed_errors = world.gather(error, root=0)
                parameter_r = world.gather(r, root=0)
                parameter_l = world.gather(Lambda_H, root=0)
                parameter_K = world.gather(CP_K, root=0)
                parameter_H = world.gather(CP_H, root=0)
                parameter_L = world.gather(CP_L, root=0)

            # Send to world
            if world.rank == 0:
                parameter_r_sorted = np.take(parameter_r,
                                             np.argsort(indexed_errors))
                parameter_l_sorted = np.take(parameter_l,
                                             np.argsort(indexed_errors))
                parameter_K_sorted = np.take(parameter_K,
                                             np.argsort(indexed_errors))
                parameter_H_sorted = np.take(parameter_H,
                                             np.argsort(indexed_errors))
                parameter_L_sorted = np.take(parameter_L,
                                             np.argsort(indexed_errors))
                indexed_errors_sorted = np.sort(indexed_errors)

                elite_errors = indexed_errors_sorted[0: N_elite]
                elite_r = parameter_r_sorted[0: N_elite]
                elite_l = parameter_l_sorted[0: N_elite]
                elite_K = parameter_K_sorted[0: N_elite]

                elite_vec = np.stack((elite_r, elite_l))

                cov_matrix = np.cov(elite_vec)
                rstats = np.array([np.mean(elite_r), np.std(
                    elite_r), np.std(parameter_r_sorted)], dtype=np.float64)
                lstats = np.array([np.mean(elite_l), np.std(
                    elite_l), np.std(parameter_l_sorted)], dtype=np.float64)
                mean_errors_all = np.mean(elite_errors)
                print('CE X-entropy iteration {}, mean interest rate {},\
						 mean lambda  {}, mean error {}, mean capital {},\
                        mean hours {} mean  labour {}'
                      .format(i, rstats[0], lstats[0], mean_errors_all,
                              np.mean(elite_K), np.mean(parameter_H_sorted),
                              np.mean(parameter_L_sorted)))
                print('Max covariance error is {}'.format(np.max(cov_matrix)))
            else:
                pass

            world.Barrier()
            world.Bcast(rstats, root=0)
            world.Bcast(lstats, root=0)
            world.Bcast(cov_matrix, root=0)
            world.bcast(mean_errors_all, root=0)

            draws = np.random.multivariate_normal(np.array([rstats[0],
                                                            lstats[0]]),
                                                  cov_matrix)
            r = eta_b * draws[0] + (1 - eta_b) * r
            Lambda_H = eta_b * draws[1] + (1 - eta_b) * Lambda_H
            mean_errors = mean_errors_all
            i += 1

        CP_Y = KL_to_Y(out[3], out[4])
        CP_gini_a = gini(out[7])
        CP_gini_i = gini(out[8])
        out.append([CP_Y, CP_gini_a, CP_gini_i])

        results_CP = {}

        CP_list = ['CP_r', 'CP_w',
                   'CP_Lambda',
                   'CP_K', 'CP_L',
                   'CP_H', 'CP_coefvar',
                   'CP_a', 'CP_z_rlz',
                   'CP_h_val', 'CP_l_val',
                   'CP_policy', 'CP_Y',
                   'CP_gini_a',
                   'CP_gini_i']

        for var, name in zip(out, CP_list):
            results_CP[name] = var

        # calculate counter-factual 1
        R = 1 + results_CP['CP_r']
        w = r_to_w(r)

        CF_r, CF_w, CF_Lambda, CF_K, CF_L, CF_H, CF_coefvar, CF_a, CF_z_rlz,\
            CF_h_val, CF_l_val, CF_policy = compute_agg_prices(
                R, w, 0, 0, social=1)
        CF_out = [CF_r, CF_w, CF_Lambda, CF_K, CF_L, CF_H, CF_coefvar, CF_a,
                  CF_z_rlz, CF_h_val, CF_l_val, CF_policy]

        print('CF calculated interest rate {}, hours are {},\
			 labour supply is {}, k_supply is {}'.format(CF_r * 100, CF_H,\
                                                     CF_L, CF_K))

        CF_Y = KL_to_Y(CF_out[3], CF_out[4])
        CF_gini_a = gini(CF_out[7])
        CF_gini_i = gini(CF_out[8])
        CF_out.append([CF_Y, CF_gini_a, CF_gini_i])
        results_CF = {}

        CF_list = ['CF_r', 'CF_w',
                   'CF_Lambda',
                   'CF_K', 'CF_L',
                   'CF_H', 'CF_coefvar',
                   'CF_a', 'CF_z_rlz',
                   'CF_h_val', 'CF_l_val',
                   'CF_policy', 'CF_Y',
                                'CF_gini_a',
                                'CF_gini_i']

        for var, name in zip(CF_out, CF_list):
            results_CF[name] = var

        return results_IM, results_CP, results_CF

    return compute_CEE, firstbest
