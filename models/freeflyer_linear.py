import sys
sys.path.append('../src/utils/')

import matplotlib.pyplot as plt
from matplotlib import rc
import time

from utils.viz import *
from utils.stats import *


import numpy as np

# 6-states, 3-controls freeflyer system
# Nonlinear dynamics with uncertain mass and inertia (linear uncertainty)

class Model:
    n_x, y_dim = 6, 6
    n_u, u_dim = 3, 3
    n_params   = 2 # number of parameters (mass, inertia)

    # Obstacle avoidance params
    robot_radius = 0.001#0.05
    nb_pos_dim   = 2  # number of positional dimensions for obs. avoid.

    # robot constants
    mass_nom,    J_nom    = 14.4, 0.15  # nominal value 
    mass_deltas, J_deltas = 2., 0.03    # param is in [p_nom - p_delta, p_nom + p_delta, ]

    masses_MC = np.zeros(0) # Monte-Carlo samples of parameters
    Js_MC     = np.zeros(0) 

    dt = 2.

    # Additive disturbances
    w_nom, w_deltas = np.zeros(n_x), np.array([1e-3,1e-3,1e-2,1e-3,1e-3,1e-2])


    def __init__(self):
        print('[freeflyer_linear::__init__] Initializing freeflyer Model \n'+
              '                             (linear, uncertain mass and inertia).')
        self.reset()

    def reset(self):
        print('[freeflyer_linear::reset] resetting mass, J, and deltas.')
        self.mass_nom,    self.J_nom    = 14.4, 0.15
        self.mass_deltas, self.J_deltas = 2., 0.03
        

    def f_dt(self, xk, uk, mass, J, wk):
        f = xk + self.dt * np.array([xk[3],
                                     xk[4],
                                     xk[5],
                                     (1./mass) * uk[0],
                                     (1./mass) * uk[1],
                                     (1./  J ) * uk[2]])
        f = f + wk
        return f
    def f_dt_batched(self, xs_k, us_k, masses, Js, ws_k):
        """
        Inputs:  xs_k     : (N_MC, n_x)
                 us_k     : (N_MC, n_u)
                 masses   : (N_MC,)
                 Js       : (N_MC,)
                 ws_k     : (N_MC, n_x)
        Outputs: xs_{k+1} : (N_MC, n_x)
        """
        fs = np.zeros_like(xs_k)
        fs[:, :3] = xs_k[:, :3] + self.dt * xs_k[:, 3:]        + ws_k[:,0:3]
        fs[:, 3]  = xs_k[:, 3]  + self.dt * us_k[:,0] / masses + ws_k[:,3]
        fs[:, 4]  = xs_k[:, 4]  + self.dt * us_k[:,1] / masses + ws_k[:,4]
        fs[:, 5]  = xs_k[:, 5]  + self.dt * us_k[:,2] / Js     + ws_k[:,5]
        return fs

    def f_dt_dx(self, xk, uk, mass, J, wk):
        A = np.eye(self.n_x)
        A[:3,3:] = self.dt*np.eye(3)
        return A
    def f_dt_dx_batched(self, xs_k, us_k, masses, Js, ws_k):
        """
        Inputs:  xs_k   : (N_MC, n_x)
                 us_k   : (N_MC, n_u)
                 masses : (N_MC,)
                 Js     : (N_MC,)
                 ws_k   : (N_MC, n_x)
        Outputs: As_k   : (N_MC, n_x,n_x)
        """
        N_MC = xs_k.shape[0]
        A           = np.repeat(np.eye(self.n_x)[np.newaxis, :, :],  N_MC, axis=0) # (N_MC, n_x,n_x)
        A[:, :3,3:] = np.repeat(self.dt*np.eye(3)[np.newaxis, :, :], N_MC, axis=0)
        return A

    def f_dt_dparams(self, xk, uk, mass, J, wk):
        """
        Outputs: f_dmass : (n_x, )
                 f_dJ    : (n_x, )
                 f_dwk   : (n_x, n_x)
        """
        f_dmass = self.dt * np.array([0.,0.,0.,
                                      -(1./mass**2)* uk[0],
                                      -(1./mass**2)* uk[1],
                                      0.])
        f_dJ    = self.dt * np.array([0.,0.,0.,0.,0.,
                                      -(1./J**2)* uk[2]])
        f_dwk = np.eye(self.n_x)
        return f_dmass, f_dJ, f_dwk
    def f_dt_dparams_batched(self, xs_k, us_k, masses, Js, wks):
        """
        Inputs:  xs_k   : (N_MC, n_x)
                 us_k   : (N_MC, n_u)
                 masses : (N_MC,)
                 Js     : (N_MC,)
                 ws_k   : (N_MC, n_x)
        Outputs: fs_dmass : (N_MC, n_x, )
                 fs_dJ    : (N_MC, n_x, )
                 fs_dwks  : (N_MC, n_x, n_x)
        """
        N_MC, n_x = len(masses), self.n_x

        fs_dmasses = np.zeros((N_MC, n_x))
        fs_dmasses[:, 3:5] = - self.dt * (us_k[:,0:2].T / (masses**2)).T

        fs_dJs = np.zeros((N_MC, n_x))
        fs_dJs[:, 5] = - self.dt * us_k[:,2] / (Js**2)

        fs_dwks = np.repeat(np.eye(n_x)[np.newaxis, :,:], N_MC, axis=0) # (N_MC, n_x, n_x)

        return fs_dmasses, fs_dJs, fs_dwks

    def Xs_dparams_MC(self, X_nom, U_nom, Xs, Us):
        """
        Returns the Jacobian matrices of the state at all times, 
        w.r.t. to all parameters
        Inputs:  k        : time id of particle to take derivative
                 X_nom    : (n_x,  N )  # currently unused, will be used with feedback 
                 U_nom    : (n_u, N-1)
                used for linearization
                 Xs       : (N_MC, n_x,  N )
                 Us       : (N_MC, n_u, N-1)
                 uses self.self.masses_MC, self.Js_MC, self.ws_MC
        Outputs: Xs_dmass : (N_MC, n_x, N)
                 Xs_dJ    : (N_MC, n_x, N)
                 Xs_dws   : (N_MC, n_x, N, N-1, n_x)
        """
        N, N_MC, n_x = X_nom.shape[1], len(self.masses_MC), self.n_x

        ms, Js, ws = self.masses_MC, self.Js_MC, self.ws_MC

        Xs_dmasses, Xs_dJs = np.zeros((N_MC, n_x,N)), np.zeros((N_MC, n_x, N))
        Xs_dws             = np.zeros((N_MC, n_x,N, ws.shape[1], ws.shape[2]))

        for j in range(N-1):
            # Jacobians w.r.t. uncertain parameters
            # fs_dmass : (N_MC, n_x)
            fjs_dms, fjs_dJs, fjs_dwks = self.f_dt_dparams_batched(Xs[:,:,j], Us[:,:,j], ms, Js, ws[:,j,:])
            # Jacobian w.r.t. state
            As_j = self.f_dt_dx_batched(Xs[:,:,j], Us[:,:,j], ms, Js, ws[:,j,:]) # (N_MC, n_x,n_x)

            # Parameters
            Xs_dmasses[:,:,j+1] = fjs_dms + np.einsum('Mxy,My->Mx', As_j, Xs_dmasses[:,:,j]) 
            Xs_dJs[:,:,j+1]     = fjs_dJs + np.einsum('Mxy,My->Mx', As_j, Xs_dJs[:,:,j]) 

            # Additive disturbances
            Xs_dws[:,:,j+1,    j  ,:] = fjs_dwks
            Xs_dws[:,:,j+1, :j,:] = np.einsum('Mxy,Myjw->Mxjw', As_j, Xs_dws[:,:,j, :j,:]) 


        return Xs_dmasses, Xs_dJs, Xs_dws


    def sample_dynamics_params(self, n_models):
        """
            Sample parameters of the uncertain model
        """
        min_mass, max_mass = self.mass_nom-self.mass_deltas, self.mass_nom+self.mass_deltas
        min_J,    max_J    = self.J_nom   -self.J_deltas,    self.J_nom   +self.J_deltas

        self.masses_MC = np.random.uniform(low=min_mass, high=max_mass, size=n_models) # (N_MC,)
        self.Js_MC     = np.random.uniform(low=min_J,    high=max_J,    size=n_models) # (N_MC,)

        return self.masses_MC, self.Js_MC

    def sample_disturbances(self, N, N_MC,
                                  beta_pdf=False):
        """
            Sample bounded disturbances ws_MC : (N_MC, N, n_x)
            N: time horizon
            N_MC: nb of particles
        """

        if not(beta_pdf):
            min_w, max_w = self.w_nom-self.w_deltas, self.w_nom+self.w_deltas
            self.ws_MC   = np.random.uniform(low=min_w, high=max_w, size=(N_MC, N, self.n_x)) # (N_MC, N, n_x)
        else:
            ws_nom = np.tile(self.w_nom,    (N_MC,N,1))
            ws_del = np.tile(self.w_deltas, (N_MC,N,1))
            vals_b = np.random.beta(0.05, 0.05, size=(N_MC, N, 6))

            self.ws_MC = (ws_nom-ws_del) + (2.0 * ws_del) * vals_b

        return self.ws_MC

    def simulate_batch(self, x_init, X_nom, U_nom,
                             N_MC = 10, 
                             B_feedback = False, B_resample = True,
                             B_beta_pdf_disturbances = False):
                             # ws=np.zeros((0,4)), masses=np.zeros(0), Js=np.zeros(0)):
        """
        Inputs:  x_init   : (n_x)
                 X_nom    : (n_x,  N )  # currently unused, will be used with feedback 
                 U_nom    : (n_u, N-1)
                 N_MC     : () nb of particles
                 B_feedback : if True, uses feedback (NOT IMPLEMENTED YET)
                 # masses   : (N_MC,)
                 # Js       : (N_MC,)
                 # ws       : (N_MC, N-1, n_x)
        Outputs: Xs_MC    : (N_MC, n_x,  N )
                 Us_MC    : (N_MC, n_u, N-1)
        """
        if B_feedback:
            raise NotImplementedError('[freeflyer_linear::simulate_batch] Feedback not implemented yet.')

        n_x, n_u, N = self.n_x, self.n_u, X_nom.shape[1]

        Xs, Us = np.zeros((N_MC, n_x, N)), np.zeros((N_MC, n_u, N-1))

        if B_resample:
            self.sample_dynamics_params(N_MC)   # populates self.masses_MC, self.Js_MC
            self.sample_disturbances(N-1, N_MC, # populates self.ws_MC
                                    beta_pdf=B_beta_pdf_disturbances)

        # multistep simulations
        Xs[:,:,0] = np.tile(x_init, (N_MC,1))
        for k in range(N-1):
            Us[:,:,k] = np.tile(U_nom[:,k], (N_MC,1))

            Xs[:,:,k+1] = self.f_dt_batched(Xs[:,:,k], Us[:,:,k], 
                                    self.masses_MC, self.Js_MC, self.ws_MC[:,k,:])

        return Xs, Us