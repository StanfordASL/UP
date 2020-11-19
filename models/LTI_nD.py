import sys
sys.path.append('../src/utils/')

import matplotlib.pyplot as plt
from matplotlib import rc
import time

from utils.viz import *
from utils.stats import *

import numpy as np



# n-states, n-controls very simple linear system

class Model:
    n_x, y_dim = 4, 4
    n_u, u_dim = 4, 4
    n_params   = 0 # number of parameters (mass, inertia)

    # dynamics
    A = np.eye(n_x)
    B = np.eye(n_u)

    # parameters bounds
    u_mag = 0.5
    x0_min, x0_max = -np.ones(n_x),       np.ones(n_x)
    u_min,  u_max  = -u_mag*np.ones(n_u), u_mag*np.ones(n_u)

    # Monte-Carlo samples of parameters
    x0s_MC      = np.zeros((0,n_x)) 
    controls_MC = np.zeros((0,n_u)) 

    # gradient descent step size for adversarial sampling
    eta_x0s      = 1.
    eta_controls = 1

    def __init__(self):
        print('[LTI_nD::__init__] Initializing model.')
        self.reset()

    def reset(self):
        print('[LTI_nD::reset] resetting dynamics.')
        

    def f_dt(self, xk, uk):
        return (self.A @ xk + self.B*uk)
    def f_dt_batched(self, xs_k, us_k):
        """
        Inputs:  xs_k     : (N_MC, n_x)
                 us_k     : (N_MC, n_u)
        Outputs: xs_{k+1} : (N_MC, n_x)
        """
        fs = (self.A@(xs_k.T)).T + (self.B@(us_k.T)).T
        return fs

    def f_dt_dx(self):
        return self.A
    def f_dt_dx_batched(self, N_MC):
        """
        Inputs:  N_MC : ()
        Outputs: As_k   : (N_MC, n_x,n_x)
        """
        As = np.repeat(self.A[np.newaxis, :, :],  N_MC, axis=0) # (N_MC, n_x,n_x)
        return As
    def f_dt_du(self):
        return self.B
    def f_dt_du_batched(self, N_MC):
        """
        Inputs:  N_MC : ()
        Outputs: Bs_k   : (N_MC, n_x,n_x)
        """
        Bs = np.repeat(self.B[np.newaxis, :, :],  N_MC, axis=0) # (N_MC, n_x,n_x)
        return Bs

    def Xs_dparams_MC(self, k, X_nom, U_nom, Xs, Us):
        """
        Returns the Jacobian matrices of the state at time k, 
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

    def sample_initial_states(self, N_MC):
        self.x0s_MC = np.random.uniform(low=self.x0_min, high=self.x0_max, size=(N_MC, self.n_x)) # (N_MC, N, n_x)
        return self.x0s_MC

    def sample_controls(self, N_MC):
        self.controls_MC = np.random.uniform(low=self.u_min, high=self.u_max, size=(N_MC, self.n_u)) # (N_MC, N, n_x)
        return self.controls_MC

    def adv_sample_params(self, Xs):
        """
          resamples initial states and controls
          Xs - (N_MC, n_x)
        """
        N_MC = Xs.shape[0]


        Cs = np.mean(Xs,0) # (n_x, N)

        # TODO ADAPT COST
        # compute cost gradient
        Jdists_dXs = np.swapaxes(Xs-Cs,1,0) / np.linalg.norm(Xs-Cs, axis=1) # (6, 1000, 14)
        # print(Jdists_dXs.shape)

        # compute gradients w.r.t. parameters
        Xs_dx0s = self.f_dt_dx_batched(N_MC)
        Xs_dus  = self.f_dt_du_batched(N_MC)

        # print('Jdists_dXs=',Jdists_dXs.shape, 'Xs_dx0s=',Xs_dx0s.shape)
        Jdist_dx0s = np.einsum('xM,Mxy->My', Jdists_dXs, Xs_dx0s) 
        Jdist_dus  = np.einsum('xM,Mxu->Mu', Jdists_dXs, Xs_dus) 
        # print('Jdist_dus=',Jdist_dsu.shape)

        # gradient ascent
        x0s, us = self.x0s_MC, self.controls_MC
        new_x0s_MC      = x0s + self.eta_x0s      * Jdist_dx0s
        new_controls_MC = us +  self.eta_controls * Jdist_dus

        new_x0s_MC      = np.clip(new_x0s_MC,      self.x0_min, self.x0_max)
        new_controls_MC = np.clip(new_controls_MC, self.u_min,  self.u_max)


        # np.linalg.norm(ws_i-ws_i_worse)
        # new_x0s_MC = np.random.uniform(low=self.x0_min, high=self.x0_max, size=(N_MC, self.n_x))
        # new_controls_MC = np.random.uniform(low=self.u_min, high=self.u_max, size=(N_MC, self.n_u))

        return new_x0s_MC, new_controls_MC
    def simulate_batch(self, N_MC = 10, 
                             B_feedback = False, B_resample = True):
        """
        Inputs:  N_MC     : () nb of particles
                 B_feedback : if True, uses feedback (NOT IMPLEMENTED YET)
        Outputs: Xs_MC    : (N_MC, n_x)
        """
        if B_feedback:
            raise NotImplementedError('[LTI_nD::simulate_batch] Feedback not implemented yet.')

        if B_resample:
            self.sample_initial_states(N_MC)
            self.sample_controls(N_MC)

        # single simulations
        Xs = self.f_dt_batched(self.x0s_MC, self.controls_MC)

        return Xs










    # # RSS METHOD CODE
    def create_deltapacking_x0s(self, N_per_dim):
        delta = (self.x0_max-self.x0_min)/(int(N_per_dim)+1)

        x0s_grid                       = np.linspace(self.x0_min+delta, self.x0_max-delta, num=N_per_dim, endpoint=True)
        x0s_00, x0s_11, x0s_22, x0s_33 = np.meshgrid(x0s_grid[:,0],x0s_grid[:,1],x0s_grid[:,2],x0s_grid[:,3])
        x0s                            = np.array([x0s_00.ravel(), x0s_11.ravel(), x0s_22.ravel(), x0s_33.ravel()]).T

        return x0s
    def compute_volume_RSS_method(self, N_per_dim):
        N_per_dim = int(N_per_dim)

        vol_RSS = 0.
        # Check overlap
        B_overlap = False
        delta = (self.x0_max-self.x0_min)/(N_per_dim+1)
        if self.u_max[0]>delta[0]/2.: 
            B_overlap = True
        if not(B_overlap):
            print(N_per_dim,' per dim: no overlap')
            vol_RSS = (N_per_dim**self.n_x) * np.prod(self.u_max-self.u_min)
        else:
            print(N_per_dim,' per dim: overlap!')
            vol_RSS = np.prod( (self.x0_max-self.x0_min) -2.*delta + (self.u_max-self.u_min) )
        return vol_RSS