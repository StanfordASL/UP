import sys
sys.path.append('../..')
# sys.path.append('../src/utils/')

import matplotlib.pyplot as plt
from matplotlib import rc
import time

# from utils.simulation import rollout_comparison, open_loop_rollout
from viz import *
from stats import *
# from freeflyer_plot import *


import numpy as np

# 13-states, 6-controls Astrobee system
# Nonlinear dynamics with uncertain mass and inertia (linear uncertainty)

class Model:
    n_x, y_dim = 13, 13
    n_u, u_dim = 6, 6
    n_params   = 4 # number of parameters (mass, inertia)

    # Obstacle avoidance params
    robot_radius = 0.001
    nb_pos_dim   = 3  # number of positional dimensions for obs. avoid.

    # robot constants
    mass_nom    = 7.2  # nominal value (J: for each diagonal element) 
    mass_deltas = 2.   # param is in [p_nom - p_delta, p_nom + p_delta, ]
    J_nom       =  0.07*np.eye(3)
    J_deltas    = 0.005*np.eye(3)

    masses_MC = np.zeros(0) # Monte-Carlo samples of parameters
    Js_MC     = np.zeros((0,3,3)) 

    dt = 2.43

    # Additive disturbances
    w_nom    = np.zeros(n_x)
    w_deltas = 0.5*np.sqrt(np.array([1e-7,1e-7,1e-7,3e-6,3e-6,3e-6, 1e-7,1e-7,1e-7,1e-7, 1e-7,1e-7,1e-7]))

    def __init__(self):
        print('[astrobee::__init__] Initializing freeflyer Model \n'+
              '                             (linear, uncertain mass and inertia).')
        self.reset()

    def reset(self):
        print('[astrobee::reset] resetting mass, J, and deltas.')
        self.mass_nom    = 7.2
        self.mass_deltas = 2. 
        self.J_nom       =  0.07*np.eye(3)
        self.J_deltas    = 0.005*np.eye(3)
        

    def f_dt(self, xk, uk, mass, J, wk):
        omega, M        = xk[10:13], uk[3:]
        qw, qx, qy, qz  = xk[6:10]
        wx, wy, wz      = xk[10:13]
        Jinv            = np.diag(1./np.diag(J)) # don't invert zero elements

        # continuous dynamics
        f      = np.zeros(self.n_x)
        f[:10] = np.array([xk[3],
                           xk[4],
                           xk[5],
                           (1./mass) * uk[0],
                           (1./mass) * uk[1],
                           (1./mass) * uk[2],
                           0.5*(-wx*qx-wy*qy-wz*qz),
                           0.5*( wx*qw-wz*qy+wy*qz),
                           0.5*( wy*qw+wz*qx-wx*qz),
                           0.5*( wz*qw-wy*qx+wx*qy)])
        f[10:] = Jinv@(M - np.cross( omega, J@omega ))
        # discretize
        f = xk + self.dt * f
        # add disturbances
        f = f + wk
        return f
    def f_dt_batched(self, xs_k, us_k, masses, Js, ws_k):
        """
        Inputs:  xs_k     : (N_MC, n_x)
                 us_k     : (N_MC, n_u)
                 masses   : (N_MC,)
                 Js       : (N_MC, 3,3)
                 ws_k     : (N_MC, n_x)
        Outputs: xs_{k+1} : (N_MC, n_x)
        """
        # extract values
        omega, M       = xs_k[:,10:13], us_k[:,3:]
        qw, qx, qy, qz = xs_k[:,6],xs_k[:,7],xs_k[:,8],xs_k[:,9]
        wx, wy, wz     = xs_k[:,10],xs_k[:,11],xs_k[:,12]
        Js_vec         = np.diagonal(Js, axis1=1, axis2=2) # (N_MC, 3)
        # skey symmetric matrix
        zeros_w = np.zeros_like(wx)
        S_w     = np.array([[zeros_w, -wz, wy],[wz, zeros_w, -wx],[-wy, wx, zeros_w]])

        # dynamics
        fs = np.zeros_like(xs_k)
        fs[:, 0:3] = xs_k[:, 3:6]
        fs[:, 3:6] = (us_k[:,0:3].T / masses).T
        fs[:, 6:10] = 0.5*np.array([-wx*qx-wy*qy-wz*qz,
                                     wx*qw-wz*qy+wy*qz,
                                     wy*qw+wz*qx-wx*qz,
                                     wz*qw-wy*qx+wx*qy]).T
        # f[10:] = Jinv@(M - np.cross( omega, J@omega ))
        Js_omega           = np.einsum('Mxy,My->Mx', Js, omega)
        omega_cross_Jomega = np.einsum('xyM,My->Mx', S_w, Js_omega) 
        fs[:, 10:]         = (M - omega_cross_Jomega) / Js_vec

        # discretize
        fs = xs_k + self.dt * fs
        # add disturbances
        fs = fs + ws_k
        return fs

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
        # extract values
        omega, M       = xs_k[:,10:13], us_k[:,3:]
        qw, qx, qy, qz = xs_k[:,6],xs_k[:,7],xs_k[:,8],xs_k[:,9]
        wx, wy, wz     = xs_k[:,10],xs_k[:,11],xs_k[:,12]
        Js_vec         = np.diagonal(Js, axis1=1, axis2=2) # (N_MC, 3)
        Jxx, Jyy, Jzz  = Js_vec[:,0], Js_vec[:,1], Js_vec[:,2]

        # (N_MC, n_x,n_x)
        A            = np.repeat(np.eye(self.n_x)[np.newaxis, :, :],  N_MC, axis=0) 
        A[:, :3,3:6] = np.repeat(self.dt*np.eye(3)[np.newaxis, :, :], N_MC, axis=0)

        zrs = np.zeros_like(wx)
        A[:, 6, 6:] = self.dt * 0.5 * np.array([zrs,-wx,-wy,-wz,-qx,-qy,-qz]).T
        A[:, 7, 6:] = self.dt * 0.5 * np.array([wx,zrs,-wz,wy,qw,qz,wy]).T
        A[:, 8, 6:] = self.dt * 0.5 * np.array([wy,wz,zrs,-wx,-qz,qw,qx]).T
        A[:, 9, 6:] = self.dt * 0.5 * np.array([wz,-wy,wx,zrs,qy,-qx,qw]).T
        A[:, 10,11] = self.dt * ( (Jyy-Jzz)*wz/Jxx)
        A[:, 10,12] = self.dt * ( (Jyy-Jzz)*wy/Jxx)
        A[:, 11,10] = self.dt * (-(Jxx-Jzz)*wz/Jyy)
        A[:, 11,12] = self.dt * (-(Jxx-Jzz)*wx/Jyy)
        A[:, 12,10] = self.dt * ( (Jxx-Jyy)*wy/Jzz)
        A[:, 12,11] = self.dt * ( (Jxx-Jyy)*wx/Jzz)

        return A

    def f_dt_dparams_batched(self, xs_k, us_k, masses, Js, wks):
        """
        Inputs:  xs_k   : (N_MC, n_x)
                 us_k   : (N_MC, n_u)
                 masses : (N_MC,)
                 Js     : (N_MC,)
                 ws_k   : (N_MC, n_x)
        Outputs: fs_dmass : (N_MC, n_x, )
                 fs_dJ    : (N_MC, n_x, 3)
                 fs_dwks  : (N_MC, n_x, n_x)
        """
        N_MC, n_x = len(masses), self.n_x
        Js_vec    = np.diagonal(Js, axis1=1, axis2=2) # (N_MC, 3)

        omega, M   = xs_k[:,10:13], us_k[:,3:]
        wx, wy, wz = xs_k[:,10],xs_k[:,11],xs_k[:,12]
        zeros_w    = np.zeros_like(wx)
        S_w        = np.array([[zeros_w, -wz, wy],[wz, zeros_w, -wx],[-wy, wx, zeros_w]])

        # Parameters
        fs_dmasses = np.zeros((N_MC, n_x))
        fs_dmasses[:, 3:6] = - self.dt * (us_k[:,0:3].T / (masses**2)).T

        fs_dJs = np.zeros((N_MC, n_x, 3))
        # Note: fs[:, 10:] = (M - omega_cross_Jomega) / Js_vec
        # part 1)
        Js_omega           = np.einsum('Mxy,My->Mx', Js, omega)
        omega_cross_Jomega = np.einsum('xyM,My->Mx', S_w, Js_omega) 
        fs_dJs[:, 10, 0]   = self.dt*((M-omega_cross_Jomega)[:,0] / (-Js_vec[:,0]**2))
        fs_dJs[:, 11, 1]   = self.dt*((M-omega_cross_Jomega)[:,1] / (-Js_vec[:,1]**2))
        fs_dJs[:, 12, 2]   = self.dt*((M-omega_cross_Jomega)[:,2] / (-Js_vec[:,2]**2))
        # part 2)
        Sw_Jw_dJ0 = S_w[:,0,:]*omega[:,0]
        Sw_Jw_dJ1 = S_w[:,1,:]*omega[:,1]
        Sw_Jw_dJ2 = S_w[:,2,:]*omega[:,2]
        fs_dJs[:, 10:, 0] += self.dt * (- Sw_Jw_dJ0.T / Js_vec)
        fs_dJs[:, 10:, 1] += self.dt * (- Sw_Jw_dJ1.T / Js_vec)
        fs_dJs[:, 10:, 2] += self.dt * (- Sw_Jw_dJ2.T / Js_vec)

        # Disturbances: additive
        fs_dwks = np.repeat(np.eye(n_x)[np.newaxis, :,:], N_MC, axis=0) # (N_MC, n_x, n_x)

        return fs_dmasses, fs_dJs, fs_dwks

    def Xs_dparams_MC(self, Xs, Us, X_nom, U_nom):
        """
        Returns the Jacobian matrices of the state TRAJECTORY
        w.r.t. to all parameters
        Inputs:  k        : time id of particle to take derivative
                 X_nom    : (n_x,  N )  # currently unused, will be used with feedback 
                 U_nom    : (n_u, N-1)
                used for linearization
                 Xs       : (N_MC, n_x,  N )
                 Us       : (N_MC, n_u, N-1)
                 uses self.self.masses_MC, self.Js_MC, self.ws_MC
        Outputs: Xs_dmass : (N_MC, n_x, N)
                 Xs_dJ    : (N_MC, n_x, N, J)
                 Xs_dws   : (N_MC, n_x, N, N-1, n_x)
        """
        N, N_MC, n_x = Xs.shape[2], len(self.masses_MC), self.n_x
        ms, Js, ws   = self.masses_MC, self.Js_MC, self.ws_MC

        Xs_dmasses, Xs_dJs = np.zeros((N_MC, n_x,N)), np.zeros((N_MC, n_x, N, 3))
        Xs_dws             = np.zeros((N_MC, n_x,N, ws.shape[1], ws.shape[2]))
        for j in range(N-1):
            # Jacobians w.r.t. uncertain parameters
            # fs_dmass : (N_MC, n_x)
            fjs_dms, fjs_dJs, fjs_dwks = self.f_dt_dparams_batched(Xs[:,:,j], Us[:,:,j], ms, Js, ws[:,j,:])
            # Jacobian w.r.t. state
            As_j = self.f_dt_dx_batched(Xs[:,:,j], Us[:,:,j], ms, Js, ws[:,j,:]) # (N_MC, n_x,n_x)

            # Parameters
            Xs_dmasses[:,:,j+1] = fjs_dms + np.einsum('Mxy,My->Mx',   As_j, Xs_dmasses[:,:,j]) 
            Xs_dJs[:,:,j+1]     = fjs_dJs + np.einsum('Mxy,MyJ->MxJ', As_j, Xs_dJs[:,:,j]) 

            # Additive disturbances
            Xs_dws[:,:,j+1,  j,:] = fjs_dwks
            Xs_dws[:,:,j+1, :j,:] = np.einsum('Mxy,Myjw->Mxjw', As_j, Xs_dws[:,:,j, :j,:]) 


        return Xs_dmasses, Xs_dJs, Xs_dws


    def sample_dynamics_params(self, n_models):
        """
            Sample parameters of the uncertain model
        """
        min_mass, max_mass = self.mass_nom-self.mass_deltas, self.mass_nom+self.mass_deltas
        min_J,    max_J    = self.J_nom   -self.J_deltas,    self.J_nom   +self.J_deltas

        self.masses_MC    = np.random.uniform(low=min_mass, high=max_mass, size=n_models) # (N_MC,)
        self.Js_MC        = np.zeros((n_models,3,3))
        self.Js_MC[:,0,0] = np.random.uniform(low=min_J[0,0], high=max_J[0,0], size=n_models)
        self.Js_MC[:,1,1] = np.random.uniform(low=min_J[1,1], high=max_J[1,1], size=n_models)
        self.Js_MC[:,2,2] = np.random.uniform(low=min_J[2,2], high=max_J[2,2], size=n_models)

        return self.masses_MC, self.Js_MC

    def sample_disturbances(self, N, N_MC):
        """
            Sample bounded disturbances ws_MC : (N_MC, N, n_x)
            N: time horizon
            N_MC: nb of particles
        """
        min_w, max_w = self.w_nom-self.w_deltas, self.w_nom+self.w_deltas

        self.ws_MC = np.random.uniform(low=min_w, high=max_w, size=(N_MC, N, self.n_x)) # (N_MC, N, n_x)

        return self.ws_MC

    def simulate_batch(self, x_init, X_nom, U_nom,
                             N_MC = 10, 
                             B_feedback = False, B_resample = True):
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
            raise NotImplementedError('[astrobee::simulate_batch] Feedback not implemented yet.')

        n_x, n_u, N = self.n_x, self.n_u, X_nom.shape[1]

        Xs, Us = np.zeros((N_MC, n_x, N)), np.zeros((N_MC, n_u, N-1))

        if B_resample:
            self.sample_dynamics_params(N_MC)   # populates self.masses_MC, self.Js_MC
            self.sample_disturbances(N-1, N_MC) # populates self.ws_MC

        # multistep simulations
        Xs[:,:,0] = np.tile(x_init, (N_MC,1))
        for k in range(N-1):
            Us[:,:,k] = np.tile(U_nom[:,k], (N_MC,1))

            Xs[:,:,k+1] = self.f_dt_batched(Xs[:,:,k], Us[:,:,k], 
                                    self.masses_MC, self.Js_MC, self.ws_MC[:,k,:])

        return Xs, Us

    def adv_sample_params(self, Xs, Us):
        """
          resamples parameters  self.masses_MC 
                                self.Js_MC 
                                self.ws_MC
                using           Xs            - (N_MC, n_x, T)
        """
        N_MC, T, x_dim = Xs.shape[0], Xs.shape[2], Xs.shape[1]
        ms, Js, ws     = self.masses_MC, self.Js_MC, self.ws_MC
        eta_w, eta_m, eta_J = 1e-2, 1e-1, 1e-1 # for traj opt
        # eta_w, eta_m, eta_J = 1e-6, 1e-6, 1e-6 # for plotting

        Cs = np.mean(Xs,0) # (n_x, T)
        Qs = np.zeros((T,x_dim,x_dim))
        for t in range(1,T): 
            Qs[t,:,:] = np.linalg.inv(np.cov(Xs[:,:,t].T))
            
        # compute cost gradient
        Jdists_dXs = np.einsum('txy,Myt->Mtx', 2*Qs, Xs-Cs) 

        # compute trajectory gradient w.r.t. parameters
        Xs_dms, Xs_dJs, Xs_dws = self.Xs_dparams_MC(Xs, Us, [], [])

        # compute cost gradient w.r.t params (average over horizon)
        Jdists_dms = np.mean(np.einsum('MTx,MxT->MT',     Jdists_dXs, Xs_dms), axis=1)
        Jdists_dJs = np.mean(np.einsum('MTx,MxTJ->MTJ',   Jdists_dXs, Xs_dJs), axis=1)
        Jdists_dws = np.mean(np.einsum('MTx,MxTtw->MTtw', Jdists_dXs, Xs_dws), axis=1)

        # gradient ascent
        ms      = self.masses_MC  + eta_m * Jdists_dms
        Js[:,0,0] = self.Js_MC[:,0,0] + eta_J * Jdists_dJs[:,0]
        Js[:,1,1] = self.Js_MC[:,1,1] + eta_J * Jdists_dJs[:,1]
        Js[:,2,2] = self.Js_MC[:,2,2] + eta_J * Jdists_dJs[:,2]
        ws      = self.ws_MC      + eta_w * Jdists_dws
     
        # clip
        ms = np.clip(ms, self.mass_nom-self.mass_deltas, self.mass_nom+self.mass_deltas)
        Js[:,0,0] = np.clip(Js[:,0,0], self.J_nom[0,0]-self.J_deltas[0,0], self.J_nom[0,0]+self.J_deltas[0,0])  
        Js[:,1,1] = np.clip(Js[:,1,1], self.J_nom[1,1]-self.J_deltas[1,1], self.J_nom[1,1]+self.J_deltas[1,1])    
        Js[:,2,2] = np.clip(Js[:,2,2], self.J_nom[2,2]-self.J_deltas[2,2], self.J_nom[2,2]+self.J_deltas[2,2])    
        for i in range(x_dim):
            ws[:,:,i] = np.clip(ws[:,:,i], self.w_nom[i]-self.w_deltas[i], 
                                           self.w_nom[i]+self.w_deltas[i])

        self.masses_MC, self.Js_MC, self.ws_MC = ms, Js, ws
        return ms, Js, ws