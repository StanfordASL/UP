import sys
sys.path.append('../..')

from utils.viz import *
from utils.stats import *

import numpy as np

# 13-states, 6-controls spacecraft system
# Nonlinear dynamics with uncertain mass and inertia (linear uncertainty)

class Model:
    n_x, y_dim = 13, 13
    n_u, u_dim = 6, 6
    n_params   = 4 # number of parameters (mass, inertia)

    # Obstacle avoidance params
    robot_radius = 0.05
    nb_pos_dim   = 2  # number of positional dimensions for obs. avoid.

    # robot constants
    mass_nom    = 7.2  # nominal value (J: for each diagonal element) 
    J_nom       = 0.07*np.eye(3)
    mass_deltas = 0.1   # param is in [p_nom - p_delta, p_nom + p_delta, ]
    J_deltas    = 0.005*np.eye(3)

    masses_MC = np.zeros(0) # Monte-Carlo samples of parameters
    Js_MC     = np.zeros((0,3,3)) 

    dt = 5#2.43

    # B_UP_method = 'randUP'
    B_UP_method = 'robUP' 

    # Additive disturbances
    w_nom    = np.zeros(n_x)
    w_deltas = 5e-1*0.5*np.sqrt(np.array([1e-7,1e-7,1e-7,3e-6,3e-6,3e-6, 1e-7,1e-7,1e-7,1e-7, 1e-7,1e-7,1e-7]))

    # Control costs
    quadratic_cost_matrix_controls = 10.*np.eye(n_u)
    quadratic_cost_matrix_state    = np.zeros((n_x,n_x))

    def __init__(self):
        print('[spacecraft::__init__] Initializing spacecraft Model \n'+
              '                       (linear, uncertain mass and inertia).')
        self.reset()

    def reset(self):
        print('[spacecraft::reset] resetting mass, J, and deltas.')
        # self.mass_nom    = 7.2
        # self.mass_deltas = 2. 
        # self.J_nom       =  0.07*np.eye(3)
        # self.J_deltas    = 0.005*np.eye(3)
        
    def get_quadratic_costs(self):
        R  = self.quadratic_cost_matrix_controls
        Q  = self.quadratic_cost_matrix_state
        QN = np.zeros([self.n_x,self.n_x])
        return Q, QN, R

    def predict_mean(self, x_k, u_k):
        """ 
            In discrete time, for one timestep k. 
            f() denotes dynamics:   x_{k+1} = f(x_k, u_k)

            Returns f(x_k, u_k) at state-control tuple (x_k, u_k)          
        """
        return self.fnom_dt(x_k, u_k)

    def predict_mean_linearized(self, x_k, u_k):
        """ 
            In discrete time, for one timestep k. 
            f() denotes dynamics:   x_{k+1} = f(x_k, u_k)

            Returns the Jacobians of f(.,.) => (df/dx, df/du). 
            linearized around a state-control tuple (x_k, u_k)          
        """
        return self.fnom_dt_dx(x_k, u_k), self.fnom_dt_du(x_k, u_k)

    def predict_confsets_monteCarlo(self, Xnom, Unom, A_all, B_all,
                                          shape="ellipsoidal", # "rectangular"
                                          N_MC=10, prob=0.9,
                                          B_feedback=False,
                                          B_reuse_presampled_dynamics=False):
        """ Computes confidence sets using randUP. (frequentist interpretation)

            Inputs/Outputs: see description in [::predict_credsets_monteCarlo]
        """
        n_x, n_u, N = self.y_dim, self.u_dim, Xnom.shape[1]

        if B_feedback: 
            raise NotImplementedError('[spacecraft::predict_credsets_monteCarlo] ' + 
                                'Feedback not supported yet.')

        # First, predict mean to ensure that feedback is accurate (useful for 1st SCP iter)
        Xmean      = np.zeros((N,n_x))
        Xmean[0,:] = Xnom[:,0]
        for k in range(0,N-1):
            Xmean[k+1,:] = self.predict_mean(Xmean[k,:], Unom[:,k])

        # (randUP) / (robUP!)
        parts = np.zeros((N_MC, n_x+n_u, N))
        Xs, Us = self.simulate_batch(Xnom[:,0], Xmean.T, Unom,
                                    N_MC      = N_MC,
                                    B_feedback= B_feedback, 
                                    B_resample= True)
        if self.B_UP_method=='robUP':
            if N_MC>200:
                raise ValueError('[spacecraft::predict_credsets_monteCarlo] ' + 
                           'N_MC (',N_MC,') is set too high.')
            self.adv_sample_params(Xs, Us)
            Xs, Us = self.simulate_batch(Xnom[:,0], Xmean.T, Unom,
                                    N_MC      = N_MC,
                                    B_feedback= B_feedback, 
                                    B_resample= False)
        elif self.B_UP_method=='randUP':
            pass
        else:   
            raise NotImplementedError('[spacecraft::predict_credsets_monteCarlo] ' + 
                                      'Unknown uncertainty prop method.')


        parts[:,:n_x,:]      = Xs 
        parts[:,n_x:,:(N-1)] = np.repeat(Unom[np.newaxis,:], N_MC, axis=0)

        if shape == "ellipsoidal":
            raise NotImplementedError('[spacecraft::predict_credsets_monteCarlo] ' + 
                                      'Ellipsoidal shapes not implemented.')
        elif shape == "rectangular":
            Deltas       = np.zeros((n_x, N))
            Deltas[:, 0] = np.zeros(n_x)
            for k in range(1,N):
                mean        = Xmean[k,:]
                deltas      = np.repeat(mean[np.newaxis,:], N_MC, axis=0) - parts[:,:n_x,k]
                Deltas[:,k] = np.max(deltas, 0)
            # _, stds_dxu  = self.propagate_standard_deviations(Xnom, Unom, A_all, B_all, B_feedback=B_feedback)
            # Deltas_dxu   = np.sqrt(p_th_quantile_chi_squared(prob, n_x)) * stds_dxu
            # print('[spacecraft::predict_confsets_monteCarlo] Deltas=',np.linalg.norm(Deltas))
            QDs     = Deltas
            QDs_dxu = np.zeros((n_x, N,n_x+n_u, N))
        else:
            raise NotImplementedError('[spacecraft::predict_credsets_monteCarlo] ' + 
                                      'Unknown UP shape.')
        return QDs, QDs_dxu, parts

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
    def fnom_dt(self, x_k, u_k):
        return self.f_dt(x_k, u_k, 
                         self.mass_nom, self.J_nom, np.zeros(self.y_dim))
    def fnom_dt_dx(self, x_k, u_k):
        omega, M        = x_k[10:13], u_k[3:]
        qw, qx, qy, qz  = x_k[6:10]
        wx, wy, wz      = x_k[10:13]
        Jxx, Jyy, Jzz   = np.diag(self.J_nom)

        A = np.zeros((self.y_dim, self.y_dim))

        A[0:3,3:6] = np.eye((3))

        A[6,7] = -wx/2
        A[6,8] = -wy/2
        A[6,9] = -wz/2
        A[6,10] = -qx/2
        A[6,11] = -qy/2
        A[6,12] = -qz/2

        A[7,6] = wx/2
        A[7,8] = -wz/2
        A[7,9] = wy/2
        A[7,10] = qw/2
        A[7,11] = qz/2
        A[7,12] = -qy/2

        A[8,6]  = wy/2
        A[8,7]  = wz/2
        A[8,9] = -wx/2
        A[8,10] = -qz/2
        A[8,11] = qw/2
        A[8,12] = qx/2

        A[9,6]  = wz/2
        A[9,7]  = -wy/2
        A[9,8]  = wx/2
        A[9,10] = qy/2
        A[9,11] = -qx/2
        A[9,12] = qw/2

        A[10,11] =  (Jyy-Jzz)*wz/Jxx
        A[10,12] =  (Jyy-Jzz)*wy/Jxx
        A[11,10] = -(Jxx-Jzz)*wz/Jyy
        A[11,12] = -(Jxx-Jzz)*wx/Jyy
        A[12,10] =  (Jxx-Jyy)*wy/Jzz
        A[12,11] =  (Jxx-Jyy)*wx/Jzz

        return (np.eye(self.y_dim) + self.dt * A)
    def fnom_dt_du(self, x_k, u_k):
        m, J, Jinv  = self.mass_nom, self.J_nom, np.diag(1./np.diag(self.J_nom))

        B            = np.zeros((self.y_dim, self.u_dim))
        B[3:6,0:3]   = (1/m) * np.eye(3)
        B[10:13,3:6] =  Jinv @ np.eye(3)

        return (self.dt*B)

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
        Inputs:  X_nom    : (n_x,  N )  # currently unused, will be used with feedback 
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
                             B_feedback = False, 
                             B_resample = True):
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
            raise NotImplementedError('[spacecraft::simulate_batch] Feedback not implemented yet.')

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
        eta_w, eta_m, eta_J = 1e-2, 1e-1, 1e-1

        Cs = np.mean(Xs,0) # (n_x, T)
        Qs = np.zeros((T,x_dim,x_dim))
        for t in range(1,T): 
            # print('np.linalg.inv(np.cov(Xs[:,:,t].T))=',np.linalg.inv(np.cov(Xs[:,:,t].T)))
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
        ms        = self.masses_MC    + eta_m * Jdists_dms
        Js[:,0,0] = self.Js_MC[:,0,0] + eta_J * Jdists_dJs[:,0]
        Js[:,1,1] = self.Js_MC[:,1,1] + eta_J * Jdists_dJs[:,1]
        Js[:,2,2] = self.Js_MC[:,2,2] + eta_J * Jdists_dJs[:,2]
        ws        = self.ws_MC        + eta_w * Jdists_dws
     
        # clip
        ms        = np.clip(ms,        self.mass_nom-self.mass_deltas,     self.mass_nom+self.mass_deltas)
        Js[:,0,0] = np.clip(Js[:,0,0], self.J_nom[0,0]-self.J_deltas[0,0], self.J_nom[0,0]+self.J_deltas[0,0])  
        Js[:,1,1] = np.clip(Js[:,1,1], self.J_nom[1,1]-self.J_deltas[1,1], self.J_nom[1,1]+self.J_deltas[1,1])    
        Js[:,2,2] = np.clip(Js[:,2,2], self.J_nom[2,2]-self.J_deltas[2,2], self.J_nom[2,2]+self.J_deltas[2,2])    
        for i in range(x_dim):
            ws[:,:,i] = np.clip(ws[:,:,i], self.w_nom[i]-self.w_deltas[i], 
                                           self.w_nom[i]+self.w_deltas[i])

        self.masses_MC, self.Js_MC, self.ws_MC = ms, Js, ws
        return ms, Js, ws













class PlanningProblem:
    """
    object that represents a planning problem
        X_goal = ball [pos, rads] centered at pos=[xdim] and of radius rads=[xdim],
    """
    def __init__(self, x_init, x_goal, Q0, N, 
                       xmin, xmax, umin, umax):
        self.x_min = xmin
        self.x_max = xmax
        self.u_min = umin
        self.u_max = umax

        self.x_init = x_init
        self.x_goal = x_goal

        # self.sphere_obstacles
        # self.poly_obstacles

        # depends on time discretization of model, as it directly
        # defines the total time to get from x_init to x_final
        self.N = N 

        # Q0: shape matrix of ellipsoid B containing x_0 with probability at least p, 
        #   where B = { x_0 | (x_0-mu_0)^T Q_0^{-1} (x_0-mu_0) < 1 }
        assert(Q0.ndim==2 and Q0.shape[0]==self.x_min.shape[0] and Q0.shape[1]==self.x_min.shape[0])
        self.Q0 = Q0
        pass
class SpacecraftProblem(PlanningProblem):
    # Goal is to get to X_goal, not a particular point.
    def __init__(self, x0, xgoal, N):
        n_x = x0.shape[0]

        # constraints
        mass, J_norm = 7.2, 0.07
        hard_limit_vel   = 0.4          # m/s
        hard_limit_accel = 0.005          # m/s^2
        hard_limit_omega = 45*3.14/180. # rad/s
        hard_limit_alpha = 50*3.14/180. # rad/s^2

        # constraints / limits
        x_max = np.array([100.,100.,100., hard_limit_vel,hard_limit_vel,hard_limit_vel,   100.,100.,100.,100.,   hard_limit_omega,hard_limit_omega,hard_limit_omega])
        u_max = np.array([mass*hard_limit_accel,mass*hard_limit_accel,mass*hard_limit_accel,  J_norm*hard_limit_alpha,J_norm*hard_limit_alpha,J_norm*hard_limit_alpha])
        x_min = np.negative(x_max)
        u_min = np.negative(u_max)

        # ---------------------
        # - ISS problem -------
        # s13     = np.sqrt(1./3.)
        # x_init  = np.array([10.,4.0,4.5,  1e-4,1e-4,1e-4,  -0.5,0.5,-0.5,0.5,  0,0,0])
        # x_goal  = np.array([10.,0.0,5.0,  1e-4,1e-4,1e-4,  s13,0.,s13,s13,     0,0,0])
        # cylindrical obstacles [(x,y),r]
        # self.sphere_obstacles = [
        #     # [[0,0,0.], 0.001],
        #     # [[11.3,3.8,4.8], 0.3],
        #     # [[10.5,5.5,5.5], 0.4],
        #     # [[8.5,0.8,5.5], 0.4],
        # ]
        # polytopic obstacles
        # self.poly_obstacles = []
        # print('Initializing the ISS.')
        # self.keepin_zones, self.keepout_zones = get_ISS_zones()
        # self.poly_obstacles = self.keepout_zones
        # additional obstacles
        # center, width = np.array([10.8,0.,5.]), 0.85*np.ones(3)
        # self.poly_obstacles.append(PolyObs(center,width))
        # center, width = np.array([11.2,1.75,4.85]), np.array([0.5,0.6,0.65])
        # self.poly_obstacles.append(PolyObs(center,width))
        # ---------------------

        # -------------------------------------------
        # Planning problem with 2 spherical obstacles
        print('Initializing problem with 2 spherical obstacles.')
        self.sphere_obstacles = [
            [[1.25,1.0,0.],0.4], 
            [[0.75,3.0,0.],0.4], 
        ]
        # polytopic obstacles
        self.poly_obstacles = []
        # -------------------------------------------


        print("[SpacecraftProblem::__init__] "+
                str(len(self.sphere_obstacles)) + " sphere obs, and "+
                str(len(self.poly_obstacles))   + " poly obs.")


        Q0 = 1e-11*np.eye(n_x)

        # Xgoal  : Goal set [pos, deltas] centered at pos=[xdim] and of half width deltas=[xdim],
        # X0safe : Safe set [pos, deltas] centered at pos=[xdim] and of half width deltas=[xdim],
        #               Xi = { x | -deltas <= (x-pos) <= deltas }, where x\in\R^{xdim}
        half_widths = 1e-3 * np.ones(n_x)
        X_safe = [x0,    half_widths]
        X_goal = [xgoal, half_widths]
        self.X_safe           = X_safe 
        self.X_goal           = X_goal
        self.B_go_to_safe_set = False # go to X_goal

        super().__init__(x_init=x0, x_goal=[], Q0=Q0, N=N, 
                         xmin=x_min, xmax=x_max, umin=u_min, umax=u_max)



class SpacecraftSimulator:
    def __init__(self, dt=5., 
                        mass_nom=7.2,
                        J_nom=0.07*np.eye(3),
                        mass_deltas=0.1,
                        J_deltas=0.005*np.eye(3),
                        w_nom=np.zeros(13),
                        w_deltas=5e-1*0.5*np.sqrt(np.array([1e-7,1e-7,1e-7,3e-6,3e-6,3e-6, 1e-7,1e-7,1e-7,1e-7, 1e-7,1e-7,1e-7]))
                        ):
        self.n_x, self.y_dim = 13, 13
        self.n_u, self.u_dim = 6, 6
        self.n_params        = 4 # number of parameters (mass, inertia)

        self.dt = dt

        # robot constants
        self.mass_nom    = mass_nom   
        self.J_nom       = J_nom      
        self.mass_deltas = mass_deltas
        self.J_deltas    = J_deltas   
        # Additive disturbances
        self.w_nom    = w_nom
        self.w_deltas = w_deltas

        # masses_MC = np.zeros(0) # Monte-Carlo samples of parameters
        # Js_MC     = np.zeros((0,3,3)) 

        # initial state
        self.x0    = np.array([0.,0.,0., 0,0,0,  0.,0.,0., 1.,  0,0,0])
        self.state = np.zeros(self.n_x)

        # states limits for sampling
        max_pos   = -15.
        max_angle = 5.          # in [rad]
        lim_vel   = 0.40        # max linear velocity
        lim_omega = 10*3.14/180 # max angular velocity
        self.states_min = ([-max_pos, -max_pos, -max_pos,    # position
                            -lim_vel, -lim_vel, -lim_vel,    # velocity
                            -1, -1, -1, -1,                  # quaternions
                            -lim_omega,-lim_omega,-lim_omega # body rate
                            ])
        self.states_max = ([max_pos, max_pos, max_pos,    # position
                            lim_vel, lim_vel, lim_vel,    # velocity
                            1, 1, 1, 1,                   # quaternions
                            lim_omega,lim_omega,lim_omega # body rate             
                            ])
        # controls limits for sampling
        mass_max    = 13.
        inertia_max = 0.2
        limit_accel = 0.1
        limit_alpha = 30*3.14/180
        F_max       = mass_max   *limit_accel # force max
        M_max       = inertia_max*limit_alpha # moment/torque max
        self.control_min = [-F_max, -F_max, -F_max, -M_max, -M_max, -M_max]
        self.control_max = [ F_max,  F_max,  F_max,  M_max,  M_max,  M_max]
        self.control_diff_min = [0.1*self.control_min[0],0.1*self.control_min[1],0.1*self.control_min[2],0.1*self.control_min[3],0.1*self.control_min[4],0.1*self.control_min[5]]
        self.control_diff_max = [0.1*self.control_max[0],0.1*self.control_max[1],0.1*self.control_max[2],0.1*self.control_max[3],0.1*self.control_max[4],0.1*self.control_max[5]]


    def reset_state(self):
        self.state = self.x0.copy()

    def f_ct(self, x, u):
        m, J     = self.mass_nom, self.J_nom
        J_mat    = J   * np.eye(3)
        Jinv_mat = 1/J[1,1] * np.eye(3)

        r, v, w         = x[0:3], x[3:6], x[10:13]
        qw, qx, qy, qz  = x[6:10]
        wx, wy, wz      = x[10:13]
        F, M            = u[0:3], u[3:6]

        f    = np.zeros(self.n_x)
        f[0] = x[3]
        f[1] = x[4]
        f[2] = x[5]
        f[3] = (1./m) * u[0]
        f[4] = (1./m) * u[1]
        f[5] = (1./m) * u[2]   
        f[6] = 1/2*(-wx*qx - wy*qy - wz*qz)
        f[7] = 1/2*( wx*qw - wz*qy + wy*qz)
        f[8] = 1/2*( wy*qw + wz*qx - wx*qz)
        f[9] = 1/2*( wz*qw - wy*qx + wx*qy)

        f[10:13] = Jinv_mat@(M[:] - np.cross(w[:],(J_mat@w)[:]))

        return f

    def f_dt(self, x_k, u_k):
        x_next = x_k + self.dt * self.f_ct(x_k, u_k)
        return x_next


    def sample_unit_quaternions_wxyz(self, n_samples=1):
        # https://www.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf
        s = np.random.uniform(low=0, high=1, size=n_samples)
        sig_1 = np.sqrt(1-s)
        sig_2 = np.sqrt(s)
        theta_1 = 2.*np.pi*np.random.uniform(low=0, high=1, size=n_samples)
        theta_2 = 2.*np.pi*np.random.uniform(low=0, high=1, size=n_samples)
        w = np.cos(theta_2) * sig_2
        x = np.sin(theta_1) * sig_1
        y = np.cos(theta_1) * sig_1
        z = np.sin(theta_2) * sig_2
        quaternions_wxyz = np.array((w,x,y,z)).T # [nb_samples x 4]
        return quaternions_wxyz
    def sample_states(self, n_samples=()):
        n_samples = ((n_samples,) if isinstance(n_samples, int) else tuple(n_samples))
        positions = np.random.uniform(low=self.states_min[0:3], 
                                      high=self.states_max[0:3], 
                                      size=n_samples + (3,))
        lin_vels = np.random.uniform(low=self.states_min[3:6], 
                                     high=self.states_max[3:6], 
                                     size=n_samples + (3,))
        quaternions = self.sample_unit_quaternions_wxyz(n_samples)
        ang_vels = np.random.uniform(low=self.states_min[10-1:13-1], 
                                     high=self.states_max[10-1:13-1], 
                                     size=n_samples + (3,))
        states = np.concatenate([positions,lin_vels,quaternions,ang_vels], axis=-1)
        return states
    def sample_controls(self, n_samples=()):
        n_samples = ((n_samples,) if isinstance(n_samples, int) else tuple(n_samples))
        nom = np.random.uniform(low=self.control_min, 
                                high=self.control_max, 
                                size=n_samples + (self.u_dim,))
        diff = np.random.uniform(low=self.control_diff_min, 
                                 high=self.control_diff_max, 
                                 size=n_samples + (self.u_dim,))
        controls = nom + diff
        controls = np.clip(controls, self.control_min, self.control_max)
        return controls