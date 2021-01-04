import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import datetime

from scipy import stats
from utils import ell_proj_np

from copy import deepcopy

def get_encoder(config, inp_dim):
    activation = nn.Tanh()
    hid_dim = config['hidden_dim']
    x_dim   = config['x_dim']
    
    encoder = nn.Sequential(
        nn.Linear(inp_dim, hid_dim),
        activation,
        nn.Linear(hid_dim, hid_dim),
        activation,
        nn.Linear(hid_dim, x_dim),
    )
    return encoder

base_config = {
    # model parameters
    'x_dim': None,
    'u_dim': None,
    'hidden_dim': 32,
    
    # training parameters
    'data_horizon': 40,
    'learning_rate': 5e-3,
    'tqdm': True,
    'learning_rate_decay': False,
    'lr_decay_rate': 5e-5,
    'l2_reg ': 0.,
}


class FeedForward(nn.Module):
    def __init__(self, config={}, model_path=None):
        super().__init__()

        self.config = deepcopy(base_config)
        if model_path is not None:
            data = torch.load(model_path)
            config = data["config"]        
            
        self.config.update(config)

        self.x_dim = self.config['x_dim']
        self.u_dim = self.config['u_dim']

        self.encoder = get_encoder(self.config, self.x_dim + self.u_dim)
        
        if model_path is not None:
            print("loading state dict")
            self.load_state_dict(data['state_dict'])        

    def loss(self, phi, y):
        """
            input:  phi: shape (..., x_dim+u_dim) - prediction
                    y:   shape (..., y_dim)   
            output: loss: sum of errors squared
        """
        err = y  - phi
        err_quadform = 0.5 * err**2
        return err_quadform


    def forward(self, x, u):
        """
            input:  x, u
            output: phi
        """
        z   = torch.cat([x,u], dim=-1)
        phi = self.encoder(z)
        return phi


class FeedForwardDynamics():
    def __init__(self, model, summary_name='FeedForward',
                              dt=1.):
        """
        Inputs:
            model: FeedForward object
        """
        super().__init__()
        self.f_nom = lambda x,u: x

        self.model = model
        self.reset()

        self.ob_dim = self.model.x_dim
        self.u_dim  = self.model.u_dim
        
        # used for annealing during training
        self.train_step = 0

        self.writer = SummaryWriter('./runs/' + summary_name + '_' + datetime.datetime.now().strftime('y%y_m%m_d%d_s%s'))

        # set up optimizer
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.model.config['learning_rate'],
                                    weight_decay=self.model.config['l2_reg'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, 
                                                         gamma=1-self.model.config['lr_decay_rate'] )

        # discretization time (for true lipschitz constant)
        self.dt = dt

        # self.B_UP_method = 'randUP'
        self.B_UP_method = 'robUP' 

        # initial conditions
        self.mu_0 = np.zeros(self.ob_dim)
        self.Q_0  = np.zeros((self.ob_dim,self.ob_dim))
        self.X0s_MC = np.zeros((0,self.ob_dim,self.ob_dim))
        self.eta_x0s = 1e-2

    def reset(self):
        pass

        

    # TRAINING FUNCTIONS
    def evaluate_loss(self, x, u, xp):
        """
        mean over time horizon/batch (dim 1), mean over batch (dim 0)
        """
        horizon = self.model.config['data_horizon']
        
        x  = x.float()
        u  = u.float()
        xp = xp.float()
        z = torch.cat([x,u], dim=-1)

        # batch compute features and targets
        phi = self.model.encoder(z)
        y   = xp - self.f_nom(x,u)

        total_loss = self.model.loss(phi, y) / horizon
        return total_loss.mean()
    
    
    def train(self, dynamics_simulator, 
                    batch_size,
                    num_train_updates, 
                    B_validation=True,
                    verbose=False):
        self.reset()
        config = self.model.config
        
        validation_freq = 100
        val_batch = 5
       
        with trange(num_train_updates, disable=(not verbose or not config['tqdm'])) as pbar:
            for idx in pbar:
                self.optimizer.zero_grad()
                self.model.train()

                x = dynamics_simulator.sample_states(batch_size)
                u = dynamics_simulator.sample_controls(batch_size)
                xp = np.zeros_like(x)
                for i in range(x.shape[0]):
                    xp[i,:] = dynamics_simulator.f_dt(x[i,:],u[i,:])
                x, u, xp = torch.from_numpy(x), torch.from_numpy(u), torch.from_numpy(xp)
                total_loss = self.evaluate_loss(x, u, xp)

                # compute validation loss
                total_loss_val=[]
                if idx % validation_freq == 0 and B_validation:
                    self.model.eval()

                    x = dynamics_simulator.sample_states(val_batch)
                    u = dynamics_simulator.sample_controls(val_batch)
                    xp = np.zeros_like(x)
                    for i in range(x.shape[0]):
                        xp[i,:] = dynamics_simulator.f_dt(x[i,:],u[i,:])
                    x, u, xp = torch.from_numpy(x), torch.from_numpy(u), torch.from_numpy(xp)
                    total_loss_val = self.evaluate_loss(x, u, xp)

                    self.writer.add_scalar('Loss/Val', total_loss_val, self.train_step)

                # grad update on logp
                total_loss.backward()
                self.optimizer.step()
                if config['learning_rate_decay']:
                    self.scheduler.step()

                # ---- logging / summaries ------
                self.train_step += 1
                step = self.train_step

                # tensorboard logging
                self.writer.add_scalar('Loss/Train', total_loss.item(), step)

                if config['learning_rate_decay']:
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], step)

                # tqdm logging
                logdict = {}
                logdict["tr_loss"] = total_loss.detach().numpy()
                if B_validation:
                    logdict["val_loss"] = total_loss_val
                    
                pbar.set_postfix(logdict)

                self.reset()


    def to_torch(self, *args):
        """
        for every argument (assumed to be a numpy array), this function
        puts the arguments into a float32 torch.Tensor and pushes it to the same device
        as self.model
        """
        device = next(self.model.parameters()).device
        return [torch.as_tensor(x, dtype=torch.float32, device=device) for x in args]
    
    def save(self, path):
        """
        Saves self.model to path.
        """
        torch.save({
            "config":self.model.config, 
            "state_dict":self.model.state_dict() }, path)
    


    def get_model_torch(self, n_samples=10):
        """
        returns function mapping x,u to mu (torch tensor)
        """
        def f(x,u):
            """
            assumes dim -2 of inputs is batch over model samples
            inputs must broadcast to (..., N, x/u dim)
            """
            z   = torch.cat([x,u], dim=-1)
            phi = self.model.encoder(z) # (N, xdim, phidim, 1)
            mu  = phi + self.f_nom(x,u)         

            return mu
        return f

    def get_model(self, with_grad=False,
                        **kwargs):
        torch_fn = self.get_model_torch(**kwargs)

        def f(x_np, u_np, eps_np=None):
            batch_shape = x_np.shape[:-1]
            x_dim       = x_np.shape[-1]
            u_dim       = u_np.shape[-1]

            x_np = x_np.reshape([-1,x_dim])
            u_np = u_np.reshape([-1,u_dim])

            x, u = self.to_torch(x_np, u_np)

            if with_grad:
                x.requires_grad = True
                u.requires_grad = True

            mu    = torch_fn(x,u)
            mu_np = mu.detach().cpu().numpy().reshape(batch_shape + (x_dim,))

            if with_grad:
                dmu_dx = torch.cat([ torch.autograd.grad(mu[..., i].sum(), x,
                                    retain_graph=True)[0][..., None, :]
                                        for i in range(x_dim) ], -2)
                dmu_dx_np = dmu_dx.detach().cpu().numpy().reshape(batch_shape + (x_dim, x_dim))

                dmu_du = torch.cat([ torch.autograd.grad(mu[..., i].sum(), u,
                                    retain_graph=(i + 1 < x_dim))[0][..., None, :]
                                        for i in range(x_dim) ], -2)

                dmu_du_np = dmu_du.detach().cpu().numpy().reshape(batch_shape + (x_dim, u_dim))

                return mu_np, dmu_dx_np, dmu_du_np

            return mu_np

        return f


    # -------------------------------------------
    # Fields for uncertainty propagation
    def Xs_dparams_MC(self, Xs, Xs_dx):
        """
        Returns the Jacobian matrices of the state TRAJECTORY
        w.r.t. to all parameters
        Inputs:  Xs       : (N_MC, N , n_x)
                 Xs_dx    : (N_MC, N , n_x, n_x)
        Outputs: Xs_dX0   : (N_MC, N, n_x, n_x)
        """
        N_MC, N, n_x = Xs.shape[0], Xs.shape[1], Xs.shape[2]

        Xs_dX0s          = np.zeros((N_MC, N, n_x, n_x))
        Xs_dX0s[:,0,:,:] = np.repeat(np.eye(n_x)[None,:], N_MC, axis=0)
        for j in range(N-1):
            # Jacobians w.r.t. Initial conditions
            Xs_dX0s[:,j+1,:,:] = np.einsum('Mxy,Myz->Mxz', Xs_dx[:,j,:,:], Xs_dX0s[:,j,:,:]) 

        return Xs_dX0s

    def adv_sample_params(self, Xs, Xs_dx):
        """
          resamples parameters  self.X0s 
                using           Xs       : (N_MC, N, n_x)
                                Xs_dx    : (N_MC, N, n_x, n_x)
        """
        N_MC, T, x_dim = Xs.shape[0], Xs.shape[1], Xs.shape[2]
        X0s            = self.X0s_MC
        eta_x0s        = self.eta_x0s

        Cs = np.mean(Xs,0) # (N, n_x)
        Qs = np.zeros((T,x_dim,x_dim))
        for t in range(1,T): 
            Qs[t,:,:] = np.linalg.inv(np.cov(Xs[:,t,:].T))
            
        # compute cost gradient
        Jdists_dXs = np.einsum('txy,Mty->Mtx', 2*Qs, Xs-Cs) 

        # compute trajectory gradient w.r.t. parameters
        Xs_dX0 = self.Xs_dparams_MC(Xs, Xs_dx)

        # compute cost gradient w.r.t params (average over horizon)
        Jdists_dX0s = np.mean(np.einsum('MTx,MTxy->MTy', Jdists_dXs, Xs_dX0), axis=1)

        # gradient ascent
        X0s  = self.X0s_MC + eta_x0s * Jdists_dX0s

        # project parameters
        mu0s       = np.repeat(self.mu_0[None,:], N_MC, axis=0)
        Q0_inv     = np.linalg.inv(self.Q_0)
        Q0s        = np.repeat(Q0_inv[None,:,:], N_MC, axis=0)
        # 1) naive projection as initial guess
        # norms_dx0s  = np.sqrt(np.einsum('Mx,xy,My->M',X0s-mu0s,Q0_inv,X0s-mu0s))
        # alphas_dx0s = np.clip(1./norms_dx0s, 0, 1)
        # projs_naive = ((X0s-mu0s).T * alphas_dx0s).T
        # 2) projection with admm
        X0s_deltas = ell_proj_np.proj_ell(Q0s, X0s-mu0s, eps=5e-4)

        X0s        = mu0s + X0s_deltas

        self.X0s_MC = X0s
        return X0s

    # ----------------------------------------------
    # Uncertainty propagation using lipschitz method

    def propagate_ellipsoid_lipschitz(self, mu_k, u_k, Q_k):
        # inputs: - mu_k : [ob_dim]           # state mean
        #         - u_k  : [u_dim]            # control input
        #         - Q_k  : [ob_dim x ob_dim]  # previous ellipsoid matrix

        # Nominal component (which is exactly linear)
        dfnom_dx = np.eye(self.ob_dim)
        Qnom     = ((dfnom_dx) @ Q_k @ (dfnom_dx).T)

        # Lipschitz part
        lip_csts     = self.dt*np.array([1.,1., 0.,0.])
        eigs_Q, vecs = np.linalg.eigh(Q_k)
        eig_max_Q    = max(eigs_Q)
        delta_x_lip  = np.zeros(self.ob_dim)
        for dim in range(self.ob_dim):
            delta_x_lip[dim] = lip_csts[dim] * eig_max_Q

        # Calibration part
        pass

        # Gaussian additive noise
        pass

        # Lipschitz + Calibration parts
        # QDelta = self.ob_dim * np.diag( (delta_x_lip + delta_x_cal + delta_eps)**2 )

        # Note: it is not sqrt(self.ob_dim), there is a small typo in the original paper
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8619572
        # this is corrected in  https://github.com/befelix/safe-exploration/
        QDelta = np.diag( self.ob_dim * delta_x_lip**2 ) # rectangle -> ellipsoid

        # bound sum of ellipsoids
        c  = np.sqrt( np.trace(Qnom) / np.trace(QDelta) )
        Qp = (c+1)/c*Qnom + (1+c)*QDelta

        return Qp

    def get_upper_bound_lipschitz_constant(self):
        for i, param in enumerate(self.model.parameters()):
            if i==0:
                W0 = param.data
            elif i==2:
                W1 = param.data
            elif i==4:
                W2 = param.data
        sig_0 = max(np.linalg.svd(W0, compute_uv=False))
        sig_1 = max(np.linalg.svd(W1, compute_uv=False))
        sig_2 = max(np.linalg.svd(W2, compute_uv=False))
        return (sig_2 * sig_1 * sig_0)

    def get_upper_bound_lipschitz_constants_vec(self):
        for i, param in enumerate(self.model.parameters()):
            if i==0:
                W0 = param.data
            elif i==2:
                W1 = param.data
            elif i==4:
                W2 = param.data
        sig_0  = max(np.linalg.svd(W0, compute_uv=False))
        sig_1  = max(np.linalg.svd(W1, compute_uv=False))
        sigs_2 = np.linalg.norm(W2, axis=1)
        return (sigs_2 * sig_1 * sig_0)