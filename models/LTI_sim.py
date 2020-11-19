import sys
sys.path.append('../..')

from utils.viz import *
from utils.stats import *

import numpy as np

""" ------------------------------------
LTI system for neural network experiment
State:   x = [px, py, vx, vy]
Control: u = [ux, uy]
----------------------------------------- """

class LTISimulator:
    def __init__(self, dt=1.0):
        self.n_x, self.x_dim = 4, 4
        self.n_u, self.u_dim = 2, 2
        self.n_params        = 0 # number of parameters

        self.dt = dt

        # initial state
        self.x0    = np.zeros(self.n_x)
        self.state = np.zeros(self.n_x)

        # dynamics (discrete time)
        self.A = np.array([[1.,0., dt,0.],
                           [0.,1., 0.,dt],
                           [0.,0., 1.,0.],
                           [0.,0., 0.,1.]])
        self.B = np.array([[0.,0.],
                           [0.,0.],
                           [dt,0.],
                           [0.,dt]])

        # states limits for sampling
        max_pos     = 5.
        limit_vel   = 1. 
        self.states_min = ([-max_pos,   -max_pos,   # position
                            -limit_vel, -limit_vel  # velocity
                            ])
        self.states_max = ([ max_pos,    max_pos,   # position
                             limit_vel,  limit_vel  # velocity
                            ])

        # controls limits for sampling
        self.control_min = [-0.4, -0.4]
        self.control_max = [ 0.4,  0.4]
        self.control_diff_min = [0.05*self.control_min[0],0.05*self.control_min[1]]
        self.control_diff_max = [0.05*self.control_max[0],0.05*self.control_max[1]]

    def reset_state(self):
        self.state = self.x0.copy()

    def f_dt(self, x_k, u_k):
        x_next = (self.A @ x_k) + (self.B @ u_k)
        return x_next

    def sample_states(self, n_samples=()):
        n_samples = ((n_samples,) if isinstance(n_samples, int) else tuple(n_samples))
        states = np.random.uniform(low=self.states_min, 
                                   high=self.states_max, 
                                   size=n_samples + (self.x_dim,))
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