import matplotlib.pyplot as plt
import numpy as np

# plotting of shapes
import matplotlib.patches as patches
import matplotlib.transforms as transforms


def plot_traj(axes, rollout_data, color='C0', linestyle='-', alpha=1.0, t0=0):
    """
    Plots each componennt of a trajectory and the actions taken
    
    rollout_data is a dictionary with keys:
        'states' -> array shape (T, obdim)
        'actions' -> shape (T-1, udim)
    
    t0: int of initial timestep of trajectory
    """        
    traj = np.array(rollout_data["states"])
    u_list = np.array(rollout_data["actions"])

    H,ob_dim = traj.shape
    u_dim = u_list.shape[-1]

    for i in range(ob_dim+u_dim):
        ax = axes[i]
        if i < ob_dim:
            ax.plot(np.arange(t0,H+t0),traj[:,i], color=color, linestyle=linestyle, alpha=alpha)
        else:
            ax.plot(np.arange(t0,t0+H-1),u_list[:,i-ob_dim], color=color, linestyle=linestyle, alpha=alpha)

def plot_trajs_2d(Xs, color='C0', linestyle='-', alpha=1.0, idx=[0,1]):
    # Xs - (N_MC, n_x, N)
    plt.plot(Xs[:,idx[0],:].T, Xs[:,idx[1],:].T, color=color, linestyle=linestyle, alpha=alpha)

def plot_pts_2d(Xs, color='C0', markerwidth=None, linestyle='-', alpha=1.0, idx=[0,1]):
    # Xs - (N_MC, n_x)
    plt.scatter(Xs[:,idx[0]], Xs[:,idx[1]], s=markerwidth, color=color, linestyle=linestyle, alpha=alpha)


def plot_state_traj(states, idx=[0,1], color='C0', linestyle='-', alpha=1.0, label=None):
    """
    Plots each x-y trajectory on a plane according to idx:
    """
    traj = states
    plt.plot(traj[:,idx[0]],traj[:,idx[1]], color=color, linestyle=linestyle, alpha=alpha, label=label)

def plot_mean_traj_with_ellipses(mus, Qs, idx=[0,1], alpha=None, color="b", label=None):
    if mus.shape[0] != len(Qs):
        raise ValueError("mus (%d), Qs (%d) and Q (%d,%d) must have same nb. of elements" %(mus.shape[0],len(Qs)))

    T = len(Qs)
    ax = plt.gca()

    plot_state_traj(mus, idx=idx, color=color, label=label)

    if alpha == None:
        alpha = 0.1 * min(1., 30/mus.shape[0])
    for k in range(T):
        mu = mus[k,idx]
        Q  = Qs[k][np.ix_(idx,idx)]
        plot_ellipse(ax, mu, Q, alpha=alpha, color=color)

def plot_mean_traj_with_rectangles(mus, deltas, idx=[0,1], alpha=None, color="b", label=None):
    # if mus.shape[0] != len(deltas):
    #     raise ValueError("mus (%d), deltas (%d) must have same nb. of elements" %(mus.shape[0],len(deltas)))

    T = len(deltas)
    ax = plt.gca()

    plot_state_traj(mus, idx=idx, color=color, label=label)

    if alpha == None:
        alpha = 0.1 * min(1., 30/mus.shape[0])
    for k in range(T):
        mu = mus[k,idx]
        ds = deltas[k][idx]
        if k==T-1:
            print('Plotted widths at T for lip = ',ds)
        plot_rectangle(ax, mu, ds, alpha=alpha, color=color)

def plot_mean_traj_with_divided_rectangles(mus, deltas, idx=[0,1], alpha=None, color="b", noFaceColor=False, label=None, B_plot_only_last=False):
    # if mus.shape[0] != len(deltas):
        # raise ValueError("mus (%d), deltas (%d) must have same nb. of elements" %(mus.shape[0],len(deltas)))

    T = len(mus)
    ax = plt.gca()

    # plot_state_traj(mus, idx=idx, color=color, label=label)
    plt.scatter(mus[0][0,idx[0]],mus[0][0,idx[1]], color=color, label=label)

    if alpha == None:
        alpha = 0.1 * min(1., 30/mus.shape[0])
    if B_plot_only_last:
        for i, (mu_ki,ds_ki) in enumerate(zip(mus[-1],deltas[-1])):
            plot_rectangle(ax, mu_ki[idx], ds_ki[idx], alpha=alpha, color=color, noFaceColor=noFaceColor)
    else:
        for k in range(T):
            for i, (mu_ki,ds_ki) in enumerate(zip(mus[k],deltas[k])):
                plot_rectangle(ax, mu_ki[idx], ds_ki[idx], alpha=alpha, color=color, noFaceColor=noFaceColor)


# Utils - plotting of geometric shapes

def plot_ellipse(ax, mu, Q, additional_radius=0., color='blue', alpha=0.1, **kwargs):
    """
    Based on
    http://stackoverflow.com/questions/17952171/not-sure-how-to-fit-data-with-a-gaussian-python.
    """

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(Q)

    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h =  2. * (np.sqrt(vals) + additional_radius)
    ellipse = patches.Ellipse(mu, w, h, theta, color=color, alpha=alpha)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ax.add_artist(ellipse) 

def plot_rectangle(ax, center, widths, additional_w=0., color='blue', alpha=0.1, noFaceColor=False, **kwargs):
    """
    Plots a rectangle with a given center and total widths
    arguments:  - center    - (2,) center
                - widths    - (2,) total widths  from and to edges of rectangle
    """
    if center.shape[0] != 2 or widths.shape[0] != 2:
        assert('plot_rectangle function can only plot in 2d.')
    facecolor = color
    if noFaceColor:
        facecolor = None

    deltas = [widths[0]+additional_w, widths[1]+additional_w]
    bottom_left = (center[0] - deltas[0]/2., center[1] - deltas[1]/2.)
    rect = patches.Rectangle((bottom_left[0],bottom_left[1]),deltas[0],deltas[1], \
                                linewidth=1,edgecolor=color,facecolor=facecolor,alpha=alpha)
    ax.add_patch(rect)
