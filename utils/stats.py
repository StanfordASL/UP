import numpy as np
import math

import matplotlib.pyplot as plt
    
def volume_ellipsoid(Q):
    """
    volume of set x.T @ Q^{-1} @ x <= 1
    
    see https://faculty.fuqua.duke.edu/~psun/bio/MVEpaper_accept.pdf
    or
        Grötschel, M., L. Lovasz, A. Schrijver. 1998.
        Geometric Algorithms and Combinatorial Optimization. 
        Springer-Verlag, Berlin, Germany.
    """
    n = Q.shape[0]
    m_pi, gamma = math.pi, math.gamma(n/2. + 1)
    det_Qinv = 1. / np.linalg.det(Q)
    vol = (m_pi**(n / 2.) / gamma) * (1./np.sqrt(det_Qinv))
    return vol

def is_in_ellipse(x, mu, Q):
#     return (x-mu).T @ np.linalg.inv(Q) @ (x-mu) < 1.
    return (x-mu).T @ np.linalg.solve(Q, (x-mu)) < 1.

def count_nb_in_ellipse(Xs, mu, Q):
    if mu.size != Q.shape[0] or Xs.shape[1] != mu.size:
        raise ValueError("Xs (%d,%d), mu (%d) and Q (%d,%d) must be the same size" %(Xs.shape[0], Xs.shape[1], mu.size, Q.shape[0], Q.shape[1]))
    
    nb_in_ellipse = 0.
    for xi in Xs:
        if is_in_ellipse(xi, mu, Q) == True:
            nb_in_ellipse += 1
    return nb_in_ellipse


def count_nb_in_ellipse_and_get_idx_inside(Xs, mu, Q):
    if mu.size != Q.shape[0] or Xs.shape[1] != mu.size:
        raise ValueError("Xs (%d,%d), mu (%d) and Q (%d,%d) must be the same size" %(Xs.shape[0], Xs.shape[1], mu.size, Q.shape[0], Q.shape[1]))
    
    nb_in_ellipse = 0.
    indices_inside = []
    for idx,xi in enumerate(Xs):
        if is_in_ellipse(xi, mu, Q) == True:
            indices_inside.append(idx)
            nb_in_ellipse += 1
    return nb_in_ellipse, indices_inside

def percentage_in_ellipse(Xs, mu, Q):
    if mu.size != Q.shape[0] or Xs.shape[1] != mu.size:
        raise ValueError("Xs (%d,%d), mu (%d) and Q (%d,%d) must be the same size" %(Xs.shape[0], Xs.shape[1], mu.size, Q.shape[0], Q.shape[1]))
        
    return count_nb_in_ellipse(Xs, mu, Q) / Xs.shape[0]



def sample_pts_ellipsoid_surface(mu, Q, NB_pts, random=True):
    """
    Uniformly samples points on the surface of an ellipsoid, specified as
    (xi-mu)^T Q^{-1} (xi-mu) == 1
    arguments: mu      - mean [dim]
                Q      - Q [dim x dim]
                NB_pts - nb of points
                random - True: Uniform sampling. 
                         False: Uniform deterministic grid
    output:    ell_pts - points on the boundary of the ellipse [xdim x NB_pts]
    """
    dim = mu.shape[0]
    if dim != Q.shape[0] or dim != Q.shape[1]:
        raise ValueError("mu (%d) and Q (%d,%d) must be the same size" %(mu.shape[0], Q.shape[0], Q.shape[1]))
    if (Q == np.zeros((dim,dim))).all():
        return np.zeros((dim,NB_pts))

    if random == False and dim > 2:
        raise ValueError("sample_pts_ellipsoid_surface: non random sampling not implemented")

    mut = np.array([mu])
    pts = sample_pts_unit_sphere(dim, NB_pts, random=random).T
    E   = np.linalg.cholesky(Q)

    ell_pts = (mut + pts @ E.T).T

    return ell_pts

def sample_pts_unit_sphere(dim, NB_pts, random=True):
    """
    Uniformly samples points on a d-dimensional sphere (boundary of a ball)
    Points characterized by    ||x||_2 = 1
    arguments:  dim    - nb of dimensions
                NB_pts - nb of points
                random - True: Uniform sampling. 
                         False: Uniform deterministic grid 
    output:     pts    - points on the boundary of the sphere [xdim x NB_pts]
    Reference: http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    """
    if dim == 2 and random == False:
        angles = np.linspace(0., 2*np.pi, num=NB_pts, endpoint=False)
        pts = np.array([np.cos(angles),np.sin(angles)])
        return pts

    if random == False and dim > 2:
        raise ValueError("sample_pts_unit_sphere: non random sampling not implemented")

    u = np.random.normal(0, 1, (dim,NB_pts))
    d = np.sum(u**2,axis=0) **(0.5)
    pts = u/d
    return pts


def sample_pts_unit_ball(dim, NB_pts):
    """
    Uniformly samples points in a d-dimensional sphere (in a ball)
    Points characterized by    ||x||_2 < 1
    arguments:  dim    - nb of dimensions
                NB_pts - nb of points
    output:     pts    - points sampled uniformly in ball [xdim x NB_pts]
    Reference: http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    """

    us    = np.random.normal(0,1,(dim,NB_pts))
    norms = np.linalg.norm(us, 2, axis=0)
    rs    = np.random.random(NB_pts)**(1.0/dim)
    pts   = rs*us / norms
    
    return pts


def sample_pts_in_ellipsoid(mu, Q, NB_pts):
    """
    Uniformly samples points in an ellipsoid, specified as
            (xi-mu)^T Q^{-1} (xi-mu) <= 1
    arguments: mu - mean [dim]
                Q - Q [dim x dim]
    output:     pts - points sampled uniformly in ellipsoid [xdim x NB_pts]
    """
    xs = sample_pts_unit_ball(mu.shape[0], NB_pts)

    E  = np.linalg.cholesky(Q)
    ys = (np.array(E@xs).T + mu).T
    
    return ys


def sample_pts_in_hyperrectangle(center, widths, NB_pts_each_dim, random=False):
    """
    arguments:  center          - (dim,  ) center
                widths          - (dim,  ) total widths from and to edges of rectangle
                NB_pts_each_dim - number of points used for sampling along each dimension
                random          - deterministic or random (uniform) sampling (not implemented yet)
    output:     uniformly sampled dim-dimensional points in the rectangle:
                        |xi - center| < width/2
                    size: [xdim x NB_total_points)]
    """
    if random==True:
        print('[sample_pts_in_hyperrectangle] non deterministic (meshgrid) sampling not implemented.')
    
    dim = center.shape[0]
    M   = NB_pts_each_dim

    edges = np.array([center-widths/2, center+widths/2]).T

    out = np.mgrid[[np.s_[edges[d,0]:edges[d,1]:(M*1j)] for d in range(dim)]]
    out = out.T.reshape(-1,dim).T
    return out


def get_hyperrectangle_volume(widths):
    return np.prod(widths)

def test_ellipsoid_function(NB_pts=100):

    mu = np.array([1,2])
    Q = np.array([[1 ,0.2], [0.2 ,1.6]])

    pts = sample_pts_ellipsoid_surface(mu, Q, NB_pts)

    plt.scatter(pts[0,:], pts[1,:])


def bound_rectangle_with_ellipsoid(widths):
    dim = widths.shape[0]
    Q = dim * np.diag(widths)
    return Q