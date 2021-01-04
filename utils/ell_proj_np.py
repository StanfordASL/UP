##
## Code written by Robert Dyro, Stanford ASL, 2020
##
## Reproduces the Self-adaptive Alternating Direction Method of Multipliers (S-ADMM) as described in
##      Z. Jia, X. Cai, and D. Han. Comparison of several fast algorithms for projection onto anellipsoid.
##          Journal of Computational and Applied Mathematics, 319(1):320–337, 2017.

##^# utils and imports #########################################################
import time
# ------------
import numpy as np, scipy.linalg as LA, math
_bmv = lambda A, x: (A @ x[..., None])[..., 0]
##$#############################################################################
##^# numpy implementation ######################################################
def _fsolve_fn(A, x, lower=True): 
    assert A.ndim == 3
    return np.stack([LA.cho_solve((A[i], lower), x[i]) for i in
        range(A.shape[0])])

def _proj_l2(F, A, AT, z, v, rho):
    return _fsolve_fn(F, z + rho[..., None] * _bmv(AT, v))

def _proj_uball(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1), 1.0)[..., None]

def proj_ell(Q, z, rho=None, max_it=10**2, eps=None, x_guess=None,
                verbose=False):
    """
        Projects points z such that they lie within the ellipsoid, i.e.,
            z.T @ Q @ z <= 1

    Inputs:      Q : ellipsoidal shape matrices - (B, n_z, n_z)
                 z : points to project          - (B, n_z)
    Outputs: new_z : projected points           - (B, n_z)
    """
    bshape = Q.shape[:-2]
    Q, z = Q.reshape((-1,) + Q.shape[-2:]), z.reshape((-1, z.shape[-1]))
    if rho is None: 
        rho = Q.shape[-1] / np.sqrt(np.linalg.norm(Q, axis=(-2, -1)))
    else: 
        rho = rho * np.ones(Q.shape[0])
    eps = eps if eps is not None else 1e-7

    A = np.stack([LA.cholesky(Q[i], lower=False) for i in range(Q.shape[0])])
    AT = A.swapaxes(-2, -1)

    P = np.eye(Q.shape[-1]) + rho[..., None, None] * Q
    F = np.stack([LA.cholesky(P[i], lower=True) for i in range(P.shape[0])])

    x = x_guess if x_guess is not None else z
    y = _fsolve_fn(A, x, lower=False)
    u = np.zeros(z.shape)

    x_prev, it = x, 0
    r_prim, r_dual = np.array(float("inf")), np.array(float("inf"))
    while it < max_it and (r_prim.mean() > eps or r_dual.mean() > eps):
        x = _proj_l2(F, A, AT, z, y - u, rho=rho)
        Ax = _bmv(A, x)
        y = _proj_uball(Ax + u)
        u = u + (Ax - y)
        r_prim = np.linalg.norm(Ax - y, axis=-1) / math.sqrt(A.shape[-1])
        r_dual = np.linalg.norm(x - x_prev, axis=-1) / math.sqrt(A.shape[-1])
        x_prev, it = x, it + 1
        # print(it)
    if verbose:
        data = dict(it=it, r_prim=r_prim, r_dual=r_dual)
        data["norms"] = np.sum(x * _bmv(Q, x), -1)
        if it == max_it and (r_prim.mean() > eps or r_dual.mean() > eps): 
            print("Ellipsoid projection did not converge")
            # print("The solution norms in the ellipse norm are:")
            # print(data["norms"])
        return x.reshape(bshape + (x.shape[-1],)), data
    else: 
        return x.reshape(bshape + (x.shape[-1],))
##$#############################################################################
##^# testing routines ##########################################################
def check(Q, z):
    import cvxpy as cp
    assert z.ndim == 1
    x_var = cp.Variable(z.shape)
    obj, cstr = cp.sum_squares(x_var - z), [cp.quad_form(x_var, Q) <= 1.0]
    prob = cp.Problem(cp.Minimize(obj), cstr)
    prob.solve(cp.ECOS, verbose=False)
    assert prob.status == "optimal"
    return x_var.value.reshape(-1)

def sample():
    n = 10**1
    N = 10
    A = np.random.randn(*(N, N, n, n))
    A[:N//2, :, :] *= 1e-3
    A[N//2:, :, :] *= 1e6
    Q = A @ A.swapaxes(-2, -1) / 2 + np.eye(A.shape[-1]) * 1e-1
    z = np.random.randn(*(N, N, n)) * 1e9
    return Q, z

def sample_2():
    n = 32
    N = 100
    A = np.random.randn(*(N, n, n))
    A[:N//2, :, :] *= 1e-3
    # A[N//2:, :, :] *= 1e6
    Q = A @ A.swapaxes(-2, -1) / 2 + np.eye(A.shape[-1]) * 1e-1
    z = np.random.randn(*(N, n)) *0.1#* 1e9
    return Q, z

def main(Q, z):
    #A = np.random.randn(*(3, 3))
    #Q = A @ A.T / 2
    #z = np.random.randn(3) * 10

    #x1 = check(Q, z)
    #print(x1)
    #print("norm =", np.sum(x1 * _bmv(Q, x1)))
    #x2 = proj_ell(Q, z, rho=1e0)
    #print(x2)

    # Naive method
    x_naive = (z.T/np.sum(z * _bmv(Q, z), -1)).T
    print("proj_dists naive =", np.linalg.norm(x_naive-z))

    #import matplotlib.pyplot as plt
    #rho_exps = range(-6, 4)
    #its = [proj_ell(Q, z, verbose=True, rho=10**float(rho_exp))[1]["it"] 
    #        for rho_exp in rho_exps]
    #plt.figure()
    #plt.plot(rho_exps, its)
    #plt.show()

    max_it = 200
    print('Norms before = ', np.sum(z * _bmv(Q, z), -1))
    print('Q=',Q.shape)
    print('z=',z.shape)

    x2, data = proj_ell(Q, z, verbose=True, eps=1e-5, max_it=max_it)
    print("norm =", data["norms"])
    print("it =", data["it"])
    # print("r_prim =", data["r_prim"])
    # print("r_dual =", data["r_dual"])
    print(x2.shape)

    print("proj_dists=", np.linalg.norm(x2-z))

# Q, z = sample()
Q, z = sample_2()
if __name__ == "__main__":
    # main(Q, z)

    start = time.time()
    main(Q, z)
    end = time.time()
    print("\n\nelapsed time = ",end-start,"\n\n")

##$#############################################################################
