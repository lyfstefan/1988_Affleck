import numpy as np
from scipy.optimize import minimize, dual_annealing
from config import SITE_N
from bandstructure import get_E12_and_N, Mu12_from_E
from joblib import Parallel, delayed
import time
from config import N_MIN, N_MAX, T_MIN, T_MAX, SITE_N, N_NUM, T_NUM, J

def Etot(x, N, t):
    chi1, chi2, chi3, chi4, phi1, phi2, phi3, phi4 = x
    E1_matrix, E2_matrix = get_E12_and_N(chi1, chi2, chi3, chi4, phi1, phi2, phi3, phi4, t)
    mu = Mu12_from_E(E1_matrix, E2_matrix, N)

    E1_matrix[E1_matrix > mu] = 0
    E2_matrix[E2_matrix > mu] = 0

    E = (E1_matrix.sum() + E2_matrix.sum()) / 2
    E += SITE_N / (2 * J) * (chi1**2 + chi2**2 + chi3**2 + chi4**2)  # J = 1
    return E / SITE_N


def Order(N, t):
    """
    Perform dual annealing + local optimization to find order parameters
    for a given particle number N and hopping t.

    Args:
        N (float): Total number of particles
        t (float): Hopping strength

    Returns:
        tuple: (N, t, chi1, chi2, chi3, chi4, phi1, phi2, phi3, phi4, Etot)
    """
    bounds = [(0, 1)] * 4 + [(-np.pi, np.pi)] * 4

    result = dual_annealing(
        Etot,
        bounds=bounds,
        args=(N, t),
        maxiter=512,
        initial_temp=500.0,
        restart_temp_ratio=1e-5,
        visit=2.6,
        accept=-4.0,
        no_local_search=True,
        seed=None
    )

    local_result = minimize(
        Etot,
        result.x,
        args=(N, t),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000}
    )

    x = local_result.x
    return (N, t, *x, local_result.fun)


def Order_multi(N, t):
    '''
    Perform dual annealing + local optimization to find order parameters for multiple times in some region
    for a given particle number N and hopping t.

    Args:
        N (float): Total number of particles
        t (float): Hopping strength

    Returns:
        tuple: (N, t, chi1, chi2, chi3, chi4, phi1, phi2, phi3, phi4, Etot)
    '''
    bounds = [(0, 1)] * 4 + [(-np.pi, np.pi)] * 4
    best_result = None
    best_fun = np.inf

    if t >= 5 and N / SITE_N <= 0:
        return Order(N, t)

    for i in range(8):
        result = dual_annealing(
            Etot,
            bounds=bounds,
            args=(N, t),
            maxiter=512,
            initial_temp=500.0,
            restart_temp_ratio=1e-5,
            visit=2.6,
            accept=-4.0,
            no_local_search=True,
            seed=i
        )

        local_result = minimize(
            Etot,
            result.x,
            args=(N, t),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000}
        )

        if local_result.fun < best_fun:
            best_fun = local_result.fun
            best_result = local_result

    x = best_result.x
    return (N, t, *x, best_result.fun)

def get_phasediagram_data():
    N_list = np.linspace(N_MIN, N_MAX, N_NUM)  
    t_list = np.linspace(T_MIN, T_MAX, T_NUM)
    task_list = [(N, t) for t in t_list for N in N_list]

    results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(Order_multi)(N, t) for N, t in task_list
    )

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), 'Order parameters found!')
    return results

