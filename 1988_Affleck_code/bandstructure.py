import numpy as np
from config import TP, KX, KY, KX_MESH, KY_MESH
# global KX, KY need to be defined externally before importing

def get_E12_and_N(chi1, chi2, chi3, chi4, phi1, phi2, phi3, phi4, t):
    '''
    Performs the calculation of E1 and E2 matrices.

    Args:
        chi1: float
        chi2: float
        chi3: float
        chi4: float
        phi1: float
        phi2: float
        phi3: float
        phi4: float
        t: float

    Returns:
        E1_matrix: 2D array
        E2_matrix: 2D array
    '''
    h12 = - (t + chi1 * np.exp(-1j * phi1)) * np.exp(1j * KX_MESH) \
          - (t + chi2 * np.exp(-1j * phi2)) * np.exp(1j * KY_MESH) \
          - (t + chi3 * np.exp(-1j * phi3)) * np.exp(-1j * KX_MESH) \
          - (t + chi4 * np.exp(-1j * phi4)) * np.exp(-1j * KY_MESH)

    common_term = -4 * TP * np.cos(KX_MESH) * np.cos(KY_MESH)
    abs_h12 = np.abs(h12) ** 0.5
    E1_matrix = common_term - abs_h12
    E2_matrix = common_term + abs_h12

    return E1_matrix, E2_matrix

def Mu12_from_E(E1, E2, N_target, ERROR_TOL=3):
    '''
    Performs a binary search to find the value of mu that satisfies N(mu) = N_target

    Args:
        E1: 2D array
        E2: 2D array
        N_target: float

    Returns:
        mu: float
    '''
    start = -1000.0
    end = 1000.0
    max_iter = 100

    for _ in range(max_iter):
        mid = 0.5 * (start + end)

        N_mid = ((E1 <= mid).sum() + (E2 <= mid).sum()) / 2

        if abs(N_mid - N_target) <= ERROR_TOL / 2:
            return mid
        elif N_mid > N_target:
            end = mid
        else:
            start = mid

    return mid  # fallback if not converged
