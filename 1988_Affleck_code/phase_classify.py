import numpy as np
from config import N_MIN, N_MAX, T_MIN, T_MAX, SITE_N, N_NUM, T_NUM

def filter_phis_by_chis(data, chi_threshold=1e-3):
    '''
    Set phi_i to zero where corresponding chi_i is below threshold.
    '''
    data = np.array(data, copy=True)
    chis = data[:, 2:6]
    phis = data[:, 6:10]
    mask = np.abs(chis) < chi_threshold
    phis[mask] = 0.0
    data[:, 6:10] = phis
    return data

def classify_phases(filtered_data, chi_tol=1e-2, phi_tol=1e-2, peierls_ratio=1.1):
    phase_names = []
    for row in filtered_data:
        t = row[1]
        chi1, chi2, chi3, chi4 = row[2:6]
        phi1, phi2, phi3, phi4 = row[6:10]

        chis = np.array([chi1, chi2, chi3, chi4])
        phis = np.array([phi1, phi2, phi3, phi4])
        theta = np.angle(t + chis * np.exp(-1j * phis))
        flux = np.abs(-theta[0] + theta[1] - theta[2] + theta[3])

        # Condition 1: Uniform
        if np.std(chis) < chi_tol and (np.all(np.abs(phis) < phi_tol) or flux/np.pi % 2 < 1e-2):
            phase_names.append("Uniform")
            continue
        
        # Condition 2: Flux (phi alternates in sign, chi almost equal)
        phi_signs = np.sign(np.real(phis))
        if np.std(chis) < chi_tol and (
            np.all(phi_signs == [+1, -1, +1, -1]) or np.all(phi_signs == [-1, +1, -1, +1])
        ):
            phase_names.append("Flux")
            continue
        
        # Condition 3: Kite (phi ≈ 0, chi grouped 2 by 2)
        chi_patterns = [
            (chis[0], chis[1], chis[2], chis[3]),  # chi1=chi2 ≠ chi3=chi4
            (chis[1], chis[2], chis[3], chis[0])  # chi2=chi3 ≠ chi4=chi1
        ]
        matched_kite = False
        for ch in chi_patterns:
            if abs(ch[0] - ch[1]) < chi_tol and abs(ch[2] - ch[3]) < chi_tol and abs(ch[0] - ch[2]) > chi_tol:
                phase_names.append("Kite")
                matched_kite = True
                break
        if matched_kite:
            continue
        
        # Condition 4: Peierls (one chi significantly larger)
        chi_abs = np.abs(chis)
        max_val = np.max(chi_abs)
        rest = np.delete(chi_abs, np.argmax(chi_abs))
        if max_val > peierls_ratio * np.max(rest):
            phase_names.append("Peierls")
            continue

        # Condition 5: Stripy (chi1=chi3≠chi2=chi4) Degenerate with kite
        if abs(chis[0] - chis[2]) < chi_tol and abs(chis[1] - chis[3]) < chi_tol and abs(chis[0] - chis[3]) > chi_tol:
            phase_names.append("Stripy")
            continue

        phase_names.append("Other")

    return np.array(phase_names).reshape((T_NUM, N_NUM))


