# config.py

import numpy as np

# Physical constants
J = 1
TP = 0  # next nearest hopping

# Lattice parameters
NX = 50
NY = 50
SITE_N = NX * NY

# Phase diagram parameters
N_NUM = 20
T_NUM = 20
N_MIN, N_MAX = 0.2 * SITE_N, 0.5 * SITE_N
T_MIN, T_MAX = 0.0, 0.5
ERROR_TOL = 3

# K-space mesh
KX = np.arange(0, 2 * np.pi, 2 * np.pi / NX) - np.pi
KY = np.arange(0, 2 * np.pi, 2 * np.pi / NY) - np.pi
KX_MESH, KY_MESH = np.meshgrid(KX, KY)

# Plot options
PLOT_FONT = "Times New Roman"
USE_TEX = True
