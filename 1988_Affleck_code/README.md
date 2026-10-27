# Affleck calculation and data guide

The [project README](../README.md) gives the bond ansatz, energy convention,
parameter ranges and references.

## Calculation

Set `J`, `TP`, `NX`, `NY`, and the filling and hopping grids in `config.py`.
From this directory:

```sh
python main.py
```

The scan minimizes the energy at each particle number and hopping value,
saves the resulting bond parameters, classifies the phases, and plots the
order parameters and phase regions.

| File | Contents |
| --- | --- |
| `config.py` | Lattice, momentum mesh and parameter ranges |
| `bandstructure.py` | Two bands and chemical-potential bisection |
| `solver.py` | Mean-field energy, dual annealing and local minimization |
| `phase_classify.py` | Amplitude filtering and phase classification |
| `data_utils.py` | Saving and loading numerical arrays |
| `plot_utils.py` | Phase, order-parameter and flux plots |
| `replot_saved.py` | Figure generation from the saved scans |
| `publication_style.py` | Shared LaTeX figure style |

## Numerical data

Each row of a saved scan has eleven columns:

```text
N, t, chi1, chi2, chi3, chi4, phi1, phi2, phi3, phi4, energy_per_site
```

$N$ is the particle number, $\nu=N/N_s$ the filling, and $\phi_i$ is in
radians. The energy and dispersion conventions are specified in the project
README. The parameter grid is recovered from the first two columns; row
order is not assumed. Historical NPY scans are in `results/data/`.

## Replotting the saved scans

```sh
python replot_saved.py
```

This exports `phase_diagram_TIMESTAMP`, `order_parameters_TIMESTAMP`, and
`flux_square_TIMESTAMP` under `results/figs/`. Each has PDF, SVG, PNG and
figure-style metadata. It reads the historical phase categories from the
original figures, so it does not depend on the current classification
thresholds. Grid sizes and site counts for the three scans are listed in
the project README.

`results/classification/` contains the phase labels and coordinate grids;
`results/figure_provenance.json` records their source checksums. Original
figures are retained in `results/figs/originals/`.
