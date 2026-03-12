# Mean-field Phase Diagram of Flux Model

This project computes the mean-field phase diagram of a square lattice model with hopping and interaction terms.

## Files
- `main.py`: Main execution script to compute and plot the phase diagram
- `solver.py`: Global + local minimization routines
- `bandstructure.py`: Band energy and filling computations
- `phase_classify.py`: Heuristic classification of phases
- `plot_utils.py`: Plot functions for order parameters and phase diagram

## Dependencies
- Python 3.11+
- numpy, scipy, matplotlib, joblib

## Usage
```bash
python main.py