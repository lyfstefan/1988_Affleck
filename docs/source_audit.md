# Source audit and release scope (October 2026)

The GitHub repository already existed and was cloned at commit
`2f9edd708353b14653c41f57c2e6c8ada602dd7f`. The eight Python files under
`1988_Affleck_code/` match the supplied local candidates; their hashes are
listed in `legacy_source_manifest.json`. They are preserved unchanged.
The parent `Affleck_1988.py` contains the same square-root band construction
and independent chi/phi ansatz; it is not a better final-paper candidate.

## Why the existing source cannot be certified as the paper model

1. The source uses `t+chi_nu*exp(-i phi_nu)`, eight independent variables
   and a quadratic chi penalty. The paper and corresponding thesis chapter
   use four real currents, `t_nu=t+i V J_nu`, with cost `V/4 sum J_nu^2`
   per physical site. These are different variational problems. Restricting
   phi without changing the cost would not establish equivalence.
2. `bandstructure.py` takes `np.abs(h12)**0.5`. For a Hermitian matrix
   `[[d,h],[h*,d]]`, the bands are `d +/- abs(h)`. This is a concrete defect
   in that candidate version; changing it alone would not resolve item 1.
3. Legacy chemical-potential bisection allows a particle-count error and
   fills all states with E<=mu, problematic for degenerate levels. Its
   stochastic searches and heuristic phase classifier were not independently
   validated as producing the published diagram.
4. Old NPY, PNG and PDF outputs have not been linked to a verified script,
   exact settings and manuscript figure. They are historical outputs, not
   published reference data. The old directory's README is also historical.

## New paper-based implementation

`loop_current/model.py` was newly reconstructed from the four-current
Hamiltonian and energy in arXiv:2402.09577v1, section II.1 (Eqs. 6-12),
cross-checked against the local thesis chapter 2. It uses the explicitly
stated dispersion modulus and Hermitian conjugate in the Bloch Hamiltonian;
it does not copy a missing conjugation in a typeset matrix element.
The source of the final journal parameter-optimization run was not recovered.

The paper uses t=1, t'=0.2 and variable V/filling for its main diagram and
describes SHGO with symmetry constraints. The example uses those hopping
values at V=4 on a small mesh, but the new generic annealing/local search
has no RLEX or zero-net-flux constraint. Its output must not be assigned a
published phase label or claimed as a reproduced phase boundary.

Tests check bands through direct matrix diagonalization and the unbroken
free-lattice limit. The zero-T canonical integral is checked against
`E_free/N=-8t/pi^2` at half filling. At t=t'=0, the staggered trial family
`(J,-J,J,-J)` has continuum energy `V*(J^2-8J/pi^2)` for J>=0; its minimum
within that family is J=4/pi^2 and E/N=-16V/pi^4. These are analytical
checks of the implementation, not digitized paper result anchors or proof
that the family is the global minimum. Numerical mesh errors are saved.

The same loop-current-only scope as the paper applies. CDW, pair density
wave and competing mean-field channels are outside the variational space.
No complete many-body or unconstrained mean-field phase diagram is claimed.
