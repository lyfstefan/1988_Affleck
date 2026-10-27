import json
import platform
from pathlib import Path
import numpy as np
import scipy
from loop_current import CurrentModel


def main():
    report={"implementation":"2026 paper-based four-current model, alongside preserved exploratory sources",
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__}}
    j=np.array([.12,-.21,.33,-.07])
    # At fixed trial current, compare mesh sizes without stochastic optimization.
    report["fixed_trial_mesh_sequence"]=[{"mesh":n,"energy_per_site":CurrentModel(n).energy(j,.25,.37)} for n in [24,40,80,120]]
    report["fixed_trial_currents"]=j.tolist()
    report["fixed_trial_t"]=.25;report["fixed_trial_filling"]=.37
    # Free square lattice at half filling has E/N=-8t/pi^2 in the continuum.
    t=.3
    exact=-8*t/np.pi**2
    computed=CurrentModel(120).energy([0]*4,t,.5)
    assert abs(computed-exact)<1e-4
    report["half_filled_free_lattice"]={"mesh":120,"t":t,"continuum_energy":exact,"computed_energy":computed,"absolute_error":abs(computed-exact)}
    # Zero hopping, staggered currents J,-J,J,-J have a known continuum
    # stationary amplitude J=4/pi^2 and energy -16*V/pi^4 at half filling.
    # This is an analytic trial family, not a claim of global optimality.
    amplitude=4/np.pi**2
    ansatz=np.array([1,-1,1,-1])*amplitude
    e=CurrentModel(120).energy(ansatz,0,.5)
    exact_staggered=-16/np.pi**4
    assert abs(e-exact_staggered)<2e-4
    report["staggered_current_trial_t0"]={"mesh":120,"currents":ansatz.tolist(),"continuum_trial_energy":exact_staggered,"computed_energy":e,"absolute_error":abs(e-exact_staggered)}
    # Small deterministic workflow smoke check, intentionally below publication
    # mesh/search budgets; save each seed and refinement status.
    result=CurrentModel(24).minimize(.2,.4,seeds=(0,1,2),maxiter=32)
    assert result["energy_per_site"]<=result["zero_current_energy"]+1e-12
    assert not result["near_search_boundary"]
    report["bounded_optimization_example"]=result
    report["scope"]="No full phase-diagram rerun; no CDW competition; fixed-trial convergence is not optimized phase-boundary convergence. Stochastic search is not a certificate of a global minimum."
    report["all_checks_passed"]=True
    Path(__file__).with_name("report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"all_checks_passed":True,"free_lattice_error":abs(computed-exact),"bounded_example_energy":result["energy_per_site"],"trial_energy":result["zero_current_energy"]},indent=2))


if __name__=="__main__":main()
