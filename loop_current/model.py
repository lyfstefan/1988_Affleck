"""Zero-temperature four-bond-current mean-field model on a square lattice.

Energies are per physical site. A full square Brillouin-zone grid double
counts the folded two-sublattice representation; each band entry has
weight 1/2. Filling is particles per physical site, in [0,1].
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import dual_annealing, minimize


@dataclass(frozen=True)
class CurrentModel:
    mesh: int = 40
    interaction: float = 1.0
    next_nearest_hopping: float = 0.0

    def __post_init__(self):
        if not isinstance(self.mesh,int) or self.mesh<4 or self.mesh%2:
            raise ValueError("mesh must be an even integer >= 4 for full-BZ folding")
        if not np.isfinite(self.interaction) or self.interaction<=0:
            raise ValueError("interaction V must be finite and positive")
        if not np.isfinite(self.next_nearest_hopping):
            raise ValueError("next-nearest hopping must be finite")
        k=np.linspace(-np.pi,np.pi,self.mesh,endpoint=False)
        kx,ky=np.meshgrid(k,k,indexing="ij")
        object.__setattr__(self,"kx",kx)
        object.__setattr__(self,"ky",ky)
        phases=np.stack([np.exp(1j*kx),np.exp(1j*ky),np.exp(-1j*kx),np.exp(-1j*ky)])
        object.__setattr__(self,"_phases",phases)

    @staticmethod
    def _currents(currents):
        currents=np.asarray(currents,dtype=float)
        if currents.shape!=(4,) or not np.isfinite(currents).all():
            raise ValueError("currents must contain four finite real J_nu")
        return currents

    def hopping(self,currents,t):
        """Paper t_nu=t+i*V*J_nu, directions +x,+y,-x,-y."""
        currents=self._currents(currents)
        if not np.isfinite(t):
            raise ValueError("t must be finite")
        return t+1j*self.interaction*currents

    def bands(self,currents,t):
        tnu=self.hopping(currents,t)
        h12=-np.sum(tnu[:,None,None]*self._phases,axis=0)
        common=-4*self.next_nearest_hopping*np.cos(self.kx)*np.cos(self.ky)
        # Eigenvalues of [[common,h12],[h12*,common]] are common +/- |h12|.
        return np.stack([common-np.abs(h12),common+np.abs(h12)])

    def energy(self,currents,t,filling,details=False):
        """Canonical zero-T energy with exact fixed filling on the discrete mesh.

        Sort band entries and fractionally fill the last entry if needed.
        This avoids the legacy chemical-potential tolerance/degeneracy error.
        Fractional filling within a degenerate shell changes no total energy.
        """
        currents=self._currents(currents)
        if not np.isfinite(filling) or not 0<=filling<=1:
            raise ValueError("filling must lie in [0,1]")
        levels=np.sort(self.bands(currents,t).ravel())
        occupied=2*self.mesh**2*filling
        full=int(np.floor(occupied))
        fraction=occupied-full
        electronic=float(levels[:full].sum())
        if full<len(levels):
            electronic+=fraction*levels[full]
        electronic/=2*self.mesh**2
        cost=self.interaction/4*float(currents@currents)
        total=electronic+cost
        if not details:
            return total
        if full==0: mu=float(levels[0])
        elif full==len(levels): mu=float(levels[-1])
        elif fraction: mu=float(levels[full])
        else: mu=float((levels[full-1]+levels[full])/2)
        return {"energy_per_site":total,"electronic_energy_per_site":electronic,
                "current_cost_per_site":cost,"filling":float(filling),
                "weighted_occupied_states":float(occupied/2),"chemical_potential":mu}

    def minimize(self,t,filling,seeds=(0,1,2),maxiter=64,current_bound=1.0):
        """Seeded annealing + local refinement within an explicit finite box.

        No claim of global optimality, phase-boundary convergence, or recovery
        of the paper's lexicographic symmetry-reduction optimizer.
        """
        if not seeds or not isinstance(maxiter,int) or maxiter<1:
            raise ValueError("provide seeds and maxiter >= 1")
        if not np.isfinite(current_bound) or current_bound<=0:
            raise ValueError("current bound must be positive and finite")
        bounds=[(-current_bound,current_bound)]*4
        fn=lambda j:self.energy(j,t,filling)
        zero=np.zeros(4)
        # Keep the feasible zero-current trial even if stochastic search is poor.
        best_j,best_energy=zero,fn(zero)
        trials=[]
        for seed in seeds:
            global_result=dual_annealing(fn,bounds,seed=int(seed),maxiter=maxiter,no_local_search=True)
            local=minimize(fn,global_result.x,method="L-BFGS-B",bounds=bounds,options={"maxiter":1000,"ftol":1e-12})
            j=local.x if local.fun<global_result.fun else global_result.x
            value=fn(j)
            trials.append({"seed":int(seed),"energy":float(value),"local_success":bool(local.success),
                           "local_message":str(local.message),"currents":j.tolist()})
            if value<best_energy:
                best_j,best_energy=j.copy(),value
        return {"currents":best_j.tolist(),**self.energy(best_j,t,filling,details=True),
                "zero_current_energy":float(fn(zero)),"t":float(t),"interaction_V":self.interaction,
                "next_nearest_hopping":self.next_nearest_hopping,"mesh":self.mesh,
                "seeds":list(seeds),"annealing_maxiter":maxiter,"current_bound":current_bound,
                "near_search_boundary":bool(max(abs(best_j))>.98*current_bound),"trials":trials}
