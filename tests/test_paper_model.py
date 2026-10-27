import unittest
import numpy as np
from loop_current import CurrentModel


class PaperModel(unittest.TestCase):
    def test_bands_against_independent_hermitian_diagonalization(self):
        model=CurrentModel(mesh=16,next_nearest_hopping=.13)
        j=np.array([.12,-.21,.33,-.07]);t=.25
        bands=model.bands(j,t)
        for i,l in [(0,0),(2,5),(7,3),(15,11)]:
            kx,ky=model.kx[i,l],model.ky[i,l]
            dx,dy=[1,0,-1,0],[0,1,0,-1]
            h=-sum((t+1j*model.interaction*j[q])*np.exp(1j*(kx*dx[q]+ky*dy[q])) for q in range(4))
            d=-4*.13*np.cos(kx)*np.cos(ky)
            exact=np.linalg.eigvalsh([[d,h],[h.conjugate(),d]])
            np.testing.assert_allclose(bands[:,i,l],exact,atol=1e-14)

    def test_free_lattice_and_folding(self):
        model=CurrentModel(mesh=40)
        t=.3
        free=(-2*t*(np.cos(model.kx)+np.cos(model.ky))).ravel()
        folded=np.sort(model.bands([0]*4,t).ravel())[::2]
        np.testing.assert_allclose(np.sort(free),folded,atol=1e-14)
        # At n=.5 occupy exactly half the single-orbital free-lattice levels.
        exact=np.sort(free)[:len(free)//2].sum()/len(free)
        self.assertAlmostEqual(model.energy([0]*4,t,.5),exact,places=13)

    def test_symmetries_and_filling(self):
        model=CurrentModel(mesh=24)
        j=np.array([.12,-.21,.33,-.07]);t=.25
        energy=model.energy(j,t,.37)
        for other in [-j,np.roll(j,1)]:
            self.assertAlmostEqual(model.energy(other,t,.37),energy,places=13)
        self.assertAlmostEqual(model.energy(j,t,.63),energy,places=13)
        for f in [0,.001,.371,.5,1]:
            result=model.energy(j,t,f,details=True)
            self.assertAlmostEqual(result["weighted_occupied_states"]/(24**2),f,places=14)
        penalty=.25*sum(j*j)
        self.assertAlmostEqual(model.energy(j,t,0),penalty,places=13)
        self.assertAlmostEqual(model.energy(j,t,1),penalty,places=13)

    def test_domain(self):
        with self.assertRaises(ValueError):CurrentModel(mesh=15)
        with self.assertRaises(ValueError):CurrentModel().energy([0]*4,.2,1.1)


if __name__=="__main__":unittest.main()
