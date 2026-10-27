"""Check historical grids and color-cell recovery without rerunning physics."""

from pathlib import Path
import sys
import unittest
import numpy as np

CODE = Path(__file__).resolve().parents[1] / "1988_Affleck_code"
sys.path.insert(0, str(CODE))
from plot_utils import _grid
from replot_saved import SCANS, recover


class SavedPlotData(unittest.TestCase):
    def test_historical_coordinates_and_row_permutation(self):
        for stamp, spec in SCANS.items():
            data = np.load(CODE / f"results/data/scan_{stamp}.npy", allow_pickle=False)
            before = data.copy()
            a = _grid(data, spec["site_n"])
            b = _grid(
                data[np.random.default_rng(11).permutation(len(data))], spec["site_n"]
            )
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)
            self.assertAlmostEqual(a[0][0], 0.2)
            self.assertAlmostEqual(a[0][-1], 0.8 if stamp == "20250628_175145" else 0.5)
            for iy, ix in [(0, 0), (-1, -1)]:
                row = data[
                    (data[:, 0] == a[0][ix] * spec["site_n"]) & (data[:, 1] == a[1][iy])
                ]
                np.testing.assert_array_equal(a[2][iy, ix], row[0, 2:])
            np.testing.assert_array_equal(data, before)

    def test_duplicate_scan_coordinate_is_rejected(self):
        data = np.load(
            CODE / "results/data/scan_20250626_234707.npy", allow_pickle=False
        )
        data[0] = data[1]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _grid(data, 2500)

    def test_original_phase_anchors(self):
        for stamp, spec in SCANS.items():
            data = np.load(CODE / f"results/data/scan_{stamp}.npy", allow_pickle=False)
            shape = (len(np.unique(data[:, 1])), len(np.unique(data[:, 0])))
            labels = recover(
                CODE / "results/figs/originals" / spec["source"], spec, shape
            )
            self.assertEqual(labels[-1, 0], "Uniform")
            if stamp == "20250628_175145":
                self.assertEqual(labels[-1, 25], "Flux")
                self.assertEqual(labels[-1, -1], "Uniform")
                self.assertEqual(labels[25, 25], "Flux")
            else:
                self.assertEqual(labels[-1, -1], "Flux")
                self.assertEqual(labels[0, -1], "Peierls")


if __name__ == "__main__":
    unittest.main()
