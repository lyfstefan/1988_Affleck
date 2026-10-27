"""Restyle historical figures and saved scans, without running the solver.

Original color cells supply the historical phase labels. Particle coordinates
come from each NPY; site counts and historical annotations are checked against
the corresponding original figure, independently of the current config.py.
"""

from pathlib import Path
import argparse
import hashlib
from io import BytesIO
import json
import numpy as np
from PIL import Image
from plot_utils import (
    PHASE_NAMES,
    plot_phase_diagram,
    plot_order_parameters,
    plot_flux_square,
)

SCANS = {
    "20250626_224949": {
        "site_n": 10000,
        "source": "phase_diagram_20250626_224949.png",
        "bounds": [209, 212, 908, 910],
    },
    "20250626_234707": {"site_n": 2500, "source": "phase_diagram_20250626_234707.pdf"},
    "20250628_175145": {
        "site_n": 2500,
        "source": "phase_diagram_20250628_175145.png",
        "bounds": [209, 213, 908, 912],
        "title": r"Phase diagram: $t'/t=0.2$",
    },
}
ORIGINAL_PALETTE = np.array(
    [
        [220, 220, 220],
        [255, 99, 71],
        [0, 191, 255],
        [255, 215, 0],
        [135, 206, 235],
        [0, 0, 0],
    ]
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(source, spec, shape):
    if source.suffix == ".pdf":
        import fitz

        with fitz.open(source) as doc:
            images = doc[0].get_images(full=True)
            if len(images) != 1:
                raise ValueError("Expected one original phase-map image")
            image = Image.open(
                BytesIO(doc.extract_image(images[0][0])["image"])
            ).convert("RGB")
        bounds = (0, 0, *image.size)
    else:
        image = Image.open(source).convert("RGB")
        bounds = spec["bounds"]
    pixels = np.asarray(image)
    x0, y0, x1, y1 = bounds
    ny, nx = shape
    labels = np.empty(shape, dtype="<U7")
    for iy in range(ny):
        for ix in range(nx):
            x = int(x0 + (ix + 0.5) * (x1 - x0) / nx)
            y = int(y0 + (ny - iy - 0.5) * (y1 - y0) / ny)
            patch = pixels[y - 1 : y + 2, x - 1 : x + 2].astype(float)
            distance = np.max(
                np.abs(patch[:, :, None, :] - ORIGINAL_PALETTE[None, None, :, :]),
                axis=3,
            )
            indices = distance.argmin(axis=2)
            if np.max(distance.min(axis=2)) > 1 or not np.all(indices == indices[0, 0]):
                raise ValueError(f"Ambiguous original cell {iy,ix}")
            labels[iy, ix] = PHASE_NAMES[int(indices[0, 0])]
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path(__file__).parent / "results"
    )
    args = parser.parse_args()
    base = args.results
    report = {}
    for timestamp, spec in SCANS.items():
        data_path = base / "data" / f"scan_{timestamp}.npy"
        data = np.load(data_path, allow_pickle=False)
        source = base / "figs/originals" / spec["source"]
        particles, hoppings = np.unique(data[:, 0]), np.unique(data[:, 1])
        shape = (len(hoppings), len(particles))
        labels = recover(source, spec, shape)
        metadata = {
            "timestamp": timestamp,
            "source_figure": str(source.relative_to(base)),
            "source_sha256": sha(source),
            "source_scan": str(data_path.relative_to(base)),
            "scan_sha256": sha(data_path),
            "provenance": "Historical plotted categories recovered without rerunning the classifier or optimization",
            "site_n_from_original_figure_axis": spec["site_n"],
            "shape": list(shape),
            "fillings": (particles / spec["site_n"]).tolist(),
            "hoppings": hoppings.tolist(),
            "phase_counts": {name: int(np.sum(labels == name)) for name in PHASE_NAMES},
            "labels": labels.tolist(),
        }
        folder = base / "classification"
        folder.mkdir(exist_ok=True)
        (folder / f"phase_regions_{timestamp}.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        plot_phase_diagram(
            labels,
            timestamp,
            folder=base / "figs",
            fillings=particles / spec["site_n"],
            hoppings=hoppings,
            title=spec.get("title", "Phase diagram"),
        )
        plot_order_parameters(
            data, timestamp, folder=base / "figs", site_n=spec["site_n"]
        )
        plot_flux_square(
            data, SITE_N=spec["site_n"], timestamp=timestamp, folder=base / "figs"
        )
        assert sha(data_path) == metadata["scan_sha256"]
        report[timestamp] = {k: v for k, v in metadata.items() if k != "labels"}
        print(timestamp, shape, metadata["phase_counts"], flush=True)
    (base / "figure_provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
