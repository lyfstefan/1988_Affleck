"""Compact LaTeX figures from existing Affleck arrays; no optimization here."""

from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
from matplotlib.patches import Patch
from config import N_MIN, N_MAX, T_MIN, T_MAX, SITE_N
from publication_style import apply_style, width, save, panel_colorbar, C

PHASE_NAMES = ["Uniform", "Flux", "Kite", "Peierls", "Stripy", "Other"]
PHASE_COLORS = [C[key] for key in ["light", "red", "blue", "yellow", "sky", "black"]]


def flux_mod_2pi(flux):
    """Oriented flux in [0, 2 pi), with 1e-10 rad endpoint roundoff removed."""
    value = np.remainder(flux, 2 * np.pi)
    endpoint = np.isclose(value, 0, atol=1e-10, rtol=0) | np.isclose(
        value, 2 * np.pi, atol=1e-10, rtol=0
    )
    return np.where(endpoint, 0.0, value)


def _finish(fig, folder, name, *, legend_fontsize=None):
    destination = Path(folder) / name
    save(fig, destination, legend_fontsize=legend_fontsize)
    plt.close(fig)
    return destination.with_suffix(".pdf")


def plot_phase_diagram(
    phases,
    timestamp,
    folder="results/figs",
    prefix="phase_diagram",
    *,
    fillings=None,
    hoppings=None,
    title="",
):
    """Display supplied categories without classifying or smoothing any cell."""
    apply_style()
    phases = np.asarray(phases)
    if phases.ndim != 2 or not np.isin(phases, PHASE_NAMES).all():
        raise ValueError("Expected a rectangular grid of known phase names")
    fillings = np.asarray(
        fillings
        if fillings is not None
        else np.linspace(N_MIN / SITE_N, N_MAX / SITE_N, phases.shape[1])
    )
    hoppings = np.asarray(
        hoppings if hoppings is not None else np.linspace(T_MIN, T_MAX, phases.shape[0])
    )
    if phases.shape != (len(hoppings), len(fillings)):
        raise ValueError("Phase grid and coordinates have different shapes")
    codes = np.empty(phases.shape, dtype=int)
    for index, label in enumerate(PHASE_NAMES):
        codes[phases == label] = index
    fig, ax = plt.subplots(figsize=(width(), width() + 0.5), layout="constrained")
    ax.pcolormesh(
        fillings,
        hoppings,
        codes,
        shading="nearest",
        rasterized=True,
        cmap=ListedColormap(PHASE_COLORS),
        norm=BoundaryNorm(np.arange(7) - 0.5, 6),
    )
    ax.set(xlabel=r"$\nu$", ylabel=r"$t/J$", title=title)
    handles = [
        Patch(facecolor=color, label=name)
        for color, name in zip(PHASE_COLORS, PHASE_NAMES)
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        handlelength=0.8,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.2,
    )
    return _finish(fig, folder, f"{prefix}_{timestamp}", legend_fontsize=7)


def _grid(data, site_n):
    """Use the coordinates saved in each row, independently of current config."""
    rows = np.asarray(data, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 11 or not np.isfinite(rows).all():
        raise ValueError("Expected finite (N,t,chi[4],phi[4],energy) rows")
    particles, hoppings = np.unique(rows[:, 0]), np.unique(rows[:, 1])
    if len(rows) != len(particles) * len(hoppings) or site_n <= 0:
        raise ValueError("Scan is not a complete rectangular grid")
    values = np.full((len(hoppings), len(particles), 9), np.nan)
    for row in rows:
        iy, ix = np.searchsorted(hoppings, row[1]), np.searchsorted(particles, row[0])
        if np.isfinite(values[iy, ix]).any():
            raise ValueError("Duplicate scan coordinate")
        values[iy, ix] = row[2:]
    if not np.isfinite(values).all():
        raise ValueError("Missing scan coordinate")
    return particles / site_n, hoppings, values


def plot_order_parameters(
    data, timestamp=None, folder="results/figs", *, site_n=SITE_N
):
    apply_style()
    fillings, hoppings, values = _grid(data, site_n)
    chis, phis = values[:, :, :4], values[:, :, 4:8].copy()
    # Preserve the original display convention for undefined tiny-amplitude phases.
    phis[np.abs(chis) < 0.001] = 0.0
    fig, axes = plt.subplots(
        2, 4, figsize=(width(2), width(2) * 0.5), layout="constrained"
    )
    for row, (field, bound, symbol) in enumerate(
        [(chis, 1.0, r"\chi"), (phis, np.pi, r"\phi")]
    ):
        for col in range(4):
            ax = axes[row, col]
            im = ax.pcolormesh(
                fillings,
                hoppings,
                field[:, :, col],
                shading="nearest",
                cmap="RdBu",
                norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
                rasterized=True,
            )
            ax.set(xlabel=r"$\nu$", ylabel=r"$t/J$", title=f"({chr(97+4*row+col)})")
            panel_colorbar(fig, ax, im, rf"${symbol}_{{{col+1}}}$")
    return _finish(
        fig, folder, "order_parameters_" + (timestamp or time.strftime("%Y%m%d_%H%M%S"))
    )


def plot_flux_square(
    data,
    N_MIN=None,
    N_MAX=None,
    T_MIN=None,
    T_MAX=None,
    SITE_N=SITE_N,
    N_NUM=None,
    T_NUM=None,
    *,
    timestamp=None,
    folder="results/figs",
):
    """Plot the oriented plaquette flux modulo 2 pi; stored arrays are untouched."""
    apply_style()
    fillings, hoppings, values = _grid(data, SITE_N)
    chis, phis = values[:, :, :4], values[:, :, 4:8]
    theta = np.angle(hoppings[:, None, None] + chis * np.exp(-1j * phis))
    flux = (
        flux_mod_2pi(-theta[:, :, 0] + theta[:, :, 1] - theta[:, :, 2] + theta[:, :, 3])
        / np.pi
    )
    fig, ax = plt.subplots(figsize=(width(), width()), layout="constrained")
    im = ax.pcolormesh(
        fillings,
        hoppings,
        flux,
        shading="nearest",
        cmap="twilight",
        vmin=0,
        vmax=2,
        rasterized=True,
    )
    ax.set(xlabel=r"$\nu$", ylabel=r"$t/J$", title="")
    panel_colorbar(fig, ax, im, r"$\Phi/\pi$")
    return _finish(
        fig, folder, "flux_square_" + (timestamp or time.strftime("%Y%m%d_%H%M%S"))
    )
