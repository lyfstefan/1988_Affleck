# Affleck mean-field calculations

This is the independent Affleck project. It is separate from the author's
[bond-order flux-phase project](https://github.com/lyfstefan/bond-order-flux-phases).

## Figures following the author's current template

The three historical scans now each have a styled phase map, an eight-panel
order-parameter figure, and a plaquette-flux figure. All text uses real LaTeX
and Computer Modern; data frames are approximately 2.5 cm square, with 10 pt
axis labels and 9 pt ticks. Continuous maps have independent short colorbars
(60% frame height, bottom-aligned, with labels above). Categorical phase maps
use a compact 7 pt color legend, avoiding a long categorical colorbar.

![Historical phase regions, latest scan](1988_Affleck_code/results/figs/phase_diagram_20250628_175145.png)

The original figures are preserved in `1988_Affleck_code/results/figs/originals/`.
Historical phase labels were recovered cell by cell from those figures, without
rerunning the heuristic classifier, changing boundaries, or interpolating cells.
The three numerical NPY scans, optimizer, model and classification code are
unchanged. Order-parameter and flux figures read the saved numerical arrays.
As in the original plotter, displayed phase angles are zeroed where the
corresponding amplitude is below 0.001; saved values are untouched.

## Reproduce the exports

```sh
python -m pip install -r requirements.txt
python 1988_Affleck_code/replot_saved.py
python -m unittest discover -s tests -v
```

LaTeX rendering also requires `latex`, `dvipng`, `type1cm`, `type1ec`, `amsmath`
and `textcomp`. On Ubuntu these are supplied by `texlive-latex-extra cm-super
dvipng ghostscript`.

This command reads only the saved scans and original figures; it never runs
`solver.py`. Coordinates come from each scan, rather than today's `config.py`:

| Scan | Grid | Historical site count | Filling range |
| --- | --- | --- | --- |
| 20250626_224949 | 40 x 40 | 10000 | 0.2–0.5 |
| 20250626_234707 | 20 x 20 | 2500 | 0.2–0.5 |
| 20250628_175145 | 50 x 50 | 2500 | 0.2–0.8 |

Site counts are inferred from the original filling-axis ranges and saved
particle numbers. The last figure retains its original `t'/t=0.2` annotation;
this is historical figure metadata, not a replacement for missing run settings.

## Files

- `1988_Affleck_code/`: scientific code and template-based plotting utilities.
- `1988_Affleck_code/results/data/scan_TIMESTAMP.npy`: original numerical scans.
- `1988_Affleck_code/results/figs/`: new PDF, SVG, PNG and typography records.
- `1988_Affleck_code/results/figs/originals/`: unchanged historical figures.
- `1988_Affleck_code/results/classification/`: recovered historical phase cells.
- `1988_Affleck_code/results/figure_provenance.json`: grid and source checksums.
- `docs/template_rules.md`: the author's current shared plotting preferences.
- `file_moves.json`: the earlier file-organization mapping, retained as history.
