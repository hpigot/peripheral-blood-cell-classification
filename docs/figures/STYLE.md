# Figure style

Every result figure comes from `src/bloodcell/plots.py`, so all figures share one
look: seaborn's `ticks` style with a quiet palette. Don't style figures by hand
in notebooks. Add a function to `plots.py`, so the next figure gets the same
treatment.

```bash
uv run bloodcell-plot runs/<run>                    # -> runs/<run>/figures/
uv run bloodcell-plot runs/<run> --out docs/figures # figures for the README
```

## Light and dark

Each figure is written twice, `<name>.png` and `<name>-dark.png`. In the README,
let GitHub pick the right one:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/reliability-dark.png">
  <img alt="Reliability diagram: …" src="docs/figures/reliability.png" width="480">
</picture>
```

The alt text states the finding (e.g. "temperature scaling cuts ECE from 3.3% to
0.9%"), not just the chart type.

## Colour has a job, or it isn't used

| Job | Encoding | Used for |
|---|---|---|
| Identity | Categorical slots 1–3: blue, orange, aqua | Models: `mobilenet_v3_small`, `efficientnet_b0`, `resnet50` (`MODEL_SLOTS`) |
| Before / after | Muted gray, then blue | Uncalibrated vs temperature-scaled; internal vs external; FP32 vs INT8 |
| Magnitude | One blue ramp, light → dark (dark mode: dark → light) | Confusion matrix |
| Categories on an axis | No colour: the axis label names them | Cell classes |

- **A model keeps its colour in every figure.** Colour follows the entity, never
  its position in a particular chart.
- **At most 3 coloured series per figure.** The three slots pass the colour-vision
  checks on all pairs in both modes: worst CVD ΔE 9.2 light / 9.4 dark,
  normal-vision ΔE ≥ 20.9. Past three, facet into small multiples. The plotting
  functions raise an error rather than invent a fourth colour.
- Light-mode aqua is below 3:1 contrast on the surface, so a figure using slot 3
  also needs direct labels or a legend.
- Text never takes a series colour. Titles and values use ink, axis labels and
  legends use secondary ink, and numeric ticks are muted. Class names on an axis
  use secondary ink because they identify the rows.
- No rainbow colormaps, and no status colours (red/green) for series.

## Forms

| Result | Form | Why not the obvious alternative |
|---|---|---|
| Confusion matrix | Row-normalized heatmap with 2px gaps between cells; cells ≥ 0.5% labelled | Raw counts hide errors in small classes |
| Per-class recall | Dot plot, weakest class at the bottom, one dot per model | Recalls sit near 100%, and bars would need a zero baseline that flattens the differences |
| Calibration | Reliability curve with the diagonal, and a bin-count strip below (log scale) | A curve with no counts overstates sparse, noisy bins |
| Training | Small multiples: loss, val balanced accuracy, val ECE | Never two y-axes on one chart |

## Marks

- Lines 2px with round caps. Dots about 8px across, with a 2px ring in the
  surface colour so they stay legible where they overlap.
- Gridlines are 1px, solid and one step off the surface, on the value axis only.
  Top and right spines are off.
- Label selectively: the endpoint of a curve, visible cells in the heatmap.
  Never a number on every point.
- One series needs no legend box, because the title names it. Two or more always
  get a legend.
- DejaVu Sans, which ships with matplotlib, so figures render the same on the
  desktop, the laptop and CI. Titles are left-aligned, sentence case, and name
  the data and the split ("… (test set)").
- 200 dpi PNG.

## Palette

| Role | Light | Dark |
|---|---|---|
| Surface | `#fcfcfb` | `#1a1a19` |
| Ink / secondary ink / muted | `#0b0b0b` / `#52514e` / `#898781` | `#ffffff` / `#c3c2b7` / `#898781` |
| Gridline / axis | `#e1e0d9` / `#c3c2b7` | `#2c2c2a` / `#383835` |
| Slot 1, 2, 3 | `#2a78d6`, `#eb6834`, `#1baf7a` | `#3987e5`, `#d95926`, `#199e70` |

If you change a series colour, re-run a colour-vision validator on the set, in
both modes, on all pairs, and update the numbers above.
