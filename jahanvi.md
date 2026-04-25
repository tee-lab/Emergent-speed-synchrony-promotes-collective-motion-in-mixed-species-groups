This repository contains Jupyter Notebooks for analysis published in:

> Tiwari et al. *Emergent speed synchrony promotes collective motion in mixed-species schools*. [JOURNAL], [YEAR]. DOI: [DOI]

| Notebook | Description |
|---|---|
| `Mix species_Group properties.ipynb` | Computes and visualises group-level properties: individual speed, group polarization, and nearest-neighbour distance across treatments and replicates |
| `Spatial sorting.ipynb` | Computes Strong sorting percent in real data and randomized null data |

---

## Data

### Source

The `.npz` input files contain precomputed arrays derived from raw tracking data.
The preprocessing code that generates these files is located in: [LINK TO OTHER REPO / NOTEBOOK — e.g. `https://github.com/[ORG]/[PREPROCESSING-REPO]`] 

---

### `pol-vel-nnd.npz` — Input for `Mix species_Group properties.ipynb`

| Key | Shape | dtype | Description |
|-----|-------|-------|-------------|

| `vel` | `(replicates, treatments, frames, individuals, 2)` | float | Per-individual 2D velocity vectors (x, y) in body lengths per second (BL/s) |
| `pol` | `(replicates, treatments, frames, 2)` | float | Group polarization vector (x, y) per frame — norm gives scalar polarization ∈ [0, 1] |
| `nnd` | `(replicates, treatments, frames, individuals)` | float | Nearest-neighbour distance per individual per frame, in body lengths (BL) |


> To test with synthetic data {synthetic data once that could be run through the preprocessing code, that is mentioned above, which generates the files ahead}:
> ```python
> import numpy as np
> np.savez('pol-vel-nnd.npz',
>     vel=np.random.randn(5, 5, [FRAMES], 16, 2),
>     pol=np.random.randn(5, 5, [FRAMES], 2),
>     nnd=np.abs(np.random.randn(5, 5, [FRAMES], 16, 16))
> )
> ```

---

### `[FILENAME].npz` — Input for `Spatial sorting.ipynb`

| Key | Shape | dtype | Description |
|-----|---|---|---|
|

---

## How to Run

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy jupyter
```

### Notebook 1 — Group Properties

1. Place `pol-vel-nnd.npz` in the same directory as the notebook.
2. Launch:
   ```bash
   jupyter notebook "Mix species_Group properties.ipynb"
   ```
3. Run all cells sequentially.

### Notebook 2 — Spatial Sorting

1. Place `sorting percent.csv` in the same directory as the notebook.
2. Launch:
   ```bash
   jupyter notebook "Spatial sorting.ipynb"
   ```
3. Run all cells sequentially.

---

## Outputs

### `Mix species_Group properties.ipynb`

| Figure | Description | Saved as |
|---|---|---|
| Fig. 1 C | KDE of nearest-neighbour distance across treatments | `near_neighbour_distance_kdeplot.png` |
| Fig. 1 D | KDE of group polarization across treatments | `group_polarization_kdeplot.png` |
| Fig. 2 A-B | Individual speed distributions — single-species trials | `individual_speed_gs16_.png` |
| Fig. 2 C-D | Individual speed distributions — mixed trials with null model | `individual_speed_gs16_null.png` |
| SI Fig. S2 | First vs. second half speed distributions per replicate × treatment | `first_vs_second_half_hist.png` |


Summary statistics (mean, median, mode, SD) for NND, polarization, and speed are printed to the notebook output; uncomment `fig.savefig(...)` lines to save figures to disk.

### `Spatial sorting.ipynb`

| Figure | Description | Saved as |
|---|---|---|
| Fig. 3 B | [DESCRIPTION] | `sorting percent plot.png` |

---

## Notes

- All `.npz` data files must be placed in the **same directory** as the notebook that reads them.
- Array shapes must match expected dimensions exactly (see Data section above).
- The `vel` and `pol` arrays use index order `[replicate][treatment]`; make sure any new data follows this convention.
- Stopping events (speed < 0.001 BL/s) are set to `NaN` before analysis.
- A `treatment` is the same as group composition and is the ratio of rosy barbs (RB) to tiger barbs (TB) in a group of 16 in this work and `replicate`is an independent experimental trial; for each replicate there are the different treatments.

## Citation

If you use this code, please cite:

```bibtex
@article{tiwari[YEAR],
  author  = {Tiwari, Jahanvi and [CO-AUTHORS]},
  title   = {Emergent speed synchrony promotes collective motion in mixed-species schools},
  journal = {[JOURNAL]},
  year    = {[YEAR]},
  doi     = {[DOI]}
}
```

