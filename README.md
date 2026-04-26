# Emergent-speed-synchrony-promotes-mixed-species-schooling

This repository contains Jupyter Notebooks used for the analysis in:

> Tiwari et al. *Emergent speed synchrony promotes collective motion in mixed-species schools*.

---

# Notebooks

| Notebook                                | Description                                                                                                                                             |
| --------------------------------------- | -----------------------------------------------------------------------------------------------------------------------------------------------------   |
| `Mix species_Group properties.ipynb`    | Computes and visualises group-level properties: individual speed, group polarization, and nearest-neighbour distance across treatments and replicates |
| `Spatial sorting analysis.ipynb`        | Computes strong sorting percent in empirical data and compares it to randomized null models                                                           |
| `Mixed species_model simulations.ipynb` | Runs mixed species model simulations of many trials, with the same model parameter set but varying initial conditions.                                 |
---

# Data

## Source

The datasets contain processed arrays derived from raw tracking data; available at  **([Zenodo](https://zenodo.org/records/19690109?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQ4NzI2ZjhlLTdhNGMtNDJjYi1hN2YzLWJmNjI4NzI3Zjg3NyIsImRhdGEiOnt9LCJyYW5kb20iOiJhNjk1ZjA2NzdjM2QyNjY3ZTM3MTVjMzkxZDgzNmNjZCJ9.PsTJIm-r98SE9VW10jY15xocLpZzFFkEtQOEV0O9DZVCxSf2qVtqGAkFUKUrh5A4EoP7RZLsS26dUKv1MeWQ8g))**

## `MS_RB+TB_pol-vel-nnd.npz`

*Input for `Mix species_Group properties.ipynb`*

| Key   | Shape                                                           | Description                                                         |
| ----- | --------------------------------------------------------------- | ------------------------------------------------------------------- |
| `vel` | `(replicates, treatments, frames, individuals, 2)`              | 2D velocity vectors (x, y) in body lengths per second (BL/s)        |
| `pol` | `(replicates, treatments, frames, 2)`                           | Group polarization vector (norm gives scalar polarization ∈ [0, 1]) |
| `nnd` | `(replicates, treatments, frames, individuals, individuals)`    | Nearest-neighbour distance per individual (in BL)                   |

---

### Synthetic test data

```python
import numpy as np

FRAMES = 1000

np.savez('MS_RB+TB_pol-vel-nnd.npz',
    vel=np.random.randn(5, 5, FRAMES, 16, 2),
    pol=np.random.randn(5, 5, FRAMES, 2),
    nnd=np.abs(np.random.randn(5, 5, FRAMES, 16))
)
```

---

## `RB12+TB4.csv`, `RB8+TB8.csv`, `RB4+TB12.csv`

*Input for `Spatial sorting analysis.ipynb`*

| Column              | Description                                             |
| ------------------- | ------------------------------------------------------- |
| `FRAME ID`          | Frame index                                             |
| `FishID`            | Fish ID                                                 |
| `Nearest Neighbour` | Nearest Neighbour ID                                    |
| `x`                 | X coordinate                                            |
| `y`                 | Y coordinate                                            |
| `SPECIES.ID`        | Species identity (RB = rosy barbs and TB = tiger barbs) |


---

## ⚙️ How to Run

### Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy jupyter
```

---

### Notebook 1 — Mix species - group Properties
1. Place `Ma_RB+TB_pol-vel-nnd.npz` in the same directory.
2. Run:

   ```bash
   jupyter notebook "Mix species_Group properties.ipynb"
   ```
3. Execute all cells.

---

### Notebook 2 — Spatial Sorting analysis

1. Place `RB12+TB4.csv`, `RB8+TB8.csv` and `RB4+TB12.csv` in the same directory.
2. Run:

   ```bash
   jupyter notebook "Spatial sorting analysis.ipynb"
   ```
3. Execute all cells.

---

### Notebook 3 — Mixed species model simulations

1. Place `spatialmodels.py` and `utilities.py` in the same directory.
2. Run:

   ```bash
   jupyter notebook "Mixed species_model simulations.ipynb"
   ```
3. Execute all cells.
4. Place simulations datset in the same directory to generate simulation plots.

---

## Outputs

### Group properties notebook

| Figure    | Description                                            | File                                  |
| --------- | ------------------------------------------------------ | ------------------------------------- |
| Fig. 1    | KDE of nearest-neighbour distance                      | `near_neighbour_distance_kdeplot.png` |
| Fig. 2    | KDE of polarization                                    | `group_polarization_kdeplot.png`      |
| Fig. 3    | First vs second half comparison of speed distribution  | `first_vs_second_half_hist.png`       |
| Fig. 4    | Individual speed distributions (single-species)        | `individual_speed_gs16_.png`          |
| Fig. 5    | Individual speed distributions (mixed-species + null)  | `individual_speed_gs16_null.png`      |
| Fig. 6    | Relationship between speed and polarization            | `fspeed_polarization.png`       |

---

### Spatial sorting notebook

| Figure  | Description                             | File                       |
| ------- | --------------------------------------- | -------------------------- |
| Fig. 1  | Strong sorting percentage vs null model | `sorting_percent_plot.png` |

---

### Mixed species model simulation notebook

| Figure  | Description                             | File                       |
| ------- | --------------------------------------- | -------------------------- |
| Fig. 1  | Single species simulations for individual speed distribution | `individual_speed_model.png` |
| Fig. 2  | Mixed species simulations for individual speed distribution  | `individual_speed_mixed species model.png` |
| Fig. 3  | Model simulation strong sorting percentage | `sorting_model.png` |
| Fig. 4  | Model simulation for group polarization | `polarization_model.png` |

---

## Notes

- All `.npz` / `.csv` data files must be placed in the **same directory** as the notebook that reads them.
- Array shapes must match expected dimensions exactly (see Data section above).
- The `vel` and `pol` arrays use index order `[replicate][treatment]`; make sure any new data follows this convention.
- Stopping events (speed < 0.001 BL/s) are set to `NaN` before analysis.
- A `treatment` is the same as group composition and is the ratio of rosy barbs (RB) to tiger barbs (TB) in a group of 16 in this work and `replicate`is an independent experimental trial.

---

## Reproducibility

- Null models are generated by randomly shuffling species identities within each frame.
- Results may vary slightly depending on random seed; set a seed for reproducibility if required.

---

## Contact

For questions or collaboration, please contact the author.
