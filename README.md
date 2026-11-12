# HN-Transformer Warehouse Study Reproduction

This project uses the HN-Transformer method proposed in the paper to reproduce the trajectories from the experimental data of the paper, the performance analysis results for different mobile phone carrying methods, and the final performance evaluation data. It includes the input recordings, reference routes, and scoring criteria. The one-command workflow saves Tables 2–4 and the four-panel trajectory comparison in `results/`.

## Quick start

Use Python 3.11 from the project root. Check the active interpreter with `python --version` before installing dependencies. The CPU run uses four threads by default.

If you already have Python 3.11 and the dependencies listed in `requirements.txt`, run the study directly.

```bash
python Run.py
```

To create a virtual environment using an installed Python 3.11 interpreter, run the following on macOS or Linux.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python Run.py
```

On Windows PowerShell, create the environment with `py -3.11 -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`. Use a fresh directory when replacing a virtual environment created with another Python version.

## Package contents

Path

Purpose

`Run.py`

Run the reconstruction and scoring entry points, then save Tables 2–4 and the comparison figure.

`reconstruct.py`

Run the frozen model and optionally the original PDR baseline.

`summarize_lengths.py`

Calculate per-record metrics and grouped means from trajectories or an existing path-length CSV.

`export_kpis.py`

Extract stop intervals and task KPIs, then recalculate scores.

`warehouse/`

Local inference, alignment, PDR, stop matching, and scoring code.

`config/`

Study recording manifest, baseline settings, and SHA-256 checksums.

`models/`

Frozen `A6_huber_gnorm.pt` checkpoint and model metadata.

`data/handheld/` and `data/carrying/`

Preprocessed IMU inputs for the two study cohorts.

`data/pdr_raw/`

Device-frame accelerometer, linear accelerometer, and gyroscope CSV files for the 36 baseline recordings.

`data/reference/`

Reference polylines and densely sampled reference points.

`paper_reference/`

Original workbook KPI inputs and scoring thresholds for recalculating Tables 3 and 4.

## Study cohorts and metrics

-   The handheld cohort contains 50 recordings, with 20 on Path1, 14 on Path2, and 16 on Path3. `Path2_Personnel3_02` and `Path2_Personnel3_03` share an input checksum. Both recording IDs are included in the manuscript analysis.
-   The additional carrying-mode cohort contains 36 recordings, with three repetitions of each of four carrying modes on each route.
    HN-Transformer uses the inputs preceding the later gyroscope-bias adjustment.
    The original PDR baseline uses the same cohort and reads the device-frame sensor files.
-   The trajectory-length ratio (TLR) is predicted path length divided by reference path length. The absolute relative path-length error is `100 × |TLR − 1|`. Group means give equal weight to each recording. The reference lengths of Path1, Path2, and Path3 are 107.6 m, 96.2 m, and 96.6 m.

## Generated files

`python Run.py` writes three CSV files and one figure in `results/`.

| File | Contents |
| --- | --- |
| `table2.csv` | Recording counts, mean TLR, and mean absolute relative path-length error by route and carrying mode, calculated from reconstructed HN-Transformer trajectories. |
| `table3.csv` | Path3 personnel KPI values from the original scoring workbook and the scores recalculated from those values. |
| `table4.csv` | Mean KPI scores and composite scores by route and carrying mode, recalculated from the original scoring workbook. |
| `warehouse_2x2_Path2_Hand_Swing01.png` | Four-panel comparison of the reference route, original PDR, and HN-Transformer at 600 dpi. |

The launcher uses temporary storage for reconstructed trajectories and cleans it up after generating the tables and figure. Use `python Run.py --out results/rerun` to run again in a separate directory.The standalone scripts can also export detailed analysis when run individually. Run `summarize_lengths.py` or `export_kpis.py` with `--paper-only` to generate only their manuscript tables. The detailed modes include comparison files when the corresponding manuscript reference CSVs are present in `paper_reference/`.

Coordinates and path lengths are in meters. CSV files use UTF-8 with a byte order mark. Read trajectory files with `numpy.load(path, allow_pickle=False)`. The length script can also process an existing per-record CSV using `--length-table`.

### Trajectory comparison figure

The panels show `Path2_Hand_Swing01`, `Path3_Trouser_Pocket02`, `Path3_Jacket_Pocket02`, and `Path1_Freedom03` in reading order. `Path3_Trouser_Pocket02` is the original recording `Path3_Trouser_Pocket05` under its review-package identifier. The figure uses the reconstructed trajectories from the current run and follows the original panel layout and plot style. Matplotlib saves the figure without opening a display window.
