# AnaCal_DESCNote

Collection of scripts and notebooks to run AnaCal on LSST data.

## Layout

| Subdir | What it is |
|---|---|
| `dp1-pre/` | Preliminary tests on DP1 (Abell 360 mask + cluster notebooks; early exploratory scripts). |
| `dp1-v1/` | First DP1 pass, run on **deep coadds** (photo-z evaluation on ecdfs, mass fit on edfs, photo-z on a360). |
| `dp1-v2/` | Current DP1 pass, run on **cell-based coadds**. Holds the `step1_*`/`step2_*`/`step3_*` scripts, per-field configs, and results for a360, edfs, ecdfs. |
| `iasim/` | Intrinsic alignment simulations. |
| `psf_tests/` | PSF tests (e.g. single-visit truncation checks). |
