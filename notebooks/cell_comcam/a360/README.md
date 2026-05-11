# AnaCal Shear Measurement on Cell-based Coadds (Abell 360, LSSTComCam DP1)

This directory contains the pipeline configuration and analysis notebooks for
running AnaCal shear measurements on LSST cell-based coadds for the Abell 360
cluster field.

## Prerequisites

Source the LSST v30.0.4.rc1 environment:

```bash
source setup_lsst_v30.bash
```

This sets up the LSST stack, `xlens`, `anacal`, and the custom DRP
pipe/tasks repos with cell-based coadd support.

## Files

| File | Description |
|---|---|
| [`measure_pipeline.yaml`](measure_pipeline.yaml) | Full pipeline: `buildCellSystematics` (mask + PSF + noise-correlation stacks), `measureCellCoadds` (detection + per-band forced FPFS), `photoZ` (FlexZBoost on the per-patch AnaCal catalog), and `mergeFpfs` (per-tract band combination + photo-z join + WCS distortion correction) |
| [`measure_pipeline_sim.yaml`](measure_pipeline_sim.yaml) | Variant pipeline for image-simulation inputs |
| [`parsl.yaml`](parsl.yaml) | BPS/Parsl configuration for batch submission on Perlmutter |
| [`parsl_sim.yaml`](parsl_sim.yaml) | BPS/Parsl configuration for the simulation pipeline |
| [`export_data.py`](export_data.py) | Helper to export cell coadds + systematics masks to scratch |
| [`abell360_cluster_cell.ipynb`](abell360_cluster_cell.ipynb) | Per-patch (pre-merge) analysis notebook: tangential shear profile and aperture-mass map, reproducing the band combination + WCS correction inline |
| [`abell360_cluster_treecorr.ipynb`](abell360_cluster_treecorr.ipynb) | TreeCorr-based tangential / cross shear profile from the per-tract merged catalogs (`a360_merged_tract*.parq`) |
| [`schemas.md`](schemas.md) | Schema documentation for the per-tract merged catalogs produced by `mergeFpfs` |
| [`abell360_reserved_star.ipynb`](abell360_reserved_star.ipynb) | PSF reserved-star residual diagnostics |

## Pipeline tasks

[`measure_pipeline.yaml`](measure_pipeline.yaml) defines four tasks.
Tasks 2–4 share the connection template `inputName = deep_coadd_cell`
so their chained dataset names are `deep_coadd_cell_*`.

1. **`buildCellSystematics`** (`xlens.processor.build_cell_systematics.BuildCellSystematicsTask`):
   for every patch, stitches the per-band cell coadds with their stored
   PSFs + noise realizations, looks up GAIA bright stars in
   `refcats/DM-39298/gaia_dr3_20230707`, and writes three per-patch
   systematics products consumed downstream by `measureCellCoadds`:

   | Output dataset | Storage class | Notes |
   |---|---|---|
   | `deep_coadd_cell_systematics_mask` | `Mask` | BAD-pixel composite + GAIA bright-star mask |
   | `deep_coadd_cell_systematics_noisecorr_6bands` | `NumpyArray` | `(6, npix, npix)` noise correlation stamps per band (`npix=49` default) |
   | `deep_coadd_cell_systematics_psfcentered_6bands` | `NumpyArray` | `(6, npix, npix)` centered PSF stamps per band |

2. **`measureCellCoadds`** (`xlens.processor.measure_cell_coadds.MeasureCellCoaddsPipe`):
   detects sources on i-band cell coadds (250×250 outer, keeping only inner
   150×150) and runs forced FPFS measurement on all bands (g, r, i, z).
   Reads `deep_coadd_cell_predetection` and `deep_coadd_cell_systematics_mask`;
   outputs the per-patch `deep_coadd_cell_anacal_catalog`.
   Key configs: `anacal.do_noise_bias_correction = True`,
   `fpfs.do_noise_bias_correction = True`, `fpfs.sigma_shapelets1 = 0.54`.
3. **`photoZ`** (`xlens.processor.photoz.photoZPipe`): runs FlexZBoost on the
   per-patch AnaCal catalog using `model_4bands_fzboost.pkl`. Outputs the
   per-patch `deep_coadd_cell_anacal_fzb_point` (point estimates) — and
   optionally `deep_coadd_cell_anacal_fzb_pdfs` (full PDFs, off by default).
   Configured with `mag_zero = 31.4`, `flux_name = fpfs1`,
   `bands = griz`, `ref_band = i`, `do_distortions = True` (so the catalog
   carries the four `±dg1`, `±dg2` photo-z perturbations needed for the
   selection-response finite difference).
4. **`mergeFpfs`** (`xlens.processor.merge.MergePipe`): per-tract task that
   stacks the per-patch (anacal + photo-z) outputs, combines the multi-band
   shapelet moments into `fpfs1_e1/e2` (and their shear derivatives) using
   `band_weights` (defaults to `1 / median(flux_err)^2`), applies the local
   tract-WCS distortion correction in place, and joins the photo-z point
   columns by `object_id`. Outputs `deep_coadd_cell_anacal_merged`
   (one parquet table per tract). Schema: see [`schemas.md`](schemas.md).
   Configured with `mag_zero = 31.4`, `fpfs_c0 = 8.4`, `bands = ["g","r","i","z"]`.

## Running the Pipeline

### 1. Update output collection

Before submitting, edit [`parsl.yaml`](parsl.yaml) and change the
`payloadName` to a unique output collection:

```yaml
payload:
  payloadName: dp1/a360_anacal    # <-- change this, e.g. dp1/a360_anacal_v2
```

BPS writes the output to `u/$USER/<payloadName>/<timestamp>` in the DP1
butler repo. For example, with `payloadName: dp1/a360_anacal` and user
`xiangchl`, the output collection is
`u/xiangchl/dp1/a360_anacal/20260331T014745Z`.

### 2. Submit with BPS

```bash
cd /path/to/this/directory
bps submit parsl.yaml
```

This submits a Slurm job that processes all available patches (~49 patches
across tracts 10463 and 10464) plus the per-tract `mergeFpfs` quanta.

### 3. Re-running mergeFpfs with pinned band weights

The `mergeFpfs` task already accepts an explicit `band_weights` list in
`measure_pipeline.yaml`; the current file pins
g/r/i/z = 0.5288 / 0.3305 / 0.1362 / 0.0045 (the values derived from
`1 / median({band}_flux_fpfs1_err)^2` on the full Abell-360 sample). To
re-run only `mergeFpfs` against an existing per-patch run, point `-i` at
that input collection and limit the pipeline to that step:

```bash
pipetask run \
  -b /global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml \
  -i u/$USER/dp1/a360_anacal \
  -o u/$USER/dp1/a360_anacal_fixedw \
  --output-run "u/$USER/dp1/a360_anacal_fixedw/$(date -u +%Y%m%dT%H%M%SZ)" \
  -p measure_pipeline.yaml#mergeFpfs \
  -d "skymap='lsst_cells_v1' AND instrument='LSSTComCam'" \
  -j 16 \
  --register-dataset-types
```

### 4. Monitor progress

```bash
# Check Slurm job status
squeue -u $USER

# Check parsl monitoring (from the submission directory)
python3 -c "
import sqlite3, os
db = os.path.join('runinfo', 'monitoring.db')
conn = sqlite3.connect(db)
cursor = conn.cursor()
cursor.execute('''
    SELECT t.task_func_name, s.task_status_name, COUNT(*)
    FROM task t JOIN status s ON t.task_id = s.task_id
    WHERE s.timestamp = (SELECT MAX(s2.timestamp) FROM status s2 WHERE s2.task_id = t.task_id)
    GROUP BY t.task_func_name, s.task_status_name
''')
for row in cursor.fetchall():
    print(f'{row[0]}: {row[1]} = {row[2]}')
"
```

### 5. Parsl configuration notes

Key parameters in [`parsl.yaml`](parsl.yaml):

| Parameter | Value | Notes |
|---|---|---|
| `max_workers` | 8 | Concurrent tasks per node (work_queue worker pool) |
| `qos` | debug | 30-min cap, higher priority; switch to `regular` for longer runs |
| `walltime` | 0:30:00 | Capped by `debug`; raise when moving to `regular` |
| `nodes_per_block` | 1 | Nodes in each Slurm allocation |
| `max_blocks` | 8 | Independent sbatch submissions (condor-style job array); scale to add nodes |
| `slurm_options` `--ntasks-per-node=8 --mem=0` | — | 8 single-CPU tasks per node sharing the whole-node RAM pool |

Quanta are farmed out across all blocks dynamically by the work_queue
master, so each new sbatch that lands picks up work immediately. To
launch e.g. 16 nodes in parallel, just bump `max_blocks` to 16.

### 6. Single-patch test run (without BPS)

For quick testing on a single patch using
[`measure_pipeline.yaml`](measure_pipeline.yaml). The CUDA env vars
suppress the libcuda-driver probe that otherwise stalls on login nodes
without a GPU (xgboost / JAX wheels are CUDA-enabled and would spawn a
worker thread that sits in `D` state).

```bash
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

pipetask run \
  -b /global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml \
  -i u/mgorsuch/metadetect/a360_extra_cells,u/mgorsuch/metadetect/a360_metadetect,u/mgorsuch/metadetect/a360_coadd,LSSTComCam/DP1,refcats/DM-39298/gaia_dr3_20230707 \
  -o u/$USER/dp1/a360_test \
  --output-run "u/$USER/dp1/a360_test/$(date -u +%Y%m%dT%H%M%SZ)" \
  -p measure_pipeline.yaml \
  -d "skymap='lsst_cells_v1' AND instrument='LSSTComCam' AND tract=10463 AND patch=31" \
  -j 16 \
  --register-dataset-types
```

## Input Data

- **Cell coadds**: `u/mgorsuch/metadetect/a360_metadetect` (assembled with
  `assembleCellCoadd`, 4 bands: g, r, i, z, with 1 noise realization)
- **Warps**: `u/mgorsuch/metadetect/a360_coadd`
- **Calibrations**: `LSSTComCam/DP1`
- **GAIA stars**: `refcats/DM-39298/gaia_dr3_20230707` (for bright star
  masking)
- **Photo-z model**: `model_4bands_fzboost.pkl` (FlexZBoost trained on g,
  r, i, z; path set by the `photoZ.model_path` config field in
  [`measure_pipeline.yaml`](measure_pipeline.yaml))

## Output Catalogs

The pipeline produces:

- **Per-patch** `deep_coadd_cell_anacal_catalog` (`measureCellCoadds`):
  detection + per-band forced measurement, with columns including
  `ra`, `dec`, `{band}_flux_fpfs1`, `{band}_dflux_fpfs1_dg1/2`,
  `{band}_flux_fpfs1_err`, `{band}_fpfs1_m00/m20/m22c/m22s`,
  `{band}_fpfs1_e1/e2`, `wsel`, `dwsel_dg1/2`, `mask_value`,
  `is_primary`, `block_id`, `object_id`.
- **Per-patch** `deep_coadd_cell_anacal_fzb_point` (`photoZ`):
  FlexZBoost point estimates with five `±dg` distortions per object
  (`zbest_0/_1p/_1m/_2p/_2m`, `zmode_*`, `z025/160/500/840/975_*`).
- **Per-tract** `deep_coadd_cell_anacal_merged` (`mergeFpfs`):
  band-combined and WCS-corrected `fpfs1_e1/e2` (+ shear responses),
  band-combined `fpfs1_m00/m20` (+ shear responses), per-band fluxes
  and flux derivatives, `wsel`, `dwsel_dg1/2`, plus the joined photo-z
  point estimates. See [`schemas.md`](schemas.md) for the full column
  list.

## Analysis Notebooks

Open the notebooks in JupyterLab using the **"LSST v30.0.4.rc1"** kernel.

- [`abell360_cluster_cell.ipynb`](abell360_cluster_cell.ipynb): works
  directly on the per-patch `deep_coadd_cell_anacal_catalog`, reproducing
  the band combination + WCS correction inline. Update the
  `anacal_collection` and `butler_config` variables at the top to match
  your run. Steps: catalog loading + magnitudes; sky coverage and GAIA
  bright-star mask; red-sequence colour selection; multi-band weighted
  shear estimation (FPFS shapelet moments); tangential shear profile with
  NFW model comparison; aperture-mass map (E-mode and B-mode).
- [`abell360_cluster_treecorr.ipynb`](abell360_cluster_treecorr.ipynb):
  loads the per-tract merged catalogs (`a360_merged_tract*.parq`),
  applies the standard cuts + photo-z cut, computes the response
  components (`R_shape`, `R_weight`, `R_sel`), and runs TreeCorr's
  `NGCorrelation` to produce the tangential / cross shear profile.

## Cleaning Up

To remove output collections from the butler, run the following script
(replace `<username>` with your username and `<payloadName>` with the
payload name used in [`parsl.yaml`](parsl.yaml)):

```bash
python3 -c "
import lsst.daf.butler as dafButler

butler_config = '/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml'
butler = dafButler.Butler(butler_config, writeable=True)

# List collections to remove
pattern = 'u/<username>/<payloadName>*'
own = [c for c in sorted(butler.registry.queryCollections(pattern))
       if c.startswith('u/<username>/')]
print(f'Found {len(own)} collections:')
for c in own:
    print(f'  {c}')

# Remove CHAINED first, then RUN
chained = [c for c in own if butler.collections.get_info(c).type.name == 'CHAINED']
runs = [c for c in own if butler.collections.get_info(c).type.name == 'RUN']
for c in chained:
    butler.registry.removeCollection(c)
for c in runs:
    butler.removeRuns([c])
print(f'Removed {len(chained)} chained + {len(runs)} run collections')
"

# Also clean up leftover files on disk
rm -rf /global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/u/<username>/<payloadName>/
```
