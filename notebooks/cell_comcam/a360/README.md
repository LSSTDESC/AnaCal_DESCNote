# AnaCal Shear Measurement on Cell-based Coadds (Abell 360, LSSTComCam DP1)

This directory contains the pipeline configuration and analysis notebook for
running AnaCal shear measurements on LSST cell-based coadds for the Abell 360
cluster field.

## Prerequisites

Source the LSST v30.0.4.rc1 environment:

```bash
source ~/setup_lsst_v30.bash
```

This sets up the LSST stack, `xlens`, `anacal`, and the custom DRP
pipe/tasks repos with cell-based coadd support.

## Files

| File | Description |
|---|---|
| [`cell_coadd_pipeline.yaml`](cell_coadd_pipeline.yaml) | Pipeline definition with two tasks: `buildCellSystematics` (mask, noise correlation, PSF) and `measureCellCoadds` (detection + forced measurement) |
| [`parsl.yaml`](parsl.yaml) | BPS/Parsl configuration for batch submission on Perlmutter |
| [`abell360_cluster_cell.ipynb`](abell360_cluster_cell.ipynb) | Analysis notebook: tangential shear profile and mass map |

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
across tracts 10463 and 10464). The pipeline defined in
[`cell_coadd_pipeline.yaml`](cell_coadd_pipeline.yaml) runs two tasks per
patch:

1. **`buildCellSystematics`** (~25s per patch): builds the mask (including
   GAIA bright star masking), noise correlation function, and stacked PSF
   from stitched cell coadd images.
2. **`measureCellCoadds`** (~50s per patch): detects sources on i-band cell
   coadds (250x250 outer, keeping only inner 150x150), then runs forced
   measurement on all bands (g, r, i, z) one band at a time.

### 3. Monitor progress

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

### 4. Parsl configuration notes

Key parameters in [`parsl.yaml`](parsl.yaml):

| Parameter | Value | Notes |
|---|---|---|
| `max_workers` | 64 | Number of concurrent tasks per node |
| `qos` | debug | 30-min limit, higher priority; use `regular` for longer runs |
| `walltime` | 0:30:00 | Sufficient for ~100 quanta on one node |
| `max_blocks` | 1 | Number of Slurm jobs (nodes); increase for more parallelism |

### 5. Single-patch test run (without BPS)

For quick testing on a single patch using
[`cell_coadd_pipeline.yaml`](cell_coadd_pipeline.yaml):

```bash
pipetask run \
  -b /global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml \
  -i u/mgorsuch/metadetect/a360_metadetect,u/mgorsuch/metadetect/a360_coadd,LSSTComCam/DP1,refcats/DM-39298/gaia_dr3_20230707 \
  -o u/$USER/dp1/a360_test \
  --output-run "u/$USER/dp1/a360_test/$(date -u +%Y%m%dT%H%M%SZ)" \
  -p cell_coadd_pipeline.yaml \
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

## Output Catalogs

The pipeline produces per-patch catalogs (`deep_cell_coadd_anacal_catalog`)
with columns including:

- `ra`, `dec` — sky coordinates
- `{band}_flux_gauss0/2/4` — Gaussianized flux measurements (g, r, i, z)
- `{band}_fpfs1_m00`, `{band}_fpfs1_m22c/s`, etc. — FPFS shapelet moments
- `{band}_fpfs1_e1/e2`, `{band}_fpfs1_de1_dg1`, etc. — FPFS ellipticities
  and shear responses
- `wsel`, `dwsel_dg1/dg2` — selection weight and derivatives
- `mask_value` — number of masked pixels in measurement stamp
- `is_primary` — deduplication flag (patch + tract inner region check)
- `block_id` — cell block assignment (always 0, one block per cell)

## Analysis Notebook

Open [`abell360_cluster_cell.ipynb`](abell360_cluster_cell.ipynb) in
JupyterLab using the **"LSST v30.0.4.rc1"** kernel. Before running, update
the `anacal_collection` variable to match your output collection:

```python
anacal_collection = "u/<username>/dp1/a360_anacal"
```

The notebook performs:
1. Catalog loading and magnitude computation
2. Sky coverage visualization with tract boundaries
3. GAIA bright star mask inspection
4. Red-sequence color selection
5. Multi-band weighted shear estimation (FPFS shapelet moments)
6. Tangential shear profile with NFW model comparison
7. Aperture mass map (E-mode and B-mode)

## Cleaning Up

To remove output collections from the butler:

```python
import lsst.daf.butler as dafButler
butler = dafButler.Butler(butler_config, writeable=True)
# List your collections
for c in sorted(butler.registry.queryCollections('u/<username>/dp1/*')):
    print(c)
# Remove (CHAINED first, then RUN)
# butler.registry.removeCollection("u/<username>/dp1/a360_anacal")
# butler.removeRuns(["u/<username>/dp1/a360_anacal/<timestamp>"])
```
