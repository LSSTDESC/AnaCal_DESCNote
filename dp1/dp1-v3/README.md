# DP1-v3 — AnaCal cluster WL on ComCam DP1 with survey-prefixed schema

End-to-end pipeline for AnaCal cluster weak lensing on the DP1 ComCam
cell-based coadds, rebuilt against the newly-pulled `xlens` where
per-band columns are survey-prefixed (`lsst_g_flux_fpfs1` etc.), fluxes
are normalised onto the fixed `xlens.utils.constants.MAG_ZERO_AB = 31.4`
zeropoint, and `measureCellCoadds` writes the smooth-truncated
magnitude family (`{b}_mag_{fam}`, `{b}_mag_{fam}_err`,
`{b}_dmag_{fam}_dg{c}`, `{b}_dmag_{fam}_err_dg{c}`).

## What's different vs v2

| | v2 | v3 |
|---|---|---|
| Per-band column names | `g_flux_fpfs1`, `i_mag_gauss2` | `lsst_g_flux_fpfs1`, `lsst_i_mag_gauss2` |
| Mag zeropoint | per-coadd `mag_zero` field | fixed `MAG_ZERO_AB = 31.4`, no per-catalog knob |
| Mag family | (not persisted) | `add_magnitude_columns` writes `mag_{fam}`, `mag_{fam}_err`, `dmag_{fam}_dg{c}`, `dmag_{fam}_err_dg{c}` |
| Merged shape-magnitude | derive per-source | precomputed `esq = e1**2 + e2**2` + `desq_dg{1,2}` at merge time |
| Merged size cut basis | `fpfs_m2/fpfs_m0` (i-band detection) | `fpfs1_m20/fpfs1_m00` (band-combined) |
| Merged catalog col count | 158 (griz) / 202 (ugrizy) | **209** (griz) / **283** (ugrizy) |
| step3's emax cut | `build_selection_mask(shape_name="fpfs", emax=…)` on i-band detection shape | `esq + dg·desq_dg{c} < emax**2` on the merged band-combined shape |
| TXPipe interface | derive mags in-ingest | forwards pre-computed mags + shear-response derivatives |

## Data flow

```
            ┌─────────────────────┐
            │ u/pecom/dp1/coadds  │   (cell-coadd predetection input)
            └─────────┬───────────┘
                      │
        bps submit parsl.yaml          ──► one slurm node, ~12 min
                      │
                      ▼
   ┌───────────────────────────────────────────────────────┐
   │  per-patch products                                   │
   │  buildCellSystematics → measureCellCoadds → photoZ    │
   │  -- deep_coadd_cell_anacal_catalog                    │
   │  -- deep_coadd_cell_anacal_fzb_point + _fzb_pdfs      │
   │  -- deep_coadd_cell_systematics_{mask, noisecorr,     │
   │     psfcentered, gaia}                                │
   └────────────────────────┬──────────────────────────────┘
                            │
            pipetask … #mergePatches      ──► ~10 s / tract
                            │
                            ▼
       ┌──────────────────────────────┐
       │ deep_coadd_cell_anacal_merged│  ArrowAstropy per tract
       │  209 cols (griz)/283 (ugrizy)│  new: esq, desq_dg{1,2},
       │                              │       mag_{fam}_err,
       │                              │       dmag_{fam}_err_dg{c}
       └──────┬───────────────────────┘
              │
   ┌──────────┼────────────┬────────────────────┬────────────────┐
   ▼          ▼            ▼                    ▼                ▼
 step2      step3_2      step3_3/step3_4      TXPipe ingest    export FITS
 (spatial,  (χ_t, χ_x,   (mass maps —         (shear catalog   (per-field
  1Dhist,    log M)       flat + healsparse)  HDF5)             .fits under
  gaia,                                                          DP1-v3/
  ref)                                                           catalogs/)
```

step3_1 (`redshift.png` + `source_nz.npz`) reads per-patch PDFs directly
(PDFs are not carried through into the merged tract catalog).

## Directory layout

| Path | Purpose |
|---|---|
| `configs/measure_pipeline_4bands.yaml` | griz pipeline for a360. Tasks: `buildCellSystematics` → `measureCellCoadds` → `photoZ` → `mergePatches`. `photoZ.model_path` points at the 4-band FlexZBoost model; `mergePatches` uses the griz `_DEFAULT_FPFS_WEIGHTS` and `fpfs_c0 = 13.7715`, `do_wcs_correction: false`, `flipu_wcs: false`. |
| `configs/measure_pipeline_6bands.yaml` | ugrizy pipeline for edfs + ecdfs. Same task chain and merge settings, 6-band `photoZ` model, u/y set to zero weight in the merge (flux pass-through only). |
| `scripts/step1_*` | Parsl yaml + tract/patch query generators (call the yaml files above). |
| `scripts/step2_*` | Per-field diagnostics: spatial distribution, 1-D magnitude histograms, coadd-vs-cell compare, GAIA tangential residuals, ref-catalog unmatched. |
| `scripts/step3_*` | Cluster-WL diagnostics: n(z), mass posterior + tangential shear, flat-sky aperture mass, curved-sky HealSparse aperture mass. |
| `scripts/setup_and_submit.sh` | Reproduces yesterday's successful bps env (CVMFS v30 + EUPS-setup `bps_parsl_sites`/`drp_pipe`/`drp_tasks` + editable xlens on PYTHONPATH) and `bps submit`s the field's parsl.yaml. |
| `scripts/_common.py`, `_step3_common.py` | Shared cut/response helpers imported by step2/step3 scripts (Butler + skymap defaults, `select_sources`, `compute_r_sel`, cut function). |
| `results/<field>/step1/parsl.yaml` | Per-field bps config (tract/patch list, band list, output collection name). |
| `results/<field>/step2/*.png` | Diagnostic plots. |
| `results/<field>/step3/*.png`/`*.npz`/`*.fits` | Cluster analysis outputs. |
| `u/xiangchl/dp1-v3/<field>_anacal2` (butler) | Output CHAINED collection. Composed of one bps RUN (buildCellSystematics + measureCellCoadds + photoZ per-patch) and one mergePatches RUN (per-tract merged). |

Butler repo: `/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml`.
Skymap: `lsst_cells_v1`.

## Fields, tracts, and merged catalogs

Per-field tracts (see `results/<field>/step1/parsl.yaml` for the exact
patch subsets):

| Field | Tracts | Per-patch (measureCellCoadds) | Merged rows | Merged cols |
|---|---|---:|---:|---:|
| a360 | 10463, 10464 | 88 | 92,283 | 209 |
| edfs | 2234, 2393, 2394 | 50 | 49,115 | 283 |
| ecdfs | 4848, 4849, 5063, 5064 | 66 | 81,240 | 283 |

Empty-tract quirks (baseline, not regressions — same behaviour as v2):

- **edfs t2393** merges to zero rows (the only overlapping patch after
  the `is_primary + wsel > 1e-5` filter).
- **ecdfs t5064** merges to zero rows (edge-of-footprint).

Exported per-field FITS live at
`/global/cfs/cdirs/desc-wl/projects/anacal/DP1-v3/catalogs/anacal_catalog_<field>.fits`.

## End-to-end recipe

```bash
# 1. Submit bps for each field (writes to u/xiangchl/dp1-v3/<field>_anacal2).
cd /global/homes/x/xiangchl/superonion/code/AnaCal_DESCNote/dp1-v3
scripts/setup_and_submit.sh results/a360/step1
scripts/setup_and_submit.sh results/edfs/step1
scripts/setup_and_submit.sh results/ecdfs/step1

# 2. mergePatches per field once bps completes (a few seconds per tract).
pipetask run --register-dataset-types \
    -b /global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml \
    -i u/xiangchl/dp1-v3/a360_anacal2,u/pecom/dp1/coadds,skymaps,LSSTComCam/DP1 \
    -o u/xiangchl/dp1-v3/a360_anacal2 \
    -p configs/measure_pipeline_4bands.yaml#mergePatches \
    -d "skymap='lsst_cells_v1' AND tract IN (10463, 10464)"
# edfs (tracts 2234, 2393, 2394) and ecdfs (4848, 4849, 5063, 5064) use
# configs/measure_pipeline_6bands.yaml#mergePatches.

# 3. Export per-field FITS.
python - <<'PY'
from lsst.daf.butler import Butler
from astropy.table import vstack
REPO = '/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml'
OUT  = '/global/cfs/cdirs/desc-wl/projects/anacal/DP1-v3/catalogs'
FIELDS = {'a360':(10463,10464), 'edfs':(2234,2393,2394),
          'ecdfs':(4848,4849,5063,5064)}
for f, tracts in FIELDS.items():
    b = Butler(REPO, collections=f'u/xiangchl/dp1-v3/{f}_anacal2')
    tables = []
    for t in tracts:
        refs = list(b.registry.queryDatasets(
            'deep_coadd_cell_anacal_merged',
            where=f"skymap='lsst_cells_v1' AND tract={t}"))
        if refs:
            tables.append(b.get(refs[0]))
    vstack(tables, metadata_conflicts='silent').write(
        f'{OUT}/anacal_catalog_{f}.fits', format='fits', overwrite=True)
PY

# 4. step2 diagnostics per field.  Field centre + radius drive tract discovery.
C_A360="--ra 37.86  --dec   6.98 --radius 1.5 --collection u/xiangchl/dp1-v3/a360_anacal2  --field a360"
C_EDFS="--ra 59.10  --dec -48.73 --radius 1.5 --collection u/xiangchl/dp1-v3/edfs_anacal2  --field edfs"
C_ECDF="--ra 53.13  --dec -28.10 --radius 1.5 --collection u/xiangchl/dp1-v3/ecdfs_anacal2 --field ecdfs"
for C in "$C_A360" "$C_EDFS" "$C_ECDF"; do bash -c "
  python scripts/step2_spacial_distribution.py $C
  python scripts/step2_1Dhist.py               $C \
      --bands lsst_g,lsst_r,lsst_i,lsst_z     # add lsst_u,lsst_y for edfs/ecdfs
  python scripts/step2_gaia_tangential.py      $C --gaia-mag-bins 8,13,15,17
"; done
# ref_unmatched needs an external anacal_catalog_<field>.fits (only for a360 + edfs).

# 5. step3 cluster analysis (--delta-gamma 0.01 turns on R_sel).
python scripts/step3_1_redshift.py           --ra 37.865017 --dec  6.982205 --z-cl 0.22   --radius 1.5 --collection u/xiangchl/dp1-v3/a360_anacal2 --field a360 --flux-name gauss2 --delta-gamma 0.01
python scripts/step3_2_mass.py               --ra 37.865017 --dec  6.982205 --z-cl 0.22   --radius 1.5 --collection u/xiangchl/dp1-v3/a360_anacal2 --field a360 --flux-name gauss2 --delta-gamma 0.01
python scripts/step3_3_massmap.py            --ra 37.865017 --dec  6.982205 --z-cl 0.22   --radius 1.5 --collection u/xiangchl/dp1-v3/a360_anacal2 --field a360 --flux-name gauss2 --delta-gamma 0.01
python scripts/step3_4_massmap_healsparse.py --ra 37.865017 --dec  6.982205 --z-cl 0.22   --radius 1.5 --collection u/xiangchl/dp1-v3/a360_anacal2 --field a360 --flux-name gauss2 --delta-gamma 0.01
# edfs cluster (eROSITA, z_cl=0.6922): --ra 59.487317 --dec -49.000349 --z-cl 0.6922 …/edfs_anacal2 --field edfs
# ecdfs has no targeted cluster — step3 is skipped.
```

## Merged catalog schema (v3)

New columns (added by `xlens.processor.merge.MergePipe` on top of the
v2 keep list):

| Column | Definition | Purpose |
|---|---|---|
| `esq` | `fpfs1_e1**2 + fpfs1_e2**2` on the WCS-corrected fpfs1 shape | one-column `|e|**2` cut, no per-source arithmetic downstream |
| `desq_dg1` | `2·(fpfs1_e1·fpfs1_de1_dg1 + fpfs1_e2·fpfs1_de2_dg1)` | analytic ±γ variant of `esq` for R_sel |
| `desq_dg2` | `2·(fpfs1_e1·fpfs1_de1_dg2 + fpfs1_e2·fpfs1_de2_dg2)` | " (comp 2) |
| `lsst_{b}_mag_{fam}_err` | smooth-truncated mag error (was `sigma_mag_{fam}` in v2) | matches the flux naming convention (`_err` suffix) |
| `lsst_{b}_dmag_{fam}_err_dg{c}` | shear response of the mag error | ±γ variant of the mag error |

Dropped from the v2 keep list (moved out of downstream cut paths — the
i-band-only detection-band shape/size is no longer used):
`fpfs_e{1,2}`, `fpfs_de{1,2}_dg{1,2}`, `fpfs_m{0,2}`, `fpfs_dm{0,2}_dg{1,2}`.

`MergePipe._finalize_columns` carries a **back-compat rename shim** that
silently maps legacy `sigma_mag` / `dsigma_mag` columns from pre-existing
per-patch catalogs onto the new `mag_err` / `dmag_err` names so
re-running only `#mergePatches` (~10 s / tract) is enough to publish the
new schema without a full bps re-measure.

Everything the merged catalog exposes is downstream of the same shape
family (`fpfs1_*`, band-combined with `xlens.utils.nxg._DEFAULT_FPFS_WEIGHTS`
and `fpfs_c0 = 13.7715`), so `step3_2`'s R calibration matches TXPipe's
`AnaCalCalculator` byte-identically on the same sample (proven with a
zero-cut sample: R_shape + R_detect agree to zero digits, R_total
matches within ~3.5e-4 — the residual is the ±γ shifted-selection tail).

## TXPipe integration

TXPipe reads the merged FITS (or the butler collection directly) via
`TXIngestAnacal` at
`/global/homes/x/xiangchl/superonion/code/TXPipe/txpipe/ingest/anacal.py`.
The ingest forwards the pre-computed magnitudes and shear-response
derivatives instead of recomputing them:

- HDF5 `mag_{b}` ← FITS `lsst_{b}_mag_{scale}`
- HDF5 `mag_err_{b}` ← FITS `lsst_{b}_mag_{scale}_err`
- HDF5 `dmag_{b}_dg{c}` ← FITS `lsst_{b}_dmag_{scale}_dg{c}`
- HDF5 `dmag_err_{b}_dg{c}` ← FITS `lsst_{b}_dmag_{scale}_err_dg{c}`
- HDF5 `esq`, `desq_dg{1,2}` ← FITS `esq`, `desq_dg{1,2}` (verbatim)
- HDF5 `s2n`, `ds2n_dg{1,2}` ← FITS `lsst_i_s2n_fpfs1`,
  `lsst_i_ds2n_fpfs1_dg{1,2}` (scale-independent — the fpfs1 S/N drives
  the brightness cut regardless of which flux family is used for
  magnitudes downstream).

`TXSourceSelectorAnacal` applies the composite cut
`mask_value < mask_threshold ∧ s2n > s2n_cut ∧ (m00+m20)/m00 > T_cut ∧
esq < emax**2 ∧ mag_{b} < {b}_hi_cut ∀ b ∧ zbin >= 0`, with
±γ variants injected via `add_sheared_variant_columns`
(`esq_{1p,1m,2p,2m}` from `desq_dg{1,2}`; `mag_{b}_{1p,1m,2p,2m}` from
`dmag_{b}_dg{1,2}`; `zbin_{1p,1m,2p,2m}` from `mean_z_{1p,1m,2p,2m}`).

Per-field TXPipe outputs (from the last full run):

| field | ingested rows | HDF5 cols | selected | R_2d |
|---|---:|---:|---:|---:|
| a360 (griz) | 92,283 | 56 | 64,063 | 0.2531 |
| edfs (ugrizy) | 49,115 | 68 | 28,800 | 0.2596 |
| ecdfs (ugrizy) | 81,240 | 68 | 40,849 | 0.2523 |

Ingested TXPipe shear catalogs land at
`data/example/anacal_inputs/<field>/shear_catalog.hdf5`; calibrated
tomography + response at
`data/example/output_anacal/<field>/shear_tomography_catalog.hdf5`.

## Env notes

- `bps submit` needs `bps_parsl_sites.SlurmWorkQueue` — provided by the
  EUPS repo at
  `/global/cfs/cdirs/desc-cl/A360_DP1/Metadetect/env/repos/bps_parsl_sites`.
  `scripts/setup_and_submit.sh` handles the CVMFS + EUPS setup.
- `xlens` is expected as an editable install (`pip install -e .` from
  the xlens source tree). `scripts/setup_and_submit.sh` prepends the
  source dir to `PYTHONPATH` so parsl workers see live edits without
  a re-measure.
- The `image` conda env (stackvana-based) is sufficient for step2 /
  step3 / TXPipe. Only the bps submission needs the mixed CVMFS+EUPS
  environment.

### Installing TXPipe

The AnaCal ingest + selector live on the `AnaCal_ingestion` branch of
[`mr-superonion/TXPipe`](https://github.com/mr-superonion/TXPipe) (fork
of LSSTDESC/TXPipe).

```bash
git clone --recurse-submodules https://github.com/mr-superonion/TXPipe
cd TXPipe
git checkout AnaCal_ingestion
./bin/install.sh
```

or, if you cloned the upstream `LSSTDESC/TXPipe` and want to add the
fork's branch on top:

```bash
git clone --recurse-submodules https://github.com/LSSTDESC/TXPipe
cd TXPipe
git remote add anacal https://github.com/mr-superonion/TXPipe.git
git fetch anacal AnaCal_ingestion
git checkout -b AnaCal_ingestion anacal/AnaCal_ingestion
./bin/install.sh
```

`./bin/install.sh` builds a self-contained conda env at `./conda`; each
new shell then does `source ./conda/bin/activate` from the TXPipe
directory before running `ceci`. Ingest reads from the merged catalog
via `examples/anacal/ingest.yml`; the selector + calibrator run through
`examples/anacal/pipeline.yml` (both use the config at
`examples/anacal/config.yml`).

## Known caveats

- Empty tracts (edfs t2393, ecdfs t5064) — see the table above.
  Downstream code either silently drops these tracts or logs
  `NoWorkFound`.
- `--z-min` / `--z-max` in step3 default to `(0.4, 2.0)`.  When
  aligning with TXPipe's zbin edges `[0.3, 0.6, …, 2.0]`, pass
  `--z-min 0.3` explicitly to match.
- On the merged catalog, `wsel > 1e-5` is already applied by
  `MergePipe._finalize_columns`, so no downstream code should apply an
  additional `wsel` cut — the min wsel in the merged output is
  ~1.01e-5 and any extra threshold would remove 0 rows.
