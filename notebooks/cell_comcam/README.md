# cell_comcam — AnaCal cluster WL on the ComCam DP1 cell coadds

End-to-end pipeline for the AnaCal cluster weak-lensing analysis on the
DP1 ComCam cell-based coadds: parsl/bps yaml generation, BPS submission,
per-field diagnostics (spatial, 1-D mag, coadd-vs-cell, GAIA-tangential
PSF residuals, BNL-reference matching), and the cluster-analysis plots
(n(z), mass posterior, tangential shear, tangent-plane aperture-mass
map + S/N histogram, curved-sky HealSparseMap aperture-mass + zoom + histogram).

## Directory layout

| Path | Purpose |
|---|---|
| `scripts/` | All pipeline + plotting code. `step1_*` build parsl yamls; `step2_*` make per-field diagnostics; `step3_1/2/3/4_*` make the cluster-analysis plots. Importable helpers live in `_common.py` and `_step3_common.py`. |
| `configs/measure_pipeline_4bands.yaml` | Cell-coadd pipeline for **griz-only** fields (a360): `buildCellSystematics` → `measureCellCoadds` (`do_measure_flux_gauss: true`) → `photoZ` (`flux_name: gauss2`, `output_pdfs: true`, 4-band model) → `mergePatches` (griz combine). |
| `configs/measure_pipeline_6bands.yaml` | Cell-coadd pipeline for **ugrizy** fields (edfs, ecdfs): same tasks but with 6-band reads, the 6-band photoZ model, and `mergePatches` that lists all 6 bands while keeping `u`/`y` at zero weight so they pass through as flux columns without affecting the band-combined shape. |
| `results/` | All per-field outputs. Subfolders are `results/<field>/{step1,step2,step3}/`. Every script writes here via `_common.FIELDS_ROOT` (or, for `step1_generate_parsl_yamls.sh`, `out_dir=$BASE/results/$field/step1`). To relocate, edit `FIELDS_ROOT` once. |
| `results/a360/` | Per-field assets for Abell 360 (griz). Subfolders: `step1/parsl.yaml`, `step1/runinfo/`, `step1/submit/` (from bps); `step2/*.png`; `step3/*.png` + `source_nz.npz` + `mass_map.hs.fits`. |
| `results/edfs/` `results/ecdfs/` | Same layout as `results/a360/` but on ugrizy. Both fields are populated (52 patches for edfs, 70 for ecdfs in `u/pecom/dp1/coadds`). The eDFS run uses the eROSITA cluster at (59.487317, −49.000349, z = 0.6922); the ecdfs field has no targeted cluster so step3 is skipped there. |
| `a360_old/` | Frozen reference assets: `setup_lsst_v30.bash` (LSST env), `compare_a360_real_vs_sim.ipynb`, `compare_a360_coadd_vs_cell.ipynb`, plus archived parsl yamls and an older README. Don't edit; only used to seed new fields. |

## Butler collections

| Collection | Role | Notes |
|---|---|---|
| `u/pecom/dp1/coadds` | INPUT — `deep_coadd_cell_predetection` per (skymap, tract, patch, band) | The only collection on the DP1 repo that has the cell-coadd predetection. ComCam coverage is **griz** for every tract we care about; `u`/`y` are absent. |
| `refcats/DM-39298/gaia_dr3_20230707` | INPUT — GAIA DR3 reference catalog | Required for bright-star masking in `BuildCellSystematicsTask`. Always chain alongside `u/pecom/dp1/coadds`. |
| `u/xiangchl/dp1-v2/<field>_anacal2` | OUTPUT — current cell-coadd runs (a360, edfs, ecdfs) | Contains `deep_coadd_cell_systematics_*`, `deep_coadd_cell_anacal_catalog` (per-patch, has FPFS+gauss fluxes), `deep_coadd_cell_anacal_fzb_point` (photo-z point estimates), `deep_coadd_cell_anacal_fzb_pdfs` (501-bin PDFs), `deep_coadd_cell_systematics_gaia` (per-patch GAIA stars), `deep_coadd_cell_anacal_merged` (per-tract; written only after `mergePatches` runs separately — see [End-to-end recipe](#end-to-end-recipe-abell-360-example)). |
| `u/xiangchl/dp1/a360_anacal_coadd` | OUTPUT — older deep-coadd run | Used by `step2_compare_coadd_vs_cell.py` and `step2_gaia_tangential_coadd.py`. |
| `LSSTComCam/DP1`, `skymaps`, `pretrained_models/...` | INPUTS chained by bps | Never modify; ignored by removal commands below. |

The butler repo is always `/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml`; the skymap is always `lsst_cells_v1`.

## End-to-end recipe (Abell 360 example)

```bash
# 0. Set up the LSST stack (also chains drp_pipe/drp_tasks/bps_parsl_sites).
source a360_old/setup_lsst_v30.bash

# 1. Regenerate per-field parsl yamls (dataQuery + payloadName + input chain).
#    Default PIPELINE_SUBSET in this script is "#buildCellSystematics,
#    measureCellCoadds,photoZ" — i.e. mergePatches is skipped so each BPS
#    run finishes in <15 min on debug qos. Clear PIPELINE_SUBSET in the
#    script to run the merge inside BPS instead.
scripts/step1_generate_parsl_yamls.sh

# 2. Submit the BPS workflow for the a360 field.
cd results/a360/step1
bps submit parsl.yaml      # ~12 minutes; 266 quanta on 1 256-core node
cd ../../..

# 2b. (Only if mergePatches was skipped above.) Materialise the per-tract
#     merged tables now — required by step2_spacial_distribution and
#     step2_1Dhist. Repeat per tract that overlaps the field.
pipetask run --register-dataset-types \
    -b /global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml \
    -i u/xiangchl/dp1-v2/a360_anacal2 \
    -o u/xiangchl/dp1-v2/a360_anacal2 \
    -p configs/measure_pipeline_4bands.yaml#mergePatches \
    -d "skymap='lsst_cells_v1' AND tract IN (<ids>)"

# 3. Per-field diagnostics (no cluster-specific knowledge).
C2="--ra 37.86 --dec 6.98 --radius 1.5 --collection u/xiangchl/dp1-v2/a360_anacal2 --field a360"
python scripts/step2_spacial_distribution.py $C2
python scripts/step2_1Dhist.py               $C2 --bands g,r,i,z
python scripts/step2_compare_coadd_vs_cell.py --ra 37.86 --dec 6.98 --radius 1.5 \
    --coll-cell  u/xiangchl/dp1-v2/a360_anacal2 \
    --coll-coadd u/xiangchl/dp1/a360_anacal_coadd --field a360
python scripts/step2_gaia_tangential.py      $C2 --gaia-mag-bins 8,13,15,17
python scripts/step2_ref_unmatched.py        $C2 \
    --ref-catalog /pscratch/sd/x/xiangchl/data/DP1-Cell/a360/anacal_catalog_a360.fits

# 4. Cluster analysis (must run step3_1 first; it persists source_nz.npz
#    that step3_2 consumes).
C3="--ra 37.865017 --dec 6.982205 --z-cl 0.22 --radius 1.5 \
    --collection u/xiangchl/dp1-v2/a360_anacal2 --field a360 --flux-name gauss2"
python scripts/step3_1_redshift.py        $C3   # redshift.png + source_nz.npz
python scripts/step3_2_mass.py            $C3   # mass.png + tangential.png  (emcee, ~40 s)
python scripts/step3_3_massmap.py         $C3   # mass_map.png + mass_map_hist.png  (~60 s, flat sky)
python scripts/step3_4_massmap_healsparse.py $C3   # mass_map.hs.fits + zoom PNG + hist (~30 s, curved sky)
```

All outputs land under `results/<field>/step1/` (parsl + bps),
`results/<field>/step2/`, `results/<field>/step3/`. Pass `--out <path>`
(or `--map-out`, `--hist-out`, `--healsparse-out`, etc.) to any
plotting script to override.

The BPS query center (used by `step1_generate_parsl_yamls.sh` to pick
patches) and the science cluster center used by step3 are intentionally
different for edfs — the BPS box is centred on the field (59.10, −48.73)
so it covers all patches with ugrizy data, while step3 is anchored on
the eROSITA cluster (59.487317, −49.000349, z = 0.6922).

## Source selection (step3 + step2_gaia_tangential)

Source cuts pass through `xlens.catalog.base.build_selection_mask` — the
same code path `ShearEstimator._measure` uses for shear-bias accounting
— wrapped by `_step3_common.select_sources(...)`. The mag/shape/size
machinery is `dg`-aware (perturbed for selection-response) at `dg=0` by
default. Every cut is CLI-tunable; defaults reproduce what the cluster
plots use today.

| Flag | Default | Meaning |
|---|---|---|
| `--flux-name` | `fpfs1` | Suffix for `{band}_flux_{flux-name}`; the cluster plots use `gauss2` (matches the BNL notebook + photoz training). |
| `--mag-zero` | `31.4` | AB zero-point for the mag cuts. |
| `--trace-min` | `0.05` | xlens size cut `fpfs_m2 / fpfs_m0 > trace_min` (detection-band moments). Set to a very negative number to disable. |
| `--emax` | `0.5` | xlens shape cut `|e|^2 < emax^2`. Set very large (e.g. `1000`) to disable. |
| `--z-col` | `zmode_0` | photo-z column for the source-z cut (BNL DP1 recipe; alt: `zbest_0`). |
| `--z-min` / `--z-max` | `0.4` / `2.0` | Source photo-z window on `--z-col`. Pass a very large `--z-max` (e.g. `100`) to drop the upper limit. |
| `--mag-max-per-band` | (built-in 4-band dict) | `'u=27.5,g=27.5,r=24.5,i=27.5,z=27.5,y=27.5'` for the BNL ugrizy recipe; default applies griz cuts {g:25.5, r:25.0, i:23.5, z:24.5}. |
| `--iband-size-min` | `None` | **Legacy** BNL-notebook size cut on `(i_fpfs1_m20 + i_fpfs1_m00)/i_fpfs1_m00`. Pass `0.1` to apply it on top of `select_sources`. |

To reproduce the BNL-notebook selection exactly (turn the xlens cuts off, add the legacy i-band cut):

```bash
python scripts/step3_2_mass.py $C3 \
    --emax 1000 --trace-min -100 --iband-size-min 0.1
# → 27,974 kept rows, χ_t = 4.36, logM = 14.92 ± 0.16, E-peak S/N = 4.42
```

Default (xlens-consistent) selection:

```bash
python scripts/step3_2_mass.py $C3
# → 24,913 kept rows, χ_t = 4.44, logM = 14.68 ± 0.16, E-peak S/N = 4.48
```

The 14.9 % drop between the two comes from the shape cut `|e|² < 0.09`;
the xlens `m2/m0` size cut at the default `0.05` removes 0 sources.

## Scripts reference

| Script | Plot # | Output(s) | Key inputs |
|---|---|---|---|
| `step1_build_tract_patch_query.py` | — | a parsl `dataQuery:` block (stdout) | `--ra/--dec/--radius`, `--bands` |
| `step1_generate_parsl_yamls.sh` | — | `results/<field>/step1/parsl.yaml` for each field | reads the in-script FIELDS array + `PIPELINE_SUBSET` |
| `step2_spacial_distribution.py` | — | `spatial.png` | merged catalog (`deep_coadd_cell_anacal_merged`) + GAIA layer (`deep_coadd_cell_systematics_gaia`) |
| `step2_1Dhist.py` | — | `1Dhist.png` | merged catalog, per-band fluxes |
| `step2_compare_coadd_vs_cell.py` | — | 5 `compare_*.png` | per-patch anacal from both `--coll-cell` and `--coll-coadd` |
| `step2_gaia_tangential.py` | — | `gaia_tangential.{png,npz}` | per-patch anacal+photoz + `deep_coadd_cell_systematics_gaia` (treecorr, 3 mag bins of GAIA lenses) |
| `step2_gaia_tangential_coadd.py` | — | `gaia_tangential_coadd.{png,npz}` | deep-coadd `deep_coadd_anacal_*` sources + GAIA from a `--gaia-collection` (default: the cell-coadd run) |
| `step2_ref_unmatched.py` | — | `ref_unmatched.png` | per-patch `deep_coadd_cell_anacal_catalog` (`is_primary` deduped) vs `--ref-catalog` (old deep-coadd FITS); 3-layer scatter of all/matched/unmatched within `--ref-match-arcsec` (default 0.6″) with `--ref-imag-max` (default i<24.0) applied to the reference side. |
| `step3_1_redshift.py` | 1 | `redshift.png` + **`source_nz.npz`** | per-patch anacal + photoz + fzb PDFs |
| `step3_2_mass.py` | 2 + 3 | `mass.png`, `tangential.png` | depends on `source_nz.npz` (run step3_1 first) |
| `step3_3_massmap.py` | 4 + 5 | `mass_map.png`, `mass_map_hist.png` | per-patch anacal (no PDF dep); flat-sky tangent-plane Schirmer aperture mass; zoom set by `--half-extent-deg` (default 0.5°) |
| `step3_4_massmap_healsparse.py` | 6 + 7 | `mass_map.hs.fits`, `mass_map_healsparse.png`, `mass_map_healsparse_hist.png` | same per-patch inputs; curved-sky Schirmer aperture mass on a HealSparseMap (`Map_E`, `Map_B`, `Map_V`, `sn_e`, `sn_b`). Zoom precedence: `--zoom-half-extent-deg` > `--physical-zoom-mpc` > default 0.5° (matches step3_3). |

Shared loaders / helpers live in `_common.py` (butler, find_tracts,
paths) and `_step3_common.py` (anacal+photoz+pdf join, `select_sources`,
`load_context`). All cluster-WL primitives live in **`xlens.utils.{nxg,
massmap, match}`** (`calibrate_shapes`, `anacal_get_tang_cross`,
`fast_bootstrap_mean`, `build_flat_wcs_grid`, `compute_mass_map`,
`compute_mass_map_healpix`, `schirmer_filter`, `sky_match`,
`append_specz_columns`, `spec_selection`).

## Reference catalogs (old deep-coadd version)

External per-field anacal catalogs from the older deep-coadd run, used
by `step2_ref_unmatched.py` (and seed inputs for `step3_*` cross-checks):

| Field | Path | Rows |
|---|---|---|
| a360 | `/pscratch/sd/x/xiangchl/data/DP1-Cell/a360/anacal_catalog_a360.fits` | 113,945 |
| edfs | `/pscratch/sd/x/xiangchl/data/DP1-Cell/edfs/anacal_catalog_edfs.fits` | 89,354 |

Same magnitude convention as our pipeline: `mag = 31.4 − 2.5·log10(i_flux_gauss2)`.

## Cleaning up runs

```bash
# Remove the CHAINED + RUN for a360_anacal2 (keep the chained inputs alone).
# Note: remove-collections must run BEFORE remove-runs, otherwise butler
# refuses with "Removing runs that are in parent CHAINED collections
# requires confirmation".
butler remove-collections <repo> u/xiangchl/dp1-v2/a360_anacal2 --no-confirm
butler remove-runs        <repo> u/xiangchl/dp1-v2/a360_anacal2/<timestamp> --no-confirm
```

`butler query-collections <repo> "u/xiangchl/dp1-v2*"` lists what's there.

## Dependencies (LSST `lsst-scipipe-12.1.0-exact-ext` env)

`numpy`, `matplotlib`, `astropy`, `scipy`, `healpy`, `healsparse`,
`treecorr`, `clmm`, `emcee`, `lsst.daf.butler`, `lsst.geom`,
`lsst.afw.image`, plus an editable install of `xlens` (selection helpers
+ cluster-WL primitives) and `bps_parsl_sites` (parsl-based BPS backend;
sourced by `setup_lsst_v30.bash`).
