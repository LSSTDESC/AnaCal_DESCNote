# cell_comcam — AnaCal cluster WL on the ComCam DP1 cell coadds

End-to-end pipeline for the AnaCal cluster weak-lensing analysis on the
DP1 ComCam cell-based coadds: parsl/bps yaml generation, BPS submission,
per-field diagnostics, and the five cluster-analysis plots (n(z), mass
posterior, tangential shear, aperture-mass map, S/N histogram).

## Directory layout

| Path | Purpose |
|---|---|
| `scripts/` | All pipeline + plotting code. `step1_*` build parsl yamls; `step2_*` make per-field diagnostics; `step3_1/2/3_*` make the cluster-analysis plots. Importable helpers live in `_common.py` and `_step3_common.py`. |
| `configs/measure_pipeline.yaml` | The cell-coadd pipeline definition: `buildCellSystematics` → `measureCellCoadds` (with `do_measure_flux_gauss: true`) → `photoZ` (`flux_name: gauss2`, `output_pdfs: true`) → `mergePatches`. |
| `a360/` | Per-field assets for Abell 360. Subfolders: `step1/parsl.yaml`, `step1/runinfo/`, `step1/submit/` (from bps), `step1/cmd_bps.xiangchl.sh`; `step2/*.png`; `step3/{*.png,source_nz.npz}`. |
| `edfs/` `ecdfs/` | Same layout as `a360/`; `step1/parsl.yaml` is empty for these two because no ugrizy-complete patch exists in `u/pecom/dp1/coadds` (ComCam DP1 is griz only). |
| `a360_old/` | Frozen reference assets: `setup_lsst_v30.bash` (LSST env), `compare_a360_real_vs_sim.ipynb`, `compare_a360_coadd_vs_cell.ipynb`, plus archived parsl yamls and an older README. Don't edit; only used to seed new fields. |

## Butler collections

| Collection | Role | Notes |
|---|---|---|
| `u/pecom/dp1/coadds` | INPUT — `deep_coadd_cell_predetection` per (skymap, tract, patch, band) | The only collection on the DP1 repo that has the cell-coadd predetection. ComCam coverage is **griz** for every tract we care about; `u`/`y` are absent. |
| `refcats/DM-39298/gaia_dr3_20230707` | INPUT — GAIA DR3 reference catalog | Required for bright-star masking in `BuildCellSystematicsTask`. Always chain alongside `u/pecom/dp1/coadds`. |
| `u/xiangchl/dp1-v2/a360_anacal2` | OUTPUT — current a360 cell-coadd run | Contains `deep_coadd_cell_systematics_*`, `deep_coadd_cell_anacal_catalog` (per-patch, has FPFS+gauss fluxes), `deep_coadd_cell_anacal_fzb_point` (photo-z point estimates), `deep_coadd_cell_anacal_fzb_pdfs` (501-bin PDFs), `deep_coadd_cell_systematics_gaia` (per-patch GAIA stars), `deep_coadd_cell_anacal_merged` (per-tract). |
| `u/xiangchl/dp1/a360_anacal_coadd` | OUTPUT — older deep-coadd run | Used only by `step2_compare_coadd_vs_cell.py` for the deep-vs-cell comparison. |
| `LSSTComCam/DP1`, `skymaps`, `pretrained_models/...` | INPUTS chained by bps | Never modify; ignored by removal commands below. |

The butler repo is always `/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml`; the skymap is always `lsst_cells_v1`.

## End-to-end recipe (Abell 360 example)

```bash
# 0. Set up the LSST stack (also chains drp_pipe/drp_tasks/bps_parsl_sites).
source a360_old/setup_lsst_v30.bash

# 1. Regenerate per-field parsl yamls (dataQuery + payloadName + input chain).
scripts/step1_generate_parsl_yamls.sh

# 2. Submit the BPS workflow for the a360 field.
cd a360/step1
bps submit parsl.yaml      # ~12 minutes; 266 quanta on 1 256-core node
cd ../..

# 3. Per-field diagnostics (no cluster-specific knowledge).
C2="--ra 37.86 --dec 6.98 --radius 1.5 --collection u/xiangchl/dp1-v2/a360_anacal2 --field a360"
python scripts/step2_spacial_distribution.py $C2
python scripts/step2_1Dhist.py               $C2 --bands g,r,i,z
python scripts/step2_compare_coadd_vs_cell.py --ra 37.86 --dec 6.98 --radius 1.5 \
    --coll-cell  u/xiangchl/dp1-v2/a360_anacal2 \
    --coll-coadd u/xiangchl/dp1/a360_anacal_coadd --field a360

# 4. Cluster analysis (must run step3_1 first; it persists source_nz.npz
#    that the other two scripts consume).
C3="--ra 37.865017 --dec 6.982205 --z-cl 0.22 --radius 1.5 \
    --collection u/xiangchl/dp1-v2/a360_anacal2 --field a360 --flux-name gauss2"
python scripts/step3_1_redshift.py  $C3       # redshift.png + source_nz.npz
python scripts/step3_2_mass.py      $C3       # mass.png + tangential.png  (emcee, ~40 s)
python scripts/step3_3_massmap.py   $C3       # mass_map.png + mass_map_hist.png (~60 s)
```

All outputs land under `<field>/step1/` (parsl + bps), `<field>/step2/` (4 plots), `<field>/step3/` (5 plots + `source_nz.npz`). Pass `--out <path>` to any plotting script to override.

## Source selection (step3 scripts)

Source cuts pass through `xlens.catalog.base.build_selection_mask` — the same code path `ShearEstimator._measure` uses for shear-bias accounting — wrapped by `_step3_common.select_sources(...)`. The mag/shape/size machinery is `dg`-aware (perturbed for selection-response) at `dg=0` by default. Every cut is CLI-tunable; defaults reproduce what the cluster plots use today.

| Flag | Default | Meaning |
|---|---|---|
| `--flux-name` | `fpfs1` | Suffix for `{band}_flux_{flux-name}`; the cluster plots use `gauss2` (matches the BNL notebook + photoz training). |
| `--mag-zero` | `31.4` | AB zero-point for the mag cuts. |
| `--trace-min` | `0.05` | xlens size cut `fpfs_m2 / fpfs_m0 > trace_min` (detection-band moments). Set to a very negative number to disable. |
| `--emax` | `0.3` | xlens shape cut `|e|^2 < emax^2`. Set very large (e.g. `1000`) to disable. |
| `--z-min` / `--z-max` | `0.4` / `2.0` | Source photo-z window on `zbest_0`. |
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

The 14.9 % drop between the two comes from the shape cut `|e|² < 0.09`; the xlens `m2/m0` size cut at the default `0.05` removes 0 sources on this dataset.

## Scripts reference

| Script | Plot # | Output(s) | Key inputs |
|---|---|---|---|
| `step1_build_tract_patch_query.py` | — | a parsl `dataQuery:` block (stdout) | `--ra/--dec/--radius`, `--bands` |
| `step1_generate_parsl_yamls.sh` | — | `<field>/step1/parsl.yaml` for each field | reads the in-script FIELDS array |
| `step2_spacial_distribution.py` | — | `spatial.png` | merged catalog (`deep_coadd_cell_anacal_merged`) + GAIA layer (`deep_coadd_cell_systematics_gaia`) |
| `step2_1Dhist.py` | — | `1Dhist.png` | merged catalog, per-band fluxes |
| `step2_compare_coadd_vs_cell.py` | — | 5 `compare_*.png` | per-patch anacal from both `--coll-cell` and `--coll-coadd` |
| `step3_1_redshift.py` | 1 | `redshift.png` + **`source_nz.npz`** | per-patch anacal + photoz + fzb PDFs |
| `step3_2_mass.py` | 2 + 3 | `mass.png`, `tangential.png` | depends on `source_nz.npz` (run step3_1 first) |
| `step3_3_massmap.py` | 4 + 5 | `mass_map.png`, `mass_map_hist.png` | per-patch anacal (no PDF dep) |

Shared loaders / helpers live in `_common.py` (butler, find_tracts, paths) and `_step3_common.py` (anacal+photoz+pdf join, `select_sources`, `load_context`). All cluster-WL primitives live in **`xlens.utils.{nxg, massmap, match}`** (`calibrate_shapes`, `anacal_get_tang_cross`, `fast_bootstrap_mean`, `build_flat_wcs_grid`, `compute_mass_map`, `schirmer_filter`, `sky_match`, `append_specz_columns`, `spec_selection`).

## Cleaning up runs

```bash
# Remove the CHAINED + RUN for a360_anacal2 (keep the chained inputs alone):
butler remove-collections <repo> u/xiangchl/dp1-v2/a360_anacal2 --no-confirm
butler remove-runs <repo> u/xiangchl/dp1-v2/a360_anacal2/<timestamp> --no-confirm
```

`butler query-collections <repo> "u/xiangchl/dp1-v2*"` lists what's there.

## Dependencies (LSST `lsst-scipipe-12.1.0-exact-ext` env)

`numpy`, `matplotlib`, `astropy`, `scipy`, `clmm`, `emcee`, `lsst.daf.butler`, `lsst.geom`, `lsst.afw.image`, plus an editable install of `xlens` (selection helpers + cluster-WL primitives) and `bps_parsl_sites` (parsl-based BPS backend; sourced by `setup_lsst_v30.bash`).
