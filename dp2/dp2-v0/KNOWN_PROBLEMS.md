# DP2 AnaCal v0 — known problems

Issues in the v0 shear catalog and its inputs, with status and the data
that fixes or works around each. [← back to README](README.md)

| # | problem | status |
|---|---|---|
| 1 | Bright stars not masked at image level | catalog-level mask available; image-level rerun pending |
| 2 | Deep fields (EDFS, ECDFS, COSMOS) not processed | **done** — separate collections |
| 3 | RA = 0 strip of the wide field missing | not fixed |
| 4 | PSF moments stored in pixel², not arcsec² | migrated catalog available |

---

## 1. Bright-star mask not applied at image level

**Problem.** The v0 systematics run (`u/xiangchl/anacal-v0/systematics`)
wrote empty GAIA tables for all 81,573 patches — no bright-star halos
were masked at the image level, so `measure`/`merge` and everything
downstream carry unmasked stellar halos.

**Cause.** The GAIA refcat (`refcats/dm-39298/gaia_dr3_20230707`) is a
standalone RUN, not a child of the `dp2` chain, and the connection was
declared `minimum=0`, so the missing input degraded to a silent warning
instead of failing.

**Workaround (v0).** A **catalog-level** mask: `build_gaia_mask.py`
flags each merged-catalog object within an HSC-derived `r(mag)` of a
GAIA star. Apply `gaia_mask == 0` to select unmasked objects.

+ data: `u/xiangchl/anacal-v0/gaia_mask_catalog` (per-tract, aligned to
  the merged catalog by `object_id`)

**Open question.** What is lost by masking at catalog vs image level?
Image-level masking zeroes the pixels before detection/measurement, so
neighbours of a bright star are re-measured on the cleaned image; the
catalog cut only removes the flagged objects. The `systematics_wbsm`
run (image-level masking, whole survey) exists and is verified, so the
fix is a `measure`/`merge` rerun on top of it.

## 2. Deep fields not processed

**Problem.** EDFS, ECDFS and COSMOS were absent from the wide-field
run.

**Solution — done.** Processed as a separate chain (43 tracts), with
**bright-star masking applied at the image level** (their systematics
were built with the refcat wired in).

+ data: `u/xiangchl/anacal-v0/{systematics,measure,merge}_deep_fields`
  — 3,460 patch catalogs, 43/43 tracts, 6.71 M merged objects
+ PSF moments here are already in **arcsec²** (see #4)

## 3. RA = 0 strip of the wide field missing

**Problem.** A one-tract-wide vertical gap at **RA ≈ 0°, Dec −5° to
+4°** is absent from the merged catalog, visible as a near-zero column
in the number-density maps. Seven fully-covered tracts are affected:
**8500, 8742, 8984, 9226, 9469, 9712, 9954**, each with ~100 riz
patches in `dp2` but in neither `patches.csv`, the 820-tract run list,
nor the merge.

**Cause.** Tract selection used `non_deep_coverage.npy > 0.1`, and that
coverage array has an **RA-wrap artefact**: every tract whose centre
sits on RA = 0 (bbox straddling 360°→0°) reports coverage ≈ 0.05–0.17
regardless of its true coverage, so these 7 fell below the 0.1 cut. An
independent estimate (mean fracdet of the r∧i∧z ≥3-visit footprint over
each tract's inner region) gives their true coverage as **0.49–0.92** —
they should have passed comfortably. The metric agrees with
`non_deep_coverage` to a few percent on every non-RA=0 tract, so the
artefact is specific to the wrap.

The neighbouring columns (RA ±1.5°) are present, so this is a genuine
hole inside otherwise contiguous, DESC-footprint-covered sky
(fracdet ≈ 0.6 there), not a survey-edge effect.

**Not fixed.** Requires adding the 7 tracts to the patch/tract lists and
running systematics → measure → merge for them (~1 node-hour), then
refreshing the merge-level products (gaia masks, diagnostics, footprint
maps). The full list of tracts with riz coverage absent from
`patches.csv` is in `DP2/tracts_missing_from_patches.csv`; the 7 here
are the fully-covered, in-footprint subset of the RA = 0 column.

## 4. PSF moments in pixel², not arcsec²

**Problem.** The `{r,i,z}_ext_shapeHSM_HsmPsfMoments_{xx,yy,xy}` columns
in the v0 merge are in **pixel²**. Any consumer that turns them into a
size (FWHM, trace, resolution) must know and apply the 0.2″/pixel
scale; the same columns from an HSC catalog are in a different pixel
scale (0.168″), so the number is not self-describing.

**Fix.** The measurement code now converts these to **arcsec²** at the
point of measurement (from the coadd WCS); higher-order moments are
dimensionless and untouched. The existing v0 merge was migrated in
place-of value (not header) to a new collection.

+ data: `u/xiangchl/anacal-v0/merge_updated` — identical to `merge`
  except the 9 PSF second-moment columns × (r,i,z) are ×0.2² → arcsec².
  `merge` (pixel²) is kept as the original.

**Future.** Record every SIZE observable in arcsec at measurement time
so no consumer needs the pixel scale — this class of bug has already
appeared twice (diagnostics `psf_fwhm`, HSC cluster script).

## 5. Galactic extinction

The v0 merged catalog's anacal fluxes and magnitudes are NOT dereddened for Galactic dust extinction. Users must apply a dereddening correction (for example using SFD or Planck dust maps) before using all flux or magnitude for colour measurements, photometric redshifts, or any analysis sensitive to extinction.
