# Abell-360 merged-catalog schema

This file documents the columns of the per-tract merged catalogs
produced by `xlens.processor.merge.MergePipe`:

| File | Tract | Rows |
| --- | --- | --- |
| `a360_merged_tract10463.parq` | 10463 | 45,176 |
| `a360_merged_tract10464.parq` | 10464 | 25,104 |
| `a360_merged_tract10704.parq` | 10704 | 3,861 |

Storage format is Parquet (`ArrowAstropy` storage class); read with
`astropy.table.Table.read("...parq")`.

Each tract catalog has 74 columns. The four magnitude bands are `g`, `r`, `i`,
`z`; the photometric zero point of the input fluxes is `mag_zero = 31.4`.

## Shapelet kernel

All FPFS shapelet quantities (`fpfs1_m00`, `fpfs1_m20`, `fpfs1_e1/2`,
`{b}_flux_fpfs1`, …) were measured with the `fpfs1` Gaussian shapelet
kernel of width

$$\sigma_{\rm shapelets1} = 0.54\,\text{arcsec}$$

set via `MeasureCellCoaddsPipeConfig.fpfs.sigma_shapelets1 = 0.54` in
`measure_pipeline.yaml`. This `σ` enters the Gaussian-aperture flux
relation `flux = m00 · 2π σ²` (so the per-band fluxes inherit the same
kernel width).

## Coordinate convention

**The local sky frame uses the GalSim convention `(+x = West, +y = North)`.
Concretely:

* `g1`, `g2`, `e1`, `e2`, … and every spin-2 quantity
  is expressed in this `+x = West` frame.
* `g2` flips sign under a `(+x = West) → (+x = East)` convention swap;
  every `*_dg2` derivative therefore also flips sign under that swap.

---

## 1. Detection / position / selection (12 columns)

| Column | Dtype | Unit | Description |
| --- | --- | --- | --- |
| `ra` | float64 | deg (ICRS) | Right ascension of the detection. |
| `dec` | float64 | deg (ICRS) | Declination of the detection. |
| `x1` | float64 | arcsec | First-moment-derived sky-plane position (`+x = West`), measured relative to the tract origin. Divide by `pixel_scale` for tract-pixel units. |
| `x2` | float64 | arcsec | Second axis of the same position pair (`+y = North`). |
| `x1_det` | float64 | arcsec | First detection-axis position used by AnaCal. |
| `x2_det` | float64 | arcsec | Second detection-axis position. |
| `wsel` | float64 | — | AnaCal selection weight `w_sel`|
| `dwsel_dg1` | float64 | — | First-order shear response of `wsel` to `g1`. |
| `dwsel_dg2` | float64 | — | First-order shear response of `wsel` to `g2`. |
| `mask_value` | int32 | bitmask | Bit-OR of bad-pixel and bright-star mask flags at the detection's pixel. |
| `object_id` | int64 | — | Stable per-object identifier. Unique across patches, tracts, and merged outputs. |

---

## 2. Band-combined ellipticity and shear response (6 columns)

The band combination is `m_combined = Σ_b w_b · m_band` with `Σ_b w_b = 1`.
`MergePipe.config.band_weights` selects the per-band weights (defaults to
`1 / median({band}_flux_fpfs1_err)^2`). `e_i = m22{c,s} / (m00 + c0)`
where `c0 = fpfs_c0 · 10^((mag_zero − 30) / 2.5)`. The local WCS shear +
rotation correction is then applied to `e1`, `e2` in place.

| Column | Dtype | Unit | Description |
| --- | --- | --- | --- |
| `fpfs1_e1` | float64 | — | Band-combined and WCS-corrected first ellipticity component. |
| `fpfs1_e2` | float64 | — | Band-combined and WCS-corrected second ellipticity component (Flips sign if x-axis is flipped). |
| `fpfs1_de1_dg1` | float64 | — | Diagonal shear response `∂e1 / ∂g1`. |
| `fpfs1_de1_dg2` | float64 | — | Off-diagonal shear response `∂e1 / ∂g2`. (Flips sign if x-axis is flipped). |
| `fpfs1_de2_dg1` | float64 | — | Off-diagonal shear response `∂e2 / ∂g1`. (Flips sign if x-axis is flipped). |
| `fpfs1_de2_dg2` | float64 | — | Diagonal shear response `∂e2 / ∂g2`. |

---

## 3. Band-combined shapelet moments and shear responses (6 columns)

Used downstream for size cuts (`(m00 + m20) / m00 > 0.1`) and weight response
calculation.

| Column | Dtype | Unit | Description |
| --- | --- | --- | --- |
| `fpfs1_m00` | float64 | — | Band-combined zeroth shapelet moment. Flux-like. |
| `fpfs1_dm00_dg1` | float64 | — | `∂m00 / ∂g1` (band-combined). |
| `fpfs1_dm00_dg2` | float64 | — | `∂m00 / ∂g2`. (Flips sign if x-axis is flipped) |
| `fpfs1_m20` | float64 | — | Band-combined size-like spin-0 shapelet moment. |
| `fpfs1_dm20_dg1` | float64 | — | `∂m20 / ∂g1`. |
| `fpfs1_dm20_dg2` | float64 | — | `∂m20 / ∂g2`. (Flips sign if x-axis is flipped) |

---

## 4. Per-band fluxes (4 columns × 4 bands = 16 columns)

`flux = m00 · 2π σ²` (Gaussian-aperture flux derived from the
shapelet `m00`; `σ = sigma_shapelets1` set by the FPFS task). Magnitudes
are derived as `mag = mag_zero − 2.5 · log10(flux)` with `mag_zero = 31.4`.

For each band `b ∈ {g, r, i, z}` the catalog has:

| Column | Dtype | Unit | Description |
| --- | --- | --- | --- |
| `{b}_flux_fpfs1` | float64 | linear flux (zero-pointed at `mag_zero`) | Per-band Gaussian-aperture flux for band `b`. |
| `{b}_dflux_fpfs1_dg1` | float64 | flux | `∂flux / ∂g1` for band `b`. |
| `{b}_dflux_fpfs1_dg2` | float64 | flux | `∂flux / ∂g2` for band `b`. (Flips sign if x-axis is flipped) |
| `{b}_flux_fpfs1_err` | float64 | flux | 1σ uncertainty on `{b}_flux_fpfs1` (Gaussian-aperture analytic propagation of `m00_err`). |

---

## 5. Photo-z point estimates (7 keys × 5 distortions = 35 columns)

Joined onto the catalog by `MergePipe._join_photoz` (`object_id`-aligned)
from the per-patch `deep_coadd_cell_anacal_fzb_point` produced by
`photoZPipe`. Each row of the photo-z table holds five FlexZBoost
estimates: the undistorted call (`_0`) plus the four `±dg` distortions
in `g1` (`_1p`, `_1m`) and `g2` (`_2p`, `_2m`), evaluated at
`dg = 0.01`. If x-axis is flipped, the `_2p` and `_2m` suffixes are
swapped, since `g2` reverses sign in the LSST convention.

The same seven keys appear with each of the five suffixes
`s ∈ {_0, _1p, _1m, _2p, _2m}`; combined that's 7 × 5 = 35 columns.

| Key (column = `key + suffix`) | Dtype | Unit | Description |
| --- | --- | --- | --- |
| `zbest{s}` | float32 | redshift | Per-object best-estimate redshift (FlexZBoost best estimate). |
| `zmode{s}` | float32 | redshift | Mode of the FlexZBoost p(z). |
| `z025{s}` | float32 | redshift | 2.5th percentile of the FlexZBoost p(z). |
| `z160{s}` | float32 | redshift | 16th percentile of the p(z). |
| `z500{s}` | float32 | redshift | 50th percentile (median) of the p(z). |
| `z840{s}` | float32 | redshift | 84th percentile of the p(z). |
| `z975{s}` | float32 | redshift | 97.5th percentile of the p(z). |
