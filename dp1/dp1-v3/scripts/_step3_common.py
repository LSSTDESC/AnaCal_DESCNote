"""Shared data-prep for the step3_*.py cluster-analysis scripts.

Loads the per-patch ``deep_coadd_cell_anacal_catalog`` for every tract
that overlaps a search box around the cluster center, joins the matching
``deep_coadd_cell_anacal_fzb_point`` photo-z table on ``object_id``,
runs ``utils.calibrate_shapes``, and returns a small dict the per-plot
scripts can consume.

Also centralises CLI argument-parsing for the field / collection /
cluster choice, so each step3 script accepts a consistent surface.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.table import Table, vstack


SCRIPTS_DIR = Path(__file__).resolve().parent
# Allow `from utils import ...` to resolve to scripts/utils/.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from xlens.utils.nxg import calibrate_shapes  # noqa: E402
from xlens.catalog.utils import flux_to_mag  # noqa: E402


from _common import (
    DEFAULT_REPO, DEFAULT_SKYMAP, FIELDS_ROOT,
    find_tracts, open_butler, resolve_field_out,
)


DEFAULT_ANACAL_DS = "deep_coadd_cell_anacal_catalog"
DEFAULT_PHOTOZ_DS = "deep_coadd_cell_anacal_fzb_point"
DEFAULT_PDF_DS = "deep_coadd_cell_anacal_fzb_pdfs"
DEFAULT_MERGED_DS = "deep_coadd_cell_anacal_merged"
# Z grid matches xlens.catalog.redshift.Z_GRIDS (np.linspace(0, 5, 501)).
Z_GRID = np.linspace(0.0, 5.0, 501)
STEP3_SUBDIR = "step3"


def add_common_args(p: argparse.ArgumentParser):
    """Attach CLI args shared by every step3 script."""
    p.add_argument("--ra", type=float, required=True, help="cluster RA in degrees (BCG)")
    p.add_argument("--dec", type=float, required=True, help="cluster Dec in degrees (BCG)")
    p.add_argument("--z-cl", type=float, required=True, help="cluster redshift")
    p.add_argument("--radius", type=float, required=True,
                   help="half-side of search square in degrees")
    p.add_argument("--collection", required=True,
                   help="butler collection holding per-patch anacal catalogs")
    p.add_argument("--field", default=None,
                   help=f"field name; output goes to <FIELDS_ROOT>/<field>/{STEP3_SUBDIR}/<out>")
    p.add_argument("--out", default=None,
                   help="explicit output path (overrides --field).")
    p.add_argument("--repo", default=DEFAULT_REPO, help="butler repo")
    p.add_argument("--skymap", default=DEFAULT_SKYMAP, help="skymap name")
    p.add_argument("--anacal-dataset", default=DEFAULT_ANACAL_DS,
                   help="per-patch anacal catalog dataset type")
    p.add_argument("--photoz-dataset", default=DEFAULT_PHOTOZ_DS,
                   help="per-patch photo-z table dataset type")
    p.add_argument("--pdf-dataset", default=DEFAULT_PDF_DS,
                   help="per-patch photo-z PDF dataset type (only used when "
                        "the calling script requests load_pdfs=True)")
    p.add_argument("--grid-step", type=float, default=0.1,
                   help="grid spacing (deg) for sampling tracts (default: 0.1)")
    p.add_argument("--mag-zero", type=float, default=31.4,
                   help="magnitude zero point (default: 31.4)")
    p.add_argument("--flux-name", default="fpfs1",
                   help="flux-column suffix for mag computation (default: fpfs1)")
    p.add_argument("--trace-min", type=float, default=0.1,
                   help="band-combined size cut "
                        "(fpfs1_m00 + fpfs1_m20) / fpfs1_m00 > trace_min "
                        "(default: 0.1; matches TXPipe's T_cut on the "
                        "same moments).")
    p.add_argument("--emax", type=float, default=0.5,
                   help="band-combined shape magnitude cut |e|<emax "
                        "(applied as esq<emax^2 on the merge-stage esq "
                        "column). Default 0.5; matches TXPipe's emax knob.")
    p.add_argument("--z-col", default=_DEFAULT_Z_COL,
                   help=f"source-z column "
                        f"(default: {_DEFAULT_Z_COL!r}; alt: zmode_0)")
    p.add_argument("--mag-max-per-band", default=None,
                   help="per-band mag upper bound, e.g. "
                        "'lsst_u=27.5,lsst_g=27.5,lsst_r=24.5,...'. Keys "
                        "are survey-prefixed. Default: 50 in every "
                        "griz band (above the xlens mag cap = no cut).")
    p.add_argument("--z-min", type=float, default=0.40,
                   help="lower bound on --z-col (default: 0.40)")
    p.add_argument("--z-max", type=float, default=2.0,
                   help="upper bound on --z-col (default: 2.0)")
    p.add_argument("--iband-size-min", type=float, default=None,
                   help="legacy size cut: drop sources with "
                        "(i_fpfs1_m20 + i_fpfs1_m00) / i_fpfs1_m00 <= this. "
                        "Pass 0.1 to reproduce the old default_selection. "
                        "Off by default (use --emax / --trace-min instead).")
    p.add_argument("--band-weights", default=None,
                   help="per-band weights for the FPFS shape combine, "
                        "e.g. 'g=0.53,r=0.33,i=0.14,z=0.005'. Default: "
                        "use xlens.utils.nxg._DEFAULT_FPFS_WEIGHTS.")
    p.add_argument("--delta-gamma", type=float, default=0.0,
                   help="half-shift γ used when computing the "
                        "selection response R_sel via ±γ variants of "
                        "the cut. Set to 0.01 to match xlens photoZPipe "
                        "DISTORTIONS + TXPipe's AnaCalCalculator. "
                        "Default 0 disables R_sel — R_total then equals "
                        "R_shape+R_detect only.")
    return p


def resolve_out(args, default_name: str) -> Optional[str]:
    """Pick output path from --out or derive from --field; mkdir -p parent."""
    return resolve_field_out(args.field, STEP3_SUBDIR, default_name, args.out)


def load_patch_pairs(butler, tract_ids, skymap_name, anacal_ds, photoz_ds,
                     pdf_ds=None):
    """Per-tract iterate patches; join anacal + photo-z by object_id, vstack.

    Returns
    -------
    cat : astropy.Table
        Concatenated anacal+photoz table (row-aligned across patches).
    pdfs : np.ndarray or None
        If ``pdf_ds`` is given and present, returns the vstacked
        undistorted PDFs (shape (N, nz)). Row order matches ``cat``.
        If the pdf dataset is absent for any patch we return ``None``
        (with a warning) so callers can fall back gracefully.
    """
    pieces = []
    pdf_pieces = [] if pdf_ds else None
    pdf_missing = False
    for tid in tract_ids:
        a_refs = list(butler.registry.queryDatasets(
            anacal_ds, where="skymap=:s AND tract=:t",
            bind={"s": skymap_name, "t": tid},
        ))
        z_refs = {
            (r.dataId["patch"]): r
            for r in butler.registry.queryDatasets(
                photoz_ds, where="skymap=:s AND tract=:t",
                bind={"s": skymap_name, "t": tid},
            )
        }
        p_refs = {}
        if pdf_ds:
            p_refs = {
                (r.dataId["patch"]): r
                for r in butler.registry.queryDatasets(
                    pdf_ds, where="skymap=:s AND tract=:t",
                    bind={"s": skymap_name, "t": tid},
                )
            }
        if not a_refs:
            print(f"#   tract={tid}: no {anacal_ds} — skipped", file=sys.stderr)
            continue
        n_kept = 0
        for ar in a_refs:
            patch = ar.dataId["patch"]
            a = Table(butler.get(ar))
            zref = z_refs.get(patch)
            if zref is None:
                print(f"#   tract={tid} patch={patch}: no {photoz_ds} — skipped",
                      file=sys.stderr)
                continue
            z = Table(butler.get(zref))
            if len(a) != len(z) or not np.array_equal(a["object_id"], z["object_id"]):
                z = z[np.argsort(z["object_id"])]
                a_sorted = a[np.argsort(a["object_id"])]
                a = a_sorted
            for c in z.colnames:
                if c != "object_id":
                    a[c] = z[c]
            pieces.append(a)
            if pdf_pieces is not None:
                pref = p_refs.get(patch)
                if pref is None:
                    pdf_missing = True
                    # placeholder of correct row count; will be discarded
                    # below if we fall back to None.
                    pdf_pieces.append(np.zeros((len(a), 501), dtype=np.float32))
                else:
                    pdfs_3d = butler.get(pref)  # (N, ndist, nz)
                    pdf_pieces.append(np.asarray(pdfs_3d[:, 0, :], dtype=np.float32))
            n_kept += len(a)
        print(f"#   tract={tid}: kept {len(a_refs)} patches, {n_kept:,} rows",
              file=sys.stderr)
    if not pieces:
        raise RuntimeError(f"no {anacal_ds} in any tract")
    cat = vstack(pieces, metadata_conflicts="silent")
    pdfs = None
    if pdf_pieces is not None:
        if pdf_missing:
            print(f"# WARN: {pdf_ds} missing for some patches — PDFs not loaded",
                  file=sys.stderr)
        else:
            pdfs = np.concatenate(pdf_pieces, axis=0)
            print(f"# loaded {pdf_ds}: shape={pdfs.shape}", file=sys.stderr)
    return cat, pdfs


_DEFAULT_MAG_MAX = {"lsst_g": 50.0, "lsst_r": 50.0, "lsst_i": 50.0, "lsst_z": 50.0}
_DEFAULT_Z_RANGE = (0.40, 2.0)
_DEFAULT_Z_COL = "zmode_0"
_DEFAULT_WSEL_MIN = 1e-5


def _has_col(table, name):
    """Duck-typed column-existence check: works for astropy.Table,
    numpy structured arrays and plain dict-of-arrays."""
    cn = getattr(table, "colnames", None)
    if cn is not None:
        return name in cn
    dt = getattr(table, "dtype", None)
    if dt is not None and getattr(dt, "names", None):
        return name in dt.names
    return name in table


def _nrows(table):
    """Duck-typed row count: astropy.Table / structured ndarray use
    ``len(table)`` (rows), but a plain dict-of-arrays has
    ``len(dict) == n_columns`` — so fall back to the length of an
    arbitrary column instead."""
    if hasattr(table, "colnames") or getattr(getattr(table, "dtype", None),
                                             "names", None):
        return len(table)
    if len(table) == 0:
        return 0
    return len(next(iter(table.values())))


def select_sources(
    table,
    *,
    flux_name="fpfs1",
    mag_zero=31.4,
    mag_max=None,
    emax=0.5,
    trace_min=0.1,
    wsel_min=_DEFAULT_WSEL_MIN,
    z_range=_DEFAULT_Z_RANGE,
    z_col=_DEFAULT_Z_COL,
    dg=0.0,
    comp=1,
    require_is_primary=True,
):
    """Source-galaxy selection for the cluster-WL scripts.

    The mag (per-band), shape (``|e|^2 < emax^2``) and size
    (``trace > trace_min``) cuts are delegated to
    ``xlens.catalog.base.build_selection_mask`` — the same code
    ``ShearEstimator._measure`` uses, so the cuts are exactly
    consistent with the shear-bias accounting.

    Layered on top: ``is_primary`` (dedup), ``wsel > wsel_min`` (the
    AnaCal selection weight) and a photo-z range. None of those need
    a dg perturbation, so they sit outside ``build_selection_mask``.

    Parameters
    ----------
    table : structured ndarray or astropy Table
        The catalog to cut. Must carry FPFS ``m0/m2/e1/e2`` columns,
        per-band ``{band}_flux_{flux_name}`` + ``_dg`` derivatives,
        ``is_primary``, ``wsel`` and (optionally) the photo-z column.
    flux_name : str
        Suffix used to read fluxes (``"fpfs1"`` or ``"gauss2"``).
    mag_zero : float
        AB-mag zero point of the fluxes (default 31.4 — DP1).
    mag_max : float | dict | None
        Per-band upper magnitude cut. ``None`` -> the built-in
        defaults ``{g: 25.5, r: 25.0, i: 23.5, z: 24.5}``. A scalar is
        broadcast across the bands.
    emax, trace_min : float
        Shape (``|e|^2 < emax^2``) and size (``m2/m0 > trace_min``)
        cuts; xlens defaults are 0.3 and 0.05.
    wsel_min : float
        Lower cut on the AnaCal selection weight.
    z_range : (float, float) or None
        Photo-z window ``(z_min, z_max)`` applied to ``z_col``. ``None``
        skips the photo-z cut.
    z_col : str
        Name of the photo-z point estimate column (default ``zbest_0``).
        Cut is silently skipped if the column isn't present.
    dg : float
        Shear value at which the mag + shape + size cuts are
        evaluated. ``0.0`` is the unperturbed selection used by
        every downstream plot; ``ShearEstimator`` callers pass
        ``+/- dg`` to get the selection-response terms.
    require_is_primary : bool
        Whether to require ``is_primary``. Default ``True``.
    """
    if mag_max is None:
        mag_max = _DEFAULT_MAG_MAX
    msk = np.ones(_nrows(table), dtype=bool)
    if require_is_primary and _has_col(table, "is_primary"):
        msk &= np.asarray(table["is_primary"], dtype=bool)
    # Photo-z at shear ±dg: xlens photoZPipe pre-computes zmode_{1p,1m,2p,2m}
    # with a fixed δγ=0.01 by re-running FlexZBoost on ±δγ-shifted fluxes,
    # so we can only pick up the correct shifted point estimate when the
    # caller's |dg| matches that internal δγ. For other |dg| values, fall
    # back to the un-shifted z_col — the induced R_sel error is small since
    # the z window (0.4, 2.0) is broad relative to the photo-z shift.
    z_variant_col = z_col
    if dg != 0.0:
        _sign_map = {(+1, 1): "1p", (-1, 1): "1m",
                     (+1, 2): "2p", (-1, 2): "2m"}
        key = (1 if dg > 0 else -1, comp)
        suf = _sign_map.get(key)
        if suf is not None:
            # Two supported base-column conventions:
            #   merged/xlens : "zmode_0"    -> "zmode_{suf}"
            #   TXPipe HDF5  : "mean_z"     -> "mean_z_{suf}"
            base = z_col[:-2] if z_col.endswith("_0") else z_col
            candidate = f"{base}_{suf}"
            if _has_col(table, candidate):
                z_variant_col = candidate
    if z_range is not None and _has_col(table, z_variant_col):
        z = np.asarray(table[z_variant_col], dtype=np.float64)
        z_lo, z_hi = z_range
        # Half-open [z_lo, z_hi) — matches TXPipe's zbin edge convention.
        msk &= (z >= z_lo) & (z < z_hi)
    # Shape magnitude cut |e|<emax on the band-combined ``esq`` column
    # (emitted by xlens.MergePipe from the WCS-corrected fpfs1 shape).
    # Shifting linearly via ``desq_dg{comp}`` gives the ±γ variants for
    # R_sel. Bypasses ``build_selection_mask``'s in-line ``emax`` term
    # so both TXPipe and step3 gate on exactly the same quantity.
    if emax is not None and _has_col(table, "esq"):
        esq_shifted = np.asarray(table["esq"], dtype=np.float64)
        dcol = f"desq_dg{comp}"
        if dg != 0.0 and _has_col(table, dcol):
            esq_shifted = esq_shifted + dg * np.asarray(
                table[dcol], dtype=np.float64
            )
        msk &= esq_shifted < emax * emax
    # Size cut on the band-combined fpfs1 moments — same quantity
    # TXPipe applies with its T_cut. Bypasses build_selection_mask's
    # trace path (which reads the detection-band fpfs_m2/fpfs_m0).
    if (trace_min is not None
            and _has_col(table, "fpfs1_m00")
            and _has_col(table, "fpfs1_m20")):
        m00 = np.asarray(table["fpfs1_m00"], dtype=np.float64)
        m20 = np.asarray(table["fpfs1_m20"], dtype=np.float64)
        if dg != 0.0:
            dm00 = f"fpfs1_dm00_dg{comp}"
            dm20 = f"fpfs1_dm20_dg{comp}"
            if _has_col(table, dm00) and _has_col(table, dm20):
                m00 = m00 + dg * np.asarray(table[dm00], dtype=np.float64)
                m20 = m20 + dg * np.asarray(table[dm20], dtype=np.float64)
        msk &= (m00 + m20) / m00 > trace_min
    # Per-band mag cut applied inline (band keys are survey-prefixed
    # column stems for v3, e.g. lsst_g / lsst_r / lsst_i / lsst_z). No
    # longer routed through build_selection_mask because that helper
    # also consumes detection-band fpfs_* moments which we no longer
    # keep in the merged catalog.
    if isinstance(mag_max, dict):
        band_cuts = mag_max.items()
    else:
        band_cuts = ((b, mag_max) for b in
                     ("lsst_g", "lsst_r", "lsst_i", "lsst_z"))
    for b, mx in band_cuts:
        col = f"{b}_flux_{flux_name}"
        if not _has_col(table, col):
            raise KeyError(
                f"missing {col!r} (needed for the per-band mag cut)"
            )
        flux = np.asarray(table[col], dtype=np.float64)
        if dg != 0.0:
            dcol = f"{b}_dflux_{flux_name}_dg{comp}"
            if _has_col(table, dcol):
                flux = flux + dg * np.asarray(table[dcol], dtype=np.float64)
        mag, _ = flux_to_mag(flux, mag_zero=mag_zero)
        msk &= mag < mx
    return msk


# Back-compat alias for any old caller still importing default_selection.
default_selection = select_sources


def add_band_mags(table, mag_zero=31.4, flux_name="fpfs1",
                  bands=("lsst_g", "lsst_r", "lsst_i", "lsst_z")):
    """Add ``{band}_mag`` columns derived from ``{band}_flux_{flux_name}``.

    No longer used by ``default_selection`` (which now computes mags
    internally via ``build_selection_mask``), but kept for any script
    that still expects ``{band}_mag`` columns on the loaded table
    (e.g. step2_compare diagnostics, step2_gaia_tangential*).
    Bad (non-positive) flux rows become NaN so plotting code
    (histogram / imshow) simply drops them instead of piling at a
    sentinel mag.
    """
    for b in bands:
        col = f"{b}_flux_{flux_name}"
        if not _has_col(table, col):
            raise KeyError(
                f"missing column {col!r} (try --flux-name fpfs1, or rerun the "
                f"pipeline with do_measure_flux_gauss=True for gauss2)"
            )
        mag, _ = flux_to_mag(
            np.asarray(table[col], dtype=np.float64),
            mag_zero=mag_zero,
        )
        table[f"{b}_mag"] = mag
    return table


def load_merged(butler, tract_ids, skymap_name,
                merged_ds=DEFAULT_MERGED_DS):
    """Load the per-tract merged anacal catalog(s) and vstack them.

    The merged catalog is already filtered by ``is_primary`` and
    ``wsel > 1e-5`` inside ``MergePipe._finalize_columns``, and carries
    the band-combined ``fpfs1_e1/e2`` (+ shear responses), the
    detection-band raw ``fpfs_*`` (for the shape/size selection cuts),
    per-band ``{b}_flux_fpfs1`` + ``gauss2`` fluxes, photo-z point
    estimates and the PSF HSM moments. PDFs are *not* in merged.
    """
    pieces = []
    for tid in tract_ids:
        refs = list(butler.registry.queryDatasets(
            merged_ds, where="skymap=:s AND tract=:t",
            bind={"s": skymap_name, "t": tid},
        ))
        if not refs:
            print(f"#   tract={tid}: no {merged_ds} — skipped", file=sys.stderr)
            continue
        # one ref per (skymap, tract); pick first if more.
        t = Table(butler.get(refs[0]))
        pieces.append(t)
        print(f"#   tract={tid}: {len(t):,} rows", file=sys.stderr)
    if not pieces:
        raise RuntimeError(f"no {merged_ds} in any candidate tract")
    return vstack(pieces, metadata_conflicts="silent")


def _compute_response_from_merged(cat, c0=50.0):
    """Reproduce ``xlens.utils.nxg.calibrate_shapes``'s response from
    the columns the merged catalog carries — works without per-band
    moments because the merge has already done the band combination.

    Returns ``e1, e2, res`` matching what ``calibrate_shapes`` produces.
    Note: the merged ``fpfs1_e1/e2`` are stored as ``m22c/(m00+c0)`` /
    ``m22s/(m00+c0)`` (already including any WCS correction the merge
    applied). For consistency with ``calibrate_shapes`` we use the same
    formula here.

    ``res`` here is the per-source *shape+detect* response term only —
    the selection-response contribution R_sel requires ±γ variants of
    the source cut, so it is computed separately by ``compute_r_sel``
    when the caller is doing full-response calibration.
    """
    e1 = np.asarray(cat["fpfs1_e1"], dtype=np.float64) * np.asarray(cat["wsel"])
    e2 = np.asarray(cat["fpfs1_e2"], dtype=np.float64) * np.asarray(cat["wsel"])
    wsel = np.asarray(cat["wsel"], dtype=np.float64)
    dwsel_dg1 = np.asarray(cat["dwsel_dg1"], dtype=np.float64)
    dwsel_dg2 = np.asarray(cat["dwsel_dg2"], dtype=np.float64)
    de1_dg1 = np.asarray(cat["fpfs1_de1_dg1"], dtype=np.float64)
    de2_dg2 = np.asarray(cat["fpfs1_de2_dg2"], dtype=np.float64)
    e1_raw = np.asarray(cat["fpfs1_e1"], dtype=np.float64)
    e2_raw = np.asarray(cat["fpfs1_e2"], dtype=np.float64)
    res = (
        de1_dg1 * wsel + dwsel_dg1 * e1_raw
        + de2_dg2 * wsel + dwsel_dg2 * e2_raw
    ) / 2.0
    return e1, e2, res


def compute_r_sel(cat, cut_fn, delta_gamma):
    """Selection-response term of the mean per-source shear calibration.

    Runs the caller-supplied ``cut_fn(cat, dg, comp)`` at ``(±delta_gamma, 1)``
    and ``(±delta_gamma, 2)``, then finite-differences the sample mean of
    ``wsel · e_raw`` between each ± pair.  This matches TXPipe's
    ``AnaCalCalculator.R_sel`` formula exactly, so total responses
    ``R_shape + R_detect + R_sel`` computed here agree with what the
    TXPipe pipeline reports (up to the choice of cut).

    Parameters
    ----------
    cat : structured ndarray or astropy Table
        Merged catalog carrying the columns ``cut_fn`` reads (fpfs
        shape/size, per-band flux, wsel, zmode variants).  Must expose
        ``fpfs1_e1/e2`` and ``wsel`` for the finite-difference numerator.
    cut_fn : callable
        ``cut_fn(cat, dg, comp) -> bool ndarray``.  ``select_sources``
        with matching keyword args is the canonical caller.
    delta_gamma : float
        Half-shift of the two variants (``±delta_gamma``); pass 0.01 to
        match xlens' photoZPipe DISTORTIONS.

    Returns
    -------
    R_sel : float
        Component-averaged selection response, per Convention A
        (numerator is ⟨wsel · e_raw⟩ over each ±γ selection sample).
    """
    wsel = np.asarray(cat["wsel"], dtype=np.float64)
    e1 = np.asarray(cat["fpfs1_e1"], dtype=np.float64)
    e2 = np.asarray(cat["fpfs1_e2"], dtype=np.float64)
    m_1p = cut_fn(cat, +delta_gamma, 1)
    m_1m = cut_fn(cat, -delta_gamma, 1)
    m_2p = cut_fn(cat, +delta_gamma, 2)
    m_2m = cut_fn(cat, -delta_gamma, 2)
    R_sel_1 = ((wsel[m_1p] * e1[m_1p]).mean()
               - (wsel[m_1m] * e1[m_1m]).mean()) / (2.0 * delta_gamma)
    R_sel_2 = ((wsel[m_2p] * e2[m_2p]).mean()
               - (wsel[m_2m] * e2[m_2m]).mean()) / (2.0 * delta_gamma)
    return 0.5 * (R_sel_1 + R_sel_2)


def load_context(args, load_pdfs=False, use_merged=True):
    """Load merged anacal catalog (or per-patch + photoz when
    ``use_merged=False``), select sources, and return shear-ready ctx.

    Parameters
    ----------
    load_pdfs : bool
        Forces ``use_merged=False`` — PDFs only live in the per-patch
        ``args.pdf_dataset`` (default ``deep_coadd_cell_anacal_fzb_pdfs``).
    use_merged : bool
        Default True: read ``deep_coadd_cell_anacal_merged`` and skip
        the in-script ``calibrate_shapes`` (the merge already did the
        band combination + soft-bias regularisation + optional WCS
        correction). Set False to fall back to the per-patch path
        (needed for step3_1 which consumes PDFs).
    """
    butler = open_butler(args.collection, repo=args.repo)
    skymap = butler.get("skyMap", skymap=args.skymap)

    tract_ids = find_tracts(skymap, args.ra, args.dec, args.radius, args.grid_step)
    print(f"# overlapping tracts: {tract_ids}", file=sys.stderr)

    if load_pdfs:
        use_merged = False

    pdfs_all = None
    if use_merged:
        merged_ds = getattr(args, "merged_dataset", DEFAULT_MERGED_DS)
        cat = load_merged(butler, tract_ids, args.skymap, merged_ds=merged_ds)
    else:
        pdf_ds = getattr(args, "pdf_dataset", DEFAULT_PDF_DS) if load_pdfs else None
        cat, pdfs_all = load_patch_pairs(
            butler, tract_ids, args.skymap,
            args.anacal_dataset, args.photoz_dataset,
            pdf_ds=pdf_ds,
        )
    print(f"# rows before selection: {len(cat):,}", file=sys.stderr)

    add_band_mags(cat, mag_zero=args.mag_zero, flux_name=args.flux_name)
    cat["index"] = np.arange(len(cat))
    cat_all = cat

    mag_max = None
    if args.mag_max_per_band is not None:
        mag_max = {
            kv.split("=")[0].strip(): float(kv.split("=")[1])
            for kv in args.mag_max_per_band.split(",")
        }
    # Bind every knob the user set once, so the baseline mask and the
    # ±γ variants used to compute R_sel apply an *identical* cut
    # surface — anything less would poison the finite difference.
    cut_kwargs = dict(
        flux_name=args.flux_name, mag_zero=args.mag_zero,
        mag_max=mag_max,
        trace_min=args.trace_min, emax=args.emax,
        z_range=(args.z_min, args.z_max),
        z_col=args.z_col,
    )
    def _cut_fn(cat_arg, dg, comp):
        return select_sources(cat_arg, dg=dg, comp=comp, **cut_kwargs)
    msk = _cut_fn(cat, 0.0, 1)
    if args.iband_size_min is not None:
        s_iband = (cat["i_fpfs1_m20"] + cat["i_fpfs1_m00"]) / cat["i_fpfs1_m00"]
        msk &= np.asarray(s_iband) > args.iband_size_min
        print(f"# legacy --iband-size-min={args.iband_size_min}: "
              f"{int(msk.sum()):,} rows after combined cut",
              file=sys.stderr)
    cat = cat[msk]
    print(f"# rows after selection: {len(cat):,}", file=sys.stderr)

    if use_merged:
        # Merged catalog already carries the band-combined fpfs1_e1/e2
        # (and responses). Reuse them directly — equivalent to running
        # calibrate_shapes on per-patch when the merge yaml's
        # band_weights + fpfs_c0 are aligned with calibrate_shapes
        # defaults.
        e1, e2, res = _compute_response_from_merged(cat)
    else:
        weights = None
        if args.band_weights is not None:
            weights = {
                kv.split("=")[0].strip(): float(kv.split("=")[1])
                for kv in args.band_weights.split(",")
            }
            print(f"# overriding calibrate_shapes weights: {weights}",
                  file=sys.stderr)
        e1, e2, res = calibrate_shapes(cat, weights=weights)
    cat["response"] = res
    # R_sel is a scalar over the selected sample: run the same cut at
    # (±args.delta_gamma, comp∈{1,2}) on the full pre-cut catalog and
    # finite-difference ⟨wsel·e⟩ between each ± pair.  Skipped when
    # ``args.delta_gamma`` is unset (older step3 callers).
    r_sel = None
    dg_sel = float(getattr(args, "delta_gamma", 0.0) or 0.0)
    if dg_sel > 0.0:
        try:
            r_sel = compute_r_sel(cat_all, _cut_fn, dg_sel)
        except KeyError as e:
            # Missing zmode_1p/... or dwsel_dg{c} — leave R_sel unset
            # and let the caller fall back to R_shape+R_detect only.
            print(f"# WARN: R_sel skipped — missing variant column {e}",
                  file=sys.stderr)
    pdfs_sel = pdfs_all[msk] if pdfs_all is not None else None
    return {
        "butler": butler,
        "skymap": skymap,
        "tract_ids": tract_ids,
        "table_all": cat_all,
        "table": cat,
        "e1": np.asarray(e1, dtype=np.float64),
        "e2": np.asarray(e2, dtype=np.float64),
        "res": np.asarray(res, dtype=np.float64),
        "r_sel": r_sel,
        "ra_bcg": args.ra,
        "dec_bcg": args.dec,
        "z_cl": args.z_cl,
        "pdfs_all": pdfs_all,
        "pdfs": pdfs_sel,
        "z_grid": Z_GRID,
    }
