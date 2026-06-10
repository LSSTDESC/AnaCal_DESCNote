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
from xlens.catalog.base import build_selection_mask  # noqa: E402


from _common import (
    DEFAULT_REPO, DEFAULT_SKYMAP, FIELDS_ROOT,
    find_tracts, open_butler, resolve_field_out,
)


DEFAULT_ANACAL_DS = "deep_coadd_cell_anacal_catalog"
DEFAULT_PHOTOZ_DS = "deep_coadd_cell_anacal_fzb_point"
DEFAULT_PDF_DS = "deep_coadd_cell_anacal_fzb_pdfs"
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
    p.add_argument("--trace-min", type=float, default=0.05,
                   help="size cut (m2/m0 > trace_min) (default: 0.05)")
    p.add_argument("--emax", type=float, default=0.3,
                   help="shape cut |e|^2 < emax^2 (default: 0.3)")
    p.add_argument("--z-min", type=float, default=0.40,
                   help="photo-z lower bound on zbest_0 (default: 0.40)")
    p.add_argument("--z-max", type=float, default=2.0,
                   help="photo-z upper bound on zbest_0 (default: 2.0)")
    p.add_argument("--iband-size-min", type=float, default=None,
                   help="legacy BNL-notebook size cut: drop sources with "
                        "(i_fpfs1_m20 + i_fpfs1_m00) / i_fpfs1_m00 <= this. "
                        "Pass 0.1 to reproduce the old default_selection. "
                        "Off by default (use --emax / --trace-min instead).")
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


_DEFAULT_MAG_MAX = {"g": 25.5, "r": 25.0, "i": 23.5, "z": 24.5}
_DEFAULT_Z_RANGE = (0.40, 2.0)
_DEFAULT_Z_COL = "zbest_0"
_DEFAULT_WSEL_MIN = 1e-5


def select_sources(
    table,
    *,
    flux_name="fpfs1",
    mag_zero=31.4,
    mag_max=None,
    emax=0.3,
    trace_min=0.05,
    wsel_min=_DEFAULT_WSEL_MIN,
    z_range=_DEFAULT_Z_RANGE,
    z_col=_DEFAULT_Z_COL,
    dg=0.0,
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
        Per-band upper magnitude cut. ``None`` -> the BNL-notebook
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
    msk = np.ones(len(table), dtype=bool)
    if require_is_primary and "is_primary" in table.colnames:
        msk &= np.asarray(table["is_primary"], dtype=bool)
    if wsel_min is not None and "wsel" in table.colnames:
        msk &= np.asarray(table["wsel"], dtype=np.float64) > wsel_min
    if z_range is not None and z_col in table.colnames:
        z = np.asarray(table[z_col], dtype=np.float64)
        z_lo, z_hi = z_range
        msk &= (z > z_lo) & (z < z_hi)
    mag_size_msk, _ = build_selection_mask(
        table, comp=1, dg=dg,
        bands="griz", mag_max=mag_max, mag_zero=mag_zero,
        flux_name=flux_name, shape_name="fpfs",
        emax=emax, trace_min=trace_min,
    )
    msk &= mag_size_msk
    return msk


# Back-compat alias for any old caller still importing default_selection.
default_selection = select_sources


def add_band_mags(table, mag_zero=31.4, flux_name="fpfs1"):
    """Add `{band}_mag` columns derived from `{band}_flux_{flux_name}`.

    No longer used by ``default_selection`` (which now computes mags
    internally via ``build_selection_mask``), but kept for any script
    that still expects ``{band}_mag`` columns on the loaded table
    (e.g. step2_compare diagnostics).
    """
    for b in ("g", "r", "i", "z"):
        col = f"{b}_flux_{flux_name}"
        if col not in table.colnames:
            raise KeyError(
                f"missing column {col!r} (try --flux-name fpfs1, or rerun the "
                f"pipeline with do_measure_flux_gauss=True for gauss2)"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            table[f"{b}_mag"] = mag_zero - 2.5 * np.log10(
                np.asarray(table[col], dtype=np.float64)
            )
    return table


def load_context(args, load_pdfs=False):
    """Load anacal + photoz (+ optional PDFs), join, select, calibrate.

    Parameters
    ----------
    load_pdfs : bool
        If True, also load per-source FlexZBoost PDFs from
        ``args.pdf_dataset`` (default ``deep_coadd_cell_anacal_fzb_pdfs``)
        and return them via ``ctx['pdfs_all']`` (row-aligned with
        ``ctx['table_all']``) and ``ctx['pdfs']`` (after the source cut,
        row-aligned with ``ctx['table']``). ``None`` if not requested or
        not present.
    """
    butler = open_butler(args.collection, repo=args.repo)
    skymap = butler.get("skyMap", skymap=args.skymap)

    tract_ids = find_tracts(skymap, args.ra, args.dec, args.radius, args.grid_step)
    print(f"# overlapping tracts: {tract_ids}", file=sys.stderr)

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

    msk = select_sources(
        cat,
        flux_name=args.flux_name, mag_zero=args.mag_zero,
        trace_min=args.trace_min, emax=args.emax,
        z_range=(args.z_min, args.z_max),
    )
    if args.iband_size_min is not None:
        s_iband = (cat["i_fpfs1_m20"] + cat["i_fpfs1_m00"]) / cat["i_fpfs1_m00"]
        msk &= np.asarray(s_iband) > args.iband_size_min
        print(f"# legacy --iband-size-min={args.iband_size_min}: "
              f"{int(msk.sum()):,} rows after combined cut",
              file=sys.stderr)
    cat = cat[msk]
    print(f"# rows after selection: {len(cat):,}", file=sys.stderr)

    e1, e2, res = calibrate_shapes(cat)
    cat["response"] = res
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
        "ra_bcg": args.ra,
        "dec_bcg": args.dec,
        "z_cl": args.z_cl,
        "pdfs_all": pdfs_all,
        "pdfs": pdfs_sel,
        "z_grid": Z_GRID,
    }
