#!/usr/bin/env python
"""Same PSF-residual / GAIA-star tangential-shear diagnostic as
``step2_gaia_tangential.py`` but for the **deep-coadd** AnaCal flavour
instead of the cell-based one.

Why a separate script?
----------------------
The cell-based and deep-coadd output collections expose slightly
different per-patch products. Three differences to handle:

1. Dataset-type prefix is ``deep_coadd_anacal_*`` (no ``_cell``).
2. ``deep_coadd_anacal_catalog`` and ``deep_coadd_anacal_fzb_point``
   typically have different row counts — we inner-join on
   ``object_id`` (the cell flavour assumes equal-length tables).
3. The deep-coadd collection has no ``deep_coadd_systematics_gaia``
   product — we read GAIA stars from a separate ``--gaia-collection``
   (default: the cell-coadd run that wrote them).

Other than the loader, the recipe is identical to
``step2_gaia_tangential.py``: select sources with the step3 selection,
calibrate shapes via ``xlens.utils.nxg.calibrate_shapes`` (which only
needs the per-band ``{band}_fpfs1_*`` moments + ``wsel``), feed
``g1, g2`` to ``treecorr.NGCorrelation`` with GAIA stars as lenses, and
emit a two-panel ``gT/gX`` plot split into 3 magnitude bins.

Usage
-----
::

    C2="--ra 37.86 --dec 6.98 --radius 1.5 \\
        --collection      u/xiangchl/dp1/a360_anacal_coadd \\
        --gaia-collection u/xiangchl/dp1-v2/a360_anacal2 \\
        --field a360"
    python scripts/step2_gaia_tangential_coadd.py $C2 \\
        --gaia-mag-bins 8,13,15,17 --sep-min 1 --sep-max 200 --nbins 12

This pulls the source galaxies from the deep-coadd run, the GAIA stars
from the cell-coadd run, and writes
``a360/step2/gaia_tangential_coadd.{png,npz}``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import treecorr
from astropy.table import Table, vstack

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _common import (  # noqa: E402
    DEFAULT_REPO, DEFAULT_SKYMAP, DEFAULT_STYLE_RC,
    find_tracts, open_butler, resolve_field_out,
)
from _step3_common import (  # noqa: E402
    add_band_mags, select_sources,
)
from xlens.utils.nxg import calibrate_shapes  # noqa: E402

# Deep-coadd dataset-type defaults (no ``_cell`` infix).
DEFAULT_ANACAL_DS = "deep_coadd_anacal_catalog"
DEFAULT_PHOTOZ_DS = "deep_coadd_anacal_fzb_point"
# GAIA is only written in the cell-coadd output, so we read it from a
# separate collection by default.
DEFAULT_GAIA_DS = "deep_coadd_cell_systematics_gaia"
DEFAULT_GAIA_COLLECTION = "u/xiangchl/dp1-v2/a360_anacal2"

STEP2_SUBDIR = "step2"
DEFAULT_OUT_NAME = "gaia_tangential_coadd.png"
DEFAULT_NPZ_NAME = "gaia_tangential_coadd.npz"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ra", type=float, required=True,
                   help="centre RA in degrees (finds overlapping tracts)")
    p.add_argument("--dec", type=float, required=True,
                   help="centre Dec in degrees")
    p.add_argument("--radius", type=float, required=True,
                   help="half-side of search square in degrees")
    p.add_argument("--collection", required=True,
                   help="butler collection with the DEEP-COADD anacal tables")
    p.add_argument("--gaia-collection", default=DEFAULT_GAIA_COLLECTION,
                   help="butler collection holding the GAIA star tables "
                        f"(default {DEFAULT_GAIA_COLLECTION!r}; the "
                        "deep-coadd collection itself doesn't have them)")
    p.add_argument("--field", default=None,
                   help=f"field name; output -> "
                        f"<FIELDS_ROOT>/<field>/{STEP2_SUBDIR}/<out>")
    p.add_argument("--out", default=None,
                   help="explicit output path (overrides --field)")
    p.add_argument("--npz", default=None,
                   help="binned-data npz path (overrides --field)")
    p.add_argument("--repo", default=DEFAULT_REPO, help="butler repo")
    p.add_argument("--skymap", default=DEFAULT_SKYMAP, help="skymap name")
    p.add_argument("--anacal-dataset", default=DEFAULT_ANACAL_DS,
                   help=f"deep-coadd anacal dataset type "
                        f"(default {DEFAULT_ANACAL_DS!r})")
    p.add_argument("--photoz-dataset", default=DEFAULT_PHOTOZ_DS,
                   help=f"deep-coadd photo-z dataset type "
                        f"(default {DEFAULT_PHOTOZ_DS!r})")
    p.add_argument("--gaia-dataset", default=DEFAULT_GAIA_DS,
                   help=f"per-patch GAIA dataset type "
                        f"(default {DEFAULT_GAIA_DS!r}; lives in "
                        f"--gaia-collection)")
    p.add_argument("--grid-step", type=float, default=0.1,
                   help="grid spacing for sampling tracts (deg, default 0.1)")
    # Source selection (mirrors step3 defaults).
    p.add_argument("--mag-zero", type=float, default=31.4)
    p.add_argument("--flux-name", default="fpfs1")
    p.add_argument("--trace-min", type=float, default=0.05)
    p.add_argument("--emax", type=float, default=0.5)
    p.add_argument("--z-min", type=float, default=0.40)
    p.add_argument("--z-max", type=float, default=2.0)
    p.add_argument("--iband-size-min", type=float, default=None,
                   help="legacy size cut (m20+m00)/m00 > this (i-band)")
    # GAIA / treecorr knobs.
    p.add_argument("--gaia-mag-bins", default="8,13,15,17",
                   help="comma-separated edges defining the GAIA G-mag bins; "
                        "the default '8,13,15,17' yields the three bins "
                        "(8-13, 13-15, 15-17).")
    p.add_argument("--sep-min", type=float, default=1.0,
                   help="min angular separation in arcsec (default 1.0)")
    p.add_argument("--sep-max", type=float, default=200.0,
                   help="max angular separation in arcsec (default 200.0)")
    p.add_argument("--nbins", type=int, default=12,
                   help="number of log-spaced separation bins (default 12)")
    return p.parse_args()


def load_anacal_photoz_inner_join(butler, tract_ids, skymap_name,
                                  anacal_ds, photoz_ds):
    """Per-tract iterate patches; **inner-join** anacal + photoz on
    ``object_id`` (the deep-coadd photo-z table is typically longer than
    the anacal catalog, so equality-of-length is not safe).

    Returns the vstacked joined astropy Table.
    """
    pieces = []
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
        if not a_refs:
            print(f"#   tract={tid}: no {anacal_ds} — skipped",
                  file=sys.stderr)
            continue
        n_kept = 0
        for ar in a_refs:
            patch = ar.dataId["patch"]
            a = Table(butler.get(ar))
            zref = z_refs.get(patch)
            if zref is None:
                print(f"#   tract={tid} patch={patch}: no {photoz_ds} "
                      "— skipped", file=sys.stderr)
                continue
            z = Table(butler.get(zref))
            # Inner join on object_id. searchsorted needs sorted keys.
            z = z[np.argsort(np.asarray(z["object_id"]))]
            z_ids = np.asarray(z["object_id"])
            a_ids = np.asarray(a["object_id"])
            idx = np.searchsorted(z_ids, a_ids)
            valid = (idx < len(z_ids)) & (
                z_ids[np.clip(idx, 0, len(z_ids) - 1)] == a_ids
            )
            a = a[valid]
            zsel = z[idx[valid]]
            for c in z.colnames:
                if c != "object_id":
                    a[c] = zsel[c]
            pieces.append(a)
            n_kept += len(a)
        print(f"#   tract={tid}: kept {len(a_refs)} patches, "
              f"{n_kept:,} rows (after inner join)",
              file=sys.stderr)
    if not pieces:
        raise RuntimeError(
            f"no {anacal_ds} survived join in any tract"
        )
    return vstack(pieces, metadata_conflicts="silent")


def load_gaia_dedup(butler, dataset_type, tract_ids, skymap_name):
    """Per-patch GAIA tables, vstacked then deduped on gaia_source_id."""
    pieces = []
    for tid in tract_ids:
        refs = list(butler.registry.queryDatasets(
            dataset_type, where="skymap=:s AND tract=:t",
            bind={"s": skymap_name, "t": tid},
        ))
        for r in refs:
            t = Table(butler.get(r))
            if len(t) == 0:
                continue
            pieces.append(t)
    if not pieces:
        raise RuntimeError(
            f"no {dataset_type} in tracts {tract_ids} in the gaia "
            "collection — pass --gaia-collection pointing at a run "
            "where BuildCellSystematicsTask wrote the GAIA tables."
        )
    cat = vstack(pieces, metadata_conflicts="silent")
    n_before = len(cat)
    _, keep = np.unique(np.asarray(cat["gaia_source_id"]), return_index=True)
    cat = cat[np.sort(keep)]
    print(f"# GAIA: {len(cat):,} unique stars "
          f"({n_before - len(cat):,} duplicates from patch overlap)",
          file=sys.stderr)
    return cat


def main():
    args = parse_args()
    plt.rcParams.update(DEFAULT_STYLE_RC)

    butler = open_butler(args.collection, repo=args.repo)
    gaia_butler = open_butler(args.gaia_collection, repo=args.repo)
    skymap = butler.get("skyMap", skymap=args.skymap)
    tract_ids = find_tracts(skymap, args.ra, args.dec, args.radius,
                            args.grid_step)
    print(f"# overlapping tracts: {tract_ids}", file=sys.stderr)

    # --- sources from the DEEP-COADD collection ---
    cat = load_anacal_photoz_inner_join(
        butler, tract_ids, args.skymap,
        args.anacal_dataset, args.photoz_dataset,
    )
    add_band_mags(cat, mag_zero=args.mag_zero, flux_name=args.flux_name)
    print(f"# rows before selection: {len(cat):,}", file=sys.stderr)
    msk = select_sources(
        cat,
        flux_name=args.flux_name, mag_zero=args.mag_zero,
        trace_min=args.trace_min, emax=args.emax,
        z_range=(args.z_min, args.z_max),
    )
    if args.iband_size_min is not None:
        m00 = cat["i_fpfs1_m00"]
        s_iband = (cat["i_fpfs1_m20"] + m00) / m00
        msk &= np.asarray(s_iband) > args.iband_size_min
        print(f"# legacy --iband-size-min={args.iband_size_min}: "
              f"{int(msk.sum()):,} rows kept", file=sys.stderr)
    cat = cat[msk]
    print(f"# rows after selection: {len(cat):,}", file=sys.stderr)

    e1, e2, res = calibrate_shapes(cat)
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    res = np.asarray(res, dtype=np.float64)
    r_mean = float(np.mean(res))
    print(f"# calibration: mean response R = {r_mean:.4f}", file=sys.stderr)
    g1 = e1 / r_mean
    g2 = e2 / r_mean

    # --- GAIA stars from the cell-coadd (or whatever --gaia-collection) ---
    stars_all = load_gaia_dedup(
        gaia_butler, args.gaia_dataset, tract_ids, args.skymap,
    )
    edges = [float(s) for s in args.gaia_mag_bins.split(",")]
    bin_specs = list(zip(edges[:-1], edges[1:]))

    src_cat = treecorr.Catalog(
        ra=np.asarray(cat["ra"]), dec=np.asarray(cat["dec"]),
        g1=g1, g2=g2, w=None,
        ra_units="deg", dec_units="deg",
    )
    arcsec_to_arcmin = 1.0 / 60.0

    # list of dicts: lo, hi, n_lens, mean_r, gT, gX, err
    per_bin = []
    mag_all = np.asarray(stars_all["gaia_g_mag"])
    for lo, hi in bin_specs:
        mask = (mag_all > lo) & (mag_all <= hi)
        stars_bin = stars_all[mask]
        label = f"({lo:g}, {hi:g}]"
        if len(stars_bin) == 0:
            print(f"# GAIA mag bin {label}: 0 stars — skipped",
                  file=sys.stderr)
            continue
        print(f"# GAIA mag bin {label}: {len(stars_bin):,} stars",
              file=sys.stderr)
        lens_cat = treecorr.Catalog(
            ra=np.asarray(stars_bin["ra"]),
            dec=np.asarray(stars_bin["dec"]),
            ra_units="deg", dec_units="deg",
        )
        ng = treecorr.NGCorrelation(
            min_sep=args.sep_min * arcsec_to_arcmin,
            max_sep=args.sep_max * arcsec_to_arcmin,
            nbins=args.nbins, sep_units="arcmin", bin_type="Log",
        )
        ng.process(lens_cat, src_cat)
        per_bin.append({
            "lo": lo, "hi": hi, "label": label,
            "n_lens": len(stars_bin),
            "mean_r_arcsec": ng.meanr / arcsec_to_arcmin,
            "gT": np.asarray(ng.xi),
            "gX": np.asarray(ng.xi_im),
            "err": np.sqrt(np.asarray(ng.varxi)),
        })
    if not per_bin:
        raise RuntimeError(
            "no GAIA stars survive any of the magnitude bins; "
            "adjust --gaia-mag-bins"
        )

    # --- save binned data ---
    npz_path = args.npz or resolve_field_out(
        args.field, STEP2_SUBDIR, DEFAULT_NPZ_NAME, None
    )
    if npz_path:
        save = {
            "edges": np.asarray(edges, dtype=np.float64),
            "n_src": np.array(len(cat), dtype=int),
            "sep_min_arcsec": np.array(args.sep_min),
            "sep_max_arcsec": np.array(args.sep_max),
        }
        for i, b in enumerate(per_bin):
            save[f"bin{i}_lo"] = np.array(b["lo"])
            save[f"bin{i}_hi"] = np.array(b["hi"])
            save[f"bin{i}_n_lens"] = np.array(b["n_lens"], dtype=int)
            save[f"bin{i}_mean_r_arcsec"] = b["mean_r_arcsec"]
            save[f"bin{i}_gT"] = b["gT"]
            save[f"bin{i}_gX"] = b["gX"]
            save[f"bin{i}_err"] = b["err"]
        np.savez(npz_path, **save)
        print(f"# wrote binned data -> {npz_path}", file=sys.stderr)

    # --- plot ---
    fig, axes = plt.subplots(
        2, 1, figsize=(7, 6.2), sharex=True, height_ratios=[1.0, 0.7]
    )
    cmap = plt.get_cmap("viridis")
    colors = [cmap(0.15), cmap(0.5), cmap(0.85)]
    markers = ["o", "s", "^"]
    offsets = np.linspace(0.92, 1.08, max(len(per_bin), 2))[:len(per_bin)]

    for b, c, m, off in zip(per_bin, colors, markers, offsets):
        label = (
            rf"$g_{{\rm gaia}} \in {b['label']}$"
            f", $N_\\star={b['n_lens']}$"
        )
        axes[0].errorbar(b["mean_r_arcsec"] * off, b["gT"], yerr=b["err"],
                         fmt=m, color=c, label=label, alpha=0.9)
        axes[1].errorbar(b["mean_r_arcsec"] * off, b["gX"], yerr=b["err"],
                         fmt=m, color=c, alpha=0.9)
    for ax in axes:
        ax.axhline(0, ls="--", color="k", alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlim(args.sep_min * 0.9, args.sep_max * 1.1)
    axes[0].set_ylabel(r"$\langle g_T \rangle$ around GAIA")
    axes[1].set_ylabel(r"$\langle g_\times \rangle$ around GAIA")
    axes[1].set_xlabel(r"angular separation [arcsec]")
    axes[0].legend(fontsize=9, loc="upper right")
    n_stars_total = sum(b["n_lens"] for b in per_bin)
    axes[0].set_title(
        f"PSF-residual diagnostic (deep coadd): "
        f"{n_stars_total:,} stars, {len(cat):,} galaxies"
    )
    plt.tight_layout()

    out = args.out or resolve_field_out(
        args.field, STEP2_SUBDIR, DEFAULT_OUT_NAME, None
    )
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"# wrote {out}", file=sys.stderr)
    else:
        plt.show()


if __name__ == "__main__":
    main()
