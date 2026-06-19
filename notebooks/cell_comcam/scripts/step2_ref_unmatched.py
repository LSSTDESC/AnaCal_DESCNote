#!/usr/bin/env python
"""
Scatter a reference anacal catalog vs our pipeline catalog around a
field center, in three layers:

  * all pipeline galaxies (background; full catalog by default,
    optionally cut via --our-imag-max)
  * reference bright (i < --ref-imag-max) matched to a pipeline source
  * reference bright (i < --ref-imag-max) unmatched (smaller markers,
    on top)

By default the i-mag cut is applied to the **reference** side only;
pass --our-imag-max to also cut the pipeline catalog.

Output: ``<field>/step2/ref_unmatched.png``.

Usage:
    python step2_ref_unmatched.py \\
        --ra 37.86 --dec 6.98 --radius 1.5 \\
        --collection u/xiangchl/dp1-v2/a360_anacal2 \\
        --field a360 \\
        --ref-catalog /pscratch/.../a360/anacal_catalog_a360.fits
"""
from __future__ import annotations

import argparse
import sys

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table, vstack
from matplotlib.lines import Line2D

from _common import (
    DEFAULT_REPO, DEFAULT_SKYMAP,
    find_tracts, open_butler, resolve_field_out,
)


DEFAULT_DATASET = "deep_coadd_cell_anacal_catalog"
DEFAULT_OUT_NAME = "ref_unmatched.png"
STEP2_SUBDIR = "step2"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ra", type=float, required=True)
    p.add_argument("--dec", type=float, required=True)
    p.add_argument("--radius", type=float, required=True,
                   help="half-side of search square (deg)")
    p.add_argument("--collection", required=True,
                   help="butler collection with the merged anacal table")
    p.add_argument("--ref-catalog", required=True,
                   help="FITS path to a reference anacal_catalog_<field>.fits "
                        "with ra/dec + i_flux_gauss2")
    p.add_argument("--repo", default=DEFAULT_REPO, help="butler repo")
    p.add_argument("--skymap", default=DEFAULT_SKYMAP, help="skymap name")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="per-patch catalog dataset type "
                        "(has gauss2 + is_primary; deduped across patches)")
    p.add_argument("--our-imag-max", type=float, default=float("inf"),
                   help="optional i-mag cut applied to the pipeline "
                        "catalog (via i_flux_gauss2; default: no cut)")
    p.add_argument("--ref-imag-max", type=float, default=24.0,
                   help="i-mag cut on the reference catalog "
                        "(default: 24.0)")
    p.add_argument("--mag-zero", type=float, default=31.4,
                   help="AB zero-point for i_flux_gauss2 -> i_mag (default: 31.4)")
    p.add_argument("--ref-match-arcsec", type=float, default=0.6,
                   help="sky-match radius for ref vs pipeline (default: 0.6\")")
    p.add_argument("--field", default=None,
                   help=f"field name; output goes to "
                        f"<FIELDS_ROOT>/<field>/{STEP2_SUBDIR}/{DEFAULT_OUT_NAME}")
    p.add_argument("--out", default=None,
                   help="explicit output path (overrides --field).")
    p.add_argument("--grid-step", type=float, default=0.1,
                   help="grid spacing (deg) for sampling tracts (default: 0.1)")
    return p.parse_args()


def load_per_patch_primary(butler, dataset_type, tract_ids, skymap_name):
    """Vstack per-patch tables, keeping only ``is_primary`` rows (dedupes
    overlap-region duplicates)."""
    tables = []
    for tid in tract_ids:
        refs = list(butler.registry.queryDatasets(
            dataset_type,
            where="skymap=:s AND tract=:t",
            bind={"s": skymap_name, "t": tid},
        ))
        if not refs:
            print(f"#   tract={tid}: no {dataset_type} refs — skipped",
                  file=sys.stderr)
            continue
        kept = 0
        for r in refs:
            t = Table(butler.get(r))
            if "is_primary" in t.colnames:
                t = t[np.asarray(t["is_primary"], dtype=bool)]
            tables.append(t)
            kept += len(t)
        print(f"#   tract={tid}: {len(refs)} patches, {kept:,} primary rows",
              file=sys.stderr)
    if not tables:
        raise RuntimeError(f"no {dataset_type} in any candidate tract")
    return vstack(tables, metadata_conflicts="silent")


def imag_from_iflux(iflux, mag_zero):
    return mag_zero - 2.5*np.log10(np.where(iflux > 0, iflux, np.nan))


def split_matched(*, ref_path, our_ra, our_dec,
                  ref_imag_max, mag_zero, match_arcsec):
    """Sky-match every ref source to ours; return (matched, unmatched).

    ``ref_imag_max=inf`` means no cut on the reference side.
    """
    with fits.open(ref_path) as h:
        d = h[1].data
        ra = np.asarray(d["ra"], dtype=np.float64)
        dec = np.asarray(d["dec"], dtype=np.float64)
        iflux = np.asarray(d["i_flux_gauss2"], dtype=np.float64)
    keep = np.ones(len(ra), dtype=bool)
    if np.isfinite(ref_imag_max):
        imag = imag_from_iflux(iflux, mag_zero)
        keep = np.isfinite(imag) & (imag < ref_imag_max)
        print(f"# ref-catalog: {len(ra):,} rows -> i<{ref_imag_max} kept: "
              f"{int(keep.sum()):,}", file=sys.stderr)
    else:
        print(f"# ref-catalog: {len(ra):,} rows (no i-mag cut)", file=sys.stderr)
    c_ref = SkyCoord(ra=ra[keep]*u.deg, dec=dec[keep]*u.deg)
    c_ours = SkyCoord(ra=our_ra*u.deg, dec=our_dec*u.deg)
    _, d2d, _ = c_ref.match_to_catalog_sky(c_ours)
    matched = d2d < match_arcsec*u.arcsec
    n_m, n_u = int(matched.sum()), int((~matched).sum())
    print(f"# matched within {match_arcsec}\" : {n_m:,}", file=sys.stderr)
    print(f"# unmatched                   : {n_u:,}  "
          f"({n_u/max(1,n_m+n_u)*100:.1f}%)", file=sys.stderr)
    return ((ra[keep][matched], dec[keep][matched]),
            (ra[keep][~matched], dec[keep][~matched]))


def main():
    args = parse_args()
    out_path = resolve_field_out(args.field, STEP2_SUBDIR, DEFAULT_OUT_NAME, args.out)

    butler = open_butler(args.collection, repo=args.repo)
    skymap = butler.get("skyMap", skymap=args.skymap)
    tract_ids = find_tracts(skymap, args.ra, args.dec, args.radius, args.grid_step)
    anacal = load_per_patch_primary(butler, args.dataset, tract_ids, args.skymap)

    our_ra_all = np.asarray(anacal["ra"], dtype=np.float64)
    our_dec_all = np.asarray(anacal["dec"], dtype=np.float64)
    our_iflux = np.asarray(anacal["i_flux_gauss2"], dtype=np.float64)
    our_imag = imag_from_iflux(our_iflux, args.mag_zero)
    if np.isfinite(args.our_imag_max):
        keep_ours = np.isfinite(our_imag) & (our_imag < args.our_imag_max)
        print(f"# pipeline: {len(our_ra_all):,} primary sources -> "
              f"i<{args.our_imag_max} kept: {int(keep_ours.sum()):,}",
              file=sys.stderr)
    else:
        keep_ours = np.ones(len(our_ra_all), dtype=bool)
        print(f"# pipeline: {len(our_ra_all):,} primary sources (no i-mag cut)",
              file=sys.stderr)
    our_ra, our_dec = our_ra_all[keep_ours], our_dec_all[keep_ours]

    (ra_m, dec_m), (ra_u, dec_u) = split_matched(
        ref_path=args.ref_catalog, our_ra=our_ra, our_dec=our_dec,
        ref_imag_max=args.ref_imag_max, mag_zero=args.mag_zero,
        match_arcsec=args.ref_match_arcsec,
    )

    our_label = (f"pipeline i<{args.our_imag_max:g} (N={len(our_ra):,})"
                 if np.isfinite(args.our_imag_max)
                 else f"all pipeline (N={len(our_ra):,})")
    ref_tag = (f"i<{args.ref_imag_max:g}"
               if np.isfinite(args.ref_imag_max) else "all")
    matched_label = (f"ref {ref_tag} matched <{args.ref_match_arcsec:g}\" "
                     f"(N={len(ra_m):,})")
    unmatched_label = f"ref {ref_tag} unmatched (N={len(ra_u):,})"

    with plt.rc_context({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
        "axes.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    }):
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(our_ra, our_dec, s=0.5, alpha=0.25, color="0.55",
                   rasterized=True)
        ax.scatter(ra_m, dec_m, s=4, alpha=0.55, color="C0",
                   rasterized=True)
        ax.scatter(ra_u, dec_u, s=0.8, alpha=0.85, color="C3", marker="x",
                   linewidths=0.5, rasterized=True)
        ax.scatter(args.ra, args.dec, s=40, c="red", marker="+",
                   linewidths=1.5, zorder=10)

        # Custom legend handles — sizes independent of scatter sizes so
        # the legend reads cleanly regardless of how small the actual
        # plot markers are.
        handles = [
            Line2D([], [], marker="o", color="0.55", linestyle="",
                   markersize=5, label=our_label),
            Line2D([], [], marker="o", color="C0", linestyle="",
                   markersize=5, alpha=0.8, label=matched_label),
            Line2D([], [], marker="x", color="C3", linestyle="",
                   markersize=5, mew=1.2, label=unmatched_label),
            Line2D([], [], marker="+", color="red", linestyle="",
                   markersize=8, mew=1.5, label="field center"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=9,
                  framealpha=0.92, edgecolor="0.6")

        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.set_xlim(args.ra - args.radius - 0.2, args.ra + args.radius + 0.2)
        ax.set_ylim(args.dec - args.radius - 0.2, args.dec + args.radius + 0.2)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(args.dec)))
        ax.invert_xaxis()
        ax.set_title(
            f"center=({args.ra:.3f}, {args.dec:.3f}), radius={args.radius} deg"
        )
        plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"# wrote {out_path}", file=sys.stderr)
    else:
        plt.show()


if __name__ == "__main__":
    main()
