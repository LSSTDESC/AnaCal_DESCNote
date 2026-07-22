#!/usr/bin/env python
"""
Scatter-plot the spatial distribution of AnaCal detections around a field
center, overlaid with the boundaries of the tracts that contain them and
(optionally) the GAIA stars from
``deep_coadd_cell_systematics_gaia`` produced by
``BuildCellSystematicsTask``.

The GAIA layer is loaded per-patch (since the same dataset is keyed on
patch); we dedup across patches by ``gaia_source_id`` so the same star
isn't drawn twice in the patch overlap region.

Usage:
    python step2_spacial_distribution.py \\
        --ra 37.86 --dec 6.98 --radius 1.5 \\
        --collection u/xiangchl/dp1-v2/a360_anacal2 \\
        --field a360
"""
from __future__ import annotations

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack
from lsst.geom import Point2D

from _common import (
    DEFAULT_REPO, DEFAULT_SKYMAP,
    find_tracts, open_butler, resolve_field_out,
)


DEFAULT_DATASET = "deep_coadd_cell_anacal_merged"
DEFAULT_GAIA_DS = "deep_coadd_cell_systematics_gaia"
DEFAULT_OUT_NAME = "spatial.png"
STEP2_SUBDIR = "step2"
# Visually-distinguishable per-tract color cycle.
_TRACT_COLORS = [
    "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple",
    "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:gray",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ra", type=float, required=True, help="center RA (deg)")
    p.add_argument("--dec", type=float, required=True, help="center Dec (deg)")
    p.add_argument("--radius", type=float, required=True,
                   help="half-side of search square (deg)")
    p.add_argument("--collection", required=True,
                   help="butler collection holding merged anacal + gaia tables")
    p.add_argument("--repo", default=DEFAULT_REPO, help="butler repo")
    p.add_argument("--skymap", default=DEFAULT_SKYMAP, help="skymap name")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="per-tract merged catalog dataset type")
    p.add_argument("--gaia-dataset", default=DEFAULT_GAIA_DS,
                   help="per-patch GAIA-stars dataset type "
                        "(default: deep_coadd_cell_systematics_gaia). "
                        "Pass --no-gaia to disable the overlay.")
    p.add_argument("--no-gaia", action="store_true",
                   help="skip the GAIA-stars overlay even if the dataset exists")
    p.add_argument("--gaia-mag-cut", type=float, default=17.0,
                   help="only plot GAIA stars brighter than this mag (default: 17)")
    p.add_argument("--field", default=None,
                   help=f"field name; output goes to "
                        f"<FIELDS_ROOT>/<field>/{STEP2_SUBDIR}/{DEFAULT_OUT_NAME} "
                        "unless --out is given")
    p.add_argument("--out", default=None,
                   help="explicit output path (overrides --field).")
    p.add_argument("--grid-step", type=float, default=0.1,
                   help="grid spacing (deg) for sampling tracts (default: 0.1)")
    p.add_argument("--xlim-pad", type=float, default=0.2,
                   help="padding (deg) added to x/y plot limits around the box")
    return p.parse_args()


def load_merged_catalogs(butler, dataset_type, tract_ids, skymap_name):
    """Load `dataset_type` per tract; return (vstacked Table, per-tract counts)."""
    tables = []
    per_tract_n = {}
    for tid in tract_ids:
        refs = list(butler.registry.queryDatasets(
            dataset_type,
            where="skymap=:s AND tract=:t",
            bind={"s": skymap_name, "t": tid},
        ))
        if not refs:
            print(f"#   tract={tid}: no {dataset_type} ref — skipped",
                  file=sys.stderr)
            continue
        t = Table(butler.get(refs[0]))
        per_tract_n[tid] = len(t)
        tables.append(t)
        print(f"#   tract={tid}: {len(t):,} rows", file=sys.stderr)
    if not tables:
        raise RuntimeError(f"no {dataset_type} datasets in any candidate tract")
    return vstack(tables, metadata_conflicts="silent"), per_tract_n


def load_gaia_dedup(butler, dataset_type, tract_ids, skymap_name):
    """Per-patch GAIA tables, vstacked then deduped on gaia_source_id.

    Returns ``None`` if the dataset type is absent for every patch — so
    a missing GAIA overlay just disables that layer and the rest of
    the plot still renders.
    """
    pieces = []
    for tid in tract_ids:
        refs = list(butler.registry.queryDatasets(
            dataset_type,
            where="skymap=:s AND tract=:t",
            bind={"s": skymap_name, "t": tid},
        ))
        for r in refs:
            t = Table(butler.get(r))
            if len(t) == 0:
                continue
            pieces.append(t)
    if not pieces:
        return None
    cat = vstack(pieces, metadata_conflicts="silent")
    if "gaia_source_id" in cat.colnames:
        n_before = len(cat)
        _, keep = np.unique(np.asarray(cat["gaia_source_id"]),
                            return_index=True)
        cat = cat[np.sort(keep)]
        print(f"# GAIA: {len(cat):,} unique stars "
              f"({n_before - len(cat):,} duplicates from patch overlap)",
              file=sys.stderr)
    else:
        print(f"# GAIA: {len(cat):,} stars (no gaia_source_id col; not deduped)",
              file=sys.stderr)
    return cat


def tract_corners_radec(tract_info):
    """5-point ra/dec outline of one tract."""
    wcs = tract_info.getWcs()
    bb = tract_info.getBBox()
    corners = [
        (bb.getMinX(), bb.getMinY()),
        (bb.getMaxX(), bb.getMinY()),
        (bb.getMaxX(), bb.getMaxY()),
        (bb.getMinX(), bb.getMaxY()),
        (bb.getMinX(), bb.getMinY()),
    ]
    ra_c = [wcs.pixelToSky(Point2D(x, y)).getRa().asDegrees() for x, y in corners]
    dec_c = [wcs.pixelToSky(Point2D(x, y)).getDec().asDegrees() for x, y in corners]
    return ra_c, dec_c


def plot_field(*, anacal_table, gaia_table, gaia_mag_cut,
               ra, dec, radius, skymap, tract_ids, per_tract_n,
               out_path=None, xlim_pad=0.2):
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(anacal_table["ra"], anacal_table["dec"],
               s=0.1, alpha=0.3, color="0.4", label="anacal detection")

    if gaia_table is not None and len(gaia_table) > 0:
        mag = np.asarray(gaia_table["gaia_g_mag"])
        bright = mag <= gaia_mag_cut
        if bright.any():
            ax.scatter(np.asarray(gaia_table["ra"])[bright],
                       np.asarray(gaia_table["dec"])[bright],
                       s=6, marker="o", color="yellow", edgecolor="none",
                       alpha=0.85, zorder=8,
                       label=f"GAIA (g≤{gaia_mag_cut:g}, N={int(bright.sum()):,})")

    ax.scatter(ra, dec, s=90, c="red", marker="x",
               zorder=10, label="field center")

    for i, tid in enumerate(tract_ids):
        color = _TRACT_COLORS[i % len(_TRACT_COLORS)]
        try:
            ti = skymap[tid]
            ra_c, dec_c = tract_corners_radec(ti)
        except Exception as e:
            print(f"# tract={tid}: skymap lookup/outline failed ({e})",
                  file=sys.stderr)
            continue
        ax.plot(ra_c, dec_c, "-", color=color, lw=1.5)
        n = per_tract_n.get(tid)
        ra_cen = float(np.mean(ra_c[:-1]))
        dec_cen = float(np.mean(dec_c[:-1]))
        label = f"{tid}\n{n:,} src" if n is not None else str(tid)
        ax.text(ra_cen, dec_cen, label,
                color=color, fontsize=11, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=color, alpha=0.8))

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_xlim(ra - radius - xlim_pad, ra + radius + xlim_pad)
    ax.set_ylim(dec - radius - xlim_pad, dec + radius + xlim_pad)
    ax.invert_xaxis()
    ax.legend(loc="upper right", fontsize=11)
    ax.set_title(
        f"center=({ra:.3f}, {dec:.3f}), radius={radius} deg, "
        f"N={len(anacal_table):,}"
    )
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"# wrote {out_path}", file=sys.stderr)
    else:
        plt.show()


def main():
    args = parse_args()
    out_path = resolve_field_out(args.field, STEP2_SUBDIR, DEFAULT_OUT_NAME, args.out)

    butler = open_butler(args.collection, repo=args.repo)
    skymap = butler.get("skyMap", skymap=args.skymap)

    tract_ids = find_tracts(skymap, args.ra, args.dec, args.radius, args.grid_step)
    print(f"# loading {args.dataset} from {args.collection}:", file=sys.stderr)
    anacal_table, per_tract_n = load_merged_catalogs(
        butler, args.dataset, tract_ids, args.skymap,
    )
    print(f"# total rows across {len(per_tract_n)} tracts: {len(anacal_table):,}",
          file=sys.stderr)

    gaia_table = None
    if not args.no_gaia:
        print(f"# loading {args.gaia_dataset} (per-patch):", file=sys.stderr)
        gaia_table = load_gaia_dedup(
            butler, args.gaia_dataset, tract_ids, args.skymap,
        )
        if gaia_table is None:
            print("# WARN: no GAIA dataset found — overlay disabled",
                  file=sys.stderr)

    plot_field(
        anacal_table=anacal_table, gaia_table=gaia_table,
        gaia_mag_cut=args.gaia_mag_cut,
        ra=args.ra, dec=args.dec, radius=args.radius,
        skymap=skymap, tract_ids=list(per_tract_n.keys()),
        per_tract_n=per_tract_n,
        out_path=out_path, xlim_pad=args.xlim_pad,
    )


if __name__ == "__main__":
    main()
