#!/usr/bin/env python
"""
1-D magnitude histograms per band for AnaCal detections in a field.

Loads ``deep_coadd_cell_anacal_merged`` (one table per tract, the
``MergePipe`` output) from a butler collection, finds every tract that
the search box around (ra, dec) overlaps, and plots a per-band magnitude
histogram derived from ``{band}_flux_{flux-name}`` and ``--mag-zero``.

Layout adapts to the band count: 4 bands -> 2x2 grid, 6 bands -> 2x3.

Usage:
    python step2_1Dhist.py \\
        --ra 37.86 --dec 6.98 --radius 1.5 \\
        --collection u/xiangchl/dp1-v2/a360_anacal2 \\
        --bands g,r,i,z \\
        --out a360_1Dhist.png
"""

import argparse
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack
from matplotlib.ticker import MultipleLocator

from _common import (
    DEFAULT_REPO, DEFAULT_SKYMAP,
    find_tracts, open_butler, resolve_field_out,
)


DEFAULT_DATASET = "deep_coadd_cell_anacal_merged"
DEFAULT_OUT_NAME = "1Dhist.png"
STEP2_SUBDIR = "step2"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ra", type=float, required=True, help="center RA in degrees")
    p.add_argument("--dec", type=float, required=True, help="center Dec in degrees")
    p.add_argument("--radius", type=float, required=True,
                   help="half-side of search square in degrees")
    p.add_argument("--collection", required=True,
                   help="butler collection holding merged anacal tables")
    p.add_argument("--bands", default="g,r,i,z",
                   help="comma-separated bands to histogram (default: g,r,i,z; "
                        "6-band example: u,g,r,i,z,y)")
    p.add_argument("--flux-name", default="fpfs1",
                   help="flux-column suffix; mag = mag_zero - 2.5*log10({band}_flux_{flux-name}) "
                        "(default: fpfs1)")
    p.add_argument("--mag-zero", type=float, default=31.4,
                   help="magnitude zero point (default: 31.4)")
    p.add_argument("--repo", default=DEFAULT_REPO, help="butler repo")
    p.add_argument("--skymap", default=DEFAULT_SKYMAP, help="skymap name")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="per-tract merged catalog dataset type")
    p.add_argument("--bins", type=int, default=60, help="histogram bins (default: 60)")
    p.add_argument("--xlim", default="20,28",
                   help="x-axis (mag) limits, comma-separated (default: 20,28)")
    p.add_argument("--field", default=None,
                   help=f"field name; output goes to {{FIELDS_ROOT}}/<field>/step2/{DEFAULT_OUT_NAME} "
                        "unless --out is given (e.g., a360, edfs, ecdfs)")
    p.add_argument("--out", default=None,
                   help="explicit output path (overrides --field-derived default). "
                        "Parent dirs auto-created. If neither --out nor --field is set, "
                        "the plot is shown interactively.")
    p.add_argument("--grid-step", type=float, default=0.1,
                   help="grid spacing (deg) for sampling tracts (default: 0.1)")
    return p.parse_args()


def load_merged_catalogs(butler, dataset_type, tract_ids, skymap_name):
    """Load `dataset_type` for each tract; return the vstacked Table."""
    tables = []
    for tid in tract_ids:
        refs = list(butler.registry.queryDatasets(
            dataset_type,
            where="skymap=:s AND tract=:t",
            bind={"s": skymap_name, "t": tid},
        ))
        if not refs:
            print(f"#   tract={tid}: no {dataset_type} ref in collection — skipping",
                  file=sys.stderr)
            continue
        t = Table(butler.get(refs[0]))
        tables.append(t)
        print(f"#   tract={tid}: {len(t):,} rows", file=sys.stderr)
    if not tables:
        raise RuntimeError(f"no {dataset_type} datasets in any candidate tract")
    return vstack(tables)


def grid_shape(n_bands):
    """Subplot grid: 4 -> (2,2); 6 -> (2,3); fallback ceil to <=3 cols."""
    if n_bands == 4:
        return 2, 2
    if n_bands == 6:
        return 2, 3
    ncols = min(n_bands, 3)
    nrows = math.ceil(n_bands / ncols)
    return nrows, ncols


def collect_band_fluxes(table, bands, flux_name):
    """Return {band: flux_array} and the list of missing column names."""
    fluxes, missing = {}, []
    for band in bands:
        col = f"{band}_flux_{flux_name}"
        if col not in table.colnames:
            missing.append(col)
            continue
        fluxes[band] = np.asarray(table[col], dtype=np.float64)
    return fluxes, missing


_BAND_COLORS = {
    "u": "#5e4fa2", "g": "#2ca02c", "r": "#d62728",
    "i": "#ff7f0e", "z": "#8c564b", "y": "#7f7f7f",
}


def plot_hists(*, anacal_table, bands, flux_name, mag_zero, bins, xlim, out_path):
    nrows, ncols = grid_shape(len(bands))

    # Restyle: larger fonts, slimmer figure per panel since axes are shared.
    with plt.rc_context({
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 6,
        "xtick.minor.size": 3.5,
        "ytick.major.size": 6,
        "ytick.minor.size": 3.5,
    }):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.6 * ncols, 3.0 * nrows),
            sharex=True, sharey=True,
        )
        axes = np.atleast_1d(axes).ravel()

        fluxes, missing = collect_band_fluxes(anacal_table, bands, flux_name)
        if missing:
            print(f"# WARN: missing flux columns {missing} — those bands get a placeholder",
                  file=sys.stderr)

        # Consistent cross-band selection: keep a row only if EVERY available
        # band has a finite, positive flux.
        n = len(anacal_table)
        good = np.ones(n, dtype=bool)
        for f in fluxes.values():
            good &= np.isfinite(f) & (f > 0)
        print(f"# cross-band selection: {good.sum():,} / {n:,} rows kept "
              f"(all of {sorted(fluxes)} positive & finite)", file=sys.stderr)

        for ax, band in zip(axes, bands):
            if band not in fluxes:
                ax.text(0.5, 0.5, f"missing\n{band}_flux_{flux_name}",
                        ha="center", va="center",
                        transform=ax.transAxes,
                        color="0.4")
                ax.text(0.96, 0.88, f"${band}$",
                        transform=ax.transAxes, fontsize=18, fontweight="bold",
                        color="0.4", va="center", ha="right")
                continue
            mag = mag_zero - 2.5 * np.log10(fluxes[band][good])
            color = _BAND_COLORS.get(band, "C0")
            ax.hist(
                mag, bins=bins, range=xlim,
                histtype="stepfilled",
                facecolor=color, edgecolor=color,
                alpha=0.35, linewidth=1.4,
            )
            ax.set_xlim(*xlim)
            # Band label inside each panel, upper-right corner.
            ax.text(0.96, 0.88, f"${band}$",
                    transform=ax.transAxes, fontsize=18, fontweight="bold",
                    color=color, va="center", ha="right",
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", ec=color, lw=1.0, alpha=0.85))
            ax.xaxis.set_major_locator(MultipleLocator(1.0))
            ax.xaxis.set_minor_locator(MultipleLocator(0.5))
            ax.grid(True, which="major", alpha=0.35, linestyle="-", linewidth=0.7)
            ax.grid(True, which="minor", alpha=0.18, linestyle=":", linewidth=0.6)

        for ax in axes[len(bands):]:
            ax.set_visible(False)

        for ax in axes[:len(bands)]:
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("count")
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel("magnitude")

        # Tight but not zero — leave a hair-line gap so panel borders read.
        plt.subplots_adjust(wspace=0.04, hspace=0.04,
                            left=0.08, right=0.98, top=0.96, bottom=0.10)
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"# wrote {out_path}", file=sys.stderr)
        else:
            plt.show()


def main():
    args = parse_args()
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    xlim = tuple(float(s) for s in args.xlim.split(","))
    if len(xlim) != 2:
        raise SystemExit("--xlim must be two comma-separated floats, e.g. 20,28")

    out_path = resolve_field_out(args.field, STEP2_SUBDIR, DEFAULT_OUT_NAME, args.out)

    butler = open_butler(args.collection, repo=args.repo)
    skymap = butler.get("skyMap", skymap=args.skymap)
    tract_ids = find_tracts(skymap, args.ra, args.dec, args.radius, args.grid_step)
    print(f"# loading {args.dataset} from {args.collection}:", file=sys.stderr)
    anacal_table = load_merged_catalogs(butler, args.dataset, tract_ids, args.skymap)
    print(f"# total rows: {len(anacal_table):,}", file=sys.stderr)

    plot_hists(
        anacal_table=anacal_table,
        bands=bands,
        flux_name=args.flux_name,
        mag_zero=args.mag_zero,
        bins=args.bins,
        xlim=xlim,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
