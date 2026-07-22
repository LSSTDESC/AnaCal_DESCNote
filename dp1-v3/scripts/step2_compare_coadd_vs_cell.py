#!/usr/bin/env python
"""Compare per-band & 2-D distributions: deep_coadd vs deep_coadd_cell.

Plots, all under ``<field>/step2/``:
  * ``compare_mag.png``         — 4-panel per-band magnitude histogram (g/r/i/z)
  * ``compare_ferr.png``        — 4-panel per-band fpfs1 flux-error histogram
  * ``compare_ri_vs_i.png``     — 2-D corner: (r-i) vs i mag
  * ``compare_iz_vs_i.png``     — 2-D corner: (i-z) vs i mag
  * ``compare_size_vs_i.png``   — 2-D corner: i-band size [arcsec] vs i mag

Loads per-patch anacal catalogs from two butler collections:
  * --coll-cell  (required) — ``deep_coadd_cell_anacal_catalog`` from MeasureCellCoaddsPipe
  * --coll-coadd (optional) — ``deep_coadd_anacal_catalog``      from MeasureCoaddsPipe (deep_coadd)

If --coll-coadd is omitted or has no refs, the plots are drawn cell-only.

Usage:
    python step2_compare_coadd_vs_cell.py \\
        --ra 37.86 --dec 6.98 --radius 1.5 \\
        --coll-cell  u/xiangchl/dp1-v2/a360_anacal2 \\
        --coll-coadd u/xiangchl/dp1/a360_anacal_coadd \\
        --field a360
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

from _common import (
    DEFAULT_REPO as REPO, DEFAULT_SKYMAP as SKYMAP, FIELDS_ROOT,
    find_tracts, open_butler,
)


DSTYPE_CELL = "deep_coadd_cell_anacal_catalog"
DSTYPE_COADD = "deep_coadd_anacal_catalog"
STEP2_SUBDIR = "step2"

MAG_ZERO = 31.4
FLUX_KIND = "fpfs1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ra", type=float, required=True, help="field center RA (deg)")
    p.add_argument("--dec", type=float, required=True, help="field center Dec (deg)")
    p.add_argument("--radius", type=float, required=True,
                   help="half-side of search square (deg)")
    p.add_argument("--coll-cell", required=True,
                   help="butler collection with cell-coadd anacal catalogs")
    p.add_argument("--coll-coadd", default=None,
                   help="butler collection with deep_coadd anacal catalogs "
                        "(optional — if missing the plots show cell only)")
    p.add_argument("--field", default=None,
                   help=f"field name; output dir is "
                        f"{{FIELDS_ROOT}}/<field>/{STEP2_SUBDIR}/compare_*.png")
    p.add_argument("--out-dir", default=None,
                   help="explicit output directory (overrides --field).")
    p.add_argument("--repo", default=REPO, help="butler repo")
    p.add_argument("--skymap", default=SKYMAP, help="skymap name")
    p.add_argument("--grid-step", type=float, default=0.1,
                   help="grid spacing (deg) for sampling tracts (default: 0.1)")
    return p.parse_args()


def load_anacal(butler, dataset_type, tract_ids, skymap_name):
    """Vstack per-patch anacal catalogs across the candidate tracts."""
    pieces = []
    for tid in tract_ids:
        refs = list(butler.registry.queryDatasets(
            dataset_type,
            where="skymap=:s AND tract=:t",
            bind={"s": skymap_name, "t": tid},
        ))
        if not refs:
            continue
        for r in refs:
            pieces.append(Table(butler.get(r)))
    if not pieces:
        return None
    cat = vstack(pieces, metadata_conflicts="silent")
    cat = cat[cat["is_primary"]] if "is_primary" in cat.colnames else cat
    return cat


def _strip_lsst_prefix(cat):
    """v3 cell catalogs prefix every band column with ``lsst_`` (e.g.
    ``lsst_i_flux_fpfs1``). This script compares against the v2 deep
    coadd catalog which uses bare-band names (``i_flux_fpfs1``); strip
    the prefix on the cell side so the shared plot helpers can look
    columns up by the same short name for both series.
    """
    for c in list(cat.colnames):
        if c.startswith("lsst_"):
            cat.rename_column(c, c[len("lsst_"):])
    return cat


def flux_col(band):
    return f"{band}_flux_{FLUX_KIND}"


def selection(t):
    """Match cell_comcam/a360_old/compare_a360_coadd_vs_cell.ipynb cell a003."""
    cond = (
        ((MAG_ZERO - 2.5 * np.log10(t[flux_col("i")])) < 24.5)
        & ((MAG_ZERO - 2.5 * np.log10(t[flux_col("g")])) < 26.0)
        & ((MAG_ZERO - 2.5 * np.log10(t[flux_col("r")])) < 25.0)
        & ((MAG_ZERO - 2.5 * np.log10(t[flux_col("z")])) < 25.0)
        & (((t["i_fpfs1_m00"] + t["i_fpfs1_m20"]) / t["i_fpfs1_m00"]) > 0.08)
    )
    return np.asarray(cond, dtype=bool)


def mag(arr):
    arr = np.asarray(arr, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return MAG_ZERO - 2.5 * np.log10(arr)


# ----- plots ----------------------------------------------------------------

def _common_rc():
    return {
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
    }


def plot_4panel_hist(*, series, value_fn, bins, rng,
                     title, xlabel, xticks_major, xticks_minor,
                     out_path, log_x=False):
    """series is a list of (name, table, mask, linestyle)."""
    with plt.rc_context(_common_rc()):
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 7), sharex=True, sharey=True,
        )
        axes = axes.ravel()
        for i, b in enumerate("griz"):
            ax = axes[i]
            for name, t, msk, ls in series:
                v = value_fn(t, b)[msk]
                v = v[np.isfinite(v)]
                ax.hist(v, bins=bins, range=rng, density=True,
                        histtype="step", lw=1.8, ls=ls, label=name)
            ax.text(0.96, 0.88, f"${b}$", transform=ax.transAxes,
                    fontsize=18, fontweight="bold", ha="right", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="0.5", lw=1.0, alpha=0.85))
            if log_x:
                ax.set_xscale("log")
            ax.set_xlim(*rng)
            ax.xaxis.set_major_locator(MultipleLocator(xticks_major))
            ax.xaxis.set_minor_locator(MultipleLocator(xticks_minor))
            ax.grid(True, which="major", alpha=0.30, ls="-", lw=0.7)
            ax.grid(True, which="minor", alpha=0.15, ls=":", lw=0.5)
            if ax.get_subplotspec().is_first_col():
                ax.set_ylabel("density")
            if ax.get_subplotspec().is_last_row():
                ax.set_xlabel(xlabel)
        axes[0].legend(loc="upper left", fontsize=11, frameon=False)
        fig.suptitle(title)
        plt.subplots_adjust(wspace=0.04, hspace=0.04,
                            left=0.08, right=0.98, top=0.93, bottom=0.10)
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"# wrote {out_path}", file=sys.stderr)
        else:
            plt.show()


class _CornerPlot:
    """2-D contour + top/right 1-D PDFs (mirrors the notebook helper)."""
    def __init__(self, *, xrange, yrange, bins, xlabel, ylabel, xlim=None, ylim=None):
        self.fig = plt.figure(figsize=(8, 8))
        gs = self.fig.add_gridspec(4, 4, wspace=0.05, hspace=0.05)
        self.ax_2d = self.fig.add_subplot(gs[1:4, 0:3])
        self.ax_2d.set_xlabel(xlabel)
        self.ax_2d.set_ylabel(ylabel)
        self.ax_top = self.fig.add_subplot(gs[0, 0:3], sharex=self.ax_2d)
        self.ax_top.set_ylabel("PDF")
        self.ax_top.tick_params(axis="x", labelbottom=False)
        self.ax_right = self.fig.add_subplot(gs[1:4, 3], sharey=self.ax_2d)
        self.ax_right.set_xlabel("PDF")
        self.ax_right.tick_params(axis="y", labelleft=False)
        self.range_2d = [xrange, yrange]
        self.range_x = xrange
        self.range_y = yrange
        self.bins = bins
        self.ax_2d.set_xlim(xlim or xrange)
        self.ax_2d.set_ylim(ylim or yrange)
        self.ax_top.set_xlim(xlim or xrange)
        self.ax_right.set_ylim(ylim or yrange)
        self._handles = []

    def add(self, x, y, *, ls, label, levels=(0.15, 0.3, 0.65)):
        hist, xe, ye = np.histogram2d(x, y, bins=self.bins,
                                      range=self.range_2d, density=True)
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        X, Y = np.meshgrid(xc, yc)
        self.ax_2d.contour(X, Y, hist.T, levels=list(levels),
                           colors="black", linestyles=ls)
        self.ax_top.hist(x, bins=self.bins, range=self.range_x,
                         histtype="step", color="black", ls=ls, density=True)
        self.ax_right.hist(y, bins=self.bins, range=self.range_y,
                           histtype="step", color="black", ls=ls,
                           density=True, orientation="horizontal")
        self._handles.append(Line2D([0], [0], color="black", ls=ls, label=label))

    def finalize(self, *, title, out_path):
        if self._handles:
            self.ax_2d.legend(handles=self._handles, loc="upper right",
                              fontsize=13, frameon=False)
        self.fig.suptitle(title, y=0.995)
        plt.tight_layout(rect=(0, 0, 1, 0.99))
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"# wrote {out_path}", file=sys.stderr)
        else:
            plt.show()


def plot_2D_color(series, *, x_band, color_bands, xrange, yrange,
                  bins, xlabel, ylabel, title, out_path):
    """series = [(name, table, mask, ls)]; y = mag[color[0]] - mag[color[1]]."""
    cp = _CornerPlot(xrange=xrange, yrange=yrange, bins=bins,
                     xlabel=xlabel, ylabel=ylabel)
    for name, t, msk, ls in series:
        x = mag(t[flux_col(x_band)])[msk]
        y = mag(t[flux_col(color_bands[0])])[msk] - mag(t[flux_col(color_bands[1])])[msk]
        m = np.isfinite(x) & np.isfinite(y)
        cp.add(x[m], y[m], ls=ls, label=name)
    cp.finalize(title=title, out_path=out_path)


def plot_2D_size(series, *, xrange, yrange, bins,
                 xlabel, ylabel, title, out_path):
    cp = _CornerPlot(xrange=xrange, yrange=yrange, bins=bins,
                     xlabel=xlabel, ylabel=ylabel)
    for name, t, msk, ls in series:
        x = mag(t[flux_col("i")])[msk]
        s = np.asarray(
            (t["i_fpfs1_m00"] + t["i_fpfs1_m20"]) / t["i_fpfs1_m00"]
        )
        size_arcsec = np.sqrt(np.maximum(s, 0.0))[msk]
        m = np.isfinite(x) & np.isfinite(size_arcsec) & (size_arcsec > 0)
        cp.add(x[m], size_arcsec[m], ls=ls, label=name)
    cp.finalize(title=title, out_path=out_path)


def main():
    args = parse_args()

    # Output dir
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif args.field is not None:
        out_dir = FIELDS_ROOT / args.field / STEP2_SUBDIR
    else:
        out_dir = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    def out(name):
        return str(out_dir / name) if out_dir is not None else None

    # Load both collections.
    butler_cell = open_butler(args.coll_cell, repo=args.repo)
    skymap = butler_cell.get("skyMap", skymap=args.skymap)
    tract_ids = find_tracts(skymap, args.ra, args.dec, args.radius, args.grid_step)
    print(f"# overlapping tracts: {tract_ids}", file=sys.stderr)

    print(f"# loading cell  collection {args.coll_cell!r}", file=sys.stderr)
    cell = load_anacal(butler_cell, DSTYPE_CELL, tract_ids, args.skymap)
    if cell is None:
        raise SystemExit(f"no {DSTYPE_CELL} datasets in {args.coll_cell}")
    _strip_lsst_prefix(cell)
    sel_cell = selection(cell)
    print(f"#   cell: {len(cell):,} rows, {sel_cell.sum():,} after selection",
          file=sys.stderr)

    coadd = None
    sel_coadd = None
    if args.coll_coadd:
        butler_coadd = open_butler(args.coll_coadd, repo=args.repo)
        print(f"# loading coadd collection {args.coll_coadd!r}", file=sys.stderr)
        coadd = load_anacal(butler_coadd, DSTYPE_COADD, tract_ids, args.skymap)
        if coadd is None:
            print(f"# WARN: no {DSTYPE_COADD} in {args.coll_coadd} — "
                  "falling back to cell-only", file=sys.stderr)
        else:
            sel_coadd = selection(coadd)
            print(f"#   coadd: {len(coadd):,} rows, {sel_coadd.sum():,} after selection",
                  file=sys.stderr)

    series = []
    if coadd is not None:
        series.append(("deep coadd", coadd, sel_coadd, "-"))
    series.append(("cell coadd", cell, sel_cell, "--"))

    # ---- plot 1: per-band 1-D magnitude histogram ----
    plot_4panel_hist(
        series=series,
        value_fn=lambda t, b: mag(t[flux_col(b)]),
        bins=50, rng=(20.0, 27.5),
        title=f"per-band magnitude ({FLUX_KIND})",
        xlabel="mag", xticks_major=1.0, xticks_minor=0.5,
        out_path=out("compare_mag.png"),
    )

    # ---- plot 2: per-band 1-D fpfs1 flux-error histogram ----
    plot_4panel_hist(
        series=series,
        value_fn=lambda t, b: np.asarray(t[f"{b}_flux_{FLUX_KIND}_err"]),
        bins=50, rng=(50.0, 900.0),
        title=f"per-band {FLUX_KIND} flux error",
        xlabel="flux error", xticks_major=200, xticks_minor=100,
        log_x=True,
        out_path=out("compare_ferr.png"),
    )

    # ---- plot 3: 2-D r-i vs i ----
    plot_2D_color(
        series, x_band="i", color_bands=("r", "i"),
        xrange=[20.0, 26.0], yrange=[-0.2, 1.3], bins=15,
        xlabel=rf"$i$ magnitude ({FLUX_KIND})", ylabel=r"$r - i$",
        title="r - i  vs  i",
        out_path=out("compare_ri_vs_i.png"),
    )

    # ---- plot 4: 2-D i-z vs i ----
    plot_2D_color(
        series, x_band="i", color_bands=("i", "z"),
        xrange=[20.0, 26.0], yrange=[-0.5, 1.0], bins=20,
        xlabel=rf"$i$ magnitude ({FLUX_KIND})", ylabel=r"$i - z$",
        title="i - z  vs  i",
        out_path=out("compare_iz_vs_i.png"),
    )

    # ---- plot 5: 2-D i-band size vs i ----
    plot_2D_size(
        series,
        xrange=[20.0, 26.0], yrange=[0.1, 1.2], bins=20,
        xlabel=rf"$i$ magnitude ({FLUX_KIND})",
        ylabel=r"$i$-band size [arcsec]",
        title="i-band size  vs  i",
        out_path=out("compare_size_vs_i.png"),
    )


if __name__ == "__main__":
    main()
