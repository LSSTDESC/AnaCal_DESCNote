#!/usr/bin/env python
"""Plots 4 + 5 — Schirmer aperture-mass S/N maps and their per-pixel
histogram.

Computes the Schirmer aperture-mass E/B/variance maps **once** and
produces both:

* ``mass_map.png``      (plot 4) — E-mode + B-mode S/N images with the
                                   E-mode peak pixel marked.
* ``mass_map_hist.png`` (plot 5) — per-pixel S/N histogram (E + B)
                                   against the unit normal reference.

Usage:
    python step3_3_massmap.py \\
        --ra 37.865017 --dec 6.982205 --z-cl 0.22 --radius 1.5 \\
        --collection u/xiangchl/dp1-v2/a360_anacal2 --field a360
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import cm
from matplotlib.ticker import MultipleLocator

from _step3_common import add_common_args, load_context, resolve_out
from xlens.utils.massmap import (
    build_flat_wcs_grid, compute_mass_map, schirmer_filter,
)


DEFAULT_MAP_OUT_NAME = "mass_map.png"
DEFAULT_HIST_OUT_NAME = "mass_map_hist.png"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--n-grid", type=int, default=121,
                   help="grid side length NxN for the map (default: 121)")
    p.add_argument("--half-extent-deg", type=float, default=0.5,
                   help="map half-extent in degrees (default: 0.5)")
    p.add_argument("--aperture-size", type=float, default=0.4,
                   help="Schirmer aperture size, deg (default: 0.4)")
    p.add_argument("--filter-a", type=float, default=15.0,
                   help="Schirmer filter `a` (default: 15)")
    p.add_argument("--filter-b", type=float, default=150.0,
                   help="Schirmer filter `b` (default: 150)")
    p.add_argument("--vmin", type=float, default=-2.0, help="map imshow vmin (default: -2)")
    p.add_argument("--vmax", type=float, default=5.0,  help="map imshow vmax (default: 5)")
    p.add_argument("--hist-xlim", default="-5,8",
                   help="histogram x (S/N) limits, comma-separated (default: -5,8)")
    p.add_argument("--map-out", default=None,
                   help="explicit output path for the S/N map "
                        "(default: <field>/step3/mass_map.png)")
    p.add_argument("--hist-out", default=None,
                   help="explicit output path for the S/N histogram "
                        "(default: <field>/step3/mass_map_hist.png)")
    return p.parse_args()


def make_map_plot(*, sn_e, sn_b, peak_idx, peak_val, extent, N,
                  vmin, vmax, out_path):
    x_vals = np.linspace(extent[0], extent[1], N)
    y_vals = np.linspace(extent[2], extent[3], N)
    with plt.rc_context({
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    }):
        fig, axs = plt.subplots(ncols=2, figsize=(11, 5.5),
                                sharex=True, sharey=True)
        kw = dict(vmin=vmin, vmax=vmax, origin="lower",
                  extent=extent, cmap="RdBu_r")
        im_e = axs[0].imshow(sn_e, **kw)
        axs[0].set_title("E-mode S/N")
        axs[0].scatter(x_vals[peak_idx[1]], y_vals[peak_idx[0]],
                       s=80, marker="x", color="black", lw=2.0,
                       label=f"peak = {peak_val:.2f}")
        axs[0].legend(loc="lower left", fontsize=11, frameon=True,
                      facecolor="white", edgecolor="0.5")
        axs[1].imshow(sn_b, **kw)
        axs[1].set_title("B-mode S/N")
        for ax in axs:
            ax.scatter(0, 0, s=70, marker="+", color="k", lw=1.5)
            ax.set_xlabel("RA offset [deg]")
            ax.set_aspect("equal")
        axs[0].set_ylabel("Dec offset [deg]")
        cb = fig.colorbar(im_e, ax=axs, fraction=0.025, pad=0.02)
        cb.set_label("S/N")
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"# wrote {out_path}", file=sys.stderr)
        else:
            plt.show()


def make_hist_plot(*, sn_e_flat, sn_b_flat, xlim, out_path):
    test_xs = np.linspace(*xlim, 1001)
    normal_dist = stats.norm(0, 1).pdf(test_xs)
    with plt.rc_context({
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    }):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        cmap = cm.coolwarm
        c_e = cmap(0.08)
        c_b = cmap(0.92)
        bins = np.linspace(*xlim, 61)
        ax.hist(sn_b_flat, bins=bins, density=True, histtype="step",
                color=c_b, lw=2.0, label="B mode")
        ax.hist(sn_e_flat, bins=bins, density=True, histtype="step",
                color=c_e, lw=2.0, label="E mode")
        ax.plot(test_xs, normal_dist, "--", color="k", lw=1.5,
                label=r"$\mathcal{N}(0,1)$")
        ax.set_xlim(*xlim)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 5e-1)
        ax.set_xlabel("aperture-mass S/N")
        ax.set_ylabel("PDF")
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.grid(True, which="major", alpha=0.30, linestyle="-", linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.15, linestyle=":", linewidth=0.5)
        ax.legend(frameon=False, fontsize=12, loc="upper right")
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"# wrote {out_path}", file=sys.stderr)
        else:
            plt.show()


def main():
    args = parse_args()
    out_map = args.map_out or resolve_out(args, DEFAULT_MAP_OUT_NAME)
    out_hist = args.hist_out or resolve_out(args, DEFAULT_HIST_OUT_NAME)
    for p in (out_map, out_hist):
        if p is not None:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    ctx = load_context(args)
    table = ctx["table"]
    e1, e2, res = ctx["e1"], ctx["e2"], ctx["res"]
    ra_bcg, dec_bcg = ctx["ra_bcg"], ctx["dec_bcg"]

    N, rr = args.n_grid, args.half_extent_deg
    x, y, x_grid, y_grid = build_flat_wcs_grid(
        ra_bcg, dec_bcg, table["ra"], table["dec"],
        n=N, half_extent_deg=rr,
    )
    global_R = float(np.mean(res))
    print(f"# global R = {global_R:.4f}", file=sys.stderr)
    g1, g2 = e1 / global_R, e2 / global_R
    weights = np.ones(len(x))

    print(f"# computing {N}x{N} Schirmer aperture map (the slow step)...",
          file=sys.stderr)
    e_ap, b_ap, v_ap = compute_mass_map(
        x_grid, y_grid, x, y, g1, g2, weights,
        schirmer_filter,
        filter_kwargs={
            "aperture_size": args.aperture_size,
            "a": args.filter_a,
            "b": args.filter_b,
        },
    )

    # Plot 4 — S/N map (normalised by the spatial-mean variance, matching
    # the BNL notebook convention).
    sn_v = np.sqrt(np.mean(v_ap))
    sn_e_map = e_ap / sn_v
    sn_b_map = b_ap / sn_v
    peak_idx = np.unravel_index(int(np.argmax(sn_e_map)), sn_e_map.shape)
    peak_val = float(sn_e_map[peak_idx])
    print(f"# E-mode peak S/N = {peak_val:.2f} at grid {peak_idx}", file=sys.stderr)
    extent = (rr, -rr, -rr, rr)
    make_map_plot(
        sn_e=sn_e_map, sn_b=sn_b_map,
        peak_idx=peak_idx, peak_val=peak_val,
        extent=extent, N=N, vmin=args.vmin, vmax=args.vmax,
        out_path=out_map,
    )

    # Plot 5 — per-pixel S/N histogram, normalised by per-pixel variance
    # (so the B mode should sit on N(0, 1)).
    sn_e_pix = (e_ap / np.sqrt(v_ap)).flatten()
    sn_b_pix = (b_ap / np.sqrt(v_ap)).flatten()
    xlim = tuple(float(s) for s in args.hist_xlim.split(","))
    make_hist_plot(sn_e_flat=sn_e_pix, sn_b_flat=sn_b_pix,
                   xlim=xlim, out_path=out_hist)


if __name__ == "__main__":
    main()
