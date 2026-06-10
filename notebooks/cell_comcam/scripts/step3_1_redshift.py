#!/usr/bin/env python
"""Plot 1 — source n(z) vs full-sample n(z), from stacked p(z) PDFs.

Loads the per-patch ``deep_coadd_cell_anacal_fzb_pdfs`` (NumPy
``(N, ndist, 501)`` arrays) via the butler, slices the undistorted
``[:, 0, :]`` PDF for every source, then:

* **source n(z)** = response-weighted sum of PDFs over the source cut
* **full n(z)** = unweighted sum of PDFs over every loaded row

Persists the normalized arrays to ``<field>/step3/source_nz.npz`` (keys
``z, src_nz, full_nz, n_src, n_full``) so ``step3_mass.py`` can read
back the same n(z) for the beta_s / beta_s_sqr calculation and the NFW
overlay — guaranteeing consistency between the two plots.

If the pipeline ran with ``output_pdfs: false`` (no PDF dataset
present), falls back to a response-weighted histogram of ``zmode_0``.

Usage:
    python step3_redshift.py \\
        --ra 37.865017 --dec 6.982205 --z-cl 0.22 --radius 1.5 \\
        --collection u/xiangchl/dp1-v2/a360_anacal2 --field a360
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import cm
from matplotlib.ticker import MultipleLocator

from _step3_common import (
    FIELDS_ROOT, STEP3_SUBDIR, Z_GRID,
    add_common_args, load_context, resolve_out,
)


DEFAULT_OUT_NAME = "redshift.png"
NZ_NPZ_NAME = "source_nz.npz"


def nz_npz_path(field: str | None, explicit: str | None = None) -> Path | None:
    """Where to persist / read back the source n(z) for step3_mass.

    Defaults to ``<FIELDS_ROOT>/<field>/step3/source_nz.npz``.
    Returns ``None`` if no field is given and no explicit path supplied.
    """
    if explicit is not None:
        return Path(explicit)
    if field is None:
        return None
    return FIELDS_ROOT / field / STEP3_SUBDIR / NZ_NPZ_NAME


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--zmode-col", default="zmode_0",
                   help="point-estimate column for the histogram fallback (default: zmode_0)")
    p.add_argument("--bins", type=int, default=80,
                   help="bins for the histogram fallback (default: 80)")
    p.add_argument("--xlim", default="0,3", help="x-axis (z) limits, comma-separated")
    p.add_argument("--nz-out", default=None,
                   help="explicit path for the saved n(z) .npz "
                        "(default: <field>/step3/source_nz.npz)")
    return p.parse_args()


def build_nz_from_pdfs(pdfs_all, pdfs_sel, response_sel):
    """source = sum(R*pdf_sel); full = sum(pdf_all). Returns normalised pair."""
    response = np.asarray(response_sel)[:, None]
    src_stack = (response * pdfs_sel).sum(axis=0)
    src_norm = src_stack / integrate.trapezoid(src_stack, Z_GRID)
    full = pdfs_all.sum(axis=0)
    full_norm = full / integrate.trapezoid(full, Z_GRID)
    return src_norm, full_norm


def build_nz_from_zmode(table_all, table, zmode_col, bins):
    """R-weighted source histogram + uniform-weight full histogram of zmode."""
    zs_src = np.asarray(table[zmode_col])
    ws_src = np.asarray(table["response"])
    zs_all = np.asarray(table_all[zmode_col])
    h_src, edges = np.histogram(zs_src, bins=bins, weights=ws_src, density=True)
    h_all, _ = np.histogram(zs_all, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, h_src, h_all


def main():
    args = parse_args()
    out_path = resolve_out(args, DEFAULT_OUT_NAME)
    xlim = tuple(float(s) for s in args.xlim.split(","))

    ctx = load_context(args, load_pdfs=True)
    pdfs_all = ctx["pdfs_all"]
    pdfs = ctx["pdfs"]
    table = ctx["table"]
    table_all = ctx["table_all"]
    z_cl = ctx["z_cl"]
    use_pdfs = pdfs_all is not None and pdfs is not None

    with plt.rc_context({
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    }):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        cmap = cm.coolwarm

        if use_pdfs:
            src_norm, full_norm = build_nz_from_pdfs(
                pdfs_all, pdfs, table["response"],
            )
            ax.plot(Z_GRID, src_norm, "-", color=cmap(0.92), lw=2.0,
                    label="source sample (R-weighted PDF stack)")
            ax.plot(Z_GRID, full_norm, "-", color=cmap(0.08), lw=2.0,
                    label="full sample (PDF stack)")

            # Persist so step3_mass.py can reuse the same n(z).
            nz_path = nz_npz_path(args.field, args.nz_out)
            if nz_path is not None:
                nz_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    nz_path,
                    z=Z_GRID, src_nz=src_norm, full_nz=full_norm,
                    n_src=len(table), n_full=len(table_all),
                )
                print(f"# wrote {nz_path}", file=sys.stderr)
        else:
            print(f"# WARN: no PDF dataset — falling back to {args.zmode_col} histogram",
                  file=sys.stderr)
            centers, h_src, h_all = build_nz_from_zmode(
                table_all, table, args.zmode_col, args.bins,
            )
            ax.plot(centers, h_src, "-", color=cmap(0.92), lw=2.0,
                    label=f"source sample ({args.zmode_col}, R-weighted)")
            ax.plot(centers, h_all, "-", color=cmap(0.08), lw=2.0,
                    label=f"full sample   ({args.zmode_col})")

        ax.axvline(z_cl, ls="--", color="k", alpha=0.5,
                   label=rf"$z_{{cl}}={z_cl:g}$")
        ax.set_xlim(*xlim)
        ax.set_xlabel("redshift")
        ax.set_ylabel(r"normalized $N(z)$")
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(True, which="major", alpha=0.30, linestyle="-", linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.15, linestyle=":", linewidth=0.5)
        ax.legend(frameon=False, fontsize=11, loc="upper right")
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"# wrote {out_path}", file=sys.stderr)
        else:
            plt.show()


if __name__ == "__main__":
    main()
