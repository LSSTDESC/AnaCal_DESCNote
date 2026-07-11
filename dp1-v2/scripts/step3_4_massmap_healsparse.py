#!/usr/bin/env python
"""Plot 6 — curved-sky Schirmer aperture-mass on a HEALPix grid,
stored as a HealSparseMap.

Unlike ``step3_3_massmap.py`` (which builds the map on a flat tangent
plane), this script computes ``Map_E``, ``Map_B`` and ``Map_V``
directly per HEALPix pixel, using the great-circle distance and the
spherical position angle between every (source, pixel) pair. That
removes the gnomonic-projection inaccuracy that grows with sky offset
and gives an output suited for stitching into a wider DESC survey map.

Outputs (under ``<field>/step3/``):
    * ``mass_map.hs.fits`` — HealSparseMap recarray with columns
        - ``Map_E``, ``Map_B``, ``Map_V`` — raw aperture-mass values
        - ``sn_e``, ``sn_b`` — divided per-pixel by ``sqrt(Map_V[pix])``
          (the B-mode of this normalisation sits on N(0, 1))
    * ``mass_map_healsparse.png`` — gnomview-style zoom around the BCG.
    * ``mass_map_healsparse_hist.png`` — per-pixel S/N histogram.

Usage:
    python step3_4_massmap_healsparse.py \\
        --ra 37.865017 --dec 6.982205 --z-cl 0.22 --radius 1.5 \\
        --collection u/xiangchl/dp1-v2/a360_anacal2 --field a360 \\
        --flux-name gauss2

Run order:
    python step3_3_massmap.py   <args>   # tangent-plane PNG plots
    python step3_4_massmap_healsparse.py <args>   # curved-sky HealSparseMap

The HealSparseMap is independent of step3_3 — feel free to run either
or both. The two output formats answer different questions: the tangent
PNG is for quick visual inspection at the BCG, the HealSparseMap is for
downstream survey-level analysis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import healpy as hp
import healsparse as hsp
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import cm
from matplotlib.ticker import MultipleLocator

from _step3_common import add_common_args, load_context, resolve_out
from xlens.utils.massmap import (
    compute_mass_map_healpix, schirmer_filter,
)


DEFAULT_HSP_OUT_NAME = "mass_map.hs.fits"
DEFAULT_PLOT_OUT_NAME = "mass_map_healsparse.png"
DEFAULT_HIST_OUT_NAME = "mass_map_healsparse_hist.png"
DEFAULT_NSIDE_SPARSE = 4096
DEFAULT_NSIDE_COVERAGE = 32
DEFAULT_PHYSICAL_ZOOM_MPC = None  # default: use fixed angular zoom (matches step3_3)
DEFAULT_ZOOM_HALF_EXTENT_DEG = 0.5  # matches step3_3's --half-extent-deg default
DEFAULT_ZOOM_NPIX = 301


def _angular_extent_from_physical(physical_mpc, z_cl,
                                  H0=70.0, Om0=0.3):
    """Convert a physical half-extent in Mpc to an angular half-extent
    in degrees, using flat-LambdaCDM angular diameter distance to z_cl.

    Same cosmology as step3_2_mass.py (H0=70, Om0=0.3, flat).
    """
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    d_a_mpc = float(cosmo.angular_diameter_distance(z_cl).to_value("Mpc"))
    return np.degrees(physical_mpc / d_a_mpc)


def make_hist_plot(*, sn_e_flat, sn_b_flat, xlim, out_path):
    """Per-pixel S/N histogram of the curved-sky map, normalised by
    per-pixel variance (sn_*_local). Same plot style as step3_3."""
    test_xs = np.linspace(*xlim, 1001)
    normal_dist = stats.norm(0, 1).pdf(test_xs)
    with plt.rc_context({
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
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
        ax.set_xlabel("aperture-mass S/N (curved sky)")
        ax.set_ylabel("PDF")
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.grid(True, which="major", alpha=0.30,
                linestyle="-", linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.15,
                linestyle=":", linewidth=0.5)
        ax.legend(frameon=False, fontsize=12, loc="upper right")
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"# wrote {out_path}", file=sys.stderr)
        else:
            plt.show()


def make_zoom_plot(
    *, hsp_map, ra_bcg, dec_bcg,
    peak_ra, peak_dec, peak_val,
    half_extent_deg, n_pix, out_path,
    vmin=None, vmax=None,
):
    """Zoom-in tangent-plane projection of the curved-sky HealSparse
    map using ``healpy.gnomview``: dense_healpy = generate_healpix_map(
    key=...) → gnomview projects it around the BCG. The peak marker
    coordinates come directly from ``peak_ra/peak_dec`` (obtained via
    ``hp.pix2ang`` on the brightest HEALPix pixel) — no re-search of
    the projected 2D array.
    """
    # Dense healpy arrays (ring scheme; healpy projections want ring).
    dense_e = hsp_map.generate_healpix_map(key="sn_e", nest=False)
    dense_b = hsp_map.generate_healpix_map(key="sn_b", nest=False)

    # gnomview parameters
    reso_arcmin = (2.0 * half_extent_deg / n_pix) * 60.0
    rot = (ra_bcg, dec_bcg)

    proj_e = hp.gnomview(
        dense_e, rot=rot, reso=reso_arcmin, xsize=n_pix, ysize=n_pix,
        no_plot=True, return_projected_map=True,
    )
    proj_b = hp.gnomview(
        dense_b, rot=rot, reso=reso_arcmin, xsize=n_pix, ysize=n_pix,
        no_plot=True, return_projected_map=True,
    )
    proj_e = np.where(proj_e == hp.UNSEEN, np.nan, proj_e)
    proj_b = np.where(proj_b == hp.UNSEEN, np.nan, proj_b)

    if vmax is None:
        # Anchor the colorbar on the E-mode peak (the signal), not on
        # max(|E|, |B|). Otherwise a single B-mode noise excursion
        # dominates and the cluster signal gets washed out for weak
        # detections (e.g. edfs at z=0.69).
        vmax = max(2.5, np.ceil(2 * abs(peak_val)) / 2)
    if vmin is None:
        vmin = -vmax
    print(f"# zoom imshow scale: vmin={vmin:.2f}, vmax={vmax:.2f}",
          file=sys.stderr)

    # Peak marker: convert the HealSparse peak (RA, Dec) into the
    # imshow data x-coordinate, matching step3_3's convention. With
    # extent=(+half, -half, ...) the imshow x-axis is reversed: the
    # LEFT edge has x_data=+half, the RIGHT edge has x_data=-half.
    # step3_3 stores its tangent-plane data with x = (α-α0)·cos(δ0)
    # in the same column order as ``s = linspace(-half, +half)``, so
    # the column at sky offset Δα ends up at x_data = Δα·cos(δ0)
    # (which is negative for west sources, putting them on the right
    # of the image — the standard "east on the left" sky view).
    cos_dec = np.cos(np.radians(dec_bcg))
    peak_x = (peak_ra - ra_bcg) * cos_dec
    peak_y = peak_dec - dec_bcg
    extent = (half_extent_deg, -half_extent_deg,
              -half_extent_deg, half_extent_deg)

    with plt.rc_context({
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    }):
        fig, axs = plt.subplots(
            ncols=2, figsize=(11, 5.5), sharex=True, sharey=True,
        )
        kw = dict(vmin=vmin, vmax=vmax, origin="lower",
                  extent=extent, cmap="RdBu_r")
        im_e = axs[0].imshow(proj_e, **kw)
        axs[0].set_title("E-mode S/N (curved sky, gnomview)")
        axs[0].scatter(peak_x, peak_y,
                       s=80, marker="x", color="black", lw=2.0,
                       label=f"peak = {peak_val:.2f}")
        axs[0].legend(loc="lower left", fontsize=11, frameon=True,
                      facecolor="white", edgecolor="0.5")
        axs[1].imshow(proj_b, **kw)
        axs[1].set_title("B-mode S/N (curved sky, gnomview)")
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


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument("--aperture-size", type=float, default=0.4,
                   help="Schirmer aperture size, deg (default: 0.4)")
    p.add_argument("--filter-a", type=float, default=6.0,
                   help="Schirmer filter `a` (default: 6)")
    p.add_argument("--filter-b", type=float, default=150.0,
                   help="Schirmer filter `b` (default: 150)")
    p.add_argument("--physical-zoom-mpc", type=float,
                   default=DEFAULT_PHYSICAL_ZOOM_MPC,
                   help="physical half-extent of the zoom window in Mpc "
                        "(default: off — use --zoom-half-extent-deg "
                        "instead, which defaults to the step3_3 value). "
                        "When set, converted to angular via D_A(z_cl). "
                        "Explicit --zoom-half-extent-deg always overrides.")
    p.add_argument("--healsparse-radius-deg", type=float, default=None,
                   help="radius of the HEALPix disc around the BCG "
                        "to compute (default: sqrt(2) × the angular "
                        "zoom-half-extent so the disc just covers the "
                        "zoom square).")
    p.add_argument("--nside-sparse", type=int,
                   default=DEFAULT_NSIDE_SPARSE,
                   help=f"healsparse value nside (default: "
                        f"{DEFAULT_NSIDE_SPARSE}, ~52 arcsec per pixel)")
    p.add_argument("--nside-coverage", type=int,
                   default=DEFAULT_NSIDE_COVERAGE,
                   help=f"healsparse coverage nside "
                        f"(default: {DEFAULT_NSIDE_COVERAGE})")
    p.add_argument("--healsparse-out", default=None,
                   help=f"output path for the HealSparseMap "
                        f"(default: <field>/step3/{DEFAULT_HSP_OUT_NAME})")
    p.add_argument("--plot-out", default=None,
                   help=f"output path for the zoom-in PNG "
                        f"(default: <field>/step3/{DEFAULT_PLOT_OUT_NAME}; "
                        f"pass '' to skip)")
    p.add_argument("--zoom-half-extent-deg", type=float, default=None,
                   help=f"half-extent of the zoom-in window in degrees "
                        f"(default: {DEFAULT_ZOOM_HALF_EXTENT_DEG}, "
                        f"matching step3_3's --half-extent-deg). When "
                        f"--physical-zoom-mpc is set, this falls back "
                        f"to the angular extent derived from it.")
    p.add_argument("--zoom-n-pix", type=int, default=DEFAULT_ZOOM_NPIX,
                   help=f"pixel side of the zoom-in window "
                        f"(default: {DEFAULT_ZOOM_NPIX})")
    p.add_argument("--hist-out", default=None,
                   help=f"output path for the per-pixel S/N histogram "
                        f"(default: <field>/step3/{DEFAULT_HIST_OUT_NAME}; "
                        f"pass '' to skip)")
    p.add_argument("--hist-xlim", default="-5,8",
                   help="histogram x (S/N) limits, comma-separated "
                        "(default: -5,8 — same as step3_3)")
    p.add_argument("--peak-edge-buffer-deg", type=float, default=None,
                   help="when finding the |sn_e| peak, ignore pixels within "
                        "this great-circle distance (deg) of the search-disc "
                        "rim — they are biased by the Schirmer aperture "
                        "kernel reaching outside the source footprint. "
                        "Default = --aperture-size (so the peak search is "
                        "restricted to the inner radius_deg - aperture_size).")
    return p.parse_args()


def main():
    args = parse_args()
    out_hsp = args.healsparse_out or resolve_out(args, DEFAULT_HSP_OUT_NAME)
    if args.plot_out == "":
        out_plot = None
    else:
        out_plot = args.plot_out or resolve_out(args, DEFAULT_PLOT_OUT_NAME)
    if args.hist_out == "":
        out_hist = None
    else:
        out_hist = args.hist_out or resolve_out(args, DEFAULT_HIST_OUT_NAME)
    for p in (out_hsp, out_plot, out_hist):
        if p is not None:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    ctx = load_context(args)
    table = ctx["table"]
    e1, e2, res = ctx["e1"], ctx["e2"], ctx["res"]
    ra_bcg, dec_bcg = ctx["ra_bcg"], ctx["dec_bcg"]
    global_r = float(np.mean(res))
    print(f"# global R = {global_r:.4f}", file=sys.stderr)
    g1, g2 = e1 / global_r, e2 / global_r
    weights = np.ones(len(e1))

    # Resolve angular extents — precedence: explicit
    # --zoom-half-extent-deg wins, then --physical-zoom-mpc (scaled by
    # D_A(z_cl)), then the fixed default that matches step3_3.
    zoom_half_extent_deg = args.zoom_half_extent_deg
    if zoom_half_extent_deg is None:
        if args.physical_zoom_mpc is not None:
            zoom_half_extent_deg = _angular_extent_from_physical(
                args.physical_zoom_mpc, ctx["z_cl"],
            )
            print(f"# physical zoom = {args.physical_zoom_mpc} Mpc at "
                  f"z={ctx['z_cl']} -> zoom_half_extent = "
                  f"{zoom_half_extent_deg:.3f} deg", file=sys.stderr)
        else:
            zoom_half_extent_deg = DEFAULT_ZOOM_HALF_EXTENT_DEG
            print(f"# zoom_half_extent = {zoom_half_extent_deg} deg "
                  f"(default, matches step3_3)", file=sys.stderr)
    radius_deg = args.healsparse_radius_deg
    if radius_deg is None:
        radius_deg = float(np.sqrt(2.0) * zoom_half_extent_deg)
    print(f"# curved-sky map: nside_sparse={args.nside_sparse}, "
          f"radius={radius_deg:.3f} deg around BCG", file=sys.stderr)
    ipix, e_ap, b_ap, v_ap = compute_mass_map_healpix(
        ra_center=ra_bcg, dec_center=dec_bcg, radius_deg=radius_deg,
        src_ra=np.asarray(table["ra"]),
        src_dec=np.asarray(table["dec"]),
        g1=g1, g2=g2, weights=weights,
        q_filter=schirmer_filter,
        filter_kwargs={
            "aperture_size": args.aperture_size,
            "a": args.filter_a,
            "b": args.filter_b,
        },
        nside=args.nside_sparse, nest=True, verbose=True,
    )

    # Per-pixel ("local") S/N: divide each pixel by its own
    # sqrt(Map_V) so the noise model is right for the B-mode null
    # test, the imshow peak, and the histogram alike.
    sn_e = e_ap / np.sqrt(np.maximum(v_ap, 1e-30))
    sn_b = b_ap / np.sqrt(np.maximum(v_ap, 1e-30))

    # Mask the outer rim before the peak search — the Schirmer kernel
    # extends ~aperture_size; pixels within that distance of the
    # search-disc edge see sources only on the inward side, biasing
    # both Map_E and Map_V and producing spurious rim peaks.
    edge_buf = (args.peak_edge_buffer_deg
                if args.peak_edge_buffer_deg is not None
                else args.aperture_size)
    peak_max_radius_deg = max(0.0, radius_deg - edge_buf)
    theta_pk, phi_pk = hp.pix2ang(
        args.nside_sparse, ipix, nest=True,
    )
    pix_ra = np.degrees(phi_pk)
    pix_dec = 90.0 - np.degrees(theta_pk)
    cos_d = (np.sin(np.deg2rad(dec_bcg)) * np.sin(np.deg2rad(pix_dec))
             + np.cos(np.deg2rad(dec_bcg)) * np.cos(np.deg2rad(pix_dec))
             * np.cos(np.deg2rad(pix_ra - ra_bcg)))
    pix_sep_deg = np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))
    inner_mask = pix_sep_deg <= peak_max_radius_deg
    n_search = int(inner_mask.sum())
    if n_search == 0:
        print(f"# WARN: peak_edge_buffer_deg={edge_buf:.3f} left zero "
              f"pixels inside radius {radius_deg:.3f} - {edge_buf:.3f} = "
              f"{peak_max_radius_deg:.3f} deg; falling back to full map",
              file=sys.stderr)
        inner_mask = np.ones_like(sn_e, dtype=bool)
        n_search = sn_e.size
    print(f"# peak search restricted to {n_search:,} / {sn_e.size:,} "
          f"pixels within {peak_max_radius_deg:.3f} deg of BCG "
          f"(rim buffer = {edge_buf:.3f} deg)",
          file=sys.stderr)
    masked_sn_e = np.where(inner_mask, sn_e, -np.inf)
    peak_arg = int(np.argmax(masked_sn_e))
    peak_ra, peak_dec = hp.pix2ang(
        args.nside_sparse, ipix[peak_arg], lonlat=True, nest=True,
    )
    print(f"# E-mode peak S/N = {sn_e[peak_arg]:.2f} at "
          f"(RA, Dec) = ({peak_ra:.4f}, {peak_dec:.4f}) deg "
          f"(BCG offset: dRA={peak_ra - ra_bcg:+.3f}, "
          f"dDec={peak_dec - dec_bcg:+.3f})",
          file=sys.stderr)
    print(f"# B-mode (local): mean={sn_b.mean():+.3f}, "
          f"std={sn_b.std():.3f} (expect ~ N(0, 1))",
          file=sys.stderr)

    dtype = np.dtype([
        ("Map_E", "f4"), ("Map_B", "f4"), ("Map_V", "f4"),
        ("sn_e", "f4"), ("sn_b", "f4"),
    ])
    hsp_map = hsp.HealSparseMap.make_empty(
        nside_coverage=args.nside_coverage,
        nside_sparse=args.nside_sparse,
        dtype=dtype, primary="Map_E",
    )
    arr = np.zeros(len(ipix), dtype=dtype)
    arr["Map_E"] = e_ap.astype("f4")
    arr["Map_B"] = b_ap.astype("f4")
    arr["Map_V"] = v_ap.astype("f4")
    arr["sn_e"] = sn_e.astype("f4")
    arr["sn_b"] = sn_b.astype("f4")
    hsp_map.update_values_pix(ipix, arr)
    if out_hsp is not None:
        hsp_map.write(out_hsp, clobber=True)
        n_valid = int(hsp_map.valid_pixels.size)
        print(f"# wrote {out_hsp}  "
              f"(nside_sparse={args.nside_sparse}, "
              f"nside_coverage={args.nside_coverage}, "
              f"valid_pix={n_valid:,})",
              file=sys.stderr)

    if out_plot is not None:
        make_zoom_plot(
            hsp_map=hsp_map, ra_bcg=ra_bcg, dec_bcg=dec_bcg,
            peak_ra=peak_ra, peak_dec=peak_dec,
            peak_val=float(sn_e[peak_arg]),
            half_extent_deg=zoom_half_extent_deg,
            n_pix=args.zoom_n_pix, out_path=out_plot,
        )

    if out_hist is not None:
        xlim = tuple(float(s) for s in args.hist_xlim.split(","))
        make_hist_plot(
            sn_e_flat=sn_e,
            sn_b_flat=sn_b,
            xlim=xlim, out_path=out_hist,
        )


if __name__ == "__main__":
    main()
