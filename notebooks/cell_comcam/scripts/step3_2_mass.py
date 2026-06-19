#!/usr/bin/env python
"""Plots 2 + 3 — the emcee posterior on log10(M500c) at fixed
concentration (plot 2) and the tangential & cross reduced shear
profile with NFW overlay (plot 3).

Both plots use the stacked source n(z) persisted by
``step3_redshift.py`` (at ``<field>/step3/source_nz.npz``), so the
``beta_s`` / ``beta_s_sqr`` driving the NFW overlay in the tangential
plot is identical to the beta used by the mass fit — no
DESC-SRD-default lineage, just the actual data n(z).

Outputs (under ``<field>/step3/``):
  * ``mass.png``       — emcee posterior on log10(M500c)              (plot 2)
  * ``tangential.png`` — reduced shear vs Mpc + NFW curve             (plot 3)

Run order:
    python step3_redshift.py <common args>   # plot 1, writes source_nz.npz
    python step3_mass.py     <common args>   # plots 2 + 3
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

import emcee
import clmm
from clmm import Cosmology
from clmm import GCData, GalaxyCluster
from clmm.support.sampler import fitters

from _step3_common import (
    FIELDS_ROOT, STEP3_SUBDIR, Z_GRID,
    add_common_args, load_context, resolve_out,
)
from xlens.utils.nxg import anacal_get_tang_cross
from step3_1_redshift import nz_npz_path, build_nz_from_pdfs


DEFAULT_TANGENTIAL_OUT_NAME = "tangential.png"
DEFAULT_MASS_OUT_NAME = "mass.png"

DEFAULT_BINS_MPC = np.array([0.3, 0.7, 1.06495522, 1.62018517, 2.46489237, 3.75, 5, 6])
DEFAULT_NFW_MASS = 6e14


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--nfw-c", type=float, default=4.0,
                   help="fixed NFW concentration (default: 4)")
    p.add_argument("--bins-mpc", default=None,
                   help="radial bin edges in Mpc. Pass either "
                        "'rmin,rmax,nbins' for log-spaced edges (e.g. "
                        "'0.2,9,7' = alternate recipe) or an explicit "
                        "comma-separated list of edges. Default: the "
                        "built-in 7-bin DEFAULT_BINS_MPC array.")
    p.add_argument("--nfw-mass", type=float, default=DEFAULT_NFW_MASS,
                   help=f"NFW M500c for the tangential-plot overlay "
                        f"(default: {DEFAULT_NFW_MASS:.0e})")
    p.add_argument("--nwalkers", type=int, default=32,
                   help="emcee walkers (default: 32)")
    p.add_argument("--nsteps", type=int, default=100,
                   help="emcee steps per walker (default: 100)")
    p.add_argument("--ci-level", type=float, default=0.68,
                   help="bootstrap CI for per-bin shear means (default: 0.68)")
    p.add_argument("--mass-bins", default="13.1,15.5,0.05",
                   help="logM histogram bins (min,max,dx; default 13.1,15.5,0.05)")
    p.add_argument("--mass-xlim", default="13,16",
                   help="x-axis limits for the mass posterior")
    p.add_argument("--nz-in", default=None,
                   help="explicit path to source_nz.npz (default: "
                        "<field>/step3/source_nz.npz)")
    p.add_argument("--tangential-out", default=None,
                   help="explicit path for the tangential plot (default: "
                        "<field>/step3/tangential.png)")
    p.add_argument("--mass-out", default=None,
                   help="explicit path for the mass-posterior plot (default: "
                        "<field>/step3/mass.png)")
    return p.parse_args()


def get_source_nz(args, ctx):
    """Load the persisted source n(z); recompute from PDFs if missing."""
    nz_path = nz_npz_path(args.field, args.nz_in)
    if nz_path is not None and nz_path.exists():
        d = np.load(nz_path)
        print(f"# loaded n(z) from {nz_path} "
              f"(N_src={int(d['n_src']):,}, N_full={int(d['n_full']):,})",
              file=sys.stderr)
        if not np.allclose(d["z"], Z_GRID):
            raise SystemExit(
                f"z-grid in {nz_path} ({d['z'].shape}) doesn't match Z_GRID"
            )
        return d["src_nz"], d["full_nz"]
    # Fallback: recompute from PDFs already in ctx.
    if ctx["pdfs_all"] is None or ctx["pdfs"] is None:
        raise SystemExit(
            f"no n(z) at {nz_path} and pipeline didn't emit PDFs — "
            "run step3_redshift.py first (or rerun the pipeline with "
            "output_pdfs: true)"
        )
    print(f"# {nz_path} missing — recomputing n(z) from PDFs", file=sys.stderr)
    return build_nz_from_pdfs(
        ctx["pdfs_all"], ctx["pdfs"], ctx["table"]["response"],
    )


def beta_moments_from_nz(z, nz, z_cl, cosmo):
    """Compute <beta_s>, <beta_s^2> by integrating against the source n(z)."""
    def beta_s_of_zs(zs):
        if zs <= z_cl:
            return 0.0
        return (
            cosmo.eval_da_z1z2(z_cl, zs) / cosmo.eval_da(zs)
            * cosmo.eval_da(1e3) / cosmo.eval_da_z1z2(z_cl, 1e3)
        )
    beta = np.array([beta_s_of_zs(zz) for zz in z])
    norm = integrate.trapezoid(nz, z)
    return (
        float(integrate.trapezoid(beta * nz, z) / norm),
        float(integrate.trapezoid(beta ** 2 * nz, z) / norm),
    )


def fit_curve(profile, beta, beta_sqr, cdelta, z_cl, cosmo, halo="nfw"):
    def fn(radius, logm):
        return clmm.compute_reduced_tangential_shear(
            r_proj=radius, mdelta=10 ** logm, cdelta=cdelta,
            z_cluster=z_cl, z_src=(beta, beta_sqr),
            z_src_info="beta", approx="order2", cosmo=cosmo,
            delta_mdef=500, massdef="critical",
            halo_profile_model=halo,
        )
    popt, pcov = fitters["curve_fit"](
        fn, profile["radius"], profile["gt"], profile["gt_err"],
        bounds=[10.0, 17.0],
    )
    return fn, {"vals": popt, "err": np.sqrt(np.diag(pcov))}


def run_emcee(profile, fn, init, nwalkers, nsteps):
    def log_prob(pars):
        if pars[0] < 10 or pars[0] > 16:
            return -np.inf
        diff = fn(profile["radius"], *pars) - profile["gt"]
        return float(-0.5 * np.sum(diff ** 2 / profile["gt_err"] ** 2))
    sampler = emcee.EnsembleSampler(nwalkers, 1, log_prob)
    p0 = np.random.normal(init["vals"], init["err"] * 0.3, (nwalkers, 1))
    p0[p0 < 0] *= -1
    sampler.run_mcmc(p0, nsteps, progress=True)
    return sampler.get_chain().flatten(), sampler.get_log_prob().flatten()


def make_tangential_plot(*, bin_mids, tang_avg, cross_avg, tang_err, cross_err,
                         rproj, gt_overlay, overlay_label, out_path,
                         chi2_t, chi2_x):
    with plt.rc_context({
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
    }):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        cmap = cm.coolwarm
        c_t = cmap(0.08)
        c_x = cmap(0.92)
        ax.plot(rproj, gt_overlay, ls="--", color="0.25", lw=1.8,
                label=overlay_label)
        ax.plot(bin_mids, tang_avg, "o", color=c_t, ms=7,
                label=rf"tangential ($\chi={np.sqrt(chi2_t):.2f}$)")
        ax.vlines(bin_mids, tang_err[:, 0], tang_err[:, 1], color=c_t, lw=1.6)
        offs = 1.05 * bin_mids
        ax.plot(offs, cross_avg, "s", color=c_x, ms=6,
                label=rf"cross ($\chi={np.sqrt(chi2_x):.2f}$)")
        ax.vlines(offs, cross_err[:, 0], cross_err[:, 1], color=c_x, lw=1.6)
        ax.axhline(0, ls=":", color="k", alpha=0.6)
        ax.semilogx()
        ax.set_ylim(-0.05, 0.10)
        ax.set_xlim(0.45, 7.5)
        ax.set_xlabel(r"$R$ [Mpc]")
        ax.set_ylabel("reduced shear")
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.grid(True, which="major", alpha=0.30, linestyle="-", linewidth=0.7)
        ax.grid(True, which="minor", alpha=0.15, linestyle=":", linewidth=0.5)
        ax.legend(frameon=False, loc="upper right", fontsize=12)
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            print(f"# wrote {out_path}", file=sys.stderr)
        else:
            plt.show()


def make_mass_plot(*, chain, ll, mass_bins, xlim, title, out_path):
    with plt.rc_context({
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.2,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
    }):
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        cmap = cm.coolwarm
        c = cmap(0.08)
        weights = np.exp(ll)
        ax.hist(chain, bins=mass_bins, weights=weights, density=True,
                linewidth=2.0, histtype="step", color=c)
        ax.hist(chain, bins=mass_bins, weights=weights, density=True,
                linewidth=0, histtype="stepfilled", color=c, alpha=0.25)
        med = float(np.average(chain, weights=weights))
        lo_q, hi_q = (float(np.percentile(chain, p)) for p in (16, 84))
        ax.axvline(med, ls="--", color=c, lw=1.2,
                   label=rf"$\log M_{{500c}}={med:.2f}_{{-{med-lo_q:.2f}}}^{{+{hi_q-med:.2f}}}$")
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"$\log_{10}(M_{500c}/M_\odot)$")
        ax.set_ylabel("normalized counts")
        ax.set_title(title)
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


def main():
    args = parse_args()
    out_tang = (args.tangential_out
                or resolve_out(args, DEFAULT_TANGENTIAL_OUT_NAME))
    out_mass = (args.mass_out or resolve_out(args, DEFAULT_MASS_OUT_NAME))
    # mkdir -p for explicit overrides
    for p in (out_tang, out_mass):
        if p is not None:
            Path(p).parent.mkdir(parents=True, exist_ok=True)

    ctx = load_context(args, load_pdfs=True)
    table = ctx["table"]
    e1, e2, res = ctx["e1"], ctx["e2"], ctx["res"]
    ra_bcg, dec_bcg, z_cl = ctx["ra_bcg"], ctx["dec_bcg"], ctx["z_cl"]

    cosmo = Cosmology(H0=70.0, Omega_dm0=0.3 - 0.045, Omega_b0=0.045, Omega_k0=0.0)

    # n(z) from step3_redshift (or recompute on the fly).
    src_nz, _full_nz = get_source_nz(args, ctx)
    beta_s, beta_s_sqr = beta_moments_from_nz(Z_GRID, src_nz, z_cl, cosmo)
    print(f"# beta_s={beta_s:.4f}  beta_s_sqr={beta_s_sqr:.4f}", file=sys.stderr)

    # ---- Plot 2 — mass posterior ----
    if args.bins_mpc is None:
        bins_mpc = DEFAULT_BINS_MPC
    else:
        parts = [float(s) for s in args.bins_mpc.split(",")]
        if len(parts) == 3:
            # 'rmin,rmax,nbins' shorthand -> log-spaced edges.
            rmin, rmax, nbins = parts[0], parts[1], int(parts[2])
            bins_mpc = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
        else:
            bins_mpc = np.asarray(parts)
    print(f"# bins_mpc = {bins_mpc.round(3).tolist()}", file=sys.stderr)
    bin_mids = 0.5 * (bins_mpc[1:] + bins_mpc[:-1])
    global_R = float(np.mean(res))
    print(f"# global R = {global_R:.4f}", file=sys.stderr)
    galcat = GCData()
    galcat["ra"] = table["ra"]
    galcat["dec"] = table["dec"]
    galcat["e1"] = e1 / global_R
    galcat["e2"] = e2 / global_R
    galcat["id"] = np.arange(len(table))
    galcat["z"] = table["zbest_0"] if "zbest_0" in table.colnames else table["zmode_0"]
    gc = GalaxyCluster(args.field or "cluster", ra_bcg, dec_bcg, z_cl, galcat)
    gc.compute_tangential_and_cross_components(add=True)
    gc.make_radial_profile(
        bins=bins_mpc, bin_units="Mpc", add=True,
        cosmo=cosmo, overwrite=True,
        use_weights=False, gal_ids_in_bins=False,
    )
    profile = gc.profile

    fn, init = fit_curve(profile, beta_s, beta_s_sqr, args.nfw_c, z_cl, cosmo)
    print(f"# curve_fit logM = {init['vals'][0]:.3f} +- {init['err'][0]:.3f}",
          file=sys.stderr)
    chain, ll = run_emcee(profile, fn, init, args.nwalkers, args.nsteps)
    fin = np.isfinite(ll)
    chain, ll = chain[fin], ll[fin]
    lo, hi, dx = (float(x) for x in args.mass_bins.split(","))
    mass_bins = np.arange(lo, hi, dx)
    xlim = tuple(float(x) for x in args.mass_xlim.split(","))
    title = (rf"{args.field or 'cluster'} — NFW mass posterior "
             rf"(fixed $c={args.nfw_c:g}$, n(z) from data)")
    make_mass_plot(
        chain=chain, ll=ll, mass_bins=mass_bins, xlim=xlim,
        title=title, out_path=out_mass,
    )

    # Best-fit logM for the tangential overlay (median of the
    # likelihood-weighted posterior).
    weights = np.exp(ll)
    logM_med = float(np.average(chain, weights=weights))
    nfw_mass_fit = 10 ** logM_med

    # ---- Plot 3 — tangential shear profile + NFW overlay ----
    source_phi = np.arctan2(
        table["dec"] - dec_bcg,
        (ra_bcg - table["ra"]) * np.cos(np.deg2rad(dec_bcg)),
    )
    ang_dist = np.sqrt(
        ((table["ra"] - ra_bcg) * np.cos(np.deg2rad(dec_bcg))) ** 2
        + (table["dec"] - dec_bcg) ** 2
    )
    Da_cl = cosmo.eval_da(z_cl)
    sky_distance = Da_cl * np.asarray(ang_dist) * (np.pi / 180.0)
    trial_shear = e1 + 1j * e2
    cl_shear = trial_shear * -np.exp(-2j * source_phi)
    shear_cl = anacal_get_tang_cross(
        cl_shear, sky_distance, bins_mpc, res, ci_level=args.ci_level,
    )
    tang_avg, cross_avg, tang_err, cross_err = shear_cl[:4]
    chi2_t = np.sum(tang_avg ** 2 / ((tang_err[:, 0] - tang_err[:, 1]) / 2.0) ** 2)
    chi2_x = np.sum(cross_avg ** 2 / ((cross_err[:, 0] - cross_err[:, 1]) / 2.0) ** 2)
    print(f"# chi(tang)={np.sqrt(chi2_t):.2f}  chi(cross)={np.sqrt(chi2_x):.2f}",
          file=sys.stderr)

    # NFW overlay at the best-fit mass from the emcee posterior, using
    # the data-derived n(z) for the beta moments.
    rproj = np.logspace(np.log10(0.3), np.log10(7.0), 100)
    moo = clmm.Modeling(massdef="critical", delta_mdef=500, halo_profile_model="nfw")
    moo.set_cosmo(cosmo)
    moo.set_concentration(args.nfw_c)
    moo.set_mass(nfw_mass_fit)
    gt_overlay = moo.eval_reduced_tangential_shear(
        rproj, z_cl, [beta_s, beta_s_sqr],
        z_src_info="beta", approx="order2",
    )
    overlay_label = (
        rf"NFW (best-fit) $\log M_{{500c}}={logM_med:.2f}$, "
        rf"$c={args.nfw_c:g}$, n(z) from data"
    )
    make_tangential_plot(
        bin_mids=bin_mids, tang_avg=tang_avg, cross_avg=cross_avg,
        tang_err=tang_err, cross_err=cross_err,
        rproj=rproj, gt_overlay=gt_overlay, overlay_label=overlay_label,
        out_path=out_tang, chi2_t=chi2_t, chi2_x=chi2_x,
    )


if __name__ == "__main__":
    main()
