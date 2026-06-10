"""Shared constants + helpers for the step2/step3 scripts.

The aim is to keep cross-script duplication low without introducing
tight coupling between step2 (per-collection plots) and step3 (cluster
analysis). Things that genuinely live in one script (selection cuts,
per-plot styling, NFW fitting, ...) stay where they are; only the
constants and small helpers everyone re-implements are gathered here.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import lsst.geom as geom
import numpy as np
from lsst.daf.butler import Butler


# Butler / skymap defaults.
DEFAULT_REPO = "/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml"
DEFAULT_SKYMAP = "lsst_cells_v1"

# Output convention: <FIELDS_ROOT>/<field>/<subdir>/<name>.
FIELDS_ROOT = Path(
    "/global/u2/x/xiangchl/superonion/code/AnaCal_DESCNote/notebooks/cell_comcam"
)


def open_butler(collections, repo=DEFAULT_REPO):
    """Thin wrapper so call sites don't repeat the repo path."""
    return Butler(repo, collections=collections)


def find_tracts(skymap, ra, dec, radius, grid_step=0.1):
    """NxN sphere-point sweep of the (ra +/- radius) box -> overlapping tracts.

    Mirrors the trick we use in step1: a corner-only sample misses
    tracts wholly inside a large search box, so we walk a dense grid.
    Spacing defaults to ~half a tract on this skymap (~1 deg wide).
    """
    n_grid = max(2, int(math.ceil(2 * radius / grid_step)) + 1)
    step = 2 * radius / (n_grid - 1)
    radec = [
        geom.SpherePoint(ra - radius + i * step, dec - radius + j * step,
                         geom.degrees)
        for i in range(n_grid) for j in range(n_grid)
    ]
    return sorted({ti.getId() for ti, _ in skymap.findTractPatchList(radec)})


def resolve_field_out(field: Optional[str], subdir: str, name: str,
                      override: Optional[str] = None) -> Optional[str]:
    """Pick output path. Explicit override wins; else <FIELDS_ROOT>/<field>/<subdir>/<name>.

    Returns ``None`` if neither override nor field is given (caller
    can fall back to plt.show()). Parent dir is auto-created when a
    concrete path is returned.
    """
    if override is not None:
        out = Path(override)
    elif field is not None:
        out = FIELDS_ROOT / field / subdir / name
    else:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)


def compute_mag(values, mag_zero=31.4):
    """``mag_zero - 2.5 * log10(values)``; non-positive / non-finite -> NaN."""
    arr = np.asarray(values, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.isfinite(arr) & (arr > 0)
        out[m] = mag_zero - 2.5 * np.log10(arr[m])
    return out


# Default plot-style rc-context, used by most step2/step3 figures.
DEFAULT_STYLE_RC = {
    "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.linewidth": 1.2,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
}
