#!/usr/bin/env python
"""
Build a parsl.yaml-style `dataQuery` block listing every (tract, patch) within
a circular region around (ra, dec) that has a `deep_coadd_cell_predetection`
dataset in `u/pecom/dp1/coadds`.

Usage:
    python build_tract_patch_query.py --ra 37.865017 --dec 6.982205 --radius 0.5

Pipes straight into a parsl.yaml payload.dataQuery.
"""

import argparse
import math
import sys
from collections import defaultdict

import lsst.geom as geom
from lsst.daf.butler import Butler


DEFAULT_REPO = "/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml"
DEFAULT_SKYMAP = "lsst_cells_v1"
DEFAULT_COLLECTION = "u/pecom/dp1/coadds"
DEFAULT_DATASET = "deep_coadd_cell_predetection"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ra", type=float, required=True, help="center RA in degrees")
    p.add_argument("--dec", type=float, required=True, help="center Dec in degrees")
    p.add_argument("--radius", type=float, required=True, help="half-side of search square in degrees")
    p.add_argument("--repo", default=DEFAULT_REPO, help="butler repo (default: DP1)")
    p.add_argument("--skymap", default=DEFAULT_SKYMAP, help="skymap name (default: lsst_cells_v1)")
    p.add_argument("--collection", default=DEFAULT_COLLECTION,
                   help=f"collection to check for {DEFAULT_DATASET} (default: {DEFAULT_COLLECTION})")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"dataset type to require per patch (default: {DEFAULT_DATASET})")
    p.add_argument("--bands", default="g,r,i,z", help="bands clause for dataQuery (default: g,r,i,z)")
    return p.parse_args()


def find_tract_patches(skymap, ra, dec, radius, grid_step=0.1):
    """NxN grid of SpherePoints across the (ra +/- radius, dec +/- radius) box.

    A 4-corner sample misses tracts that lie wholly inside a large search box
    (the 4 corners then fall in outer-ring tracts only). Use a dense grid with
    spacing <= ~half a tract so every tract intersecting the box gets hit.
    """
    n_grid = max(2, int(math.ceil(2 * radius / grid_step)) + 1)
    step = 2 * radius / (n_grid - 1)
    radec = [
        geom.SpherePoint(ra - radius + i * step, dec - radius + j * step, geom.degrees)
        for i in range(n_grid) for j in range(n_grid)
    ]
    print(f"# sampling {n_grid}x{n_grid}={n_grid * n_grid} grid points "
          f"(spacing {step:.3f} deg) across the search box", file=sys.stderr)
    tracts_and_patches = skymap.findTractPatchList(radec)

    tp_dict = defaultdict(set)
    for tract_info, patch_list in tracts_and_patches:
        tid = tract_info.getId()
        for p in patch_list:
            tp_dict[tid].add(p.sequential_index)
    return {t: sorted(ps) for t, ps in tp_dict.items()}


def filter_by_dataset(butler, tp_dict, dataset_type, skymap_name, required_bands):
    """Keep only (tract, patch) entries with refs for ALL required bands.

    One butler query per tract (not per patch). Each query returns every
    (patch, band) tuple for that tract; the band-completeness check is
    done in Python. This is ~10x faster than querying per patch.

    Partial-band patches are dropped — they would otherwise sail through
    BuildCellSystematicsTask / MeasureCellCoaddsPipe and surface as
    runtime band-mismatch errors.
    """
    required = set(required_bands)
    kept = defaultdict(set)
    candidate_patch_set = {
        tid: set(patches) for tid, patches in tp_dict.items()
    }
    for tract_id, patches in tp_dict.items():
        refs = list(butler.registry.queryDatasets(
            dataset_type,
            where="skymap=:skymap_name AND tract=:tract_id",
            bind={"skymap_name": skymap_name, "tract_id": tract_id},
        ))
        patch_bands: dict[int, set] = defaultdict(set)
        for r in refs:
            patch_bands[r.dataId["patch"]].add(r.dataId["band"])
        for patch_id in patches:
            if required.issubset(patch_bands.get(patch_id, set())):
                kept[tract_id].add(patch_id)
    return {t: sorted(ps) for t, ps in sorted(kept.items())}


def format_dataquery(kept, skymap_name, bands_csv):
    bands_clause = ",".join(f"'{b.strip()}'" for b in bands_csv.split(","))
    lines = []
    lines.append("dataQuery: >")
    lines.append(f"  (skymap='{skymap_name}')")
    lines.append(f"  AND (band in ({bands_clause}))")
    lines.append("  AND (")
    parts = []
    for tract_id, patches in kept.items():
        patches_csv = ",".join(str(p) for p in patches)
        parts.append(f"(tract={tract_id} AND patch in ({patches_csv}))")
    for i, part in enumerate(parts):
        prefix = "    " if i == 0 else "    OR "
        lines.append(prefix + part)
    lines.append("  )")
    return "\n".join(lines)


def main():
    args = parse_args()
    butler = Butler(args.repo, collections=args.collection)
    skymap = butler.get("skyMap", skymap=args.skymap)

    tp_dict = find_tract_patches(skymap, args.ra, args.dec, args.radius)
    print(f"# skymap candidates around (ra,dec)=({args.ra},{args.dec}), radius={args.radius} deg:",
          file=sys.stderr)
    for t, ps in tp_dict.items():
        print(f"#   tract={t}: {len(ps)} patches", file=sys.stderr)

    required_bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    kept = filter_by_dataset(
        butler, tp_dict, args.dataset, args.skymap, required_bands,
    )
    print(f"# after filtering on {args.dataset} in {args.collection} "
          f"(requiring ALL bands {required_bands}):", file=sys.stderr)
    for t, ps in kept.items():
        print(f"#   tract={t}: {len(ps)} patches", file=sys.stderr)
    n_total = sum(len(ps) for ps in kept.values())
    print(f"# total kept: {n_total} patches across {len(kept)} tracts", file=sys.stderr)

    if not kept:
        print("# no (tract, patch) survived the dataset filter — check --collection / --dataset",
              file=sys.stderr)
        sys.exit(1)

    print(format_dataquery(kept, args.skymap, args.bands))


if __name__ == "__main__":
    main()
