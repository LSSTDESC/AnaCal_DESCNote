"""Export cell coadds and masks for transfer to another server.

This script:
1. Finds all cell coadd and mask files in the butler
2. Copies them to a local directory
3. Creates a new standalone butler repo that can be transferred

Usage:
    python export_data.py /path/to/output_dir
"""

import os
import sys
import shutil
import lsst.daf.butler as dafButler

BUTLER_CONFIG = "/global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml"
CELL_COLLECTIONS = [
    "u/mgorsuch/metadetect/a360_metadetect",
    "u/mgorsuch/metadetect/a360_extra_cells",
]
MASK_COLLECTION = "u/xiangchl/dp1/a360_anacal"
SKYMAP_NAME = "lsst_cells_v1"


def main(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Source butler
    src_butler = dafButler.Butler(
        BUTLER_CONFIG,
        collections=CELL_COLLECTIONS + [MASK_COLLECTION, "LSSTComCam/DP1"],
    )

    # Create a new butler repo at output_dir
    dafButler.Butler.makeRepo(output_dir)
    dst_butler = dafButler.Butler(output_dir, writeable=True)

    # Register the skymap
    skymap = src_butler.get("skyMap", skymap=SKYMAP_NAME)

    # Transfer cell coadds
    print("=== Cell Coadds ===")
    cell_refs = list(src_butler.registry.queryDatasets(
        "deep_coadd_cell_predetection",
        where=f"skymap='{SKYMAP_NAME}'",
    ))
    print(f"Found {len(cell_refs)} cell coadd refs")

    # Transfer masks
    print("\n=== Masks ===")
    mask_refs = list(src_butler.registry.queryDatasets(
        "deep_coadd_cell_systematics_mask",
        where=f"skymap='{SKYMAP_NAME}'",
        collections=[MASK_COLLECTION],
    ))
    print(f"Found {len(mask_refs)} mask refs")

    # Transfer using butler
    all_refs = cell_refs + mask_refs
    print(f"\nTotal refs to transfer: {len(all_refs)}")

    # Use butler transfer
    with dst_butler.export(directory=output_dir, format="yaml") as export:
        # Register dataset types
        for dt_name in ["deep_coadd_cell_predetection", "deep_coadd_cell_systematics_mask"]:
            dt = src_butler.registry.getDatasetType(dt_name)
            try:
                dst_butler.registry.registerDatasetType(dt)
            except Exception:
                pass

        # Export
        for ref in all_refs:
            export.saveDatasets(ref)

    print(f"\nExported to {output_dir}")
    print("Transfer this directory to the target server, then import with:")
    print(f"  butler import /path/to/new_repo {output_dir}/export.yaml --transfer copy")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} /path/to/output_dir")
        sys.exit(1)
    main(sys.argv[1])
