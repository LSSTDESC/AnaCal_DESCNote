#!/usr/bin/env python

import os
import anacal
import fitsio
import healsparse
import numpy as np
from mpi4py import MPI
from astropy.io import ascii as astascii
from lsst.daf.butler import Butler


def get_image_mask_from_healsparse(exposure, hsmap):
    """
    Given an LSST exposure and a HealSparse mask file, return a 2D mask array.

    Parameters
    ----------
    exposure : lsst.afw.image.ExposureF
        The LSST exposure containing WCS info.
    hsmap : str
        HealSparse map

    Returns
    -------
    mask_array : np.ndarray (bool)
        True means the pixel is masked (i.e., bad), False is good.
    """

    wcs = exposure.getWcs()
    height, width = exposure.getDimensions()
    y0 = exposure.getY0()
    x0 = exposure.getX0()
    # Generate grid of pixel coordinates
    y_pix, x_pix = np.indices((height, width))
    x_flat = x_pix.ravel() + x0 + 0.0
    y_flat = y_pix.ravel() + y0 + 0.0
    ra, dec = wcs.pixelToSkyArray(y=y_flat, x=x_flat, degrees=True)
    mask_flat = hsmap.get_values_pos(ra, dec)
    return mask_flat.reshape((height, width))

# MPI init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Global config
obs_repo = "/repo/main"
obs_collection = "LSSTComCam/runs/DRP/DP1/v29_0_0_rc5/DM-49865"
skymap = "lsst_cells_v1"
version = "v6"

# Setup Butler
obs_butler = Butler(obs_repo, collections=obs_collection)
obs_registry = obs_butler.registry
t_skymap = obs_butler.get("skyMap", skymap=skymap)

if rank == 0:
    data_list = astascii.read("patches2.csv")
    data_array = np.array(
        [(int(row["tract"]), int(row["patch"])) for row in data_list]
    )
else:
    data_array = None

# Broadcast to all ranks
data_array = comm.bcast(data_array, root=0)

my_data_array = data_array[rank::size]
planes0 = ["BAD", "CR", "NO_DATA", "SAT", "UNMASKEDNAN"]
planes = ["BAD", "CR", "NO_DATA", "SAT", "UNMASKEDNAN"]

badplanes = {
    "g": planes0,
    "r": planes,
    "i": planes,
    "z": planes,
    "y": planes0,
}

hsmap = healsparse.HealSparseMap.read(
    "/sdf/home/a/amouroux/public_html/dp1_masks/masks_EDFS_comcam.hs"
)

for tract_id, patch_id in my_data_array:
    out_fname = f"mask2_{version}/{tract_id}-{patch_id}.fits"
    if os.path.isfile(out_fname):
        continue
    band = "i"
    ref = list(obs_registry.queryDatasets(
        "deep_coadd", skymap=skymap, patch=patch_id,
        tract=tract_id, band=band,
    ))
    if len(ref) == 0:
        print(tract_id, patch_id)
        continue
    ref = ref[0]
    exposure = obs_butler.get(ref)
    mask_array = get_image_mask_from_healsparse(exposure, hsmap)
    bitv = exposure.mask.getPlaneBitMask(badplanes[band])
    mask_array = mask_array | ((exposure.mask.array & bitv) != 0)
    for band in ["g", "r", "z", "y"]:
        try:
            ref = list(obs_registry.queryDatasets(
                "deep_coadd", skymap=skymap, patch=patch_id,
                tract=tract_id, band=band,
            ))[0]
            exposure = obs_butler.get(ref)
            bitv = exposure.mask.getPlaneBitMask(badplanes[band])
            mask_array = mask_array | ((exposure.mask.array & bitv) != 0)
        except:
            print("cannot do for %s" % band)
    mask_array = mask_array.astype(np.int16)
    if os.path.isfile(f"./brightStars/{tract_id}-{patch_id}.ecsv"):
        star_cat = np.array(
            astascii.read(f"./brightStars/{tract_id}-{patch_id}.ecsv")
        )
        anacal.mask.add_bright_star_mask(
            mask_array=mask_array,
            star_array=star_cat,
        )
    fitsio.write(out_fname, mask_array)
