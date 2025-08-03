#!/usr/bin/env python

import os
import fitsio
import numpy as np
from mpi4py import MPI
from astropy.io import ascii as astascii
from lsst.daf.butler import Butler
from xlens.process_pipe.anacal_detect import (
    AnacalDetectPipeConfig,
    AnacalDetectPipe,
)
from xlens.process_pipe.anacal_force import (
    AnacalForcePipe,
    AnacalForcePipeConfig,
)
from numpy.lib import recfunctions as rfn


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
    data_list = astascii.read("patches.csv")
    data_array = np.array(
        [(int(row["tract"]), int(row["patch"])) for row in data_list]
    )
else:
    data_array = None

# Broadcast to all ranks
data_array = comm.bcast(data_array, root=0)

my_data_array = data_array[rank::size]

config = AnacalDetectPipeConfig()
config.anacal.sigma_arcsec = 0.50
config.anacal.snr_min = 10.0
config.anacal.num_epochs = 8
config.anacal.force_size = False
config.anacal.do_noise_bias_correction = True
config.anacal.validate_psf = True
anacal_task = AnacalDetectPipe(config=config)

config = AnacalForcePipeConfig()
config.anacal.num_epochs = 8
config.anacal.do_noise_bias_correction = True
config.fpfs.do_noise_bias_correction = True
config.fpfs.npix = 64
config.fpfs.sigma_arcsec1 = 0.70
# Task and preparation
force_task = AnacalForcePipe(config=config)

for tract_id, patch_id in my_data_array:
    out_fname = f"catalog_{version}/detect_{tract_id}-{patch_id}.fits"
    if os.path.isfile(out_fname):
        continue
    mask_fname = f"mask/{tract_id}-{patch_id}.fits"
    mask_array = fitsio.read(mask_fname)
    ref = list(obs_registry.queryDatasets(
        "deep_coadd", skymap=skymap, patch=patch_id,
        tract=tract_id, band="i"
    ))[0]
    exposure = obs_butler.get(ref)
    data = anacal_task.anacal.prepare_data(
        exposure=exposure,
        seed=(tract_id * 1000 + patch_id) * 5 + 2,
        noise_corr=None,
        band="i",
        skyMap=t_skymap,
        tract=tract_id,
        patch=patch_id,
        mask_array=mask_array,
    )
    det = anacal_task.anacal.run(**data)
    mask = (det["mask_value"] < 40) & (det["is_primary"])
    det = det[mask]
    print(f"[Rank {rank}] {tract_id}-{patch_id} N={len(det)}")
    fitsio.write(out_fname, det, clobber=True)
    del exposure

    out_fname2 = f"catalog_{version}/force_{tract_id}-{patch_id}.fits"
    catalog = []
    det["a1"] = 0.3
    det["a2"] = 0.3
    det["da1_dg1"] = 0.0
    det["da1_dg2"] = 0.0
    det["da2_dg1"] = 0.0
    det["da2_dg2"] = 0.0
    for ib, band in enumerate(["g", "r", "i", "z"]):
        ref = list(obs_registry.queryDatasets(
            "deep_coadd", skymap=skymap, patch=patch_id,
            tract=tract_id, band=band,
        ))[0]
        exposure = obs_butler.get(ref)
        seed = (tract_id * 1000 + patch_id) * 5 + ib
        catalog.append(
            force_task.run_one_band(
                exposure=exposure,
                detection=det,
                band=band,
                seed=seed,
                noise_corr=None,
                skyMap=t_skymap,
                tract=tract_id,
                patch=patch_id,
                mask_array=mask_array,
            )
        )
    catalog = rfn.merge_arrays(catalog, flatten=True)
    fitsio.write(out_fname2, catalog, clobber=True)
