import fitsio
import lsst.geom as geom
import numpy as np
import os
import xlens
from astropy.io import ascii as astascii
from lsst.daf.butler import Butler
from xlens.processor.anacal import AnacalConfig, AnacalTask

# Load a recent version of DP1 for ComCam
obs_repo = "/repo/main"
obs_collection = "LSSTComCam/runs/DRP/DP1/v29_0_0_rc5/DM-49865"
skymap = "lsst_cells_v1"

obs_butler = Butler(
    obs_repo,
    collections=obs_collection,
)
obs_registry = obs_butler.registry
data_list = astascii.read("patches.csv")
t_skymap = obs_butler.get("skyMap", skymap=skymap)

config = AnacalConfig()
config.sigma_arcsec = 0.50
config.snr_min = 10.0
config.num_epochs = 8
config.force_size = False
config.npix = 32
config.do_noise_bias_correction = True

config.badMaskPlanes = [
    "BAD",
    "CR",
    "CROSSTALK",
    "NO_DATA",
    "REJECTED",
    "SAT",
    "SUSPECT",
    "UNMASKEDNAN",
    "SENSOR_EDGE",
    "STREAK",
    "VIGNETTED",
    "INTRP",
    "EDGE",
    "CLIPPED",
    "INEXACT_PSF",
]

anacal_task = AnacalTask(config=config)

ndata = len(data_list)
for data_id in range(ndata):
    tract_id = data_list["tract"][data_id]
    patch_id = data_list["patch"][data_id]
    out_fname = "noise_corrs/%d-%d.fits" % (tract_id, patch_id)
    if os.path.isfile(out_fname):
        continue
    tract = t_skymap[tract_id]
    wcs = tract.getWcs()
    patch = tract[patch_id]
    ref = list(
        obs_registry.queryDatasets(
            "deep_coadd", skymap=skymap, patch=patch_id, tract=tract_id, band="i"
        )
    )[0]
    exposure = obs_butler.get(ref)
    star_cat = np.array(
        astascii.read("./brightStars/%d-%d.ecsv" % (tract_id, patch_id))
    )
    mask = anacal_task.prepare_data(
        exposure=exposure,
        seed=tract_id * 1000 + patch_id,
        noise_corr=None,
        band="i",
        skyMap=t_skymap,
        tract=tract_id,
        patch=patch_id,
        star_cat=star_cat,
    )["mask_array"]
    npix = 49
    noise_array = np.asarray(
        exposure.image.array,
        dtype=np.float32,
    )[700:3500, 700:3500]

    window_array = np.asarray(
        (mask == 0) & (exposure.mask.array == 0) &
        (exposure.image.array ** 2.0 < exposure.variance.array * 10),
        dtype=np.float32,
    )[700:3500, 700:3500]
    noise_array[~window_array.astype(bool)] = 0.0
    pad_width = ((10, 10), (10, 10))  # ((top, bottom), (left, right))
    window_array = np.pad(
        window_array,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )
    noise_array = np.pad(
        noise_array,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0,
    )
    ny, nx = window_array.shape

    npixl = int(npix // 2)
    npixr = int(npix // 2 + 1)
    noise_corr = np.fft.fftshift(
        np.fft.ifft2(np.abs(np.fft.fft2(noise_array)) ** 2.0)
    ).real[
        ny // 2 - npixl : ny // 2 + npixr, nx // 2 - npixl : nx // 2 + npixr
    ]
    window_corr = np.fft.fftshift(
        np.fft.ifft2(np.abs(np.fft.fft2(window_array)) ** 2.0)
    ).real[
        ny // 2 - npixl : ny // 2 + npixr, nx // 2 - npixl : nx // 2 + npixr
    ]
    noise_corr = noise_corr / window_corr
    fitsio.write(
        out_fname,
        noise_corr,
    )
    del tract, patch, ref, exposure, star_cat, mask, window_array, noise_array
