# for running cell-based coadds and Metadetect
# scons is maybe not necessary everytime, but it doesn't hurt to run

export DRP_PIPE_DIR=/global/cfs/cdirs/lsst/groups/CL/A360_DP1/Metadetect/env/repos/v30/drp_pipe

# Galaxy / star truth-catalog directory used by xlens.simulator (e.g.
# OneDegSq.fits, flagship_*.fits, stars_*.fits).  galaxies.py falls back
# to cwd if CATSIM_DIR is unset, which makes BPS workers fail to find the
# files.
export CATSIM_DIR=/global/u2/x/xiangchl/superonion/code/catsim

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/v30.0.4.rc1/loadLSST-ext.sh
setup lsst_distrib -t v30_0_4_rc1
export OMP_NUM_THREADS=1

cd /global/cfs/cdirs/lsst/groups/CL/A360_DP1/Metadetect/env/repos/bps_parsl_sites
setup -r . -j
scons version

cd /global/cfs/cdirs/lsst/groups/CL/A360_DP1/Metadetect/env/repos/v30/drp_pipe # add cell-based coadds and Metadetect to pipeline
setup -r . -j
scons version

cd /global/cfs/cdirs/lsst/groups/CL/A360_DP1/Metadetect/env/repos/v30/drp_tasks # fixes bug to use visit_image
setup -r . -j
scons version
