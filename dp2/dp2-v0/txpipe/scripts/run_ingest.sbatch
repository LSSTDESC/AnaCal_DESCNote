#!/bin/bash
# Step 1/2: TXPipe AnaCal ingest of all DP2 tracts -> data/dp2_fullcatalogs/  (DP2 stack env: needs the butler)
#SBATCH --account=m1727
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=txpipe_anacal_ingest
#SBATCH --output=/pscratch/sd/x/xiangchl/dp2/txpipe_test/logs/run_ingest_%j.log
S=/global/homes/x/xiangchl/superonion/code/AnaCal_DESCNote/dp2/dp2-v0/txpipe/scripts   # pipeline/config files (this folder)
T=/pscratch/sd/x/xiangchl/dp2/txpipe_test
source ~/setup/setup_lsst_dp2.bash > /dev/null 2>&1   # before set -u: the stack setup uses unset variables
set -euo pipefail
python -c 'import txpipe, lsst.daf.butler'
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/pscratch/sd/x/xiangchl/dp2/txpipe_test/tmp   # compute-node /tmp is RAM: TXPipe's repack writes a full copy of the catalog there
cd /global/u2/x/xiangchl/superonion/code/TXPipe
echo "=== ingest  $(date)"
ceci "$S/anacal_ingestion.yml"
echo "=== ingest done  $(date)"
ls -la "$T/data/dp2_fullcatalogs/"
