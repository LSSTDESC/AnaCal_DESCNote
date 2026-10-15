#!/bin/bash
# Step 2/2 (NO MASK): TXPipe AnaCal analysis on the full ingested catalog, one z bin, no blinding -> data/anacal_dp2_nomask/  (image env: has sacc)
#SBATCH --account=m1727
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --time=12:00:00
#SBATCH --job-name=txpipe_anacal_nomask
#SBATCH --output=/pscratch/sd/x/xiangchl/dp2/txpipe_test/logs/run_analysis_nomask_%j.log
S=/global/homes/x/xiangchl/superonion/code/AnaCal_DESCNote/dp2/dp2-v0/txpipe/scripts   # pipeline/config files (this folder)
T=/pscratch/sd/x/xiangchl/dp2/txpipe_test
source ~/setup/setup_image.sh > /dev/null 2>&1   # before set -u: the setup uses unset variables
export PYTHONPATH=/global/u2/x/xiangchl/superonion/code/TXPipe:${PYTHONPATH:-}
set -euo pipefail
python -c 'import txpipe, treecorr, sacc, mpi4py'
export HDF5_USE_FILE_LOCKING=FALSE
export TMPDIR=/pscratch/sd/x/xiangchl/dp2/txpipe_test/tmp   # compute-node /tmp is RAM: TXPipe's repack writes a full copy of the catalog there
cd /global/u2/x/xiangchl/superonion/code/TXPipe
rm -f "$T"/data/anacal_dp2_nomask/twopoint_data_real_raw.sacc.checkpoint-*.pkl "$T"/data/anacal_dp2_nomask/inprogress_twopoint_data_real_raw.sacc.checkpoint-*.pkl   # never reuse old TwoPoint results
echo "=== analysis  $(date)"
ceci "$S/anacal_analysis_nomask.yml"
echo "=== analysis done  $(date)"
ls -la "$T/data/anacal_dp2_nomask/"
