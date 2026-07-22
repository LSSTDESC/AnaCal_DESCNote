#!/bin/bash
# Reproduce yesterday's successful v3 bps env (CVMFS v30 + EUPS-setup
# bps_parsl_sites + drp_pipe + drp_tasks + local xlens editable on
# PYTHONPATH), then bps-submit the parsl.yaml in $1.
set -e

FIELD_DIR="$1"
if [[ -z "$FIELD_DIR" ]]; then
    echo "usage: $0 <field_step1_dir>" >&2
    exit 2
fi

source /cvmfs/sw.lsst.eu/almalinux-x86_64/lsst_distrib/v30.0.4.rc1/loadLSST-ext.sh > /dev/null 2>&1
setup lsst_distrib -t v30_0_4_rc1

# EUPS-setup each auxiliary repo from its own directory (setup -r . -j
# is the invocation yesterday's setup script used).
cd /global/cfs/cdirs/desc-cl/A360_DP1/Metadetect/env/repos/bps_parsl_sites
setup -r . -j > /dev/null 2>&1
cd /global/cfs/cdirs/desc-cl/A360_DP1/Metadetect/env/repos/v30/drp_pipe
setup -r . -j > /dev/null 2>&1
cd /global/cfs/cdirs/desc-cl/A360_DP1/Metadetect/env/repos/v30/drp_tasks
setup -r . -j > /dev/null 2>&1

# Pick up the editable xlens tree so measureCellCoadds + mergePatches
# see the current merge.py / add_magnitude_columns edits.
export PYTHONPATH=/global/homes/x/xiangchl/superonion/code/xlens:$PYTHONPATH
export OMP_NUM_THREADS=1

cd "$FIELD_DIR"
bps submit parsl.yaml
