#!/bin/bash
# Generate per-field parsl.yaml under each field directory by running
# step1_build_tract_patch_query.py with radius=1.0 deg around the cluster
# center and pasting the resulting dataQuery block into the parsl payload.
#
# Input collection is u/pecom/dp1/coadds (predetection-bearing collection).

set -euo pipefail

BASE=/global/u2/x/xiangchl/superonion/code/AnaCal_DESCNote/notebooks/cell_comcam
QUERY_SCRIPT=$BASE/scripts/step1_build_tract_patch_query.py
COLLECTION=u/pecom/dp1/coadds,refcats/DM-39298/gaia_dr3_20230707
FILTER_COLLECTION=u/pecom/dp1/coadds
RADIUS=2.0
# Pipeline subset: blank = run everything (incl. mergePatches);
# "#labels" restricts pipetask to the listed labels. Current default
# skips the per-tract merge.
PIPELINE_SUBSET="#buildCellSystematics,measureCellCoadds,photoZ"

# field RA Dec BANDS_REQUIRED PIPELINE_TAG
# a360 (ComCam griz only)        -> 4-band pipeline (configs/measure_pipeline_4bands.yaml)
# edfs / ecdfs (ugrizy complete) -> 6-band pipeline (configs/measure_pipeline_6bands.yaml)
FIELDS=(
  "a360  37.86   6.98  g,r,i,z      4bands"
  "edfs  59.10 -48.73  u,g,r,i,z,y  6bands"
  "ecdfs 53.13 -28.10  u,g,r,i,z,y  6bands"
)

for entry in "${FIELDS[@]}"; do
  read -r field ra dec bands pipeline_tag <<< "$entry"
  out_dir=$BASE/results/$field/step1
  mkdir -p "$out_dir"
  PIPELINE_YAML=$BASE/configs/measure_pipeline_${pipeline_tag}.yaml

  echo "=== $field  (ra=$ra, dec=$dec, radius=$RADIUS deg, "\
       "bands=$bands, pipeline=$pipeline_tag) ==="
  # Capture dataQuery block, re-indent by 2 spaces to sit under `payload:`.
  # If the python helper exits non-zero (e.g. 0 patches survived the band
  # filter), skip the parsl.yaml write for this field instead of bailing
  # out the whole script.
  if ! dataquery=$(python "$QUERY_SCRIPT" \
        --ra "$ra" --dec "$dec" --radius "$RADIUS" \
        --bands "$bands" \
        --collection "$FILTER_COLLECTION" | sed 's/^/  /'); then
    echo "  SKIP $field: no patches kept; parsl.yaml not written"
    continue
  fi

  cat > "$out_dir/parsl.yaml" <<EOF
LSST_VERSION: v30.0.4.rc1
instrumnet: LSSTComCam

pipelineYaml: "${PIPELINE_YAML}${PIPELINE_SUBSET}"
wmsServiceClass: lsst.ctrl.bps.parsl.ParslService
computeSite: work_queue

pipetask:
  cmdlineArgs: ["--register-dataset-types"]

site:
    local:
        class: lsst.ctrl.bps.parsl.sites.Local
        cores: 1
        monitorEnable: true
        monitorFilename: runinfo/monitoring.db
    work_queue:
        # Debug qos by default: 1 node × 256 cores, 30-min walltime,
        # near-zero slurm queue wait. All three fields fit in this
        # envelope:
        #
        #   a360  (88 patches, 4-band): ~9 min wall, ~0.15 node-hr
        #   edfs  (52 patches, 6-band): ~10 min wall, ~0.17 node-hr
        #   ecdfs (70 patches, 6-band): ~12 min wall expected
        #
        # Caveat: with 256-way concurrency on 1 node, ~50+ python
        # workers hit cvmfs simultaneously at startup. If the
        # compute node's cvmfs cache is cold (no recent LSST jobs
        # ran on it), the FUSE daemon serialises and workers wedge
        # on `wchan: request_wait_answer`. Fixes when that happens:
        # cancel + resubmit (different node, usually warm) OR fall
        # back to the 2-node / regular-qos block commented below.
        class: bps_parsl_sites.SlurmWorkQueue
        nodes_per_block: 1
        cores_per_node: 256
        qos: debug
        constraint: cpu
        exclusive: true
        walltime: "0:30:00"
        # --- regular-qos fallback (uncomment if debug wedges on
        #     cvmfs FUSE):
        # nodes_per_block: 2
        # qos: regular
        # walltime: "2:00:00"
        account: m1727
        provider_options:
          init_blocks: 1
          min_blocks: 0
          max_blocks: 1
        worker_options: "--cores=256"
        monitorEnable: true
        monitorFilename: runinfo/monitoring.db

payload:
  payloadName: dp1-v2/${field}_anacal2
  butlerConfig: /global/cfs/cdirs/lsst/production/gen3/rubin/DP1/repo/butler.yaml
  inCollection: $COLLECTION
$dataquery
EOF
  echo "  wrote $out_dir/parsl.yaml"
done
