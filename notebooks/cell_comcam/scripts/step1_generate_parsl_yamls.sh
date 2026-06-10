#!/bin/bash
# Generate per-field parsl.yaml under each field directory by running
# step1_build_tract_patch_query.py with radius=1.0 deg around the cluster
# center and pasting the resulting dataQuery block into the parsl payload.
#
# Input collection is u/pecom/dp1/coadds (predetection-bearing collection).

set -euo pipefail

BASE=/global/u2/x/xiangchl/superonion/code/AnaCal_DESCNote/notebooks/cell_comcam
PIPELINE_YAML=$BASE/configs/measure_pipeline.yaml
QUERY_SCRIPT=$BASE/scripts/step1_build_tract_patch_query.py
COLLECTION=u/pecom/dp1/coadds,refcats/DM-39298/gaia_dr3_20230707
FILTER_COLLECTION=u/pecom/dp1/coadds
RADIUS=2.0

# field RA Dec BANDS_REQUIRED
# a360 is ComCam (griz only); edfs/ecdfs are full ugrizy fields.
FIELDS=(
  "a360  37.86   6.98  g,r,i,z"
  "edfs  59.10 -48.73  u,g,r,i,z,y"
  "ecdfs 53.13 -28.10  u,g,r,i,z,y"
)

for entry in "${FIELDS[@]}"; do
  read -r field ra dec bands <<< "$entry"
  out_dir=$BASE/$field/step1
  mkdir -p "$out_dir"

  echo "=== $field  (ra=$ra, dec=$dec, radius=$RADIUS deg, bands=$bands) ==="
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

pipelineYaml: "$PIPELINE_YAML"
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
        # NERSC debug qos: whole-node exclusive, 30-min cap, max 5 submitted.
        class: bps_parsl_sites.SlurmWorkQueue
        nodes_per_block: 1
        cores_per_node: 256
        qos: debug
        constraint: cpu
        exclusive: true
        walltime: "0:30:00"
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
