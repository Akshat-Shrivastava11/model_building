#!/bin/bash
set -euo pipefail

EOS_HOST="root://eosproject.cern.ch"
EOS_SOURCE="/eos/project/d/dragon/ashrivas/DarkHadronMassReco/SVJ_mpiEqMrho_1to250_refined_v1"
NSHARDS="${1:-150}"

echo "[INFO] Preparing refined 1to250 NPZ shard inputs"
echo "[INFO] EOS host   : ${EOS_HOST}"
echo "[INFO] EOS source : ${EOS_SOURCE}"
echo "[INFO] Shards     : ${NSHARDS}"

mkdir -p logs_refined_1to250 logs_merge_eos_refined_1to250 logs_merge_category_refined_1to250

echo "[INFO] Listing ROOT files from EOS"
xrdfs "${EOS_HOST}" ls -R "${EOS_SOURCE}" \
  | grep '/events\.root$' \
  | sort > eos_refined_1to250_events.txt

NFILES=$(wc -l < eos_refined_1to250_events.txt)
echo "[INFO] Found ${NFILES} ROOT files"

if [[ "${NFILES}" -eq 0 ]]; then
    echo "[ERROR] No ROOT files found under ${EOS_SOURCE}"
    exit 1
fi

rm -f files_[0-9][0-9][0-9]

python3 - "${NSHARDS}" <<'PY'
import sys
from pathlib import Path

nshards = int(sys.argv[1])
paths = [
    line.strip()
    for line in Path("eos_refined_1to250_events.txt").read_text().splitlines()
    if line.strip()
]

base = len(paths) // nshards
extra = len(paths) % nshards
start = 0
for idx in range(nshards):
    size = base + (1 if idx < extra else 0)
    stop = start + size
    shard_paths = paths[start:stop]
    Path(f"files_{idx:03d}").write_text("".join(f"{p}\n" for p in shard_paths))
    print(f"files_{idx:03d}: {len(shard_paths)} files")
    start = stop
PY

echo "[INFO] Repacking npz_shard_inputs.tgz"
tar -czf npz_shard_inputs.tgz \
  build_dracon_complete_genmatch_dataset_multiproc_EOS_mpiEqMrho_constituentmatch_bestdh_multidh_noptcut.py \
  files_[0-9][0-9][0-9]

ls -lh npz_shard_inputs.tgz
echo "[DONE] Inputs are ready"
