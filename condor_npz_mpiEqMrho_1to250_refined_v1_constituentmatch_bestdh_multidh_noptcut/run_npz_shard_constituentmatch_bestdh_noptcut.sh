#!/bin/bash
set -euo pipefail

PROCESS_ID="${1}"

IDX=$(printf "%03d" "${PROCESS_ID}")
FILELIST="files_${IDX}"

OUTDIR="shard_${IDX}"
OUTPUT_PREFIX="dragon_mpiEqMrho_1to250_refined_constituentmatch_bestdh_multidh_noptcut_shard_${IDX}"
TARBALL="shard_${PROCESS_ID}_1to250_refined_constituentmatch_bestdh_multidh_noptcut_output.tgz"

BUILDER="build_dracon_complete_genmatch_dataset_multiproc_EOS_mpiEqMrho_constituentmatch_bestdh_multidh_noptcut.py"

INPUT_EOS_HOST="root://eosproject.cern.ch"
OUTPUT_EOS_HOST="root://eosproject.cern.ch"
EOS_BASE="/eos/project/d/dragon/ashrivas/DarkHadronMassReco/training_data/NPZ_mpiEqMrho_1to250_refined_v1_constituentmatch_bestdh_multidh_noptcut"
EOS_SHARDS="${EOS_BASE}/shards"
EOS_SHARD_DIR="${EOS_SHARDS}/shard_${IDX}"

FULL_NPZ="${OUTDIR}/${OUTPUT_PREFIX}_full.npz"
PARTIAL_NPZ="${OUTDIR}/${OUTPUT_PREFIX}_partial.npz"
NONE_NPZ="${OUTDIR}/${OUTPUT_PREFIX}_none.npz"


echo "=========================================================="
echo "Starting DRAGON NPZ constituent best-DH shard job"
echo "Host          : $(hostname)"
echo "Date          : $(date)"
echo "Scratch       : $(pwd)"
echo "Process ID    : ${PROCESS_ID}"
echo "Shard IDX     : ${IDX}"
echo "Filelist      : ${FILELIST}"
echo "Outdir        : ${OUTDIR}"
echo "Output prefix : ${OUTPUT_PREFIX}"
echo "Tarball       : ${TARBALL}"
echo "EOS shard dir : ${EOS_SHARD_DIR}"
echo "Input EOS host: ${INPUT_EOS_HOST}"
echo "Output EOS host: ${OUTPUT_EOS_HOST}"
echo "pT cut        : disabled"
echo "=========================================================="

echo "[INFO] Sourcing LCG environment"
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh
set -u

echo "[INFO] Checking Python/Numpy/Uproot/Awkward"
python3 - <<'PY'
import numpy as np
import uproot
import awkward as ak
print("numpy OK:", np.__version__, np.__file__)
print("uproot OK:", uproot.__version__)
print("awkward OK:", ak.__version__)
PY

echo "[INFO] Unpacking inputs"
tar -xzf npz_shard_inputs.tgz

echo "[INFO] Files in scratch:"
ls -lh

if [[ ! -f "${FILELIST}" ]]; then
    echo "[ERROR] Missing filelist: ${FILELIST}"
    echo "[DEBUG] Available files:"
    ls -lh
    exit 10
fi

if [[ ! -f "${BUILDER}" ]]; then
    echo "[ERROR] Missing builder script: ${BUILDER}"
    echo "[DEBUG] Available files:"
    ls -lh
    exit 11
fi

echo "[INFO] Number of ROOT files in shard:"
wc -l "${FILELIST}"

echo "[INFO] First few files:"
head "${FILELIST}"

mkdir -p "${OUTDIR}"

echo "[INFO] Running constituent best-DH builder"
python3 "${BUILDER}" \
  --filelist "${FILELIST}" \
  --eos-host "${INPUT_EOS_HOST}" \
  --outdir "${OUTDIR}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --keep-mode all \
  --workers 4

echo "[INFO] Checking outputs"
ls -lh "${OUTDIR}"

for f in "${FULL_NPZ}" "${PARTIAL_NPZ}" "${NONE_NPZ}"; do
    if [[ ! -s "${f}" ]]; then
        echo "[ERROR] Expected NPZ was not created or is empty: ${f}"
        exit 20
    fi
    echo "[OK] Found ${f}"
done

echo "[INFO] NPZ key/shape check"
python3 - <<PY
import numpy as np
for path in ["${FULL_NPZ}", "${PARTIAL_NPZ}", "${NONE_NPZ}"]:
    d = np.load(path)
    print("\n", path)
    print("  keys:", d.files)
    print("  X:", d["X"].shape)
    print("  masses:", d["masses"].shape)
    print("  kinematics:", d["kinematics"].shape)
    print("  min pt:", float(d["kinematics"][:, 0].min()) if len(d["kinematics"]) else "EMPTY")
    print("  containment_category:", np.unique(d["containment_category"], return_counts=True))
    print("  n_visible_dark_hadrons:", np.unique(d["n_visible_dark_hadrons"], return_counts=True))
    print("  dark_hadron_p4:", d["dark_hadron_p4"].shape)
PY

echo "[INFO] Uploading shard NPZs to EOS"
xrdfs "${OUTPUT_EOS_HOST}" mkdir -p "${EOS_SHARD_DIR}"

for f in "${FULL_NPZ}" "${PARTIAL_NPZ}" "${NONE_NPZ}"; do
    base=$(basename "${f}")
    echo "  xrdcp ${base} -> ${EOS_SHARD_DIR}/${base}"
    xrdcp -f "${f}" "${OUTPUT_EOS_HOST}/${EOS_SHARD_DIR}/${base}"
    xrdfs "${OUTPUT_EOS_HOST}" stat "${EOS_SHARD_DIR}/${base}"
done

for meta in "${OUTDIR}/root_file_index_map.csv" "${OUTDIR}/${OUTPUT_PREFIX}_summary.csv"; do
    if [[ -s "${meta}" ]]; then
        base=$(basename "${meta}")
        echo "  xrdcp ${base} -> ${EOS_SHARD_DIR}/${base}"
        xrdcp -f "${meta}" "${OUTPUT_EOS_HOST}/${EOS_SHARD_DIR}/${base}"
    fi
done

echo "[INFO] Packing shard output"
tar -czf "${TARBALL}" "${OUTDIR}"

ls -lh "${TARBALL}"

echo "=========================================================="
echo "[SUCCESS] Finished constituent best-DH shard ${IDX}"
echo "End time: $(date)"
echo "=========================================================="
