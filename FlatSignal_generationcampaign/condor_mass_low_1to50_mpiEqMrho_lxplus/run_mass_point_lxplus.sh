#!/bin/bash
set -euo pipefail

POINT_ID="${1}"
POINT_FILE="${2}"
EVENTS="${3}"
EOS_OUT_BASE="${4}"
EOS_HOST="${5:-root://eoscms.cern.ch}"

echo "=========================================================="
echo "Starting refined SVJ mpi=mrho production point on lxplus"
echo "Host              : $(hostname)"
echo "Date              : $(date)"
echo "Initial dir       : $(pwd)"
echo "CONDOR scratch    : ${_CONDOR_SCRATCH_DIR:-unknown}"
echo "POINT_ID          : ${POINT_ID}"
echo "POINT_FILE        : ${POINT_FILE}"
echo "EVENTS            : ${EVENTS}"
echo "EOS_OUT_BASE      : ${EOS_OUT_BASE}"
echo "EOS_HOST          : ${EOS_HOST}"
echo "=========================================================="

echo "[INFO] Files transferred into scratch"
ls -lah

echo "[INFO] Unpacking model_building_production.tgz"
mkdir -p work
tar -xzf model_building_production.tgz -C work
cd work

echo "[INFO] Repo contents inside Condor scratch"
pwd
ls -lah

LINE="$(awk -v id="${POINT_ID}" '($1 !~ /^#/ && $1 == id) {print $0}' "../${POINT_FILE}")"
if [[ -z "${LINE}" ]]; then
    echo "[ERROR] Could not find POINT_ID=${POINT_ID} in ${POINT_FILE}"
    exit 10
fi

read -r PID MRHO_IN MPI_IN SEED REPEAT MASS_INDEX MASS_LABEL <<< "${LINE}"

if [[ "${MRHO_IN}" != "${MPI_IN}" ]]; then
    echo "[ERROR] Mass point is not mpi=mrho: mrho=${MRHO_IN}, mpi=${MPI_IN}"
    exit 11
fi

MMED="2000"
RINV="0.3"
MASS="${MRHO_IN}"
MRHO="${MASS}"
MPI="${MASS}"
TAG=$(printf "point_%06d_mpiEqMrho_%s_rep%02d" "${PID}" "${MASS_LABEL}" "${REPEAT}")
LOCAL_OUT="SVJ_${TAG}"
EOS_OUT_DIR="${EOS_OUT_BASE}/${TAG}"
EOS_OUT_URL="${EOS_HOST}/${EOS_OUT_DIR}"

echo "[INFO] PID         = ${PID}"
echo "[INFO] CMS model   = generated from configs/model_cms.py"
echo "[INFO] MASS        = ${MASS} (mpi = mrho)"
echo "[INFO] MRHO        = ${MRHO}"
echo "[INFO] MPI         = ${MPI}"
echo "[INFO] SEED        = ${SEED}"
echo "[INFO] REPEAT      = ${REPEAT}"
echo "[INFO] MASS_INDEX  = ${MASS_INDEX}"
echo "[INFO] LOCAL_OUT   = ${LOCAL_OUT}"
echo "[INFO] EOS_OUT_DIR = ${EOS_OUT_DIR}"
echo "[INFO] EOS_OUT_URL = ${EOS_OUT_URL}"

echo "[INFO] Sourcing init.sh"
set +u
source init.sh
set -u

echo "[INFO] Running generation"
rm -rf "${LOCAL_OUT}"
mkdir -p "${LOCAL_OUT}"

PYTHIA_SEED=$((SEED % 900000000))
if [[ "${PYTHIA_SEED}" -le 0 ]]; then
    PYTHIA_SEED=1
fi

cat > pythia_seed_${PID}.txt <<EOF
Random:setSeed = on
Random:seed = ${PYTHIA_SEED}
EOF

CONFIG_FILE="configs/generated_model_cms_mpiEqMrho_${PID}.py"
CONFIG_OBJ="point_${PID}"
read -r SCALE MQ < <(python3 - "${CONFIG_FILE}" "${CONFIG_OBJ}" "${MASS}" "${MMED}" "${RINV}" <<'PYEOF'
import sys
from svjHelper import scale_cms

config_file, config_obj, mass_s, mmed_s, rinv_s = sys.argv[1:6]
mass = float(mass_s)
mmed = float(mmed_s)
rinv = float(rinv_s)
scale = scale_cms(mpi=mass)
mq = mass / 2.0

with open(config_file, "w") as handle:
    handle.write(
        "from magiconfig import MagiConfig\n"
        "from svjHelper import scale_cms, gchi_lhcdm\n\n"
        "config = MagiConfig()\n"
        f"config.{config_obj} = MagiConfig()\n"
        f"config.{config_obj}.channel = 's'\n"
        f"config.{config_obj}.mmed = {mmed:g}\n"
        f"config.{config_obj}.Nc = 2\n"
        f"config.{config_obj}.Nf = 2\n"
        f"config.{config_obj}.mpi = {mass:.12g}\n"
        f"config.{config_obj}.mrho = config.{config_obj}.mpi\n"
        f"config.{config_obj}.scale = scale_cms(mpi=config.{config_obj}.mpi)\n"
        f"config.{config_obj}.mq = config.{config_obj}.mpi / 2.\n"
        f"config.{config_obj}.pvector = 0.75\n"
        f"config.{config_obj}.rinv = {rinv:.12g}\n"
        f"config.{config_obj}.spectrum = 'cms'\n"
        f"config.{config_obj}.gq = 0.25\n"
        f"config.{config_obj}.gchi = gchi_lhcdm(gDM=1.0, Nc=config.{config_obj}.Nc, Nf=config.{config_obj}.Nf)\n"
    )

print(f"{scale:.12g} {mq:.12g}")
PYEOF
)

echo "[INFO] CONFIG_FILE = ${CONFIG_FILE}"
echo "[INFO] CONFIG_OBJ  = config.${CONFIG_OBJ}"
echo "[INFO] SCALE       = ${SCALE}"
echo "[INFO] MQ          = ${MQ}"

./run_model helper -C "${CONFIG_FILE}" -O "config.${CONFIG_OBJ}" \
    --steps pythia delphes \
    --events "${EVENTS}" \
    --pythia "pythia_seed_${PID}.txt" cards/CMS_Common.txt cards/CMS_Tune_CP5.txt \
    --dir "${LOCAL_OUT}" \
    --quiet

echo "[INFO] Local output files"
find "${LOCAL_OUT}" -type f -printf "%p %s bytes\n" || true

echo "[INFO] Creating EOS output directory"
xrdfs "${EOS_HOST}" mkdir -p "${EOS_OUT_DIR}"

echo "[INFO] Copying outputs to EOS"
find "${LOCAL_OUT}" -type f | while read -r FILE; do
    REL="${FILE#${LOCAL_OUT}/}"
    EOS_SUBDIR="$(dirname "${EOS_OUT_DIR}/${REL}")"
    xrdfs "${EOS_HOST}" mkdir -p "${EOS_SUBDIR}"
    echo "[COPY] ${FILE} -> ${EOS_HOST}/${EOS_OUT_DIR}/${REL}"
    xrdcp -f "${FILE}" "${EOS_HOST}/${EOS_OUT_DIR}/${REL}"
done

echo "[INFO] Writing metadata"
cat > "metadata_${TAG}.txt" <<METAEOF
point_id=${PID}
mrho=${MRHO}
mpi=${MPI}
scale=${SCALE}
mq=${MQ}
seed=${SEED}
pythia_seed=${PYTHIA_SEED}
repeat=${REPEAT}
mass_index=${MASS_INDEX}
events=${EVENTS}
mmed=${MMED}
rinv=${RINV}
local_out=${LOCAL_OUT}
eos_out_dir=${EOS_OUT_DIR}
eos_host=${EOS_HOST}
date=$(date)
host=$(hostname)
METAEOF

xrdcp -f "metadata_${TAG}.txt" "${EOS_HOST}/${EOS_OUT_DIR}/metadata_${TAG}.txt"

echo "=========================================================="
echo "[SUCCESS] Finished POINT_ID=${PID}"
echo "[SUCCESS] Output: ${EOS_OUT_URL}"
echo "End time: $(date)"
echo "=========================================================="
