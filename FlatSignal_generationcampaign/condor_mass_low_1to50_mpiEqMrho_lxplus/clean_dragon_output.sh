#!/usr/bin/env bash
set -euo pipefail

# Run on lxplus before submitting a fresh campaign.
# Default is dry-run. Pass --execute to move any existing dataset aside.

MODE="dry-run"
EOS_HOST="root://eosproject.cern.ch"
DATASET="/eos/project/d/dragon/ashrivas/DarkHadronMassReco/DataForDragon_lowMass_1to50"
ARCHIVE_PARENT="/eos/project/d/dragon/ashrivas/DarkHadronMassReco/_archive_before_signal_campaign"

usage() {
  cat <<EOF
Usage: $0 [--execute]

Moves an existing refined signal dataset out of the way on dragon EOS:
  ${DATASET}

Default is dry-run. Nothing is changed unless --execute is passed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      MODE="execute"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

run() {
  echo "+ $*"
  if [[ "${MODE}" == "execute" ]]; then
    "$@"
  fi
}

timestamp="$(date +%Y%m%d_%H%M%S)"
archive="${ARCHIVE_PARENT}/DataForDragon_lowMass_1to50_${timestamp}"

echo "=========================================================="
echo "dragon EOS output cleanup"
echo "Mode     : ${MODE}"
echo "EOS host : ${EOS_HOST}"
echo "Dataset  : ${DATASET}"
echo "Archive  : ${archive}"
echo "=========================================================="

if xrdfs "${EOS_HOST}" stat "${DATASET}" >/dev/null 2>&1; then
  echo "[INFO] Existing dataset found."
  run xrdfs "${EOS_HOST}" mkdir -p "${ARCHIVE_PARENT}"
  run xrdfs "${EOS_HOST}" mv "${DATASET}" "${archive}"
else
  echo "[INFO] No existing dataset found. Fresh output area is already clean."
fi

run xrdfs "${EOS_HOST}" mkdir -p "${DATASET}"

echo "=========================================================="
echo "[DONE] Output area ready:"
echo "  ${DATASET}"
echo "=========================================================="
