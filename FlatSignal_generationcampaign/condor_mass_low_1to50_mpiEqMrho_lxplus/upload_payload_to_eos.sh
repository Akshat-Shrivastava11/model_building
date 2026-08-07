#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EOS_HOST="${EOS_HOST:-root://eosproject.cern.ch}"
EOS_DIR="${EOS_DIR:-/eos/project/d/dragon/ashrivas/DarkHadronMassReco/condor_payloads/FlatSignal_generationcampaign/condor_mass_low_1to50_mpiEqMrho_lxplus}"
EOS_LOG_DIR="${EOS_LOG_DIR:-/eos/project/d/dragon/ashrivas/DarkHadronMassReco/condor_logs/FlatSignal_generationcampaign/low/test}"

xrdfs "${EOS_HOST}" mkdir -p "${EOS_DIR}"
xrdfs "${EOS_HOST}" mkdir -p "${EOS_LOG_DIR}"
xrdcp -f "${HERE}/run_mass_point_lxplus.sh" "${EOS_HOST}/${EOS_DIR}/run_mass_point_lxplus.sh"
xrdcp -f "${HERE}/model_building_production.tgz" "${EOS_HOST}/${EOS_DIR}/model_building_production.tgz"
xrdcp -f "${HERE}/mass_points_low_1to50.txt" "${EOS_HOST}/${EOS_DIR}/mass_points_low_1to50.txt"

echo "[DONE] Uploaded low-mass payload to ${EOS_HOST}/${EOS_DIR}"
