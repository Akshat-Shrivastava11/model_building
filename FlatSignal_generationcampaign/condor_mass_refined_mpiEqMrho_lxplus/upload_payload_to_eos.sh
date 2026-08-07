#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EOS_HOST="${EOS_HOST:-root://eosproject.cern.ch}"
EOS_DIR="${EOS_DIR:-/eos/project/d/dragon/ashrivas/DarkHadronMassReco/condor_payloads/FlatSignal_generationcampaign/condor_mass_refined_mpiEqMrho_lxplus}"
EOS_LOG_DIR="${EOS_LOG_DIR:-/eos/project/d/dragon/ashrivas/DarkHadronMassReco/condor_logs/FlatSignal_generationcampaign/refined/test}"

xrdfs "${EOS_HOST}" mkdir -p "${EOS_DIR}"
xrdfs "${EOS_HOST}" mkdir -p "${EOS_LOG_DIR}"
xrdcp -f "${HERE}/run_mass_point_lxplus.sh" "${EOS_HOST}/${EOS_DIR}/run_mass_point_lxplus.sh"
xrdcp -f "${HERE}/model_building_production.tgz" "${EOS_HOST}/${EOS_DIR}/model_building_production.tgz"
xrdcp -f "${HERE}/mass_points_refined_1to250.txt" "${EOS_HOST}/${EOS_DIR}/mass_points_refined_1to250.txt"
xrdcp -f "${HERE}/point_ids_part1.txt" "${EOS_HOST}/${EOS_DIR}/point_ids_part1.txt"
xrdcp -f "${HERE}/point_ids_part2.txt" "${EOS_HOST}/${EOS_DIR}/point_ids_part2.txt"

echo "[DONE] Uploaded refined payload to ${EOS_HOST}/${EOS_DIR}"
