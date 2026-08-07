#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_BUILDING="$(cd "${HERE}/../.." && pwd)"
CAMPAIGN_REL="FlatSignal_generationcampaign/condor_mass_low_1to50_mpiEqMrho_lxplus"

cd "${MODEL_BUILDING}"

set +u
source init.sh
set -u

if ! python3 -c "import magiconfig" >/dev/null 2>&1; then
    echo "[INFO] Installing Python package dependencies into install/python_packages"
    (cd install && bash python_packages.sh)
fi

if [[ ! -x install/pythia8/examples/main42 ]]; then
    if [[ -f "${MODEL_BUILDING}/../pythia8_prebuilt_lxplus.tgz" ]]; then
        echo "[INFO] Unpacking prebuilt Pythia8 from ../pythia8_prebuilt_lxplus.tgz"
        tar -xzf "${MODEL_BUILDING}/../pythia8_prebuilt_lxplus.tgz" -C install
    fi
fi

if [[ ! -x install/pythia8/examples/main42 ]]; then
    echo "[INFO] Installing Pythia8 into install/pythia8"
    export PYTHIA8MINOR=317
    sed -i.bak \
        -e 's/PYTHIA_VERSION=.*/PYTHIA_VERSION=pythia8317/' \
        -e 's#https://pythia.org/releases/pythia83/#https://www.pythia.org/download/pythia83/#' \
        -e 's#https://www.pythia.org/download/pythia83/#https://pythia.org/download/pythia83/#' \
        -e 's#https://pythia8.web.cern.ch/releases/pythia83/#https://pythia.org/download/pythia83/#' \
        install/pythia8.sh
    (cd install && bash -e pythia8.sh)
fi

if [[ ! -x install/pythia8/examples/main42 ]]; then
    echo "[ERROR] Pythia8 runner was not built: install/pythia8/examples/main42" >&2
    exit 20
fi

if [[ -f install/pythia8/mb_init.sh ]] && ! grep -q "PYTHIA8MINOR" install/pythia8/mb_init.sh; then
    echo "export PYTHIA8MINOR=317" >> install/pythia8/mb_init.sh
fi

python3 "${HERE}/make_refined_mass_points.py" \
    --min-mass 1.0 \
    --max-mass 50.0 \
    --step 0.01 \
    --repeats 4 \
    --seed 50001 \
    --out "${HERE}/mass_points_low_1to50.txt"

payload=(
    cards
    configs
    install
    Histogram.py
    common.py
    init.sh
    run_model
    svjHelper.py
)

if [[ -e models ]]; then
    payload+=(models)
fi

tar -czf "${HERE}/model_building_production.tgz" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude="${CAMPAIGN_REL}/logs" \
    --exclude="${CAMPAIGN_REL}/model_building_production.tgz" \
    "${payload[@]}"

mkdir -p "${HERE}/logs"

cat > "${HERE}/upload_payload_to_eos.sh" <<'EOSUPLOAD'
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
EOSUPLOAD
chmod +x "${HERE}/upload_payload_to_eos.sh"

EOS_URL_BASE="root://eosproject.cern.ch//eos/project/d/dragon/ashrivas/DarkHadronMassReco/condor_payloads/FlatSignal_generationcampaign/condor_mass_low_1to50_mpiEqMrho_lxplus"
cat > "${HERE}/submit_low_1to50_eos.sub" <<EOF
universe = vanilla

Executable = /bin/bash

Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT

Transfer_Input_Files = ${EOS_URL_BASE}/run_mass_point_lxplus.sh, ${EOS_URL_BASE}/model_building_production.tgz, ${EOS_URL_BASE}/mass_points_low_1to50.txt

Output = /dev/null
Error  = /dev/null
Log    = /dev/null

request_cpus   = 1
request_memory = 4 GB
request_disk   = 20 GB

+JobFlavour = "workday"

Arguments = run_mass_point_lxplus.sh \$(Process) mass_points_low_1to50.txt 300 /eos/project/d/dragon/ashrivas/DarkHadronMassReco/DataForDragon_lowMass_1to50 root://eosproject.cern.ch

Queue 19604
EOF

cat > "${HERE}/submit_low_1to50_test_one_eos.sub" <<EOF
universe = vanilla

Executable = /bin/bash

Should_Transfer_Files = YES
WhenToTransferOutput = ON_EXIT

Transfer_Input_Files = ${EOS_URL_BASE}/run_mass_point_lxplus.sh, ${EOS_URL_BASE}/model_building_production.tgz, ${EOS_URL_BASE}/mass_points_low_1to50.txt

Output = /dev/null
Error  = /dev/null
Log    = /dev/null

request_cpus   = 1
request_memory = 4 GB
request_disk   = 20 GB

+JobFlavour = "workday"

Arguments = run_mass_point_lxplus.sh 0 mass_points_low_1to50.txt 300 /eos/project/d/dragon/ashrivas/DarkHadronMassReco/DataForDragon_lowMass_1to50 root://eosproject.cern.ch

Queue 1
EOF

echo "Prepared lxplus campaign in ${HERE}"
echo "Jobs: $(awk '($1 !~ /^#/) {n++} END {print n+0}' "${HERE}/mass_points_low_1to50.txt")"
echo "Events per job: 300"
echo "Total requested events: $(awk '($1 !~ /^#/) {n++} END {print (n+0)*300}' "${HERE}/mass_points_low_1to50.txt")"
echo
echo "Upload payload to EOS with:"
echo "  ${CAMPAIGN_REL}/upload_payload_to_eos.sh"
echo
echo "Test one job first with:"
echo "  condor_submit ${CAMPAIGN_REL}/submit_low_1to50_test_one_eos.sub"
echo
echo "Submit from model_building with:"
echo "  condor_submit ${CAMPAIGN_REL}/submit_low_1to50_eos.sub"
