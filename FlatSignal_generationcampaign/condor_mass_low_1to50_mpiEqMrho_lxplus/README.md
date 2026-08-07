# Low-Mass Refined mpi=mrho lxplus Campaign

Target production:

- Mass range: `1.0` to `50.0` GeV
- Grid spacing: `0.01` GeV
- Repeats per mass: `4`
- Events per job: `300`
- Total jobs: `19604`
- Total requested events: `5,881,200`
- EOS output base: `/eos/project/d/dragon/ashrivas/DarkHadronMassReco/DataForDragon_lowMass_1to50`
- EOS host: `root://eosproject.cern.ch`

Prepare the filelist and payload tarball:

```bash
cd ~/Dark_Sector/Darkhardon/model_building
./condor_mass_low_1to50_mpiEqMrho_lxplus/prepare_lxplus_campaign.sh
```

Copy this `model_building` checkout or at least this campaign directory to lxplus, then submit from the `model_building` directory:

```bash
condor_submit condor_mass_low_1to50_mpiEqMrho_lxplus/submit_low_1to50_lxplus.sub
```

Before submitting a fresh signal campaign, clean the output dataset so old dragon files are not mixed with the new refined grid:

```bash
./condor_mass_low_1to50_mpiEqMrho_lxplus/clean_dragon_output.sh
./condor_mass_low_1to50_mpiEqMrho_lxplus/clean_dragon_output.sh --execute
```

To change the total statistics, regenerate `mass_points_low_1to50.txt` with a different grid or `--repeats`, or edit the events argument in `submit_low_1to50_lxplus.sub`.
