# Refined mpi=mrho lxplus Campaign

Target production:

- Mass range: `1.0` to `250.0` GeV
- Grid spacing: `0.025` GeV
- Repeats per mass: `4`
- Events per job: `300`
- Total jobs: `39844`
- Total requested events: `11,953,200`
- EOS output base: `/eos/project/d/dragon/ashrivas/DarkHadronMassReco/DataForDragon`
- EOS host: `root://eosproject.cern.ch`

Prepare the filelist and payload tarball:

```bash
cd /uscms/home/ashrivas/nobackup/Dark_Sector/Darkhardon/model_building
./condor_mass_refined_mpiEqMrho_lxplus/prepare_lxplus_campaign.sh
```

Copy this `model_building` checkout or at least this campaign directory to lxplus, then submit from the `model_building` directory:

```bash
condor_submit condor_mass_refined_mpiEqMrho_lxplus/submit_refined_1to250_lxplus.sub
```

Before submitting a fresh signal campaign, clean the output dataset so old dragon files are not mixed with the new refined grid:

```bash
./condor_mass_refined_mpiEqMrho_lxplus/clean_dragon_output.sh
./condor_mass_refined_mpiEqMrho_lxplus/clean_dragon_output.sh --execute
```

To change the total statistics, regenerate `mass_points_refined_1to250.txt` with a different `--repeats`, or edit the events argument in `submit_refined_1to250_lxplus.sub`.
