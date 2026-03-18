# Dark QCD Model Building

## Installation

```bash
git clone git@github.com:cms-svj/model_building
cd model_building
./install.sh
```

## Environment

```bash
source init.sh
```

This repository is built on the [LCG 106](https://lcginfo.cern.ch/release_packages/106/x86_64-el9-gcc13-opt/) environment.
Important software versions:
* Pythia8 3.10
* Delphes 3.5.1pre09
* HepMC 2.06.11
* ROOT 6.32.02
* coffea 2025.12.0
* uproot 5.7.2
* awkward 2.9.0

## Predefined model configurations

A few predefined model configuration files are provided in the [configs](./configs) folder:
1. The CMS model used in [EXO-19-020](https://arxiv.org/abs/2112.11125) (largely based on [arXiv:1503.00009](https://www.arxiv.org/abs/1503.00009) and [arXiv:1707.05326](https://arxiv.org/abs/1707.05326)).
2. One of the Snowmass benchmark models from [arXiv:2203.09503](https://arxiv.org/abs/2203.09503).

## Running

An example run command is:
```bash
./run_model helper -C configs/model_cms.py --steps all --events 10 --verbose
```

This command will, in order:
1. load the CMS model configuration
2. generate the Pythia and Delphes cards for this configuration
3. run Pythia
4. run Delphes
5. make histograms

If the `--steps` command is omitted, items 3, 4, and 5 will be skipped.
Items 3, 4, and 5 can be run separately by specifying just one of them in `--steps`.

The input model configuration can be modified using command-line arguments, and the resulting configuration file will be generated along with the Pythia and Delphes cards.

Once histograms are created, an example of how to plot them, comparing different models, can be found in [Plots.py](./Plots.py).

### Alternative method

An external Pythia card can be used (instead of generating a model by providing the helper with a parameter configuration):
```bash
./run_model external --card my_card.cmnd --stableIDs 53 4900211 --darkHadronIDs 4900111 4900113 4900211 4900213 --darkHadronFinalIDs 4900111 4900211 --pythia '' --steps all --events 10 --verbose
```

As shown, the lists of stable particle IDs, dark hadron IDs, and final dark hadron IDs must be provided manually in order for the Delphes output to be correct.
(The argument `--pythia ''` prevents appending common settings to the Pythia card, which are included by default.)

## Analysis

The notebook [example_MT.ipynb](./example_MT.ipynb) shows an example of interactively analyzing the generated events using [coffea](https://github.com/CoffeaTeam/coffea/):
* applying selections
* computing new quantities
* filling histograms
* making plots

For programmatic analysis, follow [Histogram.py](./Histogram.py) and [Plots.py](./Plots.py) as noted above.
