#!/bin/bash

if [ -z "PYTHIA8MINOR" ]; then
	export PYTHIA8MINOR=10
fi

PYTHIA_VERSION=pythia83${PYTHIA8MINOR}
PYTHIA_EXE=main42
if [ "$PYTHIA8MINOR" -ge 12 ]; then
	PYTHIA_EXE=main132
fi

wget "https://pythia.org/releases/pythia83/${PYTHIA_VERSION}.tgz"
tar -xzf ${PYTHIA_VERSION}.tgz
mv ${PYTHIA_VERSION} pythia8
rm ${PYTHIA_VERSION}.tgz
cd pythia8

HEPMC_BASE=(/cvmfs/sft.cern.ch/lcg/releases/${LCG_VIEW}/HepMC/*/${LCG_ARCH})
LHAPDF_BASE=(/cvmfs/sft.cern.ch/lcg/releases/${LCG_VIEW}/MCGenerators/lhapdf/*/${LCG_ARCH})

./configure --with-hepmc2=${HEPMC_BASE} --with-lhapdf6=${LHAPDF_BASE} --with-python --with-gzip
make -j 8
make install

(cd examples
make ${PYTHIA_EXE}
)

cat << 'EOF' > mb_init.sh
export PYTHIA8=${MODEL_BUILDING}/install/pythia8
export PATH=${PYTHIA8}/bin/:${PATH}
export LD_LIBRARY_PATH=${PYTHIA8}/lib/:${LD_LIBRARY_PATH}
export PYTHIA8DATA=$(pythia8-config --xmldoc)
export PYTHIA8MINOR=${PYTHIA8MINOR}
export PYTHIA8RUNNER=${PYTHIA8}/examples/${PYTHIA_EXE}
EOF
