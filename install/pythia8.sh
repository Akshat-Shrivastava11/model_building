#!/bin/bash

PYTHIA_VERSION=pythia8310

wget "https://www.pythia.org/download/pythia83/${PYTHIA_VERSION}.tgz"
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
make main42
)

cat << 'EOF' > mb_init.sh
export PYTHIA8=${MODEL_BUILDING}/install/pythia8
export PATH=${PYTHIA8}/bin/:${PATH}
export LD_LIBRARY_PATH=${PYTHIA8}/lib/:${LD_LIBRARY_PATH}
export PYTHIA8DATA=$(pythia8-config --xmldoc)
export PYTHIA8RUNNER=${PYTHIA8}/examples/main42
EOF
