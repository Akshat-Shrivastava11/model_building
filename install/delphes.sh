#!/bin/bash

git clone git@github.com:cms-svj/delphes -b DarkHadronJets
cd delphes

make -j 8

cat << 'EOF' > mb_init.sh
export DELPHES=${MODEL_BUILDING}/install/delphes
export PATH=${DELPHES}/:${PATH}
export LD_LIBRARY_PATH=${DELPHES}/:${LD_LIBRARY_PATH}
EOF
