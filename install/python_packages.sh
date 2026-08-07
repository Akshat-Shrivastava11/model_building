#!/bin/bash

mkdir -p python_packages
cd python_packages

wget https://github.com/kpedro88/lcg-venv/raw/refs/heads/main/lcg-venv
chmod +x lcg-venv
./lcg-venv mbenv

cat << 'EOF' > mb_init.sh
export MODEL_BUILDING_PYTHON=${MODEL_BUILDING}/install/python_packages/mbenv
export VIRTUAL_ENV=${MODEL_BUILDING_PYTHON}
export PATH=${MODEL_BUILDING_PYTHON}/bin:${PATH}
export PYTHONPATH=${MODEL_BUILDING_PYTHON}/lib/python3.11/site-packages:${PYTHONPATH:-}
EOF

source mb_init.sh

PKGS_UPGRADE=(
coffea==2025.12.0 \
mplhep \
)

for PKG in ${PKGS_UPGRADE[@]}; do
	pip install --upgrade $PKG
done

PKGS=(
magiconfig \
fastjet \
)

for PKG in ${PKGS[@]}; do
	pip install $PKG
done
