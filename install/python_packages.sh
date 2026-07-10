#!/bin/bash

mkdir -p python_packages
cd python_packages

wget https://github.com/kpedro88/lcg-venv/raw/refs/heads/main/lcg-venv
chmod +x lcg-venv
./lcg-venv mbenv

cat << 'EOF' > mb_init.sh
source ${MODEL_BUILDING}/install/python_packages/mbenv/bin/activate
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
