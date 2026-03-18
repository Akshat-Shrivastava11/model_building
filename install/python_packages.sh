#!/bin/bash

mkdir -p python_packages
cd python_packages

PKGS_UPGRADE=(
coffea==2025.12.0 \
mplhep \
)

for PKG in ${PKGS_UPGRADE[@]}; do
	HOME=$PWD pip install --user --no-cache-dir --upgrade $PKG
done

PKGS=(
magiconfig \
fastjet \
)

for PKG in ${PKGS[@]}; do
	HOME=$PWD pip install --user --no-cache-dir $PKG
done

cat << 'EOF' > mb_init.sh
export PYTHONPATH=${MODEL_BUILDING}/install/python_packages/.local/lib/python3.11/site-packages/:${PYTHONPATH}
EOF
