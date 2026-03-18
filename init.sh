#!/bin/bash

export LCG_VIEW=LCG_106
export LCG_ARCH=x86_64-el9-gcc13-opt
source /cvmfs/sft.cern.ch/lcg/views/${LCG_VIEW}/${LCG_ARCH}/setup.sh

export MODEL_BUILDING=$PWD
export MODEL_BUILDING_EXTERNALS="pythia8 python_packages"

for EXTERNAL in $MODEL_BUILDING_EXTERNALS; do
	EXTERNAL_DIR=${MODEL_BUILDING}/install/${EXTERNAL}
	if [ -d ${EXTERNAL_DIR} ]; then
		source ${EXTERNAL_DIR}/mb_init.sh
	fi
done
