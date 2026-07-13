#!/bin/bash

source init.sh

while getopts "p:" opt; do
	case "$opt" in
		p) export PYTHIA8MINOR="$OPTARG"
		;;
	esac
done

cd install

for EXTERNAL in $MODEL_BUILDING_EXTERNALS; do
	echo $EXTERNAL
	./${EXTERNAL}.sh
done
