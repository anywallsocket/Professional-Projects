#!/usr/bin/env bash

# ALA sample script
# note that one can optimize NNs
# for clusters or adsorption sites

START=0
LOOPS=10
GAs=10

# global iteration loop
for ((I=$START; I<=$LOOPS; I++));
do
	echo ">> $I"
	rm -r DBs/$I
	mkdir DBs/$I

	# parallelizable stoichiometry loop
	for ((J=0; J<$GAs; J++));
	do
		python3 ads.py $I $J
  		#python3 clust.py $I $J
		wait
	done

	rm -r NNs/$I
	mkdir NNs/$I

	python3 train.py $I $GAs
	wait
	python3 analyze.py $I
	echo "$I <<"

done
