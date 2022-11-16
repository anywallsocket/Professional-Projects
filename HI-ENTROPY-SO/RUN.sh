#!/usr/bin/env bash
#Here is the main control for the self-optimizer
#It simply assumes INITIALIZE has run succesfully

ATOMS=$1
CALC='EMT'
START=0
LOOPS=10

##LOOP
for ((i=$START; i<=$LOOPS; i++));
do
	printf "\n>>ITERATION $i starting\n"
	
	#run GA
	printf ">[RUNNING GA.py]\n"
	mkdir DBs/$i
	if python3 CODE/GA.py $i $ATOMS; then
		printf ">[DONE]\n"
		rm DBs/work*
	else
		exit 1
	fi

	printf ">[RUNNING validate.py]\n"
	if python3 CODE/validate.py $i; then
		printf ">[DONE]\n"
	else
		exit 1
	fi
	#run training
	printf ">[RUNNING train.py]\n"
	mkdir NNs/$i
	if python3 CODE/train.py $i; then
		printf ">[DONE]\n"
	else
		exit 1
	fi

	#rinse and repeat
	printf ">>ITERATION $i complete\n"
done

##
