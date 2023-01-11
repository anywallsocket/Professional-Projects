#!/usr/bin/env bash
#Here is the main control for the self-optimizer
#It simply assumes INITIALIZE has run succesfully

ATOMS=13
CALC='LAMMPS'
START=7
LOOPS=10
touch mae
touch output

##LOOP
for ((i=$START; i<=$LOOPS; i++));
do
	printf "\n>>ITERATION $i starting\n" >> output
	
	#run GA
	echo ">[RUNNING torchGA.py]" >> output
	if python3 GRAMS/torchGA.py $i $ATOMS; then
		echo ">[DONE]" >> output
		rm DBs/working*
	else
		exit 1
	fi

	#run validation
        echo ">[RUNNING validate.py]" >> output
	if python3 GRAMS/validate.py $i $CALC; then
                echo ">[DONE]" >> output
		rm TRAJ/*
		rm DBs/working*
		#rm -r checkpoints/2021*
        else
                exit 1
        fi

        #run training
        echo ">[RUNNING train.py]" >> output
        if python3 GRAMS/train.py $i; then
                echo ">[DONE]" >> output
		mv checkpoints/2021* checkpoints/NN$i
        else
                exit 1
        fi

	#rinse and repeat
	printf ">>ITERATION $i complete\n" >> output
done
