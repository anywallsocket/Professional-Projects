#/usr/bin/env bash

ATOMS=15
CALC='LAMMPS'
i=0

echo '>>RUNNING initial_GA.py'
if python3 GRAMS/initial_GA.py $ATOMS $CALC; then
        echo '>>DONE'
        rm DBs/working*
else
        exit 1
fi

echo '>>RUNNING initial_train.py'
if python3 GRAMS/initial_train.py $i; then
        echo '>>DONE'
        mv checkpoints/2021* checkpoints/NN$i
else
        exit 1
fi

###