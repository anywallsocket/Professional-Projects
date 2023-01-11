#!/usr/bin/env python3

import sys, time
import torch
import numpy as np
from os import listdir
from ase.db import connect
from ase.io.trajectory import Trajectory
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from programs import *

iteration = int(sys.argv[1])
#load current train DB
d = 'DBs/train'+str(iteration)+'.db'
#d = 'DBs/GA'+str(iteration)+'.db'
DB = connect(d)
if len(DB) == 0: myprint('bad database'); exit()
images = [DB.get_atoms(id=i+1) for i in range(len(DB))]

#TRAINING

#load previous NN
trainer = LoadAmpTorch('new', iteration)
myprint('[Training on new data]')
start_time = time.time()
trainer.train()
myprint('Training Complete:',round(time.time()-start_time,6),"(s)\n")
predictions = trainer.predict(images)

#ANALYSIS

#Energy plot
xe = np.array([image.get_potential_energy() for image in images])
ye = np.array(predictions["energy"])
myplot('energy', xe, ye, iteration)

#Force plot (This assumes atom/adsorbate count 13/3)
x = np.array([image.get_forces() for image in images]).flatten()
y = np.array(predictions["forces"]).flatten()
mask = np.array([[[False,False,False]]*13+[[True,True,True]]*3]*len(images)).flatten()
newx = x[mask]; newy = y[mask]
myplot('force', np.array(newx), np.array(newy), iteration)

###