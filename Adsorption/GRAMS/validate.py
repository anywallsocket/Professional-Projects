#!/usr/bin/env python3
#Validate relaxes all structures in given DB to specified extent
#via NN potential and specified calculator, and uses the difference
#to generate train.db for further NN optimization

from programs import *
import numpy as np
from ase.db import connect
from os import listdir
import matplotlib.pyplot as plt
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.calculators.lammpsrun import LAMMPS
from ase.io.trajectory import Trajectory
from ase.io import read, write
import sys, random
from tqdm import tqdm

#set-up
iteration = int(sys.argv[1])
c = sys.argv[2]

#load current database
DB = connect('DBs/GA'+str(iteration)+'.db')
if len(DB) == 0: print('bad database'); exit()

#load specific calculator
if c == 'EMT': calc = EMT()

elif c == 'LAMMPS':
	#currently assumes AuPd
	files = ['AuPd.set']
	parameters = {'pair_style': 'eam/alloy', 'pair_coeff': ['* * AuPd.set Au Pd']}
	calc = LAMMPS(files=files, parameters=parameters)

else: myprint('bad calc argument for current validation'); exit()

#load up the last NN
NN_calc = LoadAmpTorch('old', iteration)

#relax all the structures according to the calc and NN, and compare
steps = 10*iteration
fmin = 0.001
true_energies = []
pred_energies = []
delta = []

#First we relax all the structures with the NN and the calc
#We save the calc's trajectory files, and add species from them
#To train.db later if they are sufficently distinct

myprint('Relaxing', len(DB), 'images according to', c, '(true) and NN (pred)')
myprint('fmin =', fmin, 'steps =', steps)

for i in tqdm(range(len(DB))):

	pred = DB.get_atoms(id=i+1)
	pred.calc = NN_calc
	true = pred.copy()
	true.calc = calc

	#Relax with calc (save these trajectories for train.db)
	test = BFGS(true, trajectory='TRAJ/'+str(i)+'.traj', logfile=None)
	test.run(fmax=fmin, steps=steps)
	relaxed = Trajectory('TRAJ/'+str(i)+'.traj')[-1]
	tf = relaxed.get_potential_energy()

	#Relax with NN (don't save these, only get final energy)
	test = BFGS(pred, trajectory='TRAJ/temp.traj', logfile=None)
	test.run(fmax=fmin, steps=steps)
	relaxed = Trajectory('TRAJ/temp.traj')[-1]
	pf = relaxed.get_potential_energy()

	diff = round(abs(tf-pf), 6)

	#myprint('['+str(i)+']','\ttf:',round(tf,3),'\tpf:',round(pf,3),'\tdiff:', diff)
	#Filter unreasonable results upstream
	if diff < 10:
		delta.append(diff)
		true_energies.append(tf)
		pred_energies.append(pf)

#Quick plot for convergence report
#quickplot(true_energies, pred_energies, iteration)
myplot('relax', np.array(true_energies), np.array(pred_energies), iteration)

#Save mean absolute error
MAE = round(np.mean(delta),6)
if MAE > 10: myprint('MAE likely biased, rejecting Outliers')
myprint('MAE =', MAE)
with open('mae', 'a') as f: f.write(str(iteration)+' '+str(MAE)+'\n')

#To train on difference, we first generate the train.db
upperthreshold = MAE * 2.0
lowerthreshold = MAE / 2.0

#Save good and poor fit images for building train.db
good_index = []
poor_index = []
myprint('Comparing Relaxed Energies')

for i in range(len(delta)):
	if delta[i] > lowerthreshold and delta[i] < upperthreshold:
		poor_index.append(i)
	elif delta[i] < lowerthreshold:
		good_index.append(i)
	#and we simply skip those above upperthreshold

#Write sufficently different relaxations from trajectory files to train.db
working_DB = connect('DBs/working.db')
train_DB = connect('DBs/train'+str(iteration)+'.db')
miss = int(len(DB)-(len(poor_index)+len(good_index)))
myprint(len(poor_index), 'poor fits,')
myprint(len(good_index), 'good fits,')
myprint('and', miss, 'skipped')

#Add ratio of what NN guesses well and what it guesses poorly
for i in poor_index:
	T = Trajectory('TRAJ/'+str(i)+'.traj')
	sample = random.sample(list(np.arange(1, len(T))), int(len(T)/4))
	for j in sample:
		working_DB.write(T[int(j)])

for i in good_index:
	T = Trajectory('TRAJ/'+str(i)+'.traj')
	sample = random.sample(list(np.arange(1, len(T))), int(len(T)/2))
	for j in sample:
		working_DB.write(T[int(j)])

myprint('Working with', len(working_DB), 'images')

#Sort and prune for bad images / duplicates
write_train(iteration)

##
