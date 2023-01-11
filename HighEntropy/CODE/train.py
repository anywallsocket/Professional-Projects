#!/usr/bin/env python3

import os, sys, time
import torch
import numpy as np
from ase.db import connect
from programs import *

###------------------------------------------
### SET UP
###------------------------------------------start = time.time()

start = time.time()
print('[trainGPU.py Starting]')

iteration = int(sys.argv[1])
directory = 'NNs/'+str(iteration)

#Build on previous images
images = []
#for i in reversed(range(iteration+1)):
if iteration>0:
	db = connect('DBs/NN-'+str(iteration-1)+'.db')
	for j in range(len(db)): images.append(db.get_atoms(id=j+1))

print(len(images), 'images read')
dataset = images2AtomsData(images, directory)

###------------------------------------------
### SCHNET TRAINING
###------------------------------------------

if torch.cuda.is_available(): device = 'cuda'
else: device = 'cpu'

rho = 0.2
bs = 32
lr = 5E-4
split = [0.6, 0.2, 0.2]
ratios = [int(i*100) for i in split]

trainer, sets = loadTrainer(dataset, directory, rho, bs, lr, split)

print('Training on new data...')

t1 = time.time()
trainer.train(device=device, n_epochs=500)
t2 = time.time()

minutes = round((t2-t1)/60,2)
print('Training complete --', minutes, 'min')

###------------------------------------------
### ANANYSIS
###------------------------------------------

save = True
idx = 2 #[train, valid, test]
kind = 'test'
energies, forces = trainMAE(sets, directory)
trueE, predE = energies[idx][0], energies[idx][1]
trueF, predF = forces[idx][0], forces[idx][1]

MAEs = [energies[0][2], forces[0][2],
		energies[1][2], forces[1][2],
		energies[2][2], forces[2][2]]

#Remove outstanding errors
limit = 10
oopsE, oopsF = 0, 0
newTE, newPE, newTF, newPF = [], [], [], []
for i in range(len(trueE)):
	if abs(trueE[i]-predE[i]) <= limit: 
		newTE.append(trueE[i])
		newPE.append(predE[i])
	else: oopsE += 1
for i in range(len(trueF)):
	if abs(trueF[i]-predF[i]) <= limit: 
		newTF.append(trueF[i])
		newPF.append(predF[i])
	else: oopsF += 1
print('oops:', oopsE, oopsF)

###------------------------------------------
### PLOTTING
###------------------------------------------

print('PLOTTING...')
trainPlot(iteration, directory, save)
ratioplot(iteration, 'energy', np.array(newTE), np.array(newPE), kind, save)
ratioplot(iteration, 'force', np.array(newTF), np.array(newPF), kind, save)
deltaplot(iteration, 'energy', np.array(newTE), np.array(newPE), kind, save)
deltaplot(iteration, 'force', np.array(newTF), np.array(newPF), kind, save)
histoplot(iteration, 'energy', np.array(newTE), np.array(newPE), kind, save)
histoplot(iteration, 'force', np.array(newTF), np.array(newPF), kind, save)
barplot(iteration, MAEs, ratios, save)

print('DONE')
print('[trainGPU.py Finished -', round(time.time()-start, 3),'(s)]')

###