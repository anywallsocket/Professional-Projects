#!/usr/bin/env python3

import os, sys, time, torch
import numpy as np
import matplotlib.pyplot as plt
import schnetpack as spk
import schnetpack.train as trn
from ase.io import read, write
from torch import tensor
from ase.db import connect
from torch.optim import Adam
from schnetpack import AtomsData
from matplotlib.cm import ScalarMappable
from ase.calculators.singlepoint import SinglePointCalculator
from scipy.stats import gaussian_kde, lognorm
from collections.abc import Iterable
from ase.optimize import BFGS

###------------------------------------------
### UTILITY FUNCTIONS
###------------------------------------------

def N2S(N):
	S = []
	for n in N:
		if n==28: S.append('Ni')
		elif n==29: S.append('Cu')
		elif n==46: S.append('Pd')
		elif n==47: S.append('Ag')
		elif n==78: S.append('Pt')
		elif n==79: S.append('Au')
	return str(S)

def myprint(*kargs):
	s = ''
	for k in kargs: s=s+' '+str(k)
	with open('output', 'a') as f:
		f.write(s+'\n')

def trueFlat(A):
	for a in A:
		if isinstance(a,Iterable) and not isinstance(a,(str,bytes)): yield from trueFlat(a)
		else: yield a

def center(p, a):
	meanx = np.mean(p[:,0]); meany = np.mean(p[:,1]); meanz = np.mean(p[:,2])
	dx = a/2-meanx; dy = a/2-meany; dz = a/2-meanz
	q = [[i[0]+dx, i[1]+dy, i[2]+dz] for i in p]
	return q

def NchooseM(N, M):
	#would certainly want a different approach for larger numbers
	assert M<N and M>1
	order = np.random.choice(np.array(M), M, replace=False)
	choices = [int(np.random.uniform(1, N-M+2))]
	for i in range(1, M-1):
		choices.append(int(np.random.uniform(1, N-np.sum(choices)-M+i+2)))
	choices.append(N-np.sum(choices))
	selection = [choices[order[i]] for i in range(M)]
	return selection

#Return lognorm sample of size s from y, scaled to y
def normlog(y, s):
	
	rng = np.random.default_rng()
	assert len(y)>s, 'sample larger than space'
	yabs = np.abs(y)
	x = np.arange(len(y))
	p = lognorm(yabs).pdf(x)
	p = p/np.sum(p)
	
	idx = np.sort(rng.choice(x, size=s, replace=False, p=p))
	sample = [y[i] for i in idx]
	return idx, sample

#Trim cuts out images based on their forces standard deviations
def trim(images, nstd):

	F = [i.get_forces().flatten() for i in images]

	#forces are evaluated per cluster, rather than per set
	meanF, stdF = [np.mean(f) for f in F], [np.std(f) for f in F]

	valid = []
	good = True
	for i in range(len(F)):
		for j in range(len(F[i])):
			if abs(F[i][j]-meanF[i]) < stdF[i]*nstd:
				good = True
			else: good = False
		if good: valid.append(images[i])

	print(len(images)-len(valid), 'forces trimed out of', len(images))
	return valid

#Prune cuts images based on energetic degeneracy, also assures unit cell bounds
def prune(images, decimals):

	energies = [i.get_potential_energy() for i in images]
	indexsort = np.argsort(energies)

	#Unique energies (via rounding)
	uniqueindexsort = np.unique(np.round(np.sort(energies), decimals), 
		return_index=True)[1]
	C = [images[int(indexsort[int(i)])] for i in uniqueindexsort]
	E = [i.get_potential_energy() for i in C]
	mean = np.mean(E)

	valid = []
	for i in range(len(C)):

		#Re-centering requires re-assigning calc energy and forces
		e = C[i].get_potential_energy()
		f = C[i].get_forces()

		#Check bounds (extra cautious)
		pos = center(C[i].get_positions(), 20)
		for j in pos:
			for k in j:
				if k <= 0 or k >= 20: break
			else: continue
			break
		else:
			#Finally, write good structures (copy won't copy calc)
			c = C[i].copy()
			c.set_positions(pos)
			c.calc = SinglePointCalculator(c)
			c.calc.results['energy'] = e
			c.calc.results['forces'] = f   
			valid.append(c)

	print(len(images)-len(valid), 'energies pruned out of', len(images))
	return valid

###------------------------------------------
### SCHNET TRAINING FUNCTIONS
###------------------------------------------

def images2AtomsData(images, directory):
	
	db = os.path.join(directory, 'new_dataset.db')
	property_list = []

	for i in images:
		energy = np.array(i.get_potential_energy(), dtype=np.float32)
		forces = np.array(i.get_forces(), dtype=np.float32)
		property_list.append({'energy': energy, 'forces': forces})
	
	dataset = AtomsData(db, available_properties=['energy', 'forces'])
	dataset.add_systems(images, property_list)
	
	return dataset

def LoadSchNetCalc(NN_path):

	model = torch.load(os.path.join(NN_path, 'best_model'), map_location='cpu')
	calculator = spk.interfaces.SpkCalculator(
		model=model,
		device='cpu',
		energy='energy',
		forces='forces',
		energy_units='eV',
		forces_units='eV/Angstrom'
	)
	return calculator

def loadTrainer(dataset, directory, rho, bs, lr, split):

	train, valid, test = spk.train_test_split(
			data=dataset,
			num_train=int(len(dataset)*split[0]),
			num_val=int(len(dataset)*split[1]),
			split_file=os.path.join(directory, 'split.npz'),
		)

	train_loader = spk.AtomsLoader(train, batch_size=bs, shuffle=True)
	valid_loader = spk.AtomsLoader(valid, batch_size=bs, shuffle=True)
	test_loader = spk.AtomsLoader(test, batch_size=bs, shuffle=True)

	n_features = 128
	means, stdevs = train_loader.get_statistics('energy', divide_by_atoms=True)

	schnet = spk.representation.SchNet(
		n_atom_basis=n_features,
		n_filters=n_features,
		n_gaussians=25,
		n_interactions=3,
		cutoff=5.,
		cutoff_network=spk.nn.cutoff.CosineCutoff
	)

	energy_model = spk.atomistic.Atomwise(
		n_in=n_features,
		property='energy',
		mean=means['energy'],
		stddev=stdevs['energy'],
		derivative='forces',
		negative_dr=True
	)

	model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)
	optimizer = Adam(model.parameters(), lr=lr)
	metrics = [spk.metrics.MeanAbsoluteError('energy'), 
				spk.metrics.MeanAbsoluteError('forces')]

	hooks = [
		trn.CSVHook(log_path=directory, metrics=metrics),
		trn.ReduceLROnPlateauHook(
			optimizer, patience=5, factor=0.8, min_lr=1E-5,
			stop_after_min=True
		)
	]
	
	def loss(batch, result):

		diff_energy = batch['energy']-result['energy']
		err_sq_energy = torch.mean(diff_energy ** 2)
		diff_forces = batch['forces']-result['forces']
		err_sq_forces = torch.mean(diff_forces ** 2)
		return rho*err_sq_energy + (1-rho)*err_sq_forces

	trainer = trn.Trainer(
		model_path=directory,
		model=model,
		hooks=hooks,
		loss_fn=loss,
		optimizer=optimizer,
		train_loader=train_loader,
		validation_loader=valid_loader,
	)

	return trainer, [train_loader, valid_loader, test_loader]

def trainMAE(sets, directory):

	Edata = []; Fdata = []
	model = torch.load(os.path.join(directory, 'best_model'), map_location='cpu')
	
	for i in range(len(sets)):
		
		energy_error = []
		forces_error = []
		
		predE = []; trueE = []
		predF = []; trueF = []  
		
		for batch in sets[i]:
						
			#apply prediction model per batch
			pred = model(batch)
			
			#counting atoms by length of force arrays to calculate EPA and FPA
			batch_size = len(batch['forces'])
			
			for n in range(batch_size):
				
				#atom count (PA) could be different within each batch
				#therefore EPA needs to be done per image - forces are automatically PA
				#PA = len(batch['forces'].detach().cpu().numpy()[n])

				#collecting energies
				predE.append(pred['energy'].detach().cpu().numpy()[n])
				trueE.append(batch['energy'].detach().cpu().numpy()[n])

				#collecting forces (batched forces are arrayed)
				predF.append(pred['forces'].detach().cpu().numpy()[n])
				trueF.append(batch['forces'].detach().cpu().numpy()[n])

				#calculate errors
				tmp_energy = torch.abs(pred['energy']-batch['energy']).detach().cpu().numpy()[n][0]
				energy_error.append(tmp_energy)

				tmp_forces = torch.mean(torch.abs(pred['forces']-batch['forces']), 
										dim=(1,2)).detach().cpu().numpy()[n]
				forces_error.append(tmp_forces)

		#MAEs are taken per batch per set
		energy_error = np.mean(energy_error)
		forces_error = np.mean(forces_error)

		#Energies and Forces are also per set (for barplot)
		trueE = np.array(list(trueFlat(trueE)))
		predE = np.array(list(trueFlat(predE)))
		trueF = np.array(list(trueFlat(trueF)))
		predF = np.array(list(trueFlat(predF)))
		
		#This happens 3 times (train, valid, test)
		Edata.append([trueE, predE, energy_error])
		Fdata.append([trueF, predF, forces_error])

	return Edata, Fdata

###------------------------------------------
### PLOTTING FUNCTIONS
###------------------------------------------

def trainPlot(iteration, directory, save=False):

	results = np.loadtxt(os.path.join(directory, 'log.csv'), skiprows=1, delimiter=',')
	T = np.array(results[:,0]-results[0,0])/60
	E = results[:,4]
	F = results[:,5]

	MAEe = round(E[-1],4); MAEf = round(F[-1],4)

	fig, ax = plt.subplots(figsize=(5,5), nrows=2, ncols=1, sharex=True)

	ax[0].set_title('iteration = '+str(iteration)+' - valid')
	ax[0].plot(T, E, c='b', label='energy: '+str(MAEe), zorder=2)
	ax[0].set_ylabel('MAE (eV)')
	ax[0].grid(ls=':', zorder=0)
	ax[0].tick_params(labelbottom=False)    
	ax[0].legend()

	ax[1].plot(T, F, c='r', label='forces: '+str(MAEf), zorder=2)
	ax[1].set_ylabel('MAE (eV/atom/\u212B)')
	ax[1].grid(ls=':', zorder=0)
	ax[1].legend()
	
	plt.subplots_adjust(wspace=0.0, hspace=0.05)
	plt.xlabel('time (min)')
	plt.tight_layout(pad=1)
	if save: plt.savefig('PICS/train-'+str(iteration)+'.png', dpi=200)
	plt.clf()

def barplot(iteration, measures, ratios, save=False):

	Max = np.max(measures)
	
	plt.figure(figsize=(5,5))
	plt.bar(np.arange(3)-0.2, [measures[0],measures[2],measures[4]], 
		label='energy', width=0.4, color='b', edgecolor='k', alpha=0.75, zorder=2)
	plt.bar(np.arange(3)+0.2, [measures[1],measures[3],measures[5]], 
		label='forces', width=0.4, color='r', edgecolor='k', alpha=0.75, zorder=2)

	t1 = plt.text(0-0.1, 0.1*Max, str(ratios[0])+'%', fontsize=10)
	t1.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
	t2 = plt.text(1-0.1, 0.1*Max, str(ratios[1])+'%', fontsize=10)
	t2.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
	t3 = plt.text(2-0.1, 0.1*Max, str(ratios[2])+'%', fontsize=10)
	t3.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))

	plt.xticks(np.arange(3), ['train', 'valid', 'test'])
	plt.ylabel('MAE')
	plt.title('iteration = '+str(iteration))
	plt.legend(); plt.grid(ls=':', zorder=0); plt.tight_layout(pad=1)
	if save: plt.savefig('PICS/bar'+str(iteration)+'.png', dpi=200)
	plt.clf()

def ratioplot(iteration, kind, x, y, title, save=False):

	line = [np.min(y), np.max(y)]
	if kind=='energy' or kind=='relax': unit=' (eV)'
	elif kind=='force': unit=' (eV/atom/\u212B)'
	else: print('[either energy, forces, or relax]'); return
	
	MAE = np.round(np.mean(np.abs(x-y)), 4)

	xy = np.vstack([x,y])
	z = gaussian_kde(xy)(xy)
	idx = z.argsort()
	x, y, z = x[idx], y[idx], z[idx]
	norm = plt.Normalize()
	colors = plt.cm.turbo(norm(z))
	
	plt.figure(figsize=(5,5))
	plt.plot(line, line, c='black', linestyle='dotted', 
		linewidth=0.75, zorder=3)
	plt.scatter(x, y, facecolor=colors, edgecolor='None', 
		s=20, alpha=0.75, label='MAE '+str(MAE), zorder=2)
	plt.xlabel('DFT '+kind+unit)
	plt.ylabel('NN '+kind+unit)
	plt.title('iteration = '+str(iteration)+' - '+title)
	plt.legend(); plt.grid(ls=':', zorder=0); plt.tight_layout(pad=1)
	if save: plt.savefig('PICS/ratio-'+kind+'-'+str(iteration)+'.png', dpi=200)
	plt.clf()

def deltaplot(iteration, kind, y1, y2, title, save=False):

	if kind=='energy' or kind=='relax': unit=' (eV)'
	elif kind=='force': unit=' (eV/atom/\u212B)'
	else: print('[either energy, forces, or relax]'); return
	
	MAE = np.round(np.mean(np.abs(y1-y2)), 4)

	index = np.argsort(y1)
	y1 = y1[index]; y2 = y2[index]
	line = [np.min(y2), np.max(y2)]
	
	fig, ax = plt.subplots(figsize=(5,5), nrows=2, ncols=1, sharex=True)
	x = np.arange(len(y1))

	ax[0].set_title('iteration = '+str(iteration)+' - '+title)
	ax[0].scatter(x, y1, label='DFT', marker='o', 
		s=100, c='r', alpha=0.5, zorder=2)
	ax[0].scatter(x, y2, label='NN', 
		s=1, c='b', alpha=0.5, zorder=3)
	ax[0].legend()
	ax[0].set_ylabel('EPA'+unit)
	ax[0].tick_params(labelbottom=False)    
	ax[0].grid(ls=':', zorder=0)

	delta = (y1-y2)
	ax[1].plot(delta, c='r', linewidth=0.5, alpha=1.0, zorder=2)
	ax[1].scatter(x,delta,label='MAE '+str(MAE), facecolors='w',
		alpha=0.75, edgecolors='k', s=5, zorder=3)
	ax[1].set_ylabel('$DFT_i - NN_i$'+unit)
	ax[1].legend(loc='upper left')
	ax[1].grid(ls=':', zorder=0)

	plt.subplots_adjust(wspace=0.0, hspace=0.05)
	plt.xlabel('species number (sorted by energy)')
	plt.tight_layout(pad=1)
	if save: plt.savefig('PICS/delta-'+str(kind)+'-'+str(iteration)+'.png', dpi=200)
	plt.clf()
	
def histoplot(iteration, kind, x, y, title, save):

	if kind=='energy' or kind=='relax': 
		c = 'b'; unit=' (eV/atom)'
	elif kind=='force': 
		c = 'r'; unit=' (eV/atom/\u212B)'
	else: print('[either energy, forces, or relax]'); return
	
	lab = True
	delta = x-y
	ME = np.round(np.mean(delta),3)
	if abs(ME) < 0.001: lab = False
	
	dmin, dmax = round(np.min(np.abs(delta)),3), round(np.max(np.abs(delta)),3)
	buffer = (dmax-dmin)/10
	print('absolute', kind, 'difference range =', '['+str(dmin)+','+str(dmax)+']')
	res = 100

	#Fit gaussian to data
	#rule of thumb for covariance factor
	h = 1.06*np.std(delta)*len(delta)**(-0.2)
	distribution = gaussian_kde(delta)
	distribution.covariance_factor = lambda: h
	distribution._compute_covariance()
	x = np.linspace(np.min(delta)-buffer, np.max(delta)+buffer, res)
	density = distribution(x)
	density = density/np.max(density)
	
	#Plotting
	plt.figure(figsize=(5,5))
	plt.vlines(0, 0, 1, color='k', alpha=0.75, lw=1, zorder=3)
	if lab: 
		plt.vlines(ME, 0, 1, ls='--', color='k', alpha=0.75,
					label='ME '+str(ME), lw=1, zorder=3)
	else: 
		plt.vlines(ME, 0, 1, ls='--', color='k', alpha=0.75,
					label='ME $\\approx$ 0.0', lw=1, zorder=3)
	plt.plot(x, density, linewidth=1, c=c, zorder=2)
	plt.fill_between(x, density, color=c, lw=1, alpha=0.5, zorder=2)
	
	#plt.xlim(-2,2); plt.ylim(-0.2, 5.2)
	
	plt.title('iteration = '+str(iteration)+' - '+title)
	plt.xlabel('$DFT_i-NN_i$ '+kind+unit)
	plt.ylabel('Normalized Density')
	plt.legend(); plt.grid(ls=':'); plt.tight_layout(pad=1)
	if save: plt.savefig('PICS/histo-'+kind+'-'+str(iteration)+'.png', dpi=200)
	plt.clf()

###------------------------------------------
