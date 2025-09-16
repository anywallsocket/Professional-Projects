#!/usr/bin/env python3

import sys, os, time
import numpy as np
import schnetpack as spk
from tqdm import tqdm
from ase.io import read
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde, lognorm
from scipy.signal import fftconvolve
from collections import Counter
from collections.abc import Iterable

###------------------------------------------
### ANALYSIS FUNCTIONS
###------------------------------------------

def trueFlat(A): #flattens ragged nested arrays
	for a in A:
		if isinstance(a,Iterable) and not isinstance(a,(str,bytes)): 
			yield from trueFlat(a)
		else: yield a


def LoadSchNetCalc(directory, cutoff):

	calculator = spk.interfaces.SpkCalculator(
		model_file=os.path.join(directory, 'best_inference_model'),
		neighbor_list=spk.transform.ASENeighborList(cutoff=cutoff),
		energy_key='energy',
		force_key='forces',
		energy_unit='eV',
		force_units='eV/Ang',
		position_unit='Ang',
	)
	return calculator

def internal(directory, calc, limit=10.0):

	images = read(os.path.join(directory, 'new_dataset.db'), index=':')
	splitfile = np.load(os.path.join(directory, 'split.npz'))
	splits = [splitfile[i] for i in splitfile]
	measures = []
	names = ['train', 'valid', 'test']
	oops = 0

	for i in range(len(splits)):

		trueE = []; predE = []
		trueF = []; predF = []

		for j in tqdm(range(len(splits[i])), desc='predicting '+names[i]+' data'):
			image = images[splits[i][j]]
			tE = image.get_potential_energy()
			tF = image.get_forces()
			copy = image.copy()
			copy.calc = calc
			pE = copy.get_potential_energy()
			pF = copy.get_forces()
			if abs(tE-pE)<limit and (abs(tF-pF)<limit).all():
				trueE.append(tE)
				trueF.append(tF)
				predE.append(pE)
				predF.append(pF)
			else: oops+=1

		print('limiters:', oops)
		trueE = np.array(list(trueFlat(trueE)))
		predE = np.array(list(trueFlat(predE)))
		trueF = np.array(list(trueFlat(trueF)))
		predF = np.array(list(trueFlat(predF)))
		MAEe = np.mean(np.abs(trueE-predE))
		MAEf = np.mean(np.abs(trueF-predF))
		measures.append([trueE, predE, trueF, predF, MAEe, MAEf])

	return measures

###------------------------------------------
### PLOTTING FUNCTIONS
###------------------------------------------

def trainplot(directory, save=False):

	format = ticker.FormatStrFormatter('%.3f')

	data = read_csv(directory+'/lightning_logs/version_0/metrics.csv')
	data.ffill(inplace=True)
	epochs = data['epoch']
	metrics = [[data['train_loss'], data['val_loss']],
			   [data['train_energy_MAE'], data['val_energy_MAE']],
			   [data['train_forces_MAE'], data['val_forces_MAE']]]
	skip = int(0.05*len(epochs))
	
	fig, axes = plt.subplots(3, 1, figsize=(3,3), sharex=True)
	labs = ['train', 'valid']
	colors = ['b', 'g', 'r']
	ylabs = ['loss', 'energy', 'forces']
	for i in range(3):
		ax = axes[i]
		for j in range(2):
			last = np.round(metrics[i][j].iloc[-1], 3)
			ax.plot(epochs[skip:], metrics[i][j][skip:], color=colors[i], 
					alpha=0.5*(j+1), label=labs[j]+': '+str(last))
			ax.set_ylabel(ylabs[i])
			ax.grid(ls=':')
			ax.legend(loc='upper right', facecolor='w')
			ax.yaxis.set_major_formatter(format)
	fig.subplots_adjust(wspace=0.03, hspace=0.1)
	if save: plt.savefig(directory+'/train.png', dpi=100, bbox_inches='tight')
	else: plt.show()
	plt.clf()

def barplot(directory, measures, save=False):

	Max = np.max(measures)

	plt.figure(figsize=(3,3))
	plt.bar(np.arange(3)-0.2, [measures[0],measures[2],measures[4]], 
		label='energy', width=0.4, color='b', edgecolor='k', alpha=0.75, zorder=2)
	plt.bar(np.arange(3)+0.2, [measures[1],measures[3],measures[5]], 
		label='forces', width=0.4, color='r', edgecolor='k', alpha=0.75, zorder=2)

	t1 = plt.text(0-0.1, 0.1*Max, '70%', fontsize=10)
	t1.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
	t2 = plt.text(1-0.1, 0.1*Max, '15%', fontsize=10)
	t2.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))
	t3 = plt.text(2-0.1, 0.1*Max, '15%', fontsize=10)
	t3.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='k'))

	plt.xticks(np.arange(3), ['train', 'valid', 'test'])
	plt.ylabel('MAE')
	plt.legend(facecolor='w'); plt.grid(ls=':', zorder=0)
	if save: plt.savefig(directory+'/bar.png', dpi=100, bbox_inches='tight')
	else: plt.show()
	plt.clf()

def fast_kde(x, y):

	n = len(x)
	factor = n**(-1.0/6.0)
	hx = np.std(x, ddof=1)*factor
	hy = np.std(y, ddof=1)*factor
	bw = np.sqrt(hx*hy)
	bw = max(bw, 1e-3)
	hx = max(hx, 1e-3)
	hy = max(hy, 1e-3)
	bx = min(int(np.ceil((x.max() - x.min()) / hx)), 300)
	by = min(int(np.ceil((y.max() - y.min()) / hy)), 300)

	counts, xedges, yedges = np.histogram2d(x, y, bins=(bx,by), density=False)

	dx = xedges[1] - xedges[0]
	dy = yedges[1] - yedges[0]
	sx = bw/dx
	sy = bw/dy

	nx = int(6*sx) + 1
	ny = int(6*sy) + 1
	xi = np.linspace(-3*sx, 3*sx, nx)
	yi = np.linspace(-3*sy, 3*sy, ny)
	Xk, Yk = np.meshgrid(xi, yi)
	kernel = np.exp(-0.5*((Xk/sx)**2 + (Yk/sy)**2))
	kernel /= kernel.sum()

	smooth = fftconvolve(counts, kernel, mode='same')
	ix = np.clip(np.searchsorted(xedges, x) - 1, 0, bx-1)
	iy = np.clip(np.searchsorted(yedges, y) - 1, 0, by-1)

	return smooth[ix, iy]

def newplot(directory, kind, x, y, name, save=False):

	rmse = np.sqrt(np.mean((y - x)**2))
	mae  = np.mean(np.abs(y - x))

	xy = np.vstack([x, y])
	if kind=='energy': z = gaussian_kde(xy)(xy)
	else: z = fast_kde(x, y)

	idx = z.argsort()
	x_plot, y_plot, z_plot = x[idx], y[idx], z[idx]
	
	fig, ax = plt.subplots(figsize=(3,3))
	
	sc = ax.scatter(x_plot, y_plot, c=z_plot, s=20, cmap='turbo', edgecolor='None', alpha=0.75, zorder=2)
	
	mn = min(ax.get_xlim()[0], ax.get_ylim()[0])
	mx = max(ax.get_xlim()[1], ax.get_ylim()[1])

	print('% MAE relative to range:', np.round(mae/np.abs(np.max(x)-np.min(x))*100, 3))
	ax.plot([mn, mx], [mn, mx], ls='--', lw=0.75, color='k', alpha=0.75, zorder=3)

	text = f"RMSE: {rmse:.2f}\nMAE:  {mae:.2f}"
	ax.text(
		0.05, 0.95, text,
		transform=ax.transAxes,
		va='top', ha='left',
		bbox=dict(boxstyle='round, pad=0.3', edgecolor='gray', facecolor='w', alpha=0.75)
	)

	plt.title(name)
	if kind=='energy': 
		ax.set_xlabel('True energy (eV)')
		ax.set_ylabel('NN energy (eV)')
	else:
		ax.set_xlabel('True forces (eV/atom/\u212B)')
		ax.set_ylabel('NN forces (eV/atom/\u212B)')

	ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
	ax.set_aspect('equal', 'box')
	ax.tick_params(direction='out')
	plt.grid(ls=':', alpha=0.5, zorder=0)
	if save: plt.savefig(directory+'/ratio-'+kind+'.png', dpi=100, bbox_inches='tight')
	else: plt.show()
	plt.clf()

def deltaplot(directory, kind, y1, y2, title, save=False):

	if kind=='energy' or kind=='relax': unit=' (eV)'
	elif kind=='force': unit=' (eV/atom/\u212B)'
	else: print('[either energy, forces, or relax]'); return

	ME = np.round(np.mean(np.abs(y1-y2)), 4)

	index = np.argsort(y1)
	y1 = y1[index]; y2 = y2[index]
	line = [np.min(y2), np.max(y2)]

	fig, ax = plt.subplots(figsize=(3,3), nrows=2, ncols=1, sharex=True)
	x = np.arange(len(y1))

	ax[0].scatter(x, y1, label='True', marker='o', s=100, c='r', alpha=0.5, zorder=2)
	ax[0].scatter(x, y2, label='NN', s=1, c='b', alpha=0.5, zorder=3)
	ax[0].legend()
	ax[0].set_ylabel('energy'+unit)
	ax[0].tick_params(labelbottom=False)    
	ax[0].grid(ls=':', zorder=0)

	delta = (y1-y2)
	ax[1].plot(delta, c='r', linewidth=0.5, alpha=1.0, zorder=2)
	ax[1].scatter(x,delta,label='ME '+str(ME), facecolors='w',
		alpha=0.75, edgecolors='k', s=5, zorder=3)
	ax[1].set_ylabel('$True - NN_i$'+unit)
	ax[1].legend(loc='upper left', facecolor='w')
	ax[1].grid(ls=':', zorder=0)

	plt.subplots_adjust(wspace=0.0, hspace=0.05)
	plt.xlabel('sorted images')
	if save: plt.savefig(directory+'/delta-'+str(kind)+'.png', dpi=100, bbox_inches='tight')
	else: plt.show()
	plt.clf()

###------------------------------------------
### ANALYSIS
###------------------------------------------

I = sys.argv[1]

directory = 'NNs/'+str(I)
calc = LoadSchNetCalc(directory, 5.5)

results = internal(directory, calc)
MAEs = [results[0][4], results[0][5], 
		results[1][4], results[1][5], 
		results[2][4], results[2][5]]

print([np.round(m,3) for m in MAEs])

testE = [results[2][0], results[2][1]]
testF = [results[2][2], results[2][3]]

save = True

if save: np.save(directory+'/maes.npy', MAEs)
title = 'test-'+str(I)

barplot(directory, MAEs, save)
newplot(directory, 'energy', testE[0], testE[1], title, save)
newplot(directory, 'force', testF[0], testF[1], title, save)
deltaplot(directory, 'energy', testE[0], testE[1], title, save)
trainplot(directory, save)

###------------------------------------------
### END
###------------------------------------------