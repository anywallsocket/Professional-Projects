#!/usr/bin/env python3
import sys, time, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from mace.cli.eval_configs import main as mace_eval_main
from ase.io import read, write
from tqdm import tqdm
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

def evaluate_mace_cli(model, configs, output, device='cpu', batch_size=1):
	cli_args = [
		'mace_eval_configs',
		'--default_dtype', 'float32',
		'--model', str(model),
		'--configs', str(configs),
		'--output', str(output),
		'--device', str(device),
		'--batch_size', str(batch_size)
	]
	with patch('sys.argv', cli_args): mace_eval_main()

def analyze_split(split, e_limit=5.0, f_limit=20.0, save=True):

	images = read(NN+split+'-eval.xyz', index=':')
	assert len(images) > 0, 'no images found'

	good, bad = [], []
	E_true, E_pred = [], []
	F_true, F_pred = [], []

	for atoms in tqdm(images):
		e_true = atoms.info['REF_energy']
		e_pred = atoms.info['MACE_energy']
		f_true = atoms.arrays['REF_forces']
		f_pred = atoms.arrays['MACE_forces']

		n_atoms = len(atoms)
		E_true.append(e_true/n_atoms)
		E_pred.append(e_pred/n_atoms)
		F_true.append(f_true)
		F_pred.append(f_pred)

		e_err = np.abs(e_true - e_pred)
		f_err_max = np.max(np.abs(f_true - f_pred))

		if e_err < e_limit and f_err_max < f_limit: good.append(atoms)
		else:
			atoms.info['error_E'] = e_err
			atoms.info['error_F_max'] = f_err_max
			bad.append(atoms)

	E_true = np.array(E_true)
	E_pred = np.array(E_pred)
	F_true = np.concatenate(F_true)
	F_pred = np.concatenate(F_pred)

	e_mae = np.mean(np.abs(E_true - E_pred))
	f_mae = np.mean(np.abs(F_true - F_pred))
	
	f_max_overall = np.max(np.abs(F_true - F_pred)) 

	N = len(images)
	p_good = 100*len(good)/N
	p_bad = 100*len(bad)/N

	print('['+split+']')
	print(f'Total: {N} images | {p_good:.1f}% Good | {p_bad:.1f}% Bad')
	print(f'Energy MAE: {e_mae:.4f} eV/atom')
	print(f'Force MAE:  {f_mae:.4f} eV/Å (Max error: {f_max_overall:.4f} eV/Å)')

	if save:
		parity(E_true, E_pred, 'energy', title=split)
		parity(F_true, F_pred, 'force', title=split)
		if bad: write(NN+split+'_bad.xyz', bad)
		
	return good, bad

###------------------------------------------
### PLOTTERS
###------------------------------------------

def parse_training_log(log_path):
	data = []
	with open(log_path, 'r') as f:
		for line in f:
			try: data.append(json.loads(line))
			except json.JSONDecodeError: continue

	df = pd.DataFrame(data)
	if df.empty: raise ValueError(f"No valid data found in {log_path}")

	df['epoch'] = df['epoch'].ffill()
	eval_df = df[df['mode'] == 'eval'].copy()
	eval_df['epoch'] = eval_df['epoch'].fillna(0).astype(int)

	# Group by epoch and average the metrics
	plot_df = eval_df.groupby('epoch').agg({
		'loss': 'mean',
		'rmse_e': 'mean',
		'rmse_f': 'mean',
	}).reset_index()

	return plot_df

def plot_metrics(df, save=True):
	fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
	
	metrics = [
		('loss', 'Loss', 'b'),
		('rmse_e', 'Energy RMSE (eV)', 'b'),
		('rmse_f', 'Force RMSE (eV/Å)', 'r')
	]
	
	end_vals = []
	
	for ax, (col, ylabel, color) in zip(axs, metrics):
		mean_val = df[col].mean()
		last_val = df[col].iloc[-1]
		end_vals.append(last_val)
		
		ax.plot(df['epoch'], df[col], color=color)
		
		# Dynamic Y-limit: Ignore the first 10% of epochs to bypass the initial spike
		ignore_idx = int(len(df) * 0.1)
		if ignore_idx < len(df):
			ymax = df[col].iloc[ignore_idx:].max() * 1.05
			ax.set_ylim(0, ymax)
		
		ax.set_xlabel('Epoch')
		ax.set_ylabel(ylabel)
		ax.grid(True, linestyle='--', alpha=0.6)
		
		# Place average and final values cleanly inside the plot
		text_str = f'Avg: {mean_val:.3f}\nEnd: {last_val:.3f}'
		ax.text(0.95, 0.95, text_str, transform=ax.transAxes, 
				fontsize=10, verticalalignment='top', horizontalalignment='right',
				bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
	
	print(f"[Training Errors] L: {end_vals[0]:.3f}, E: {end_vals[1]:.3f}, F: {end_vals[2]:.3f}")

	if save: plt.savefig(NN+'plot-train.png', dpi=200, bbox_inches='tight')
	plt.clf()

def get_fast_density(x, y, bins=150, smooth=2.0):
	
	H, xedges, yedges = np.histogram2d(x, y, bins=bins)
	H_smooth = gaussian_filter(H, sigma=smooth)
	
	xidx = np.clip(np.searchsorted(xedges, x) - 1, 0, bins - 1)
	yidx = np.clip(np.searchsorted(yedges, y) - 1, 0, bins - 1)
	
	return H_smooth[xidx, yidx]

def parity(x, y, kind, title=''):

	try: x = x.ravel(); y = y.ravel()
	except: pass
	rmse = np.sqrt(np.mean((y - x)**2))
	mae = np.mean(np.abs(y - x))
	maep = np.round(mae / np.abs(np.max(x) - np.min(x)) * 100, 3)

	if kind == 'energy':
		xy = np.vstack([x, y])
		z = gaussian_kde(xy)(xy)
	else:
		z = get_fast_density(x, y)
		
	z = (z - z.min()) / (z.max() - z.min())
	idx = z.argsort()
	x_plot, y_plot, z_plot = x[idx], y[idx], z[idx]

	fig, ax = plt.subplots(figsize=(3, 3))
	ax.scatter(x_plot, y_plot, c=z_plot, s=20, cmap='turbo', edgecolor='none', alpha=0.75, zorder=2)

	lims = [
		np.min([ax.get_xlim(), ax.get_ylim()]),  
		np.max([ax.get_xlim(), ax.get_ylim()]),  
	]
	ax.plot(lims, lims, 'k--', alpha=0.5, zorder=1)
	ax.set_xlim(lims)
	ax.set_ylim(lims)
	
	text = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nMAE%: {maep:.2f}'
	ax.text(
		0.05, 0.95, text, transform=ax.transAxes,
		va='top', ha='left',
		bbox=dict(boxstyle='round, pad=0.3', edgecolor='gray', facecolor='w', alpha=0.75)
	)

	labels = {
		'energy': ('REF energy (eV/atom)', 'NN energy (eV/atom)'),
		'force': ('REF forces (eV/Å)', 'NN forces (eV/Å)')
	}
	ax.set_xlabel(labels.get(kind)[0])
	ax.set_ylabel(labels.get(kind)[1])
	ax.tick_params(direction='out')
	ax.grid(ls=':', alpha=0.5, zorder=0)
	plt.title(str(I)+'-'+title)
	plt.savefig(NN+title+'-'+str(kind)+'.png', dpi=200, bbox_inches='tight')
	plt.clf()

###------------------------------------------
### CONFIG
###------------------------------------------

I = int(sys.argv[1])
seed = 123
NN = 'NNs/'+str(I)+'/'

t1 = time.time()

print('[Evaluating Training Data]')
evaluate_mace_cli(
	model=NN+str(I)+'_compiled.model',
	configs=NN+'train.xyz',
	output=NN+'train-eval.xyz'
)
print('[Evaluating Validation Data]')
evaluate_mace_cli(
	model=NN+str(I)+'_compiled.model',
	configs=NN+'valid.xyz',
	output=NN+'valid-eval.xyz'
)

#Training Plot
log_path = NN+str(I)+'_run-'+str(seed)+'_train.txt'
df = parse_training_log(log_path)
plot_metrics(df)

#Parity Plots
train_good, train_bad = analyze_split('train', e_limit=1.0, f_limit=2.0)
valid_good, valid_bad = analyze_split('valid', e_limit=1.0, f_limit=2.0)

# Option to define which data to perpetuate here.

t2 = time.time()

minutes = round((t2-t1)/60,2)
print('[Analysis Complete --', minutes, 'min]')


###------------------------------------------
### END
###------------------------------------------
