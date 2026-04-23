#!/usr/bin/env python3

import yaml, time, os, sys
from ase.io import read, write
import numpy as np
from mace.cli.run_train import main as mace_run_train_main
from mace.cli.eval_configs import run as mace_eval_configs_run
import logging
from tqdm import tqdm

I = int(sys.argv[1])
GAs = int(sys.argv[2])

directory = 'NNs/'+str(I)+'/'
try: os.mkdir(directory)
except: pass

from mace.calculators import MACECalculator
calc = MACECalculator(model_paths='r2scan.model', default_dtype='float64')

print('[Collecting Images]')
#Here we can either re-train whole network or continue.

images = []
for i in tqdm(range(GAs)):
	try:
		ga = read('DBs/'+str(I)+'/ga-'+str(i)+'.db', index=':')
		for g in ga:
			new = g.copy()
			new.calc = calc
			new.info['REF_energy'] = new.get_total_energy()
			new.arrays['REF_forces'] = new.get_forces()
			new.calc = None
			images.append(new)
	except: print('failed to read', I, 'ga-'+str(i)+'.db')
print('New Found:', len(images))
if I>0:
	for split in ['train', 'valid']:
		prev = read('NNs/'+str(I-1)+'/'+split+'.xyz', index=':')
		for p in prev: 
			images.append(p)
assert len(images)>0, 'no images read'
print('Total Found:', len(images))

N = len(images)
n_train = int(N*0.8)
index = np.random.permutation(N)
train = [images[i] for i in index[:n_train]]
valid = [images[i] for i in index[n_train:]]

write(directory+'train.xyz', train)
write(directory+'valid.xyz', valid)

config_file_path = directory+"config.yml"
os.makedirs(os.path.dirname(config_file_path), exist_ok=True)

config_data = {
	"model": "MACE",
	"num_channels": 100,
	"max_L": 1,
	"r_max": 5.5,
	"energy_weight": 1.0,
	"forces_weight": 2.0,
	"loss": "weighted",
	"optimizer": "schedulefree",
	"default_dtype": "float32",
	"name": str(I),
	"model_dir": directory,
	"log_dir": directory,
	"checkpoints_dir": directory,
	"results_dir": directory,
	"train_file": directory+"train.xyz",
	"valid_file": directory+"valid.xyz",
	"energy_key": "REF_energy",
	"forces_key": "REF_forces",
	"E0s": "average",
	"device": "cuda",
	"batch_size": 4,
	"max_num_epochs": 20,
	"swa": False,
	#"start_swa": 40,
	#"swa_energy_weight": 1000,
	#"swa_forces_weight": 100,
	"ema": True,
	"ema_decay": 0.99,
	"seed": 123,
	"enable_cueq": False,
	"plot": False
}

with open(config_file_path, 'w') as f:
	yaml.dump(config_data, f, default_flow_style=False) 

def train_mace(config_file_path):
	logging.getLogger().handlers.clear() 
	sys.argv = ["program", "--config", config_file_path] 
	mace_run_train_main()

t1 = time.time()
train_mace(config_file_path)
t2 = time.time()

minutes = round((t2-t1)/60,2)
print('Training complete --', minutes, 'min')

###