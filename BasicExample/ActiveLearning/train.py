#!/usr/bin/env python3

import numpy as np
import schnetpack as spk

import sys, os, time, random
import torch, torchmetrics
from ase.io import read, write
import pytorch_lightning as pl
from schnetpack.data import ASEAtomsData, AtomsDataModule
from pytorch_lightning.callbacks import RichProgressBar

###------------------------------------------
### NETWORK FUNCTIONS
###------------------------------------------

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

#Analyzes NN preformance against itself
def internal(directory, calc, limit=10.0):

	db = connect(os.path.join(directory, 'new_dataset.db'))
	images = [db.get_atoms(id=j+1) for j in range(len(db))]
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

		print('oopsies:', oops)
		trueE = np.array(list(trueFlat(trueE)))
		predE = np.array(list(trueFlat(predE)))
		trueF = np.array(list(trueFlat(trueF)))
		predF = np.array(list(trueFlat(predF)))
		MAEe = np.mean(np.abs(trueE-predE))
		MAEf = np.mean(np.abs(trueF-predF))
		measures.append([trueE, predE, trueF, predF, MAEe, MAEf])

	return measures

# Analyzes NN against external dataset
def external(images, calc, limit=10.0):
	
	trueE = []; predE = []
	trueF = []; predF = []
	valid = []; inval = []
	oops = 0
	
	for i in tqdm(range(len(images)), desc='predicting images'):
		
		image = images[i]
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
			valid.append(image)
		else: 
			oops+=1
			inval.append(image)
			
	print('oopsies:', oops)
	trueE = np.array(list(trueFlat(trueE)))
	predE = np.array(list(trueFlat(predE)))
	trueF = np.array(list(trueFlat(trueF)))
	predF = np.array(list(trueFlat(predF)))
	MAEe = np.mean(np.abs(trueE-predE))
	MAEf = np.mean(np.abs(trueF-predF))
	measures = [trueE, predE, trueF, predF, MAEe, MAEf]

	return valid, inval, measures

def myModule(directory, images, bs):

	#First convert images to ASEAtomsData
	try: os.remove(os.path.join(directory, 'new_dataset.db'))
	except: pass

	property_list = []
	for i in images:
		energy = np.array([i.get_potential_energy()])
		forces = np.array(i.get_forces())
		property_list.append({'energy': energy, 'forces': forces})

	new_dataset = ASEAtomsData.create(
		os.path.join(directory, 'new_dataset.db'), 
		distance_unit='Ang',
		property_unit_dict={'energy':'eV/Ang', 'forces':'eV/Ang'}
	)

	new_dataset.add_systems(property_list, images)

	#Now wrap AtomsData in DataModule class
	DM = AtomsDataModule(
		os.path.join(directory, 'new_dataset.db'),
		split_file = os.path.join(directory, 'split.npz'),
		batch_size=bs, val_batch_size=bs, test_batch_size=bs,
		num_train=int(0.7*len(images)), 
		num_val=int(0.15*len(images)), 
		num_test=int(0.15*len(images)),
		load_properties = ['energy', 'forces'],
		transforms=[
			spk.transform.ASENeighborList(cutoff=5.5),
			spk.transform.RemoveOffsets('energy', remove_mean=True, remove_atomrefs=False),
			spk.transform.CastTo32()
		],
		num_workers=0, # threads for dataloader
		pin_memory=False, # set to false, when not using a GPU
	)

	DM.prepare_data()
	DM.setup()

	return DM

def myModel(lr, features, interactions, gaussians):

	cutoff = 5.5
	
	#Build model
	radial_basis = spk.nn.GaussianRBF(n_rbf=gaussians, cutoff=cutoff)
	schnet = spk.representation.SchNet(
		n_atom_basis=features, n_interactions=interactions,
		radial_basis=radial_basis,
		cutoff_fn=spk.nn.CosineCutoff(cutoff)
	)
	
	#Assign training features
	pred_energy = spk.atomistic.Atomwise(n_in=features, output_key='energy')
	pred_forces = spk.atomistic.Forces(energy_key='energy', force_key='forces')
	
	#Build potential
	nnpot = spk.model.NeuralNetworkPotential(
		representation=schnet,
		input_modules=[spk.atomistic.PairwiseDistances()],
		output_modules=[pred_energy, pred_forces],
		postprocessors=[
			spk.transform.CastTo64(),
			spk.transform.AddOffsets('energy', add_mean=True, add_atomrefs=False)
		]
	)
	
	#Define loss weights for features
	output_energy = spk.task.ModelOutput(
		name='energy',
		loss_fn=torch.nn.HuberLoss(delta=1.0),
		loss_weight=0.1,
		metrics={"MAE": torchmetrics.MeanAbsoluteError()}
	)
	output_forces = spk.task.ModelOutput(
		name='forces',
		loss_fn=torch.nn.HuberLoss(delta=1.0),
		loss_weight=0.9,
		metrics={"MAE": torchmetrics.MeanAbsoluteError()}
	)
	
	scheduler_args = {'mode': 'min', 'factor': 0.75, 'patience': 5, 
						'threshold': 5e-3, 'threshold_mode': 'rel', 
						'cooldown': 5, 'min_lr': 0,
	}

	#Define task
	task = spk.task.AtomisticTask(
		model=nnpot,
		outputs=[output_energy, output_forces],
		optimizer_cls=torch.optim.AdamW,
		optimizer_args={'lr': lr},
		scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
		scheduler_monitor='val_loss',
		scheduler_args=scheduler_args,
	)

	return task

def myTrainer(directory, epochs):

	#Assign logs
	logger = pl.loggers.CSVLogger(save_dir=directory)
	callbacks = [
		RichProgressBar(),
		spk.train.ModelCheckpoint(
			model_path=os.path.join(directory, 'best_inference_model'),
			save_top_k=1,
			monitor='val_loss'
		),
		pl.callbacks.EarlyStopping(monitor='val_loss', patience=15),
	]
	
	#Define trainer
	trainer = pl.Trainer(
		accelerator='auto',
		callbacks=callbacks,
		logger=logger,
		log_every_n_steps=10,
		default_root_dir=directory,
		max_epochs=epochs,
	)
	
	return trainer

###------------------------------------------
### TRAINING
###------------------------------------------

I = int(sys.argv[1])
GAs = int(sys.argv[2])

# Set up the hyperparameters
epochs = 50
batch_size = 20
learning_rate = 1e-3
features = 50
interactions = 10
gaussians = 10

directory = 'NNs/'+str(I)+'/'
try: 
	os.remove('splitting.lock')
	os.mkdir(directory)
except: pass

print('[Collecting Images]')
images = []
for i in range(I+1):
	for j in range(GAs):
		try:
			db = read('DBs/'+str(i)+'/ga-'+str(j)+'.db', index=':')
			for d in db: images.append(d)
		except: pass #print(j, end=',')
print('Images Found:', len(images))

dm = myModule(directory, images, batch_size)
task = myModel(learning_rate, features, interactions, gaussians)
trainer = myTrainer(directory, epochs)

print('[Training]')
t1 = time.time()

trainer.fit(task, datamodule=dm)

t2 = time.time()
minutes = round((t2-t1)/60,2)
print('Training Time:', minutes, '(m)')

###------------------------------------------
### DONE
###------------------------------------------