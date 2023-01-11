#!/usr/bin/env python3
import sys
import torch
import numpy as np
from ase.db import connect
from ase.io.trajectory import Trajectory
from amptorch.ase_utils import AMPtorch
from amptorch.trainer import AtomsTrainer
from programs import *

iteration = sys.argv[1]
DB = connect('DBs/GA0.db')
if len(DB) == 0: print('bad database'); exit()
images = [DB.get_atoms(id=i+1) for i in range(len(DB))]

#CONFIGURATION

Gs = {
    "default": {
        "G2": {
            "etas": np.logspace(np.log10(0.01), np.log10(10.0), num=10),
            "rs_s": [0],
        },
        "G4": {"etas": [0.005], "zetas": [1.0, 4.0], "gammas": [1.0, -1.0]},
        "cutoff": 5,
    },
}

config = {
    "model": {
        "num_layers": 3,
        "num_nodes": 7,
        "get_forces": True,
        "batchnorm": False,
        "activation": torch.nn.Tanhshrink,
    },
    "optim": {
        "gpus": 0,
        "force_coefficient": 1.0,
        "lr": 0.00005,
        "batch_size": 32,
        "epochs": 1000,
        "optimizer": torch.optim.Adam,
        "loss": "mse",
        "metric": "mae",
    },
    "dataset": {
        "raw_data": images,
        "val_split": 0.0,
        "fp_scheme": "gaussian",
        #"fp_scheme": "mcsh",
        "fp_params": Gs,
        #"cutoff_params": {"cutoff_func": "Polynomial", "gamma": 2.0},
        "cutoff_params": {"cutoff_func": "Cosine"},
        "save_fps": False,
        "scaling": {"type": "normalize", "range": (0, 1)}
        #"scaling": {"type": "standardize"}
    },
    "cmd": {
        "debug": False,
        "dtype": torch.FloatTensor,
        "run_dir": "./",
        "seed": 0,
        "identifier": None,
        "verbose": False,
        "logger": True,
    },
}

#TRAINING

trainer = AtomsTrainer(config)
trainer.train()
predictions = trainer.predict(images)

#ANALYSIS

#Energy plot
xe = np.array([image.get_potential_energy() for image in images])
ye = np.array(predictions["energy"])
myplot('energy', xe, ye, iteration)

#Force plot
xf = np.array([image.get_forces() for image in images]).flatten()
yf = np.array(predictions["forces"]).flatten()
myplot('force', np.array(xf), np.array(yf), iteration)

###
