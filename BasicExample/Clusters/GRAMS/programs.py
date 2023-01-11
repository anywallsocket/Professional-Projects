#!/usr/bin/env python3
#Mini Library of useful functions -- most algorithms are in-place

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from ase.db import connect
from amptorch import AtomsTrainer
from amptorch import AMPtorch
import torch

def myprint(*kargs):
    s = ''
    for a in kargs: s=s+' '+str(a)
    with open('output', 'a') as f:
        f.write(s+'\n')

def myplot(kind, x, y, iteration):

    MAE = round(np.mean(np.abs(x-y)),3)
    line=[np.min(y), np.max(y)]
    if kind=='energy' or kind=='relax': unit=' (eV)'
    elif kind=='force': unit=' (eV/A)'
    else: print('[either energy, forces, or relax]'); return
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]
    plt.plot(line, line, c='black', linestyle='dotted', linewidth=0.5, label=' MAE', zorder=3)
    plt.scatter(x, y, c=z, s=50, alpha=0.5, label=str(MAE), zorder=2)
    plt.colorbar().set_ticks([])
    plt.title(str(iteration))
    plt.xlabel('LMP '+kind+unit)
    plt.ylabel('NN '+kind+unit)
    plt.legend()
    plt.grid(zorder=1)
    if iteration !=0: plt.savefig('PICS/'+kind+str(iteration)+'.png', dpi=200)
    plt.close()

def LoadAmpTorch(kind, iteration):

    checkpoint = 'checkpoints/NN'+str(iteration-1)
    config = torch.load(checkpoint+'/config.pt')
    
    if iteration > 0:
        config['optim']['force_coefficient'] = int(10/iteration)
        config['cmd']['identifier'] = 'LMP'+str(iteration)
        if kind == 'old': config['cmd']['logger'] = False
        else: config['cmd']['logger'] = False
        torch.save(config, checkpoint+'/config.pt')

    if kind == 'old':
        trainer = AtomsTrainer()
        trainer.load_pretrained(checkpoint)
        calc = AMPtorch(trainer)
        return calc
    
    if kind == 'new':
        #load new images 
        data = 'DBs/train'+str(iteration)+'.db'
        db = connect(data)
        images = []
        for i in range(len(db)): images.append(db.get_atoms(id=i+1))
        if (len(images)==0): print('empty atoms object'); exit(1)
        config['dataset']['raw_data'] = images
        trainer = AtomsTrainer(config)
        return trainer
    
    else: print('kind must be new or old'); exit(1)

def write_train(iteration):

    #Sort and prune for bad images / duplicates
    work = connect('DBs/working.db')
    train = connect('DBs/train'+str(iteration)+'.db')
    energies = []
    for i in range(len(work)):
        energies.append(work.get_atoms(id=i+1).get_potential_energy())

    indexsort = np.argsort(energies)
    uniqeindexsort = np.unique(np.round(np.sort(energies), 3), return_index=True)[1]

    #Filter edge cases
    for i in uniqeindexsort:
        c = work.get_atoms(id=int(indexsort[int(i)])+1)

        #skip any positions outside the box (Ang)
        pos = c.positions
        for j in pos:
            for k in j:
                if k <= 0 or k >= 20: break
            else: continue
            break
        #skip unstable structures (eV)
        else: 
            if c.get_potential_energy() < -10 and c.get_potential_energy() > -50: 
                train.write(c)

    myprint(len(train), 'images written to train.db')

###
