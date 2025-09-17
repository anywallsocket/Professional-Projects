#!/usr/bin/env python3

#Basic imports
import os, sys
import numpy as np
from tqdm import tqdm
from ase.atoms import Atoms
from ase.constraints import FixAtoms, FixBondLength, FixBondLengths
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_numbers
from ase.build import molecule
from ase.db import connect

#GA imports
from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.offspring_creator import OperationSelector
from ase.ga.standardmutations import *
from ase.ga.convergence import GenerationRepetitionConvergence
import schnetpack as spk

###------------------------------------------
### UTILITY FUNCTIONS
###------------------------------------------

EMT_energies = {'H' : 3.21, 'H2' : 1.1588634908331565, 'N' : 5.1, 'N2' : 0.5487647766267809, 'NH3' : 3.3665072530369713}

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

def genCluster(test):

	cluster = test.copy()
	iterations = 10000

	dmin = 2.2
	if len(test)<20: cell = 6
	elif len(test)<30: cell = 7
	elif len(test)<40: cell = 8 
	else: cell = 10
	dmax = cell*np.sqrt(3)
	
	positions = []
	
	while len(positions) < len(cluster) and iterations>0:
	
		point = np.random.uniform(0, cell, size=3)
		if positions:
			distances = np.linalg.norm(np.array(positions)-point, axis=1)
			if not all(dmin < d < dmax for d in distances):
				iterations -= 1
				continue
		positions.append(point)
		iterations-=1
	
	if iterations>0: 
		cluster.positions = positions
		cluster.center()
		return cluster
	else: print('failed'); return None

###------------------------------------------
### INPUTS
###------------------------------------------

I = int(sys.argv[1])
J = int(sys.argv[2])
Z = ['Ag', 'Au', 'Cu', 'Ni', 'Pd', 'Pt']
N = np.random.randint(10,20)

pop_size = 50

if I==0: calc = EMT()
else: calc = LoadSchNetCalc('NNs/'+str(I-1), 5.5)

stoi = np.random.choice(Z, N)
cluster = Atoms('', cell=[20, 20, 20], pbc=False)
box = [(3.,3.,3.), ((14.,0.,0.), (0.,14.,0.), (0.,0.,14.))]

db_file = 'DBs/'+str(I)+'/work-'+str(J)+'.db'
try: os.remove(db_file)
except: pass

print('[J='+str(J)+' Start]')

atom_numbers = [atomic_numbers[s] for s in stoi]
unique_atom_types = get_all_atom_types(cluster, atom_numbers)
blmin = closest_distances_generator(atom_numbers=unique_atom_types, ratio_of_covalent_radii=0.75)

print('[Generating Population]')
sg = StartGenerator(cluster, stoi, blmin, box_to_place_in=box)
d = PrepareDB(db_file_name=db_file, simulation_cell=cluster, stoichiometry=atom_numbers)
for i in tqdm(range(pop_size)):	
	c = genCluster(Atoms(stoi, cell=[20,20,20], pbc=False))
	d.add_unrelaxed_candidate(c)

da = DataConnection(db_file)
comp = InteratomicDistanceComparator(n_top=N, 
				pair_cor_cum_diff=0.03, 
				pair_cor_max=0.7, dE=0.01, mic=False)

pairing = CutAndSplicePairing(cluster, N, blmin)
mutations = OperationSelector([0.2, 0.3, 0.5],
			[MirrorMutation(blmin, N),
			RattleMutation(blmin, N),
			PermutationMutation(N)])

###------------------------------------------
### INITIAL RELAXATION
###------------------------------------------

fmax = 0.001
i_steps = 50

print('[Relaxing Population]')
print('fmax='+str(fmax), 'steps='+str(i_steps))

while da.get_number_of_unrelaxed_candidates() > 0:

	a = da.get_an_unrelaxed_candidate()
	a.calc = calc
	dyn = BFGS(a, trajectory=None, logfile=None)
	dyn.run(fmax=fmax, steps=i_steps)
	e = a.get_potential_energy()
	f = a.get_forces()
	a.info['key_value_pairs']['raw_score'] = e
	a.calc = SinglePointCalculator(a)
	a.calc.results['energy'] = np.float64(e)
	a.calc.results['forces'] = np.float64(f)
	da.add_relaxed_step(a)

population = Population(data_connection=da, population_size=pop_size, comparator=comp)

#population.update()

###------------------------------------------
### POOL OPTIMIZATION
###------------------------------------------

steps = 10
n2e = pop_size
mutation = 0.4

print('[Evolving Population]')
print('fmax='+str(fmax), 'steps='+str(steps))

for i in tqdm(range(n2e)):

	a1, a2 = population.get_two_candidates()
	a3, desc = pairing.get_new_individual([a1, a2])
	if a3 is None: continue
	da.add_unrelaxed_candidate(a3, description=desc)

	if np.random.random() < mutation:
			a3_mut, desc = mutations.get_new_individual([a3])
			if a3_mut is not None:
					da.add_unrelaxed_step(a3_mut, description=desc)
					a3 = a3_mut

	a3.calc = calc
	dyn = BFGS(a3, trajectory=None, logfile=None)
	dyn.run(fmax=fmax, steps=steps)
	e = a3.get_potential_energy()
	f = a3.get_forces()
	a3.info['key_value_pairs']['raw_score'] = e
	a3.calc = SinglePointCalculator(a3)
	a3.calc.results['energy'] = np.float64(e)
	a3.calc.results['forces'] = np.float64(f)
	da.add_relaxed_step(a3)
	population.update()

print('[J='+str(J)+' Done]')

###------------------------------------------
### OUPUTS
###------------------------------------------

images = [i for i in da.get_all_relaxed_candidates()]
write('DBs/'+str(I)+'/ga-'+str(J)+'.db', images[:n2e])
try: os.remove(db_file)
except: pass

###------------------------------------------
### END
###------------------------------------------
