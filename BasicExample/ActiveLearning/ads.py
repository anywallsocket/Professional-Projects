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

def adsE(image):
	
	clust = image[:N]
	clust.calc = calc
	image.calc = calc
	TE = image.get_total_energy()
	CE = clust.get_total_energy()
	AdsE = TE - CE - EMT_energies[names[J%5]]
	return AdsE

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
	positions = []
	
	while len(positions) < len(cluster) and iterations>0:
	
		point = np.random.uniform(0, box, size=3)
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

def relaxCluster(test, calc, steps=250):
	
	cluster = test.copy()
	cluster.calc = calc
	sim = BFGS(cluster, trajectory=None, logfile=None)
	sim.run(fmax=0.01, steps=steps)
	cluster.center()
	return cluster

def fix(image):
	# set up for molecules of 1 or 2 bonds
	cluster_indices = np.arange(N)
	image.set_constraint(None)
	bond_indices = []
	if len(adsorbate)==2:
		image.set_constraint([FixAtoms(cluster_indices), FixBondLength(N, N+1)])
	elif len(adsorbate)>2:
		bond_indices = [[N,N+1], [N,N+2], [N,N+3], [N+1, N+2], [N+2, N+3], [N+1, N+3]]
		image.set_constraint([FixAtoms(cluster_indices), FixBondLengths(bond_indices)])
	else: image.set_constraint([FixAtoms(cluster_indices)])
	return image


###------------------------------------------
### INPUTS
###------------------------------------------

I = int(sys.argv[1])
J = int(sys.argv[2])
Z = ['Ag', 'Au', 'Cu', 'Ni', 'Pd', 'Pt']
N = np.random.randint(10,50)

pop_size = 50

if N<20: box = 6
elif N<30: box = 7
elif box<40: box = 8
dmin = 2.2
dmax = box*np.sqrt(3)

if I==0:
	calc = EMT()
else:
	calc = LoadSchNetCalc('NNs/'+str(I-1), 5.5)

stoi = np.random.choice(Z, N)
gen = genCluster(Atoms(stoi, cell=[20,20,20], pbc=False))
cluster = relaxCluster(gen, EMT(), 500)

box = [(0.,0.,0.), ((20.,0.,0.), (0.,20.,0.), (0.,0.,20.))]

db_file = 'DBs/'+str(I)+'/work-'+str(J)+'.db'
try: os.remove(db_file)
except: pass

names = ['H', 'H2', 'N', 'N2', 'NH3']
mol = molecule(names[J%5])
print('[J='+str(J)+' Start]')

adsorbate = Atoms(mol.symbols, positions=mol.positions, cell=[20,20,20], pbc=False)
adsorbate.center()
unit = [adsorbate]
atom_numbers = adsorbate.get_atomic_numbers()
unique_atom_types = get_all_atom_types(cluster, atom_numbers)
blmin = closest_distances_generator(atom_numbers=unique_atom_types, ratio_of_covalent_radii=1.0)

print('[Initializing Candidates]')
sg = StartGenerator(cluster, unit, blmin, box_to_place_in=box)
d = PrepareDB(db_file_name=db_file, simulation_cell=cluster, stoichiometry=atom_numbers)
images = [sg.get_new_candidate() for i in tqdm(range(pop_size))]
for im in images: d.add_unrelaxed_candidate(im)

da = DataConnection(db_file)
atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
n_to_optimize = len(adsorbate)

comp = InteratomicDistanceComparator(n_top=n_to_optimize, pair_cor_cum_diff=0.03, pair_cor_max=0.7, dE=0.05, mic=False)
pairing = CutAndSplicePairing(cluster, n_to_optimize, blmin, use_tags=True)
rattlemut = RattleMutation(blmin, n_to_optimize, rattle_prop=0.3, rattle_strength=0.5, use_tags=True)
rotmut = RotationalMutation(blmin, fraction=0.3, min_angle=0.5*np.pi)
rattlerotmut = RattleRotationalMutation(rattlemut, rotmut)
mutations = OperationSelector([1], [rattlemut])

population = Population(data_connection=da, 
			population_size=pop_size, 
			comparator=comp, use_extinct=True)

###------------------------------------------
### INITIAL RELAXATION
###------------------------------------------

print('[Relaxing Population]')
while da.get_number_of_unrelaxed_candidates() > 0:

	a = da.get_an_unrelaxed_candidate()
	a = fix(a)
	a.calc = calc
	e = a.get_total_energy()
	f = a.get_forces()
	a.info['key_value_pairs']['raw_score'] = adsE(a)
	a.calc = SinglePointCalculator(a)
	a.calc.results['energy'] = np.float64(e)
	a.calc.results['forces'] = np.float64(f)
	da.add_relaxed_step(a)

population.update()

###------------------------------------------
### POOL OPTIMIZATION
###------------------------------------------

fmin = 0.001
steps = 10
n2e = pop_size

print('[Evolving Population]')
print('fmin='+str(fmin), 'steps='+str(steps))

for i in tqdm(range(n2e)):

	a = None
	while a is None:
		a = population.get_one_candidate()
		a, desc = mutations.get_new_individual([a])
	a = fix(a)
	da.add_unrelaxed_candidate(a, description='mutation: rattle')
	a.calc = calc
	dyn = BFGS(a, trajectory=None, logfile=None)
	dyn.run(fmax=fmin, steps=steps)
	e = a.get_total_energy()    
	f = a.get_forces()
	a.info['key_value_pairs']['raw_score'] = adsE(a)
	a.calc = SinglePointCalculator(a)
	a.calc.results['energy'] = np.float64(e)
	a.calc.results['forces'] = np.float64(f)
	da.add_relaxed_candidate(a)
	population.update()

print('[J='+str(J)+' Done]')

###------------------------------------------
### OUPUTS
###------------------------------------------

images = [i for i in da.get_all_relaxed_candidates()]
index = np.argsort([adsE(i) for i in images])
ordered = [images[i] for i in index]
write('DBs/'+str(I)+'/ga-'+str(J)+'.db', ordered[:n2e])
try: os.remove(db_file)
except: pass

###------------------------------------------
### END
###------------------------------------------