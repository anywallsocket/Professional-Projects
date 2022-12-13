#!/usr/bin/env python3

#General imports
from programs import *
import sys, time, random, os
import numpy as np
from tqdm import tqdm
from random import random as rand
from ase import Atoms
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.db import connect
from ase.data import atomic_numbers
from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.constraints import FixAtoms

#GA imports
from ase.ga.utilities import *
from ase.ga import get_parametrization
from ase.ga.data import DataConnection
from ase.ga.population import Population
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.ga.offspring_creator import OperationSelector
from ase.ga.standardmutations import MirrorMutation
from ase.ga.standardmutations import RattleMutation
from ase.ga.standardmutations import PermutationMutation

###------------------------------------------
### HELPER FUNCTIONS
###------------------------------------------

def combine_parameters(conf):
	#Get and combine selected parameters
	parameters = []
	gets = [get_atoms_connections(conf) + get_rings(conf) +
			get_angles_distribution(conf) + get_atoms_distribution(conf)]
	for get in gets:
		parameters += get
	return parameters

def skip(conf, comparison_energy, weights):
	parameters = combine_parameters(conf)
	#Return if weights not defined (too few completed
	#calculated structures to make a good fit)
	if weights is None: return False
	regression_energy = sum(p * q for p, q in zip(weights, parameters))
	# Skip with 90% likelihood if energy appears to go up 5 eV or more
	if (regression_energy - comparison_energy) > 5 and rng.random() < 0.9: return True
	else: return False

###------------------------------------------
### SET UP
###------------------------------------------

#Update RNG
rng = np.random.default_rng()
start_time = time.time()

#BASIC SET UP
iteration = int(sys.argv[1])
meta = int(sys.argv[2])

#Load up pre-randomized stoichiometries
S = np.load('S13.npy')

if meta==0 and iteration==0: calc = EMT()
else: calc = LoadSchNetCalc('NNs/'+str(iteration-1))

ALL = [] #Collect all images per metalcount per stoichiometry

###------------------------------------------
### ENTROPY LOOP
###------------------------------------------

for m in range(4):

	print('Generating', m+2, 'out of 6 elements per cluster...')

	###------------------------------------------
	### STOICHIOMETRIC LOOP
	###------------------------------------------

	Z = [] #Collect all images per stoichiometry

	for s in range(len(S[m])):

		#Define the composition of the atoms to optimize
		stoichiometry = [int(i) for i in list(S[m][s])]
		print(stoichiometry)

		db_file = 'DBs/work-'+str(m)+'-'+str(s)+'.db'

		if meta==0 and iteration==0: pop_size = 100
		else:
			prev_dir = connect('DBs/'+str(iteration-1)+'/temp-'+str(m)+'-'+str(s)+'.db')
			images = [prev_dir.get_atoms(id=i+1) for i in range(len(prev_dir))]
			pop_size = len(images)

		#ratio of covalent radii
		rcr = 0.8
		#create the cell / slab
		slab = Atoms('', cell=[20, 20, 20], pbc=False)

		#define the volume in which the adsorbed cluster is optimized
		box = [(5.,5.,5.), ((10.,0.,0.), (0.,10.,0.), (0.,0.,10.))]
		#the first vector is the origin and extending into the other 3 vectors

		#Define the closest distance two atoms of a given species can be to each other
		unique_atom_types = get_all_atom_types(slab, stoichiometry)
		blmin = closest_distances_generator(atom_numbers=unique_atom_types, 
			ratio_of_covalent_radii=rcr)

		if meta==0 and iteration==0:
			#Create the starting population
			sg = StartGenerator(slab, stoichiometry, blmin, box_to_place_in=box)
			starting_population = [sg.get_new_candidate() for i in range(pop_size)]
			d = PrepareDB(db_file_name=db_file, simulation_cell=slab, stoichiometry=stoichiometry)
			for i in starting_population: d.add_unrelaxed_candidate(i)
		else:
			#Load the starting population
			d = PrepareDB(db_file_name=db_file, simulation_cell=slab, stoichiometry=stoichiometry)
			for i in images: d.add_unrelaxed_candidate(i)

		#Initialize the different components of the GA
		da = DataConnection(db_file)
		atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
		n_to_optimize = len(atom_numbers_to_optimize)
		all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)

		#Comparatot, Cross-over, and Mutation options
		comp = InteratomicDistanceComparator(n_top=n_to_optimize, pair_cor_cum_diff=0.03,
											 pair_cor_max=0.7, dE=0.05, mic=False)

		pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
		mutations = OperationSelector([1., 1., 1.],
									  [MirrorMutation(blmin, n_to_optimize),
									   RattleMutation(blmin, n_to_optimize),
									   PermutationMutation(n_to_optimize)])
		#PARAMETERIZATION

		n2e = 100
		mp = 0.3
		fmax = 0.001
		steps = 10

		#GENERATE POPULATION

		#Relax all unrelaxed structures (so that da has its proper dictionary)
		while da.get_number_of_unrelaxed_candidates() > 0:
			a = da.get_an_unrelaxed_candidate()
			a.calc = calc    
			e = a.get_potential_energy()
			f = a.get_forces()
			a.info['key_value_pairs']['raw_score'] = -e
			a.calc = SinglePointCalculator(a)
			a.calc.results['energy'] = np.float64(e)
			a.calc.results['forces'] = np.float64(f)    
			da.add_relaxed_step(a, perform_parametrization=combine_parameters)

		#create the population
		population = Population(data_connection=da, population_size=pop_size, comparator=comp)

		#SELECTION

		#create the regression expression for estimating the energy
		all_trajs = da.get_all_relaxed_candidates()
		sampled_points = []
		sampled_energies = []
		for conf in all_trajs:
			no_of_conn = list(get_parametrization(conf))
			if no_of_conn not in sampled_points:
				sampled_points.append(no_of_conn)
				sampled_energies.append(conf.get_potential_energy())

		sampled_points = np.array(sampled_points)
		sampled_energies = np.array(sampled_energies)

		if len(sampled_points) > 0 and len(sampled_energies) >= len(sampled_points[0]):
			weights = np.linalg.lstsq(sampled_points, sampled_energies, rcond=-1)[0]
		else: weights = None

		#CROSSOVER AND MUTATION

		for i in tqdm(range(n2e), desc='Evolving Population'):

			#pairing
			a1, a2 = population.get_two_candidates()

			#selecting the "worst" parent energy to compare with child
			ce_a1 = da.get_atoms(a1.info['relax_id']).get_potential_energy()
			ce_a2 = da.get_atoms(a2.info['relax_id']).get_potential_energy()
			comparison_energy = min(ce_a1, ce_a2)

			a3, desc = pairing.get_new_individual([a1, a2])
			if a3 is None: continue
			if skip(a3, comparison_energy, weights): continue
			da.add_unrelaxed_candidate(a3, description=desc)

			#mutation
			if rng.random() < mp:
				a3_mut, desc = mutations.get_new_individual([a3])
				if (a3_mut is not None and not skip(a3_mut, comparison_energy, weights)):
					da.add_unrelaxed_step(a3_mut, description=desc)
					a3 = a3_mut

			#relaxation
			a3.calc = calc
			dyn = BFGS(a3, trajectory=None, logfile=None)
			dyn.run(fmax=fmax, steps=steps)
			e = a3.get_potential_energy()
			f = a3.get_forces()
			a3.info['key_value_pairs']['raw_score'] = -e
			a3.calc = SinglePointCalculator(a3)
			a3.calc.results['energy'] = np.float64(e)
			a3.calc.results['forces'] = np.float64(f)
			da.add_relaxed_step(a3, perform_parametrization=combine_parameters)
			population.update()
			#

		#Re-write stoichiometric images
		images = da.get_all_relaxed_candidates()[:n2e]
		write('DBs/'+str(iteration)+'/temp-'+str(m)+'-'+str(s)+'.db', images)

		for i in images: Z.append(i)

	###------------------------------------------
	### STOICHIOMETRIC LOOP
	###------------------------------------------

	idx, sample = normlog(np.sort([z.get_potential_energy() for z in Z]), 100)
	images = [Z[i] for i in idx]
	for i in images: ALL.append(i)

###------------------------------------------
### ENTROPY LOOP
###------------------------------------------

write('DBs/GA-'+str(iteration)+'.db', trim(prune(ALL, 3), 2))
print('DONE - initial GA - '+str(round(time.time()-start_time,3)))




###