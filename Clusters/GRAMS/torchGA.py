#!/usr/bin/env python3

#General imports
from programs import *
import sys, time
import numpy as np
from tqdm import tqdm
from random import random
from ase import Atoms
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.db import connect
from ase.data import atomic_numbers
from ase.ga.data import PrepareDB
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import closest_distances_generator
from ase.ga.utilities import get_all_atom_types
from ase.constraints import FixAtoms

#GA imports
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

#AmpTorch imports
from amptorch import AtomsTrainer
from amptorch import AMPtorch
import torch

start_time = time.time()

#BASIC SET UP

iteration = int(sys.argv[1])
atoms = int(sys.argv[2])

#Load the calculator object
calc = LoadAmpTorch('old', iteration)
ALL = []

for s in range(1, atoms):

    db_file = 'DBs/working'+str(s)+'.db'
    #ratio of covalent radii
    rcr = 0.8

    #define size of initial population
    population_size = 100

    #create the cell / slab
    slab = Atoms('', cell=[20, 20, 20], pbc=False)
    #define the volume in which the adsorbed cluster is optimized
    box = [(6.,6.,6.), ((8.,0.,0.), (0.,8.,0.), (0.,0.,8.))]
    #the first vector is the origin and extending into the other 3 vectors
    #Define the composition of the atoms to optimize
    atom_numbers = s*[79] + int(atoms-s)*[46]

    #GA CONFIGURATION

    #Define the closest distance two atoms of a given species can be to each other
    unique_atom_types = get_all_atom_types(slab, atom_numbers)
    blmin = closest_distances_generator(atom_numbers=unique_atom_types, ratio_of_covalent_radii=rcr)

    #create and load the starting population
    sg = StartGenerator(slab, atom_numbers, blmin, box_to_place_in=box)
    starting_population = [sg.get_new_candidate() for i in range(population_size)]
    d = PrepareDB(db_file_name=db_file, simulation_cell=slab, stoichiometry=atom_numbers)
    for a in starting_population: d.add_unrelaxed_candidate(a)

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

    #This is necessary for the hack*
    from ase.calculators.singlepoint import SinglePointCalculator

    #First, change the following parameters to suit your needs
    n_to_test = 100 #testing
    mutation_probability = 0.3
    fmin = 0.0001
    steps = 2

    #MAIN ALGORITHM

    #Relax all unrelaxed structures
    myprint("Initializing Candidates")
    while da.get_number_of_unrelaxed_candidates() > 0:
        a = da.get_an_unrelaxed_candidate()
        a.calc = calc    
        dyn = BFGS(a, trajectory=None, logfile=None)
        dyn.run(fmax=fmin, steps=steps)
        e = a.get_potential_energy()
        f = a.get_forces()
        a.info['key_value_pairs']['raw_score'] = -e
        a.calc = SinglePointCalculator(a)
        a.calc.results['energy'] = np.float64(e)
        a.calc.results['forces'] = np.float64(f)    
        da.add_relaxed_step(a)

    #create the population
    population = Population(data_connection=da, population_size=population_size, comparator=comp)

    #Test n_to_test new candidates
    myprint("Evolving Candidates:")
    steps = 5*iteration
    myprint('fmin =', fmin, ' steps =', steps)

    for i in tqdm(range(n_to_test)):
        a1, a2 = population.get_two_candidates()
        a3, desc = pairing.get_new_individual([a1, a2])
        if a3 is None: continue
        da.add_unrelaxed_candidate(a3, description=desc)

        #check if we want to do a mutation
        if random() < mutation_probability:
            a3_mut, desc = mutations.get_new_individual([a3])
            if a3_mut is not None:
                da.add_unrelaxed_step(a3_mut, desc)
                a3 = a3_mut

        a3.calc = calc
        dyn = BFGS(a3, trajectory=None, logfile=None)
        dyn.run(fmax=fmin, steps=steps)
        e = a3.get_potential_energy()
        f = a3.get_forces()
        a3.info['key_value_pairs']['raw_score'] = -e
        a3.calc = SinglePointCalculator(a3)
        a3.calc.results['energy'] = np.float64(e)
        a3.calc.results['forces'] = np.float64(f)
        da.add_relaxed_step(a3)
        population.update()

    ALL.append(da.get_all_relaxed_candidates())

images = []
for a in ALL:
    for i in a:
        images.append(i)

print('Writing', len(images), 'images to GA', str(iteration))
write('DBs/GA'+str(iteration)+'.db', images)
print('DONE - initial GA - '+str(round(time.time()-start_time,3)))

"""
#AmpTorch bug prevents AmpTorch calculator object from being read from ASE database
#For a proper database to be stored therefore, a trajectory file must be written/read

images = da.get_all_relaxed_candidates()
E = []
F = []
for c in images: 
	c.calc = calc
	E.append(c.get_potential_energy())
	F.append(c.get_forces())

write('TRAJ/temp.traj', images)
t = Trajectory('TRAJ/temp.traj')
ga = connect('DBs/GA'+str(iteration)+'.db')
for i in range(len(t)): 
    c = t[i]
    c.calc.results['energy'] = np.float64(E[i])
    c.calc.results['forces'] = np.float64(F[i])
    ga.write(c)
##
"""