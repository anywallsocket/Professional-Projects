#!/usr/bin/python3

#Basic imports
from programs import *
import time, sys
import numpy as np
from tqdm import tqdm
from random import random
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import BFGS
from ase.calculators.lammpsrun import LAMMPS

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
from ase.ga.standardmutations import MirrorMutation
from ase.ga.standardmutations import RattleMutation
from ase.ga.standardmutations import PermutationMutation
from ase.ga.parallellocalrun import ParallelLocalRun

import time
start_time = time.time()

#define adorbate #
iteration = int(sys.argv[1])
s = int(sys.argv[2])

db_file = 'DBs/working.db'

#Create the molecule and adsorbate
unit = [('Pd',s)]
slab = read('best.vasp')
slab.pbc = False
slab.set_constraint(FixAtoms(mask=len(slab)*[True]))
box = [(6.,6.,6.), ((8.,0.,0.), (0.,8.,0.), (0.,0.,8.))]
atom_numbers = [46]*s

#Define the closest distance two atoms of a given species can be to each other
unique_atom_types = get_all_atom_types(slab, atom_numbers)
blmin = closest_distances_generator(atom_numbers=unique_atom_types, ratio_of_covalent_radii=1.0)

#Create the starting population
sg = StartGenerator(slab, unit, blmin, box_to_place_in=box)
population_size = 100
myprint('[Initializing Candidates]')
starting_population = [sg.get_new_candidate() for i in tqdm(range(population_size))]

#Create the database to store information in
d = PrepareDB(db_file_name=db_file, simulation_cell=slab, stoichiometry=atom_numbers)
for a in starting_population: d.add_unrelaxed_candidate(a)

#Initialize the different components of the GA
da = DataConnection(db_file)
atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
n_to_optimize = len(atom_numbers_to_optimize)
all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)

#Cross-over, and Mutation options
comp = InteratomicDistanceComparator(n_top=n_to_optimize, pair_cor_cum_diff=0.03,
                                     pair_cor_max=0.7, dE=0.05, mic=False)

pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
mutations = OperationSelector([1., 1., 0.0],
                              [MirrorMutation(blmin, n_to_optimize),
                               RattleMutation(blmin, n_to_optimize),
                               PermutationMutation(n_to_optimize)])

#PARAMETERIZATION

#This is necessary for the hack*
from ase.calculators.singlepoint import SinglePointCalculator
calc = LoadAmpTorch('old', iteration)

n_to_test = 100
mutation_probability = 0.4
fmin = 0.01
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
steps = 10*iteration
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
##

write('DBs/GA'+str(iteration)+'.db', da.get_all_relaxed_candidates())

print('DONE - initial GA - '+str(round(time.time()-start_time,3))+'s')
##