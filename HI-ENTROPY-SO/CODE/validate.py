#!/usr/bin/env python3

import sys
from ase.io import write
from ase.db import connect
from ase.calculators.emt import EMT

iteration = int(sys.argv[1])

db = connect('DBs/GA-'+str(iteration)+'.db')

images = [db.get_atoms(id=i+1) for i in range(len(db))]

valid = []
for i in images:

	c = i.copy()
	c.calc = EMT()
	e = c.get_potential_energy()
	f = c.get_forces()
	valid.append(c)

write('DBs/NN-'+str(iteration)+'.db', valid)

print('valid DONE')
##