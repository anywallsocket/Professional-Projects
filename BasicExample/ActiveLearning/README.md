## Here we showcase how the Active-Learning loop can find optimial adsorption sites on randomized clusters.
These states are comprised of multiple elements, and thus their composition, size, stoichiometry all together multiply the state space.
To explore spaces like this automatically, we employ the AL algorithm with minimal adjustment.
This time the neural networks are encoded with SchentPack (https://github.com/atomistic-machine-learning/schnetpack)

The genetic algorithm is encoded with atomic simulations environment, as is the database archetecture (https://gitlab.com/ase/ase)

### Here we see the energy and force predictions improving over global loops.

<p align="center">
  <img width="600" height="550" src="PICS/Emae.png">
</p>

<p align="center">
  <img width="600" height="550" src="PICS/Fmae.png">
</p>

### And here is an in-depth analysis into the final iteration's preformance.

<p align="center">
  <img width="600" height="550" src="PICS/10-energy.png">
</p>

<p align="center">
  <img width="600" height="550" src="PICS/10-force.png">
</p>

<p align="center">
  <img width="600" height="550" src="PICS/d-1-energy.png">
</p>

<p align="center">
  <img width="600" height="550" src="PICS/d-2-energy.png">
</p>

<p align="center">
  <img width="600" height="550" src="PICS/histo-energy.png">
</p>

<p align="center">
  <img width="600" height="550" src="PICS/histo-force.png">
</p>


### Finally we showcase some analysis into adsorption energy trends for each adsorbate.


<p align="center">
  <img width="600" height="550" src="PICS/H-top.png">
</p>

<p align="center">
  <img width="600" height="550" src="PICS/H2-top.png">
</p>

<p align="center">
  <img width="600" height="550" src="PICS/N-top.png">
</p>

<p align="center">
  <img width="600" height="550" src="PICS/N2-top.png">
</p>

<p align="center">
  <img width="600" height="550" src="PICS/NH3-top.png">
</p>

