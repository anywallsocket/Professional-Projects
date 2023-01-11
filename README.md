# Genetic Algorithm coupled with Neural Network

This Active Learning algorithm is built on Atomic Simulations Environment (ASE)
as well as the Torch interpreter for Atomistic Machine-Learning Package (AMP).

The point is materials discovery, and works with nanoclusters as well as adsorption sites.

It can synthesize a calculator like DFT, LAMMPS, or EMT with a neural network.
Currently it's configured for LAMMPS, but I'm working with a DFT version as well.

Structures are generated with the genetic algorithm which train a growing neural network.
The network then feeds back into the GA for producing more structures.
This methodology allows one to bootstrap a self-optimizing NN functional,
whilst avoiding all the heavy computational costs (e.g., DFT) of traditional calculators.

./INIT will initialize the proceedure

./RUN.sh will run the global loops

GRAMS/ contains all programs
<t>including phase.py and volume.py for 3d plots

Here below we see fixed 13 atom AuPd cluster trained on GAs and NNs to find adsorption sites for C2.
The ideal sites end up forming a dodecahedron -- avoiding the inner 13 atom icosahedron verticies. 
  
![](dodec.gif)

Here the same system for the same 8 iterations, except a heatmap has been overlayed and more images seen.
  Note that the heatmap is relative to the range of each iteration, and so the scale itself changes.
  
![](13grid.gif)
  
Something to note is that it only takes 8 iterations to find the absolute global minima.
  This is seen as the Au12Pd cluster, with Pd at the core of an Au icosahedron.
