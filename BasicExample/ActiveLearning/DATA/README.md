Final Iteration's DFT values.

Parsing Example:

```
from ase.io import read

test = read('../../TIME/DFT-iteration_10.extxyz', index='0')
test.info

{'index': np.int64(0),
 'nearest_neighbor_distance': np.float64(1.5628799672719595),
 'nearest_neighbor_element': 'Pt',
 'adsorbate': 'H',
 'adsorbate_site': 'top',
 'cluster_size': np.int64(10),
 'cluster_energy': np.float64(-320.4773421),
 'adsorbate_energy': np.float64(-1.07452648),
 'adsorption_energy': np.float64(-3.061959420000025)}

test.arrays

{'numbers': array([ 1, 46, 46, 46, 46, 46, 46, 78, 78, 78, 78]),
 'positions': array([[ 7.54931, 11.63296, 10.40569],
        [12.48554,  9.60482, 12.54978],
        [12.25646,  8.10204, 10.19413],
        [10.06787,  8.6052 , 11.76829],
        [11.56973,  9.65489,  7.84258],
        [ 8.83538, 10.04476,  7.49053],
        [10.29658, 11.8495 ,  8.90499],
        [10.40141, 11.13888, 11.80447],
        [12.26779, 10.65031, 10.14597],
        [ 9.85156,  8.34328,  9.19847],
        [ 8.54233, 10.4529 , 10.15278]])}

```
