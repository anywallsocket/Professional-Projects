"""Semi-grand Monte Carlo Evolutionary Search (SG-MCES).

Search a fixed-size multicomponent cluster for low total formation energy
using Metropolis chemical mutations, short coordinate relaxations, and
elitist population selection. This is an optimization procedure; its complete
trajectory is not an equilibrium semi-grand canonical ensemble sample.

Scientific operations and default parameters follow the supplied GC.py.
The original function names are retained for straightforward comparison.
See SG_MCES_methods.md for equations, pseudocode, and interpretation.

Run from the same working directory as the original script:
    python /path/to/SG_MCES.py

The default demo profile retains GC.py's CPU and 10-step settings. An optional
full profile uses CUDA and a configurable larger relaxation budget:
    python /path/to/SG_MCES.py --profile full --relax-steps 1000

Requires NumPy, tqdm, ASE, and MACE with the supplied trained model.
Model and database paths below are relative to the working directory.
Each run resets DB_NAME, exactly as the original script intended.
"""

import argparse
import os

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.cluster import Icosahedron
from ase.data import atomic_numbers
from ase.db import connect
from ase.optimize import LBFGS
from mace.calculators import MACECalculator
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration: all numerical defaults and element ordering are unchanged.
# ---------------------------------------------------------------------------

# The order matters for reproducing a seeded sequence of random species draws.
ELEMENTS = ['Al', 'Co', 'Ni', 'Mn', 'W', 'Mg', 'Cr', 'Fe', 'Cu']
MODEL_PATH = 'FT-2.model'
DEVICE = 'cpu'
DEFAULT_DTYPE = 'float32'

# Elemental reference energies in eV/atom, copied verbatim from GC.py.
# These must be consistent with the energy convention of the trained model.
# During transmutation they also act as fixed chemical-potential offsets.
# Their provenance is not specified in the original file.
mu_refs = {
    'Al': -6.721347604257971,
    'Co': -13.229259057674634,
    'Mn': -14.680311021740309,
    'Ni': -11.957086694846831,
    'W': -51.339848407944416,
    'Mg': -4.171991533876267,
    'Cr': -14.846916372344474,
    'Fe': -14.441847033931484,
    'Cu': -10.839152680840218,
}

I = 0                         # Run identifier used in the database filename.
N_SHELLS = 3                   # ASE icosahedron shell count (55 atoms here).
POP_SIZE = 10                  # Number of retained candidates and offspring.
N_GENERATIONS = 10
DB_NAME = 'DBs/GCS-' + str(I) + '.db'

TEMP_K = 500                   # Metropolis search temperature, in kelvin.
N_MC_ATTEMPTS = 15             # Proposal slots per child, including no-ops.
TRANSMUTE_PROBABILITY = 0.5
POSITION_NOISE_STD = 0.1       # Initial Gaussian displacement scale, in angstrom.
RELAX_FMAX = 0.05              # LBFGS force stopping threshold, in eV/angstrom.
RELAX_STEPS = 10               # Maximum steps; convergence is not guaranteed.
KB_EV_PER_K = 8.617333262e-5    # Original Boltzmann constant, in eV/K.

# The full profile is an execution preset, not a different search algorithm.
# Its 500-step cap is a suggested starting point, not a convergence guarantee.
# Device and step cap can be overridden independently on the command line.
RUN_PROFILES = {
    'demo': {'device': DEVICE, 'relax_steps': RELAX_STEPS},
    'full': {'device': 'cuda', 'relax_steps': 500},
}


def formE(atoms):
    """Return total formation energy in eV, using the attached calculator.

    E_form = E_cluster - sum_i N_i * mu_i.

    Summing once per atom is equivalent to summing over species counts. The
    original per-atom summation order is retained to avoid changing rounding.
    This is a total energy, rather than an energy per atom. All candidates
    have the same atom count, so either normalization gives the same ranking.
    """
    e_cluster = atoms.get_potential_energy()
    e_ref_total = 0.0
    for symbol in atoms.get_chemical_symbols():
        e_ref_total += mu_refs[symbol]
    return e_cluster - e_ref_total


def mc_swap(atoms, calc, temp_K=500, n_swaps=15):
    """Return a child after composition-preserving Metropolis swap attempts.

    The parent is copied, so its symbols and coordinates remain unchanged.
    Each attempt selects two distinct indices uniformly. If the species are
    identical, that proposal slot is consumed without an energy evaluation.
    Otherwise the identities are exchanged at fixed coordinates.

    Downhill moves are accepted; other moves are accepted with probability
    exp(-delta_E_form / (k_B*T)). Rejected swaps are immediately reversed.
    No coordinate relaxation takes place inside this MC sequence.
    """
    child = atoms.copy()
    child.calc = calc
    kT = temp_K * KB_EV_PER_K
    current_E = formE(child)

    for _ in range(n_swaps):
        idx1, idx2 = np.random.choice(len(child), 2, replace=False)
        if child.numbers[idx1] == child.numbers[idx2]:
            continue

        # Changing atomic identities leaves positions and species counts fixed.
        child.numbers[idx1], child.numbers[idx2] = (
            child.numbers[idx2], child.numbers[idx1]
        )
        new_E = formE(child)
        dE = new_E - current_E

        # Preserve the original short circuit and random-number consumption.
        # A zero-energy move follows the probability branch (probability one).
        if dE < 0 or np.random.rand() < np.exp(-dE / kT):
            current_E = new_E
        else:
            child.numbers[idx1], child.numbers[idx2] = (
                child.numbers[idx2], child.numbers[idx1]
            )

    # The child is evaluated afresh after its subsequent coordinate relaxation.
    child.calc = None
    return child


def mc_transmute(atoms, calc, elements, temp_K=500, n_attempts=15):
    """Return a child after fixed-size, composition-changing MC attempts.

    Select an atom uniformly and propose a species uniformly from the complete
    allowed pool, including its current species. Identical-species proposals
    consume a slot and are skipped. This preserves the original proposal law.

    For an old species a and new species b, the acceptance energy is
    delta_E_form = delta_E_cluster - (mu_b - mu_a). The symmetric proposal
    probabilities require no additional proposal-ratio factor. Total atom
    count and coordinates stay fixed throughout these attempts.
    """
    child = atoms.copy()
    child.calc = calc
    kT = temp_K * KB_EV_PER_K
    z_pool = [atomic_numbers[el] for el in elements]
    current_E = formE(child)

    for _ in range(n_attempts):
        idx = np.random.randint(len(child))
        old_z = child.numbers[idx]
        new_z = np.random.choice(z_pool)
        if new_z == old_z:
            continue

        child.numbers[idx] = new_z
        new_E = formE(child)
        dE = new_E - current_E

        if dE < 0 or np.random.rand() < np.exp(-dE / kT):
            current_E = new_E
        else:
            # Restore the last accepted configuration on rejection.
            child.numbers[idx] = old_z

    child.calc = None
    return child


def evaluate_and_store(atoms, calc, db, relax_steps=RELAX_STEPS):
    """Briefly relax coordinates, assign fitness, and store one candidate.

    Chemical identities remain fixed during LBFGS. The optimizer stops at
    fmax <= RELAX_FMAX or the supplied relax_steps cap. The score is computed even if
    that force threshold was not reached, as in the original algorithm.

    Fitness is -E_form, so larger scores correspond to lower formation energy.
    A SinglePointCalculator stores the post-relaxation energy and forces for
    this geometry. Every evaluated candidate is written to the database,
    including candidates that will later be discarded by elitist selection.
    """
    atoms.calc = calc
    opt = LBFGS(atoms, logfile=None)
    opt.run(fmax=RELAX_FMAX, steps=relax_steps)

    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    score = -formE(atoms)

    # Retain the existing ASE metadata convention and database score field.
    atoms.info['key_value_pairs'] = {'raw_score': score}
    atoms.calc = SinglePointCalculator(atoms, energy=e, forces=f)
    db.write(atoms, raw_score=score)
    return atoms, score


def main(device=DEVICE, relax_steps=RELAX_STEPS, model_path=MODEL_PATH,
         db_name=DB_NAME, default_dtype=DEFAULT_DTYPE):
    """Run the search; defaults reproduce the local demonstration settings.

    Increasing relax_steps changes the coordinate-relaxation budget, while
    device and default_dtype select calculator execution settings. These
    adjustments can change energies and the stochastic trajectory; the
    mutation, fitness, parent selection, and survival rules are unchanged.
    """
    calc = MACECalculator(
        model_paths=[model_path], device=device, default_dtype=default_dtype
    )

    # Make a fresh database, retaining the original reset-on-run behavior.
    # Operational fixes: create a missing output directory and only suppress
    # a missing-file error. Other deletion failures must not silently append
    # a new run to an old database.
    os.makedirs(os.path.dirname(db_name) or '.', exist_ok=True)
    try:
        os.remove(db_name)
    except FileNotFoundError:
        pass
    db = connect(db_name)
    population = []

    print('--- Generating Initial Random Population ---')
    for _ in tqdm(range(POP_SIZE)):
        # Each seed begins with the same Cu-based geometry. The Cu template
        # sets the initial lattice scale; species are then assigned uniformly.
        template = Icosahedron('Cu', noshells=N_SHELLS)
        template.set_chemical_symbols(np.random.choice(ELEMENTS, len(template)))
        pos = template.get_positions()
        template.set_positions(
            pos + np.random.normal(0, POSITION_NOISE_STD, pos.shape)
        )
        template.pbc = False
        atoms, score = evaluate_and_store(template, calc, db, relax_steps)
        population.append((score, atoms))

    # Stable sorting preserves insertion order for equal scores.
    population.sort(key=lambda x: x[0], reverse=True)

    print('--- Starting Dynamic Stoichiometry Loop ---')
    for gen in range(N_GENERATIONS):
        print(f'Generation {gen}')
        next_gen_candidates = []

        for _ in tqdm(range(POP_SIZE)):
            # Rank 0 is best. A folded Gaussian favors low ranks. Modulo
            # wraps out-of-range draws; it is not clipping or resampling.
            # Keep this exact rule because changing it changes the search.
            parent_idx = int(np.abs(np.random.normal(0, POP_SIZE / 3))) % POP_SIZE
            parent = population[parent_idx][1]

            # One operator is chosen per child; its complete 15-slot sequence
            # runs before relaxation. Parents are sampled with replacement.
            if np.random.rand() < TRANSMUTE_PROBABILITY:
                child = mc_transmute(
                    parent, calc, ELEMENTS,
                    temp_K=TEMP_K, n_attempts=N_MC_ATTEMPTS,
                )
            else:
                child = mc_swap(
                    parent, calc, temp_K=TEMP_K, n_swaps=N_MC_ATTEMPTS
                )

            child, score = evaluate_and_store(child, calc, db, relax_steps)
            next_gen_candidates.append((score, child))

        # Parents stay eligible alongside offspring. Keep the best POP_SIZE
        # entries; there is no crossover, deduplication, or diversity filter.
        population.extend(next_gen_candidates)
        population.sort(key=lambda x: x[0], reverse=True)
        population = population[:POP_SIZE]

        best_score = population[0][0]
        best_atoms = population[0][1]
        formula = best_atoms.get_chemical_formula()
        print(f'Best E_form: {-best_score:.4f} eV | Composition: {formula}')


def parse_arguments():
    """Expose demonstration and full-run settings without altering MC rules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=RUN_PROFILES, default='demo',
                        help='demo: CPU/10 steps; full: CUDA/500 steps')
    parser.add_argument('--device', default=None,
                        help='MACE device override, e.g. cpu, cuda, or cuda:0')
    parser.add_argument('--relax-steps', type=int, default=None,
                        help='LBFGS step-cap override; convergence is not guaranteed')
    parser.add_argument('--model-path', default=MODEL_PATH,
                        help='Path to the existing trained MACE model')
    parser.add_argument('--db-name', default=DB_NAME,
                        help='Output database path; an existing file is deleted')
    parser.add_argument('--dtype', choices=('float32', 'float64'),
                        default=DEFAULT_DTYPE, help='MACE calculator precision')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional NumPy seed; default leaves the RNG unseeded')
    args = parser.parse_args()
    profile = RUN_PROFILES[args.profile]
    if args.device is None:
        args.device = profile['device']
    if args.relax_steps is None:
        args.relax_steps = profile['relax_steps']
    if args.relax_steps < 1:
        parser.error('--relax-steps must be at least 1')
    if args.seed is not None and not 0 <= args.seed <= 2**32 - 1:
        parser.error('--seed must be between 0 and 2**32 - 1')
    return args


# Importing the module exposes its functions without loading a model, deleting
# a database, or starting a search. Running it as a script starts the search.
if __name__ == '__main__':
    args = parse_arguments()
    if args.seed is not None:
        np.random.seed(args.seed)
    main(device=args.device, relax_steps=args.relax_steps,
         model_path=args.model_path, db_name=args.db_name,
         default_dtype=args.dtype)

