from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from rdkit.Chem.EState import EState, EState_VSA
def get_morse_descriptors(mol, id=1,weights='mass', num_bins=32, max_dist=12.0):
    """
    Custom MoRSE-like descriptor with atomic contributions.
    Returns descriptor vector and individual atomic pair contributions.
    """
    conf = mol.GetConformer(id)

    num_atoms = mol.GetNumAtoms()
    dist_matrix = np.zeros((num_atoms, num_atoms))
    weight_vector = []

    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        if weights == 'mass':
            weight_vector.append(atom.GetMass())
        elif weights == 'number':
            weight_vector.append(atom.GetAtomicNum())
        else:
            weight_vector.append(1.0)  # uniform weight

    weight_vector = np.array(weight_vector)

    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            pos_i = conf.GetAtomPosition(i)
            pos_j = conf.GetAtomPosition(j)
            dist = np.linalg.norm(pos_i - pos_j)
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist  # symmetric

    # MoRSE calculation
    sk_vals = np.linspace(0.5, max_dist, num_bins)
    morse_vector = np.zeros(num_bins)
    pair_contributions = np.zeros((num_bins, num_atoms, num_atoms))

    for k, sk in enumerate(sk_vals):
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                rij = dist_matrix[i][j]
                contrib = weight_vector[i] * weight_vector[j] * np.cos(rij * sk)
                morse_vector[k] += contrib
                pair_contributions[k, i, j] = contrib
                pair_contributions[k, j, i] = contrib

    return morse_vector, pair_contributions


def get_morse_sinc_descriptors(mol, weights='mass', num_bins=32, s_min=0.5, s_max=12.0):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()

    # Get atomic weights
    if weights == 'mass':
        A = np.array([mol.GetAtomWithIdx(i).GetMass() for i in range(num_atoms)])
    elif weights == 'number':
        A = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(num_atoms)])
    else:
        A = np.ones(num_atoms)

    # Compute distance matrix
    R = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i):
            ri = conf.GetAtomPosition(i)
            rj = conf.GetAtomPosition(j)
            R[i, j] = R[j, i] = np.linalg.norm(ri - rj)

    # Compute MoRSE descriptors
    s_vals = np.linspace(s_min, s_max, num_bins)
    morse_vector = np.zeros(num_bins)
    contributions = np.zeros((num_bins, num_atoms, num_atoms))

    for k, s in enumerate(s_vals):
        for i in range(num_atoms):
            for j in range(i):
                r_ij = R[i, j]
                if r_ij == 0:
                    continue
                sinc_val = np.sin(s * r_ij) / (s * r_ij)
                contrib = A[i] * A[j] * sinc_val
                morse_vector[k] += contrib
                contributions[k, i, j] = contributions[k, j, i] = contrib

    return morse_vector, contributions, s_vals
