import networkx as nx
import numpy as np

def longest_simple_paths(G):
    """
    Compute longest simple paths between all pairs of nodes in the graph G.
    
    Parameters:
    G : networkx.Graph
    
    Returns:
    dict : {(u, v): length of longest simple path}
    """
    def dfs(current, visited, length):
        nonlocal longest
        visited.add(current)

        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                dfs(neighbor, visited, length + 1)
        
        if length > longest[current_start, current]:
            longest[current_start, current] = length
        visited.remove(current)

    longest = {}
    nodes = list(G.nodes())
    for u in nodes:
        current_start = u
        for v in nodes:
            longest[u, v] = 0
        dfs(u, set(), 0)
    
    # Make it symmetric (undirected graph)
    return {(min(u, v), max(u, v)): max(w, longest.get((v, u), 0))
            for (u, v), w in longest.items()}
    

def detour_matrix(G):
    """
    Compute the detour matrix for an undirected graph G.
    
    Returns:
    numpy.ndarray : Detour matrix D where D[i, j] is the length of the longest simple path between nodes i and j.
    """
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    D = np.zeros((n, n), dtype=int)
    
    lsp = longest_simple_paths(G)

    for (u, v), length in lsp.items():
        i, j = index[u], index[v]
        D[i, j] = D[j, i] = length
    
    return D

import numpy as np
from rdkit import Chem

def build_connectivity_matrix_B(mol):
    """
    Constructs the symmetric connectivity matrix [B] for a molecule,
    following the chemical bonding and weighting rules provided.

    Parameters:
    mol : RDKit Mol object (with or without hydrogens)

    Returns:
    np.ndarray : The connectivity matrix [B] (N x N) with N = # heavy atoms
    """
    
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    N = len(atoms)

    # Initialize B with default off-diagonal value 0.001
    B = np.full((N, N), 0.001)

    # Diagonal: atomic numbers
    for i, atom in enumerate(atoms):
        B[i, i] = atom.GetAtomicNum()

    # Fill in bond values
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        # Determine base bond weight
        if bond_type == Chem.rdchem.BondType.SINGLE:
            value = 0.1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            value = 0.2
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            value = 0.3
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            value = 0.15
        else:
            value = 0.001  # fallback or unknown type

        # Augment if either atom is terminal (only one bond)
        if atoms[i].GetDegree() == 1 or atoms[j].GetDegree() == 1:
            value += 0.01

        # Update symmetric entries
        B[i, j] = B[j, i] = value

    return B


import numpy as np
from rdkit import Chem

def build_modified_adjacency_matrix(mol):
    """
    Build the modified adjacency matrix described in the literature,
    incorporating empirical diagonal values and square-root bond order off-diagonals.

    Parameters:
    mol : RDKit Mol object (with hydrogens included)

    Returns:
    np.ndarray : Modified adjacency matrix (N x N)
    """
    # Empirical diagonal values (example from literature)
    empirical_diagonal = {
        1: 0.08,    # H
        6: 0.00,    # C
        7: 0.22,    # N
        8: 0.34,    # O
        9: 0.51,    # F
        17: 0.51,   # Cl (halogens grouped)
        35: 0.51,   # Br (halogens grouped)
        # Add others as needed
    }

    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    N = len(atoms)

    # Initialize the matrix
    A = np.zeros((N, N))

    # Diagonal entries: electronic environment
    for i, atom in enumerate(atoms):
        atomic_num = atom.GetAtomicNum()
        A[i, i] = empirical_diagonal.get(atomic_num, 0.1)  # Default small value

    # Off-diagonal: square root of bond order
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()

        if bond_type == Chem.rdchem.BondType.SINGLE:
            value = 1.0
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            value = np.sqrt(2)
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            value = np.sqrt(3)
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            value = np.sqrt(1.5)
        else:
            value = 0.0  # unknown

        A[i, j] = A[j, i] = value

    return A

def get_pendent_matrix(graph: nx.Graph, distance_matrix: np.ndarray):
    """
    Extracts the pendent matrix from the full graph distance matrix.
    
    Parameters:
    graph : networkx.Graph
        The molecular or general graph
    distance_matrix : np.ndarray
        The AxA full distance matrix
    
    Returns:
    np.ndarray : The A x m pendent matrix (m = number of terminal nodes)
    """
    A = graph.number_of_nodes()
    
    # Identify terminal vertices (degree == 1)
    terminal_nodes = [i for i, d in dict(graph.degree()).items() if d == 1]
    if len(terminal_nodes) == 0:
        return [] 
    else:
        pendent_matrix = distance_matrix[:, list(terminal_nodes)]
    return pendent_matrix


def compute_laplacians(G: nx.Graph):
    """
    Computes the unnormalized, symmetric normalized, and random-walk Laplacian matrices for a given graph.

    Parameters:
        G (nx.Graph): An undirected NetworkX graph.

    Returns:
        dict: A dictionary containing:
            - 'adjacency': Adjacency matrix A
            - 'unnormalized': Laplacian L = D - A
            - 'normalized_symmetric': L_sym = I - D^(-1/2) A D^(-1/2)
            - 'random_walk': L_rw = I - D^(-1) A
    """
    A = nx.to_numpy_array(G)
    degrees = A.sum(axis=1)
    D = np.diag(degrees)

    # Unnormalized Laplacian
    L = D - A

    # Symmetric Normalized Laplacian
    with np.errstate(divide='ignore'):
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
    L_sym = np.eye(G.number_of_nodes()) - D_inv_sqrt @ A @ D_inv_sqrt

    # Random Walk Laplacian
    with np.errstate(divide='ignore'):
        D_inv = np.diag(1.0 / degrees)
        D_inv[np.isinf(D_inv)] = 0
    L_rw = np.eye(G.number_of_nodes()) - D_inv @ A

    return {
        "adjacency": A,
        "unnormalized": L,
        "normalized_symmetric": L_sym,
        "random_walk": L_rw,
    }




def get_distance_path_count_matrix(graph: nx.Graph):
    """
    Computes the distance path count matrix based on the shortest path distances.
    
    Parameters:
    graph : networkx.Graph
        A molecular or general graph
    
    Returns:
    np.ndarray : A x A matrix where Dp[i, j] = (d_ij^2 + d_ij)/2, and 0 on the diagonal
    """
    dist_matrix = nx.floyd_warshall_numpy(graph)
    A = dist_matrix.shape[0]
    
    # Compute the triangular number matrix
    distance_path_matrix = (dist_matrix ** 2 + dist_matrix) / 2
    
    # Zero out the diagonal
    np.fill_diagonal(distance_path_matrix, 0)
    
    return distance_path_matrix
