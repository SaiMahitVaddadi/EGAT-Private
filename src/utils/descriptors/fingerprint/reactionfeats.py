from rdkit import Chem
from ....graph.reaction.Reaction import ReactionFeaturizer
from ....graph.reaction.ReactionGeometry import ReactionFeaturizerwithGeometry
from ....graph.reaction.ReactionGlobal import ReactionFeaturizerwithPadding
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdchem
import networkx as nx
from .gedhelpers import node_subst_cost, node_del_cost, node_ins_cost, edge_subst_cost, edge_del_cost, edge_ins_cost
from networkx.algorithms.similarity import graph_edit_distance,optimize_graph_edit_distance
import numpy as np

'''
To add:
- Grab 3D features 
    - Delta V, Delta SA
    - Delta of other Features 

'''

class ReactionFeatures:
    def __init__(self,featurizer):
        #Featurizer is a ReactionFeaturizer with subfeats reactant and product
        self.featurizer = featurizer
        self.reactant_mol = self.featurizer.reactant.matrixdescriptors.new_mol
        self.product_mol = self.featurizer.product.matrixdescriptors.new_mol
        self.reactant_g = self.to_nx(self.reactant_mol)
        self.product_g = self.to_nx(self.product_mol)

    
    def to_nx(self,mol):
        """
        Convert a molecule to a networkx graph.
        :param mol: RDKit molecule object.
        :return: Networkx graph.
        """
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum())
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_order = bond.GetBondTypeAsDouble()
            G.add_edge(i, j, bond_order=bond_order)
        return G
    

    def GEDWithBO(self):
        return graph_edit_distance(self.reactant_g, self.product_g,node_subst_cost=node_subst_cost,node_del_cost=node_del_cost, node_ins_cost=node_ins_cost,edge_subst_cost=edge_subst_cost,edge_del_cost=edge_del_cost,edge_ins_cost=edge_ins_cost)
        
    def GEDwoBO(self):
        return graph_edit_distance(self.reactant_g, self.product_g,node_subst_cost=node_subst_cost,node_del_cost=node_del_cost, node_ins_cost=node_ins_cost)
    
    def OptimalGEDwithBO(self):
        return optimize_graph_edit_distance(self.reactant_g, self.product_g,node_subst_cost=node_subst_cost,node_del_cost=node_del_cost, node_ins_cost=node_ins_cost,edge_subst_cost=edge_subst_cost,edge_del_cost=edge_del_cost,edge_ins_cost=edge_ins_cost)
    
    def OptimalGEDwoBO(self):
        return optimize_graph_edit_distance(self.reactant_g, self.product_g,node_subst_cost=node_subst_cost,node_del_cost=node_del_cost, node_ins_cost=node_ins_cost)
    
    


from rdkit import Chem
from rdkit.Chem import rdFMCS

def mcs_distance(mol1, mol2, timeout=10):
    """
    Computes Maximum Common Substructure Distance (MCS-Distance)
    between two RDKit Mol objects.

    MCS-D = 1 - |MCS| / max(|mol1|, |mol2|)
    where |.| = number of atoms

    Parameters:
        mol1, mol2: RDKit Mol objects
        timeout: Time limit for the MCS search (in seconds)

    Returns:
        mcs_distance (float)
    """
    # Find MCS
    mcs_result = rdFMCS.FindMCS([mol1, mol2], timeout=timeout)
    if mcs_result.canceled:
        raise TimeoutError("MCS computation timed out.")

    # Convert SMARTS to Mol and count atoms
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    if mcs_mol is None:
        raise ValueError("Failed to parse MCS SMARTS.")

    mcs_size = mcs_mol.GetNumAtoms()
    max_size = max(mol1.GetNumAtoms(), mol2.GetNumAtoms())

    # Compute MCS-Distance
    mcs_distance = 1.0 - (mcs_size / max_size)
    return mcs_distance

import numpy as np
import networkx as nx
from collections import Counter

def degree_distribution_distance(G1, G2):
    deg1 = Counter(dict(G1.degree()).values())
    deg2 = Counter(dict(G2.degree()).values())
    max_deg = max(max(deg1.keys(), default=0), max(deg2.keys(), default=0))
    
    vec1 = np.array([deg1.get(i, 0) for i in range(max_deg+1)])
    vec2 = np.array([deg2.get(i, 0) for i in range(max_deg+1)])
    
    return np.sum(np.abs(vec1 - vec2))  # L1 norm

def spectral_distance(G1, G2, k=None):
    L1 = nx.normalized_laplacian_matrix(G1).todense()
    L2 = nx.normalized_laplacian_matrix(G2).todense()
    
    eig1 = np.sort(np.linalg.eigvalsh(L1))
    eig2 = np.sort(np.linalg.eigvalsh(L2))
    
    if k:
        eig1 = eig1[:k]
        eig2 = eig2[:k]
    
    # Pad shorter vector
    len_diff = len(eig1) - len(eig2)
    if len_diff > 0:
        eig2 = np.pad(eig2, (0, len_diff))
    elif len_diff < 0:
        eig1 = np.pad(eig1, (0, -len_diff))
    
    return np.linalg.norm(eig1 - eig2)


import zss

class NodeWrapper(zss.Node):
    def __init__(self, idx, label):
        super().__init__(label)
        self.idx = idx

def mol_to_tree(mol, root_idx=0):
    from rdkit import Chem
    visited = set()
    root = NodeWrapper(root_idx, mol.GetAtomWithIdx(root_idx).GetSymbol())
    queue = [(root, root_idx)]
    atom_nodes = {root_idx: root}
    
    while queue:
        parent_node, idx = queue.pop(0)
        visited.add(idx)
        atom = mol.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            nid = neighbor.GetIdx()
            if nid not in visited:
                child = NodeWrapper(nid, neighbor.GetSymbol())
                parent_node.addkid(child)
                queue.append((child, nid))
    return root

def tree_edit_distance(mol1, mol2):
    tree1 = mol_to_tree(mol1)
    tree2 = mol_to_tree(mol2)
    return zss.distance(tree1, tree2)

def line_graph_edit_distance(G1, G2):
    L1 = nx.line_graph(G1)
    L2 = nx.line_graph(G2)
    return nx.graph_edit_distance(L1, L2)


import networkx as nx
import numpy as np
from karateclub import GraphletSampling

def graphlet_frequency_vector(G):
    model = GraphletSampling()
    model.fit([G])
    return model.get_embedding()[0]

def graphlet_frequency_distance(G1, G2):
    vec1 = graphlet_frequency_vector(G1)
    vec2 = graphlet_frequency_vector(G2)
    return np.linalg.norm(vec1 - vec2)  # Euclidean distance
