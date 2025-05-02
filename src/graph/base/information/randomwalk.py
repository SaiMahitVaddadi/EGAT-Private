from ..base import BaseFeaturizer
from rdkit import Chem
import networkx as nx
import numpy as np
from scipy.linalg import pinv
import random
from scipy.linalg import eigvalsh
from collections import Counter
from math import factorial, log
from dataclasses import dataclass
from typing import Optional


#TO-DO: Add Weights by Valence Electrons, Hybidization, Coulomb Matrix


@dataclass
class Params:
    wt_adj_mat_by: Optional[str] = None  # Options: 'bond_order', 'atomic_mass', 'valence', 'hybridization', 'coulomb', 'all'
    rw_weight_by: Optional[str] = None  # Options: 'atomic_mass', 'bond_order', 'hyb', 'valence', 'coulomb', 'all'
    sp_box_size: Optional[float] = None  # For periodic boundary conditions
    hop_radius: int = 1  # For k-hop subgraph extraction


class PathDistanceMetrics(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def SPDFunc(self,edge):
        try:
            path_length = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            return [path_length]
        except:
            return [-1]  # Return a large value if the edge does not exist or an error occurs
    
    def getAllPaths(self, start_atom, end_atom, max_length=1000):
        try:
            graph = nx.Graph()
            for bond in self.matrixdescriptors.new_mol.GetBonds():
                graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            
            all_paths = dict()
            all_paths['path'] = []
            all_paths['length'] = []    
            for path in nx.all_simple_paths(graph, source=start_atom, target=end_atom, cutoff=max_length):
                atoms = [self.matrixdescriptors.new_mol.GetAtomWithIdx(int(idx)) for idx in path]
                bonds = []
                for i in range(len(path) - 1):
                    bond = self.matrixdescriptors.new_mol.GetBondBetweenAtoms(path[i], path[i + 1])
                    if bond:
                        bonds.append(bond)
                path_length = len(path) - 1
                all_paths['length'].append(path_length)
                all_paths['path'].append({"atoms": atoms, "bonds": bonds})
            
            if all_paths['length']:
                min_length = min(all_paths['length'])
                min_paths = [all_paths['path'][i] for i, length in enumerate(all_paths['length']) if length == min_length]
                all_paths['shortest_paths'] = min_paths
                all_paths['shortest_length'] = min_length
            return all_paths
        except Exception as e:
            print(f"Error in getAllPaths: {e}")
            return []
    

    def WtSPDByMass(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                atoms = path['atoms']
                total_weight = 0
                for atom in atoms:
                    total_weight += atom.GetMass() / 128
            
                weighted_distance = distance * total_weight
                wdm.append(weighted_distance)
            return [min(wdm)]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]

    def WtSPDByBO(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                bonds = path['bonds']
                total_weight = 0
                for bond in bonds:
                    total_weight += bond.GetBondTypeasDouble() / 8
            
                weighted_distance = distance * total_weight
                wdm.append(weighted_distance)
            weighted_distance = min(wdm)
            return [weighted_distance]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]


    def WTSPDByHYB(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                atoms = path['atoms']
                total_weight_atoms = 0
                total_weight_hyb = 0
                for atom in atoms:
                    total_weight_atoms += atom.GetMass() / 128
                    hyb = next(key for key, value in atom.GetHybridization().values.items() if value == atom.GetHybridization())
                    total_weight_hyb += hyb / 9
                
                total_weight = total_weight_atoms + total_weight_hyb
                wdm.append(total_weight*distance)
            return [min(wdm)]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]
        

    def WTSPDByVE(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                atoms = path['atoms']
                total_weight_atoms = 0
                total_weight_hyb = 0
                for atom in atoms:
                    total_weight_atoms += atom.GetMass() / 128
                    try:
                        total_weight_hyb += self.properties.el_valence[atom.GetSymbol()] / 8
                    except: 
                        total_weight_hyb += self.properties.el_valence[atom.GetSymbol().lower()] / 8
                total_weight = total_weight_atoms + total_weight_hyb
                wdm.append(total_weight*distance)
            return [min(wdm)]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]
        
    def WTSPDByHYBVE(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                atoms = path['atoms']
                total_weight_atoms = 0
                total_weight_hyb = 0
                total_weight_ve = 0
                for atom in atoms:
                    total_weight_atoms += atom.GetMass() / 128
                    hyb = next(key for key, value in atom.GetHybridization().values.items() if value == atom.GetHybridization())
                    total_weight_hyb += hyb / 9
                    try:
                        total_weight_ve += self.properties.el_valence[atom.GetSymbol()] / 8
                    except: 
                        total_weight_ve += self.properties.el_valence[atom.GetSymbol().lower()] / 8
                
                total_weight = total_weight_atoms + total_weight_hyb + total_weight_ve
                wdm.append(total_weight*distance)
            return [min(wdm)]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]

    def getCoulombMatrix(self, mol,id=1):
        try:
            num_atoms = mol.GetNumAtoms()
            coulomb_matrix = np.zeros((num_atoms, num_atoms))

            for i in range(num_atoms):
                atom_i = mol.GetAtomWithIdx(int(i))
                Z_i = atom_i.GetAtomicNum()
                pos_i = np.array(mol.GetConformer(conf_id=id).GetAtomPosition(i))

                for j in range(num_atoms):
                    if i == j:
                        coulomb_matrix[i, j] = 0.5 * Z_i ** 2.4  # Diagonal elements
                    else:
                        atom_j = mol.GetAtomWithIdx(int(j))
                        Z_j = atom_j.GetAtomicNum()
                        pos_j = np.array(mol.GetConformer(conf_id=id).GetAtomPosition(j))
                        distance = np.linalg.norm(pos_i - pos_j)
                        coulomb_matrix[i, j] = (Z_i * Z_j) / distance  # Off-diagonal elements

            return coulomb_matrix
        except Exception as e:
            print(f"Error in getCoulombMatrix: {e}")
            return np.zeros((0, 0))
        

    def WTSPDbyCM(self,edge):
        try:
            mol = self.matrixdescriptors.new_mol
            coulomb_matrix = self.getCoulombMatrix(mol)
            distance = Chem.GetDistanceMatrix(mol)[edge[0], edge[1]]
            weight = coulomb_matrix[edge[0], edge[1]]
            return [distance * weight]
        except Exception as e:
            print(f"Error in WTSPDbyCM: {e}")
            return [-1]


    def usePBC(self,edge,id=1):
        mol = self.matrixdescriptors.new_mol
        try:
            pos1 = mol.GetConformer(conf_id=id).GetAtomPosition(edge[0])
            pos2 = mol.GetConformer(conf_id=id).GetAtomPosition(edge[1])
            delta = pos2 - pos1
            delta -= self.params.sp_box_size * np.round(delta / self.params.sp_box_size)  # Apply periodic boundary conditions
            distance = np.linalg.norm(delta)
            return [distance]
        except:
            return [-1]
            

    def WtSPDByAll(self,edge):
        all_paths = self.getAllPaths(edge[0], edge[1])
        try:
            distance = Chem.GetDistanceMatrix(self.matrixdescriptors.new_mol)[edge[0], edge[1]]
            wdm = []
            for path in all_paths['shortest_paths']:
                atoms = path['atoms']
                bonds = path['bonds']
                total_weight_bonds = 0
                total_weight_atoms = 0
                total_weight_hyb = 0
                total_weight_ve = 0
                for atom in atoms:
                    total_weight_atoms += atom.GetMass() / 128
                    hyb = next(key for key, value in atom.GetHybridization().values.items() if value == atom.GetHybridization())
                    total_weight_hyb += hyb / 9
                    try:
                        total_weight_ve += self.properties.el_valence[atom.GetSymbol()] / 8
                    except: 
                        total_weight_ve += self.properties.el_valence[atom.GetSymbol().lower()] / 8
                for bond in bonds:
                    total_weight_bonds += bond.GetBondTypeasDouble() / 8

                total_weight = total_weight_atoms + total_weight_bonds + total_weight_hyb + total_weight_ve
                wdm.append(total_weight*distance)
            return [min(wdm)]
        except Exception as e:
            print(f"Error in WtSPDByMass: {e}")
            return [-1]


class WeightedAdjacencyMatrix(PathDistanceMetrics):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def getmasses(self):
        try:
            masses = [atom.GetMass() for atom in self.matrixdescriptors.element]
            return masses
        except Exception as e:
            print(f"Error in getmasses: {e}")
            return []

    def getvalence(self):
        try:
            valence = [] 
            for atom in self.matrixdescriptors.element:
                try:
                    valence.append(self.properties.el_valence[atom.GetSymbol()])
                except:
                    valence.append(self.properties.el_valence[atom.GetSymbol().lower()])
            return valence
        except Exception as e:
            print(f"Error in getvalence: {e}")
            return []
    
    def gethyb(self):
        try:
            hyb = []
            for atom in self.matrixdescriptors.element:
                hyb.append(next(key for key, value in atom.GetHybridization().values.items() if value == atom.GetHybridization()))
            return hyb
        except Exception as e:
            print(f"Error in gethyb: {e}")
            return []

    def WtAdjMatByBO(self):
        self.G = (self.matrixdescriptors.bond_mat / 8) * self.matrixdescriptors.adj_mat
    
    def WtAdjMatByAM(self):
        masses = self.getmasses()
        self.G = self.matrixdescriptors.adj_mat * np.sqrt(np.outer(masses, masses))
    
    def WtAdjMatByValence(self):
        valence = self.getvalence()
        self.G = self.matrixdescriptors.adj_mat * np.sqrt(np.outer(valence, valence))
    
    def WtAdjMatByHYB(self):
        hyb = self.gethyb()
        self.G = self.matrixdescriptors.adj_mat * np.sqrt(np.outer(hyb, hyb))

    def WtAdjMatByCoulomb(self):
        try:
            mol = self.matrixdescriptors.new_mol
            coulomb_matrix = self.getCoulombMatrix(mol)
            self.G = self.matrixdescriptors.adj_mat * coulomb_matrix
        except Exception as e:
            print(f"Error in WtAdjMatByCoulomb: {e}")
            self.G = self.matrixdescriptors.adj_mat

    def WtAdjMatByAll(self):
        masses = self.getmasses()
        valence = self.getvalence()
        hyb = self.gethyb()
        try:
            mol = self.matrixdescriptors.new_mol
            coulomb_matrix = self.getCoulombMatrix(mol)
            self.G = self.matrixdescriptors.adj_mat * np.sqrt(np.outer(masses, masses)) + \
                     self.matrixdescriptors.adj_mat * np.sqrt(np.outer(valence, valence)) + \
                     self.matrixdescriptors.adj_mat * np.sqrt(np.outer(hyb, hyb)) + \
                     self.matrixdescriptors.adj_mat * coulomb_matrix
        except Exception as e:
            print(f"Error in WtAdjMatByAll: {e}")
            self.G = self.matrixdescriptors.adj_mat

    def getWeightedMatrix(self):
        if self.params.wt_adj_mat_by == 'bond_order':
            self.WtAdjMatByBO()
        elif self.params.wt_adj_mat_by == 'atomic_mass':
            self.WtAdjMatByAM()
        elif self.params.wt_adj_mat_by == 'valence':
            self.WtAdjMatByValence()
        elif self.params.wt_adj_mat_by == 'hybridization':
            self.WtAdjMatByHYB()
        elif self.params.wt_adj_mat_by == 'coulomb':
            self.WtAdjMatByCoulomb()
        elif self.params.wt_adj_mat_by == 'all':
            self.WtAdjMatByAll()
        else:
            self.G = self.matrixdescriptors.adj_mat


class CommuteTimes(WeightedAdjacencyMatrix):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    
    def SetupCommuteTimeMatrix(self):
        self.getWeightedMatrix()
        if self.params.wt_adj_mat_by is not None:
            self.G = nx.from_numpy_array(self.G, create_using=nx.Graph)
        else:
            self.G = nx.from_numpy_array(self.matrixdescriptors.adj_mat, create_using=nx.Graph)
    
    # --- Spectral Approximation using Laplacian Pseudoinverse ---
    def spectral_commute_time(self,u, v):
        if not nx.is_connected(self.G):
            raise ValueError("Graph must be connected to compute commute time.")

        L = nx.laplacian_matrix(self.G).todense()
        L_pinv = pinv(L)

        vol_G = self.G.number_of_edges() * 2
        ct = vol_G * (L_pinv[u, u] + L_pinv[v, v] - 2 * L_pinv[u, v])
        return float(ct)

    # --- Random Walk Approximation ---
    def approximate_commute_time(self,u, v, num_walks=100, max_steps=1000):
        total_time = 0
        for _ in range(num_walks):
            current = u
            steps = 0
            while current != v and steps < max_steps:
                neighbors = list(self.G.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                steps += 1
            while current != u and steps < max_steps:
                neighbors = list(self.G.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                steps += 1
            total_time += steps
        return total_time / num_walks if num_walks > 0 else float('inf')
    
    # --- Personalized PageRank Approximation ---
    def ppr_score(self, u, v, alpha=0.85):
        ppr = nx.pagerank(self.G, alpha=alpha, personalization={u: 1})
        return 1 / ppr.get(v, 1e-9)  # Inverse to mimic distance

    # --- Hitting Time Approximation ---
    def approximate_hitting_time(self,u, v, num_walks=100, max_steps=1000):
        total_steps = 0
        for _ in range(num_walks):
            current = u
            steps = 0
            while current != v and steps < max_steps:
                neighbors = list(self.G.neighbors(current))
                if not neighbors:
                    break
                current = random.choice(neighbors)
                steps += 1
            total_steps += steps
        return total_steps / num_walks if num_walks > 0 else float('inf')

    def approximate_commute_time_via_hitting(self,u, v, **kwargs):
        return self.approximate_hitting_time(u, v, **kwargs) + self.approximate_hitting_time(v, u, **kwargs)

    # --- Truncated Random Walk Matrix Power ---
    def truncated_rw_score(self,u, v, steps=100):
        A = nx.to_numpy_array(self.G)
        P = A / A.sum(axis=1, keepdims=True)
        Pk = np.linalg.matrix_power(P, steps)
        return 1 / (Pk[u][v] + 1e-9)

    def compute_hitting_times(self):
        n = self.G.shape[0]
        D = np.diag(self.G.sum(axis=1))
        L = D - self.G
        L_plus = pinv(L)
        degrees = np.diag(D)
        pi = degrees / degrees.sum()

        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    H[i, j] = (L_plus[j, j] - L_plus[i, j]) / pi[j]
        return H

    def mean_first_passage_time(self):
        # Average over all i ≠ j
        H = self.compute_hitting_times()
        n = H.shape[0]
        return np.sum(H) / (n * (n - 1))

    def compute_resistance_distance(self, edge):
        try:
            return nx.resistance_distance(self.G, edge[0], edge[1])
        except Exception as e:
            print(f"Error computing resistance distance: {e}")
            return -1

    def compute_clustering_coefficient(self, node):
        try:
            return nx.clustering(self.G, node)
        except Exception as e:
            print(f"Error computing clustering coefficient: {e}")
            return -1

    def compute_closeness_centrality(self, node):
        try:
            return nx.closeness_centrality(self.G, node)
        except Exception as e:
            print(f"Error computing closeness centrality: {e}")
            return -1

    def compute_degree_centrality(self, node):
        try:
            return nx.degree_centrality(self.G)[node]
        except Exception as e:
            print(f"Error computing degree centrality: {e}")
            return -1

    def _get_ring_info(self):
        return self.matrixdescriptors.new_mol.GetRingInfo().AtomRings()


    def _same_ring(self, u, v):
        ring_info = self._get_ring_info()
        return any(u in ring and v in ring for ring in ring_info)

    def _same_aromatic(self, u, v):
        atom_u = self.matrixdescriptors.new_mol.GetAtomWithIdx(int(u))
        atom_v = self.matrixdescriptors.new_mol.GetAtomWithIdx(int(v))

        shortest_path = nx.shortest_path(self.G, source=u, target=v)
        atoms_in_path = [self.matrixdescriptors.new_mol.GetAtomWithIdx(int(idx)) for idx in shortest_path]
        return all(atom.GetIsAromatic() for atom in atoms_in_path)

    
    def _topological_overlap(self, u, v):
        common_neighbors = self._get_common_neighbors(u,v)
        neighbors_u = list(self.G.neighbors(u))
        neighbors_v = list(self.G.neighbors(v))
        return len(common_neighbors) / (len(neighbors_u) + len(neighbors_v) - len(common_neighbors))

    def _edge_clustering_coefficient(self, u, v):
        return nx.clustering(self.G, u, v) if nx.has_path(self.G, u, v) else 0

    def _forman_ricci_curvature(self, u, v):
        """
        Computes the Forman-Ricci curvature for edge (edge[0],edge[1]).
        The Forman-Ricci curvature is based on local graph geometry.
        """
        # Compute the degree of each atom
        deg_u = self.G.degree(u)
        deg_v = self.G.degree(v)

        # Find common neighbors
        common_neighbors = list(nx.common_neighbors(self.G, u, v))

        # Calculate the Forman-Ricci curvature using the formula
        curvature = (len(common_neighbors) - 1) / (deg_u + deg_v - 2)

        return curvature

class SubgraphFunctions(WeightedAdjacencyMatrix):
    def SetupCommuteTimeMatrix(self):
        self.getWeightedMatrix()
        if self.params.wt_adj_mat_by is not None:
            self.G = nx.from_numpy_array(self.G, create_using=nx.Graph)
        else:
            self.G = nx.from_numpy_array(self.matrixdescriptors.adj_mat, create_using=nx.Graph)
  
    def _get_k_hop_subgraph(self, center):
        nodes_within_k = [
            node for node, dist in nx.single_source_shortest_path_length(self.G, center, cutoff=self.params.hop_radius).items()
        ]
        subgraph = self.G.subgraph(nodes_within_k)
        neighbors = list(self.G.neighbors(center))
        return subgraph, neighbors, center

    def _density(self, G_sub):
        N = G_sub.number_of_nodes()
        E = G_sub.number_of_edges()
        return (2 * E) / (N * (N - 1)) if N > 1 else 0

    def _avg_deg_neighbors(self, neighbors):
        return np.mean([self.G.degree(n) for n in neighbors]) if neighbors else 0

    def _clustering(self, G_sub, i):
        return nx.clustering(G_sub, i) if G_sub.number_of_nodes() > 2 else 0

    def _assortativity(self, i, neighbors):
        if len(neighbors) <= 1:
            return 0
        neighbor_degs = [self.G.degree(j) for j in neighbors]
        return np.corrcoef([self.G.degree(i)] * len(neighbor_degs), neighbor_degs)[0, 1]

    def _spectral_radius(self, G_sub):
        A = nx.to_numpy_array(G_sub)
        eigs = eigvalsh(A)
        return eigs[-1]


    def _fiedler_value(self, G_sub):
        A = nx.to_numpy_array(G_sub)
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        eigs = eigvalsh(L)
        return eigs[1] if len(eigs) > 1 else 0

    def _degree_entropy(self, G_sub):
        degrees = [d for _, d in G_sub.degree()]
        N = len(degrees)
        counts = Counter(degrees)
        probs = np.array([v / N for v in counts.values()])
        return -np.sum(probs * np.log(probs + 1e-10))

    def _local_bertzct(self, G_sub):
        degrees = [d for _, d in G_sub.degree()]
        term1 = sum([log(factorial(d)) if d > 1 else 0 for d in degrees])
        term2 = sum([
            log(G_sub.degree(u) * G_sub.degree(v))
            for u, v in G_sub.edges()
            if G_sub.degree(u) > 0 and G_sub.degree(v) > 0
        ])
        return term1 + term2



class RandomWalk(CommuteTimes,SubgraphFunctions):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def weightfcn(self,current_atom,neighbors,atomic_masses=None,mol=None):
        if self.params.rw_weight_by == 'atomic_mass':
            weights = [atomic_masses[neighbor]/128 for neighbor in neighbors]
        elif self.params.rw_weight_by == 'bond_order':
            weights = [mol.GetBondBetweenAtoms(current_atom, neighbor).GetBondTypeAsDouble()/8 for neighbor in neighbors]
        elif self.params.rw_weight_by == 'hyb':
            weights = [next(key for key, value in neighbor.GetHybridization().values.items() if value == neighbor.GetHybridization())/9 for neighbor in neighbors]
        elif self.params.rw_weight_by == 'valence':
            try:
                weights = [self.properties.el_valence[neighbor.GetSymbol()] / 8 for neighbor in neighbors]
            except:
                weights = [self.properties.el_valence[neighbor.GetSymbol().lower()] / 8 for neighbor in neighbors]
        elif self.params.rw_weight_by == 'coulomb':
            coulomb_matrix = self.getCoulombMatrix(mol)
            weights = [coulomb_matrix[current_atom, neighbor] for neighbor in neighbors]
        elif self.params.rw_weight_by == 'all':
            weights = []
            for neighbor in neighbors:
                bond = mol.GetBondBetweenAtoms(current_atom, neighbor)
                bond_order = bond.GetBondTypeAsDouble() / 8 if bond else 0
                atomic_mass = atomic_masses[neighbor] / 128
                try:
                    valence = self.properties.el_valence[neighbor.GetSymbol()] / 8
                except:
                    valence = self.properties.el_valence[neighbor.GetSymbol().lower()] / 8
                hybridization = next(
                    key for key, value in neighbor.GetHybridization().values.items()
                    if value == neighbor.GetHybridization()
                ) / 9
                coulomb_matrix = self.getCoulombMatrix(mol)
                coulomb_weight = coulomb_matrix[current_atom, neighbor]

                total_weight = (
                    bond_order + atomic_mass + valence + hybridization + coulomb_weight
                )
                weights.append(total_weight)
        return weights
                        
    def RWFunc(self,edge,steps=1000):
        try:
            mol = self.matrixdescriptors.new_mol
            if self.params.rw_weight_by == 'atomic_mass' or self.params.rw_weight_by == 'all':
                atomic_masses = [atom.GetMass() for atom in mol.GetAtoms()]

            current_atom = edge[0]
            path = [current_atom]
            
            for _ in range(steps):
                neighbors = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(int(current_atom)).GetNeighbors()]
            
                if not neighbors:
                    break
                
                if edge[1] in neighbors:
                    path.append(edge[1])
                    break
                
                if self.params.rw_weight_by is not None:
                    weights = self.weightfcn(current_atom, neighbors, atomic_masses, mol)
                    total_weight = sum(weights)
                    probabilities = [weight / total_weight for weight in weights]
                    next_atom = np.random.choice(neighbors, p=probabilities)
                else:
                    next_atom = np.random.choice(neighbors)
                path.append(next_atom)
                current_atom = next_atom
        
            return path
        except Exception as e:
            print(f"Error in RW: {e}")
            return []
            
    # --- Unified Interface ---
    def CommuteTime(self,u, v, method="spectral", fallback_to_random=True):
        try:
            if method == "spectral":
                return self.spectral_commute_time(u, v)
            elif method == "random_walk":
                return self.approximate_commute_time(u,v)
            elif method == "ppr":
                return self.ppr_score(u,v)
            elif method == "hitting":
                return self.approximate_commute_time_via_hitting(u,v)
            elif method == "truncated_rw":
                return self.truncated_rw_score(u,v)
            else:
                raise ValueError(f"Unknown method: {method}")
        except Exception as e:
            print(f"Method {method} failed: {e}")
            if fallback_to_random:
                return self.approximate_commute_time(u,v)
            else:
                return -1  # large fallback

    def FirstPassageTime(self,edge):
        fpt = self.mean_first_passage_time()
        return fpt[edge[0], edge[1]]
    
    def ResistanceDistance(self,edge):
        return self.compute_resistance_distance(edge)

    def ClusteringCoefficient(self,node):
        return self.compute_clustering_coefficient(node)

    def ClosenessCentrality(self, node):
        return self.compute_closeness_centrality(node)
    
    def DegreeCentrality(self, node):
        return self.compute_degree_centrality(node)
    
    def SameRing(self, edge):
        return self._same_ring(edge[0],edge[1])

    def SameAromaticSequence(self,edge):
        return self._same_aromatic(edge[0],edge[1])
    
    def TopologicalOverlap(self,edge):
        return self._topological_overlap(edge[0],edge[1])
    
    def EdgeClusteringCoefficient(self,edge):
        return self._edge_clustering_coefficient(edge[0],edge[1])
    
    def FormanRicciCurvature(self,edge):
        return self._forman_ricci_curvature(edge[0],edge[1])
    
    def Density(self,ind):
        G_sub, neighbors, center = self._get_k_hop_subgraph(ind)
        return self._density(G_sub)
    
    def AvgDegreeNeighbors(self,ind):
        G_sub, neighbors, center = self._get_k_hop_subgraph(ind)
        return self._avg_deg_neighbors(neighbors)
    
    def Clustering(self,ind):
        G_sub, neighbors, center = self._get_k_hop_subgraph(ind)
        return self._clustering(G_sub, center)
    
    def Assortativity(self,ind):
        G_sub, neighbors, center = self._get_k_hop_subgraph(ind)
        return self._assortativity(center, neighbors)
    
    def SpectralRadius(self,ind):
        G_sub, neighbors, center = self._get_k_hop_subgraph(ind)
        return self._spectral_radius(G_sub)
    
    def DegreeEntropy(self,ind):
        G_sub, neighbors, center = self._get_k_hop_subgraph(ind)
        return self._degree_entropy(G_sub)

    def LocalBertzCT(self,ind):
        G_sub, neighbors, center = self._get_k_hop_subgraph(ind)
        return self._local_bertzct(G_sub)
        

