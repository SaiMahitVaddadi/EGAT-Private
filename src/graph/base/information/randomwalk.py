from dataclasses import dataclass
from typing import Optional
from .helpers.commutetimes import CommuteTimes
from .helpers.subgfuncs import SubgraphFunctions


# Add all of whats in commute time to this.
# Add of nxfuncs into this. 


@dataclass
class Params:
    wt_adj_mat_by: Optional[str] = None  # Options: 'bond_order', 'atomic_mass', 'valence', 'hybridization', 'coulomb', 'all'
    rw_weight_by: Optional[str] = None  # Options: 'atomic_mass', 'bond_order', 'hyb', 'valence', 'coulomb', 'all'
    sp_box_size: Optional[float] = None  # For periodic boundary conditions
    hop_radius: int = 1  # For k-hop subgraph extraction


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
        

