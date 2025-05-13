from .weightedadjmat import WeightedAdjacencyMatrix
import networkx as nx
import numpy as np
from scipy.linalg import pinv
import random

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
            return nx.closeness_centrality(self.G, node) # Can add Distance 
        except Exception as e:
            print(f"Error computing closeness centrality: {e}")
            return -1
        
    def compute_closeness_centrality_dist(self, node,d=1):
        try:
            return nx.closeness_centrality(self.G, node,distance=d) # Can add Distance 
        except Exception as e:
            print(f"Error computing closeness centrality: {e}")
            return -1
    
    def compute_edgebetweeness_centrality(self, edge,wt=None):
        try:
            return nx.edge_betweenness_centrality(self.G,weight=wt)[edge] # Can add Distance 
        except Exception as e:
            print(f"Error computing edge betweenness centrality: {e}")
            return -1
        
    def compute_currentflowcloseness_centrality_dist(self, node,wt=None):
        try:
            return nx.current_flow_closeness_centrality(self.G,weight =wt)[node] # Can add Distance 
        except Exception as e:
            print(f"Error computing closeness centrality: {e}")
            return -1
        
    def compute_currentflowbetweeness_centrality_dist(self, node,wt=None):
        try:
            return nx.current_flow_betweenness_centrality(self.G,weight =wt)[node] # Can add Distance 
        except Exception as e:
            print(f"Error computing closeness centrality: {e}")
            return -1
    def compute_communicability_betweenness_centrality(self, node):
        try:
            return nx.communicability_betweenness_centrality(self.G)[node]
        except Exception as e:
            print(f"Error computing communicability betweenness centrality: {e}")
            return -1

    def compute_load_centrality(self, node, cutoff=None, weight=None):
        try:
            return nx.load_centrality(self.G, v=node, cutoff=cutoff, weight=weight)
        except Exception as e:
            print(f"Error computing load centrality: {e}")
            return -1

    def compute_edge_load_centrality(self, edge, cutoff=None, weight=None):
        try:
            return nx.edge_load_centrality(self.G, cutoff=cutoff, weight=weight)[edge]
        except Exception as e:
            print(f"Error computing edge load centrality: {e}")
            return -1
        
    def compute_subgraph_centrality(self, node):
        try:
            return nx.subgraph_centrality(self.G)[node]
        except Exception as e:
            print(f"Error computing subgraph centrality: {e}")
            return -1

    def compute_harmonic_centrality(self, node, distance=None):
        try:
            return nx.harmonic_centrality(self.G, distance=distance)[node]
        except Exception as e:
            print(f"Error computing harmonic centrality: {e}")
            return -1

    def compute_dispersion(self, u, v=None):
        try:
            return nx.dispersion(self.G, u, v)
        except Exception as e:
            print(f"Error computing dispersion: {e}")
            return -1

    def compute_local_reaching_centrality(self, node, weight=None):
        try:
            return nx.local_reaching_centrality(self.G, node, weight=weight)
        except Exception as e:
            print(f"Error computing local reaching centrality: {e}")
            return -1

    def compute_percolation_centrality(self, node,weight=None):
        try:
            return nx.percolation_centrality(self.G, weight=weight)[node]
        except Exception as e:
            print(f"Error computing percolation centrality: {e}")
            return -1
    
    def second_order_centrality(self, node):
        try:
            return nx.second_order_centrality(self.G)[node]
        except Exception as e:
            print(f"Error computing second order centrality: {e}")
            return -1

    def trophic_levels(self,edge):
        try:
            return nx.trophic_levels(self.G)[edge]
        except Exception as e:
            print(f"Error computing trophic levels: {e}")
            return {}

    def trophic_differences(self,edge):
        try:
            return nx.trophic_differences(self.G)[edge]
        except Exception as e:
            print(f"Error computing trophic differences: {e}")
            return {}
        
    def edmonds_karp(self, edge):
        try:
            return nx.algorithms.flow.edmonds_karp(self.G, edge[0], edge[1])
        except Exception as e:
            print(f"Error in edmonds_karp: {e}")
            return None

    def shortest_augmenting_path(self, edge):
        try:
            return nx.algorithms.flow.shortest_augmenting_path(self.G, edge[0], edge[1])
        except Exception as e:
            print(f"Error in shortest_augmenting_path: {e}")
            return None

    def preflow_push(self, edge):
        try:
            return nx.algorithms.flow.preflow_push(self.G, edge[0], edge[1])
        except Exception as e:
            print(f"Error in preflow_push: {e}")
            return None

    def dinitz(self, edge):
        try:
            return nx.algorithms.flow.dinitz(self.G, edge[0], edge[1])
        except Exception as e:
            print(f"Error in dinitz: {e}")
            return None

    def boykov_kolmogorov(self, edge):
        try:
            return nx.algorithms.flow.boykov_kolmogorov(self.G, edge[0], edge[1])
        except Exception as e:
            print(f"Error in boykov_kolmogorov: {e}")
            return None

    def max_flow_min_cost(self, edge):
        try:
            return nx.algorithms.flow.min_cost_flow(self.G, demand={edge[0]: -1, edge[1]: 1})
        except Exception as e:
            print(f"Error in max_flow_min_cost: {e}")
            return None

    def EffectiveSize(self, node, weight=None):
        try:
            return nx.effective_size(self.G, nodes=node, weight=weight)
        except Exception as e:
            print(f"Error computing effective size: {e}")
            return -1
    
    
    def approximate_current_flow_betweenness_centrality(self, weight=None):
        try:
            return nx.approximate_current_flow_betweenness_centrality(
                self.G,
                weight=weight
            )
        except Exception as e:
            print(f"Error in approximate_current_flow_betweenness_centrality: {e}")
            return {}
    

    
    def compute_edgecurrflowbetweeness_centrality(self, edge,wt=None):
        try:
            return nx.edge_current_flow_betweenness_centrality(self.G,weight=wt)[edge] # Can add Distance 
        except Exception as e:
            print(f"Error computing edge betweenness centrality: {e}")
            return -1

    def compute_information_centrality_dist(self, node,wt=None):
        try:
            return nx.algorithms.centrality.information_centrality(self.G,weight =wt)[node] # Can add Distance 
        except Exception as e:
            print(f"Error computing closeness centrality: {e}")
            return -1
    
    def compute_incr_closeness_centrality(self, edge):
        try:
            return nx.incremental_closeness_centrality(self.G, edge)
        except Exception as e:
            print(f"Error computing closeness centrality: {e}")
            return -1

    def compute_degree_centrality(self, node):
        try:
            return nx.degree_centrality(self.G)[node]
        except Exception as e:
            print(f"Error computing degree centrality: {e}")
            return -1
    
    def compute_indegree_centrality(self, node):
        try:
            return nx.in_degree_centrality(self.G)[node]
        except Exception as e:
            print(f"Error computing degree centrality: {e}")
            return -1
    
    def compute_outdegree_centrality(self, node):
        try:
            return nx.out_degree_centrality(self.G)[node]
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