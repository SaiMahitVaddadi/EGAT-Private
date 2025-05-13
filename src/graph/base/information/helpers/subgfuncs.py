
from .weightedadjmat import WeightedAdjacencyMatrix
import networkx as nx
import numpy as np
from scipy.linalg import pinv
import random

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

