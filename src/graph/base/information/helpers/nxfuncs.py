from networkx.algorithms.similarity import simrank_similarity,panther_similarity
from networkx.algorithms.chordal import chordal_graph_cliques,chordal_graph_treewidth,find_induced_nodes
from networkx.algorithms.asteroidal import find_asteroidal_triple
import numpy as np
from networkx.algorithms.bridges import bridges, has_bridges, local_bridges
from networkx.algorithms.bipartite import is_bipartite,sets, color, density, degrees
    

'''
To add: 
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.flow.gomory_hu_tree.html#networkx.algorithms.flow.gomory_hu_tree
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.google_matrix.html#networkx.algorithms.link_analysis.pagerank_alg.google_matrix
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.cn_soundarajan_hopcroft.html#networkx.algorithms.link_prediction.cn_soundarajan_hopcroft
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.ra_index_soundarajan_hopcroft.html#networkx.algorithms.link_prediction.ra_index_soundarajan_hopcroft
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.common_neighbor_centrality.html#networkx.algorithms.link_prediction.common_neighbor_centrality
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.within_inter_cluster.html#networkx.algorithms.link_prediction.within_inter_cluster
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.reciprocity.overall_reciprocity.html#networkx.algorithms.reciprocity.overall_reciprocity
https://networkx.org/documentation/stable/reference/algorithms/polynomials.html
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.estrada_index.html#networkx.algorithms.centrality.estrada_index
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.trophic_incoherence_parameter.html#networkx.algorithms.centrality.trophic_incoherence_parameter
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.regular.k_factor.html#networkx.algorithms.regular.k_factor
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.richclub.rich_club_coefficient.html#networkx.algorithms.richclub.rich_club_coefficient
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.smallworld.sigma.html#networkx.algorithms.smallworld.sigma
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.smallworld.omega.html#networkx.algorithms.smallworld.omega
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.smetric.s_metric.html#networkx.algorithms.smetric.s_metric
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.voronoi.voronoi_cells.html#networkx.algorithms.voronoi.voronoi_cells
To Add To Edges:
- SimRank
- Panther
- # of Induced Nodes 


To Add as Node:
- Chordal Graph Clique
- Induced Node Count
- Asteroidal Node
- Bipartite Color
- Bipartite Density
- Bipartite Degrees

To Add as Molecule:
- Chordal Graph Treewidth
- 

'''
def SimRankMatrix(G):
    sim_dict = simrank_similarity(G)
    rows = sorted(sim_dict.keys())
    cols = sorted(next(iter(sim_dict.values())).keys())
    # Fill matrix
    sim_matrix = np.array([[sim_dict[i][j] for j in cols] for i in rows])
    return sim_matrix

def PantherMatrix(G):
    nodes = sorted(G.nodes())
    n = len(nodes)
    sim_matrix = np.zeros((n, n))

    # Compute similarity from each node to all others
    for i, source in enumerate(nodes):
        scores = panther_similarity(G, source)
        for j, target in enumerate(nodes):
            sim_matrix[i, j] = scores.get(target, 0.0)
    return sim_matrix

def ChordalGraphClique(G):
    """
    Compute the clique number of the graph.
    :param G: Networkx graph.
    :return: Clique number.
    """
    cliques = [c for c in chordal_graph_cliques(G)]
    return cliques

def ChordalGraphTreeWidth(G):
    """
    Compute the treewidth of the graph.
    :param G: Networkx graph.
    :return: Treewidth.
    """
    return chordal_graph_treewidth(G)

def InducedNodes(G, treewidth_bound=9223372036854775807):
    """
    Compute the induced nodes of the graph for all pairs of nodes.
    :param G: Networkx graph.
    :param treewidth_bound: Treewidth bound (default is maximum integer value).
    :return: A tuple of two matrices:
             - A matrix of values showing the number of induced nodes.
             - A matrix of lists showing the nodes in the induced paths.
    """
    nodes = list(G.nodes())
    induced_nodes_data = {}

    # Compute induced nodes for all pairs of nodes
    for i, s in enumerate(nodes):
        for j, t in enumerate(nodes):
            if s != t:
                induced_nodes_data[(s, t)] = find_induced_nodes(G, s, t, treewidth_bound)

    # Initialize matrices
    num_nodes_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
    nodes_list_matrix = [[[] for _ in range(len(nodes))] for _ in range(len(nodes))]
    node_count = {node: 0 for node in nodes}  # Dictionary to count occurrences of each node

    # Populate matrices
    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            if source != target:
                induced_nodes = induced_nodes_data.get((source, target), [])
                num_nodes_matrix[i, j] = len(induced_nodes)
                nodes_list_matrix[i][j] = induced_nodes
                # Update the count for each node in the induced nodes
                for node in induced_nodes:
                    node_count[node] += 1

    # Convert the node_count dictionary to a list of tuples (node, count)
    node_count_list = sorted(node_count.items(), key=lambda x: x[0])

    return num_nodes_matrix, nodes_list_matrix,node_count_list

def AsteroidalNodes(G):
    """
    Compute the asteroidal nodes of the graph.
    :param G: Networkx graph.
    :return: A list of asteroidal nodes.
    """
    asteroidal_nodes = find_asteroidal_triple(G)
    return asteroidal_nodes


def IsBipartite(G):
    """
    Check if the graph is bipartite.
    :param G: Networkx graph.
    :return: True if the graph is bipartite, False otherwise.
    """
    return is_bipartite(G)

def BipartiteSets(G, top_nodes=None):
    """
    Get the bipartite node sets of the graph.
    :param G: Networkx graph.
    :param top_nodes: Optional set of top nodes.
    :return: A tuple of two sets representing the bipartite node sets.
    """
    return sets(G, top_nodes=top_nodes)

def BipartiteColor(G):
    """
    Get a two-coloring of the bipartite graph.
    :param G: Networkx graph.
    :return: A dictionary mapping nodes to colors (0 or 1).
    """
    return color(G)

def BipartiteDensity(B, nodes):
    """
    Compute the density of the bipartite graph.
    :param B: Networkx bipartite graph.
    :param nodes: Set of nodes in one partition.
    :return: Density of the bipartite graph.
    """
    return density(B, nodes)

def BipartiteDegrees(B, nodes, weight=None):
    """
    Compute the degrees of the two node sets in the bipartite graph.
    :param B: Networkx bipartite graph.
    :param nodes: Set of nodes in one partition.
    :param weight: Optional edge attribute to use as weight.
    :return: A dictionary of node degrees.
    """
    return degrees(B, nodes, weight=weight)

def GenerateBridges(G, root=None):
    """
    Generate all bridges in a graph.
    :param G: Networkx graph.
    :param root: Optional root node to start the search.
    :return: A generator of bridges (edges that, if removed, increase the number of connected components).
    """
    return list(bridges(G, root=root))

def HasBridges(G, root=None):
    """
    Decide whether a graph has any bridges.
    :param G: Networkx graph.
    :param root: Optional root node to start the search.
    :return: True if the graph has any bridges, False otherwise.
    """
    return has_bridges(G, root=root)

def LocalBridges(G, with_span=False, weight=None):
    """
    Iterate over local bridges of G optionally computing the span.
    :param G: Networkx graph.
    :param with_span: If True, include the span of each local bridge.
    :param weight: Optional edge attribute to use as weight.
    :return: A generator of local bridges (edges that, if removed, increase the shortest path distance between their endpoints).
    """
    return list(local_bridges(G, with_span=with_span, weight=weight))

