import networkx as nx
from collections import defaultdict


# Function to calculate shortest path distance between two edges
def edge_distance(graph, edge1, edge2):
    u1, v1 = edge1
    u2, v2 = edge2
    
    # Find the shortest distance between any vertex of edge1 and any vertex of edge2
    return nx.shortest_path_length(graph, source=u1, target=u2)  # Shortest path between u1 and u2

# Function to calculate the Edge Wiener index for a given edge
def edge_wiener_index(graph, edge):
    total_distance = 0
    all_edges = list(graph.edges())
    
    for other_edge in all_edges:
        if edge != other_edge:
            distance = edge_distance(graph, edge, other_edge)
            total_distance += distance
    
    return total_distance

def edge_wiener_index_byorder(graph, edge,order= 1):
    total_distance = 0
    all_edges = list(graph.edges())
    
    for other_edge in all_edges:
        if edge != other_edge:
            distance = edge_distance(graph, edge, other_edge)
            if distance == order:
                total_distance += distance
            
    return total_distance

# Function to calculate the Hyper-Wiener index for a given edge
def hyper_wiener_index(graph, edge):
    total_distance = 0
    all_edges = list(graph.edges())
    
    for other_edge in all_edges:
        if edge != other_edge:
            distance = edge_distance(graph, edge, other_edge)
            # Apply the hyperbolic weight: distance / (2^distance)
            hyper_wiener_weighted_distance = distance / (2 ** distance)
            total_distance += hyper_wiener_weighted_distance
    
    return total_distance

def hyper_wiener_index_byorder(graph, edge,order=1):
    total_distance = 0
    all_edges = list(graph.edges())
    
    for other_edge in all_edges:
        if edge != other_edge:
            distance = edge_distance(graph, edge, other_edge)
            # Apply the hyperbolic weight: distance / (2^distance)
            if distance == order:
                hyper_wiener_weighted_distance = distance / (2 ** distance)
                total_distance += hyper_wiener_weighted_distance
    
    return total_distance


def vertex_edge_distance(G, edge, vertex):
    # Edge is a tuple (u, v), vertex is an int
    return min(nx.shortest_path_length(G, source=vertex, target=u) for u in edge)

# Vertex-Edge Wiener Index for a given edge
def vertex_edge_wiener_for_edge(G, edge):
    return sum(vertex_edge_distance(G, edge, v) for v in G.nodes)

# Vertex-Edge Wiener Index for a given vertex
def vertex_edge_wiener_for_vertex(G, vertex):
    return sum(vertex_edge_distance(G, e, vertex) for e in G.edges)




# Group vertex distances from each edge by distance value
def vertex_edge_distance_histogram_for_edges(G):
    vewi_dist_edge = {}
    
    for edge in G.edges:
        dist_count = defaultdict(int)
        for v in G.nodes:
            d = vertex_edge_distance(G, edge, v)
            dist_count[d] += 1
        vewi_dist_edge[edge] = dict(dist_count)

    

    return vewi_dist_edge

# Group edge distances from each vertex by distance value
def vertex_edge_distance_histogram_for_vertices(G):
    vewi_dist_vertex = {}
    for v in G.nodes:
        dist_count = defaultdict(int)
        for edge in G.edges:
            d = vertex_edge_distance(G, edge, v)
            dist_count[d] += 1
        vewi_dist_vertex[v] = dict(dist_count)
    return vewi_dist_vertex




