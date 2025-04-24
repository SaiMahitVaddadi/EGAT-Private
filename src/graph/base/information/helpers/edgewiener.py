import networkx as nx

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








