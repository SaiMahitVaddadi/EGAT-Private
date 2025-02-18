import random

def mask_uv_vectors(u,v):
    size_uv = list(u.size())[0]
    #Zhao's note: randint is inclusive
    index = random.randint(0, size_uv-1)
    #print(f"chosen index: {index}, chosen from {size_uv}\n")
    #Zhao's note: there are a few options to try, 0 , -1, size_uv
    #0: there is atom zero originally, it may be confusing to the model
    #-1: may not be recognizable by the graph (dgl), IndexError: index 40 is out of bounds for dimension 0 with size 40
    #size_uv: fails
    #could try random
    max_u = max(list(u))
    max_v = max(list(v))
    u[index] = random.randint(0, max_u)
    v[index] = random.randint(0, max_v)
    return u, v

def assign_zero(x):
    return 0

def mask_node_features(g):
    n_nodes = g.num_nodes()
    index = random.randint(0, n_nodes-1) # select a random node
    g.ndata['x'][index] = g.ndata['x'][index].apply_(assign_zero)
    return index

def mask_node_features_v2(g, num_nodes):
    n_nodes = g.num_nodes()
    indices = random.sample(range(n_nodes), num_nodes)
    #index = random.randint(0, n_nodes-1) # select a random node
    for index in indices:
        g.ndata['x'][index] = g.ndata['x'][index].apply_(assign_zero)
    return 0


def mask_edge_features(g):
    n_edges = g.num_edges()
    index = random.randint(0, n_edges-1) # select a random edge
    g.edata['x'][index] = g.edata['x'][index].apply_(assign_zero)
    return index

def mask_edge_features_v2(g, num_edges):
    n_edges = g.num_edges()
    indices = random.sample(range(n_edges), num_edges)
    for index in indices:
        g.edata['x'][index] = g.edata['x'][index].apply_(assign_zero)
    return 0

def mask_node_and_edges(g):
    n_nodes = g.num_nodes()
    index = random.randint(0, n_nodes-1) # select a random node

    # Mask the node features
    g.ndata['x'][index] = g.ndata['x'][index].apply_(assign_zero)

    # Mask the edges connected to the node
    in_edges = g.in_edges(index, form='eid')
    out_edges = g.out_edges(index, form='eid')
    all_edges = list(in_edges) + list(out_edges)
    for edge in all_edges:
        g.edata['x'][edge] = g.edata['x'][edge].apply_(assign_zero)

    return index, all_edges


def mask_node_and_edges_v2(g, num_nodes):
    n_nodes = g.num_nodes()
    indices = random.sample(range(n_nodes), num_nodes)
    for index in indices:
        g.ndata['x'][index] = g.ndata['x'][index].apply_(assign_zero)
        in_edges = g.in_edges(index, form='eid')
        out_edges = g.out_edges(index, form='eid')
        all_edges = list(in_edges) + list(out_edges)
        for edge in all_edges:
            g.edata['x'][edge] = g.edata['x'][edge].apply_(assign_zero)
    return 0


def mask_node_and_edges_v3(g, num_nodes, num_edges):
    n_nodes = g.num_nodes()
    indices = random.sample(range(n_nodes), num_nodes)
    for index in indices:
        g.ndata['x'][index] = g.ndata['x'][index].apply_(assign_zero)
    n_edges = g.num_edges()
    indices = random.sample(range(n_edges), num_edges)
    for index in indices:
        g.edata['x'][index] = g.edata['x'][index].apply_(assign_zero)
    return 0

def mask_node_and_neighbors(g, num_nodes, neighbor_level):
    n_nodes = g.num_nodes()
    indices = random.sample(range(n_nodes), num_nodes)
    for index in indices:
        # Mask the node features
        g.ndata['x'][index] = g.ndata['x'][index].apply_(assign_zero)
        
        # Mask the neighbors up to the specified level
        neighbors = set([index])
        for _ in range(neighbor_level):
            new_neighbors = set()
            for node in neighbors:
                in_edges = g.in_edges(node, form='eid')
                out_edges = g.out_edges(node, form='eid')
                all_edges = list(in_edges) + list(out_edges)
                for edge in all_edges:
                    g.edata['x'][edge] = g.edata['x'][edge].apply_(assign_zero)
                    src, dst = g.find_edges(edge)
                    new_neighbors.update(src.tolist() + dst.tolist())
            neighbors.update(new_neighbors)
        
        for neighbor in neighbors:
            g.ndata['x'][neighbor] = g.ndata['x'][neighbor].apply_(assign_zero)
    
    return indices