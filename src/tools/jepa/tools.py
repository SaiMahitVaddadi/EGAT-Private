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