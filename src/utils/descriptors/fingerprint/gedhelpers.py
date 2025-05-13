def node_subst_cost(n1, n2):
    return 0 if n1['atomic_num'] == n2['atomic_num'] else 1

def node_del_cost(n):
    return 1

def node_ins_cost(n):
    return 1

def edge_subst_cost(e1, e2):
    return abs(e1['bond_order'] - e2['bond_order']) * 0.5  # e.g., single → double = 0.5

def edge_del_cost(e):
    return 1

def edge_ins_cost(e):
    return 1