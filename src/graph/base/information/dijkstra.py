from rdkit import Chem
import networkx as nx
from ..base import BaseFeaturizer
import random
from dataclasses import dataclass
import heapq


class DijkstraFeaturizer(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)



    def is_rotatable(self,bond):
        """Heuristic check for rotatable bond."""
        # Single, non-ring bond between non-terminal heavy atoms
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            return False
        if bond.IsInRing():
            return False

        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()

        if begin.GetDegree() == 1 or end.GetDegree() == 1:
            return False

        return True

    def get_weighted_graph_with_stereo(self,id=1):
        G = nx.Graph()
        conf = self.conformer
        if isinstance(conf,Chem.rdchem.Mol): conf = conf.GetConformer()
        mol = self.matrixdescriptors.new_mol

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            # Base weight from bond order

            if self.params.usebofactor_dijkstra:
                order = bond.GetBondTypeAsDouble()
                weight = 1.0 / order  # Higher order = stronger bond = lower weight


            if self.params.useconjfactor_dijkstra:
                # Conjugation factor
                conjugation_factor = 1.5 if bond.GetIsConjugated() else 1.0
            else:
                conjugation_factor = 1.0

            if self.params.userotfactor_dijkstra:
                # Rotatability factor
                rotatable = self.is_rotatable(bond)
                rotatability_factor = 0.75 if rotatable else 1.25
            else:
                rotatability_factor = 1.0
            
            if self.params.usestereofactor_dijkstra:
                stereo = bond.GetStereo()
                if stereo == Chem.rdchem.BondStereo.STEREOE:
                    stereo_factor = 1.1
                elif stereo == Chem.rdchem.BondStereo.STEREOZ:
                    stereo_factor = 1.4
                else:
                    stereo_factor = 1.0
            else:
                stereo_factor = 1.0

            if self.params.usedistfactor_dijkstra:
                # Distance factor
                begin_pos = conf.GetAtomPosition(i)
                end_pos = conf.GetAtomPosition(j)
                distance = (begin_pos - end_pos).Length()
                if distance < 1.5:
                    dist_factor = 0.5
                elif distance < 2.5:
                    dist_factor = 1.0
                else:
                    dist_factor = 1.5

            # Combined weight
            if self.params.combo_dijkstra == 'product':
                combined_weight = weight * conjugation_factor * rotatability_factor * stereo_factor * dist_factor
            elif self.params.combo_dijkstra == 'sum':
                combined_weight = weight + conjugation_factor + rotatability_factor + stereo_factor + dist_factor
            elif self.params.combo_dijkstra == 'combo_product':
                combined_weight = weight * conjugation_factor * rotatability_factor * stereo_factor / dist_factor
            elif self.params.combo_dijkstra == 'combo_sum':
                combined_weight = weight/dist_factor + conjugation_factor + rotatability_factor + stereo_factor 
            G.add_edge(i, j, weight=combined_weight)

        return G

    def biased_random_walk(self,G, start, end, max_steps=100):
        path = [start]
        current = start
        visited = set([start])

        for _ in range(max_steps):
            neighbors = [n for n in G.neighbors(current) if n not in visited]
            if not neighbors:
                break

            # Probabilities inversely proportional to weight
            weights = [1.0 / G[current][n]['weight'] for n in neighbors]
            total = sum(weights)
            probs = [w / total for w in weights]

            next_node = random.choices(neighbors, probs)[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node

            if current == end:
                break

        return path

    def GetRandomWalkWDijkstra(self,edge,id=1):
        G = self.get_weighted_graph_with_stereo(id)
        start, end = edge
        path = self.biased_random_walk(G, start, end)
        return path
    


    def floyd_warshall_random_walk(self, edge, id=1):
        G = self.get_weighted_graph_with_stereo(id)
        start, end = edge

        # Compute shortest paths using Floyd-Warshall algorithm
        shortest_paths = dict(nx.floyd_warshall(G, weight='weight'))

        # Extract the shortest path distance between start and end
        if start in shortest_paths and end in shortest_paths[start]:
            shortest_distance = shortest_paths[start][end]
        else:
            return []  # No path exists

        # Perform a random walk biased by the shortest path distance
        path = [start]
        current = start
        visited = set([start])

        while current != end:
            neighbors = [n for n in G.neighbors(current) if n not in visited]
            if not neighbors:
                break

            # Probabilities inversely proportional to weight and biased by shortest distance
            weights = [
                1.0 / (G[current][n]['weight'] + abs(shortest_paths[n][end] - shortest_distance))
                for n in neighbors
            ]
            total = sum(weights)
            probs = [w / total for w in weights]

            next_node = random.choices(neighbors, probs)[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path
    
    def bellman_ford_random_walk(self, edge, id=1):
        G = self.get_weighted_graph_with_stereo(id)
        start, end = edge

        # Compute shortest paths using Bellman-Ford algorithm
        try:
            shortest_paths = nx.single_source_bellman_ford_path_length(G, start, weight='weight')
        except nx.NetworkXUnbounded:
            return []  # Handle negative weight cycles if any

        # Extract the shortest path distance between start and end
        if end in shortest_paths:
            shortest_distance = shortest_paths[end]
        else:
            return []  # No path exists

        # Perform a random walk biased by the shortest path distance
        path = [start]
        current = start
        visited = set([start])

        while current != end:
            neighbors = [n for n in G.neighbors(current) if n not in visited]
            if not neighbors:
                break

            # Probabilities inversely proportional to weight and biased by shortest distance
            weights = [
                1.0 / (G[current][n]['weight'] + abs(shortest_paths.get(n, float('inf')) - shortest_distance))
                for n in neighbors
            ]
            total = sum(weights)
            probs = [w / total for w in weights]

            next_node = random.choices(neighbors, probs)[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def johnson_random_walk(self, edge, id=1):
        G = self.get_weighted_graph_with_stereo(id)
        start, end = edge

        # Compute shortest paths using Johnson's algorithm
        try:
            shortest_paths = nx.johnson(G, weight='weight')
        except nx.NetworkXError:
            return []  # Handle errors if any

        # Extract the shortest path distance between start and end
        if start in shortest_paths and end in shortest_paths[start]:
            shortest_distance = shortest_paths[start][end]
        else:
            return []  # No path exists

        # Perform a random walk biased by the shortest path distance
        path = [start]
        current = start
        visited = set([start])

        while current != end:
            neighbors = [n for n in G.neighbors(current) if n not in visited]
            if not neighbors:
                break

            # Probabilities inversely proportional to weight and biased by shortest distance
            weights = [
                1.0 / (G[current][n]['weight'] + abs(shortest_paths[start][n] + shortest_paths[n][end] - shortest_distance))
                for n in neighbors
            ]
            total = sum(weights)
            probs = [w / total for w in weights]

            next_node = random.choices(neighbors, probs)[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def yen_random_walk(self, edge, id=1, k=3):
        G = self.get_weighted_graph_with_stereo(id)
        start, end = edge

        # Compute k-shortest paths using Yen's algorithm
        try:
            k_shortest_paths = list(nx.shortest_simple_paths(G, start, end, weight='weight'))
            k_shortest_paths = k_shortest_paths[:k]  # Limit to k paths
        except nx.NetworkXNoPath:
            return []  # No path exists

        # Perform a random walk biased by the k-shortest paths
        path = [start]
        current = start
        visited = set([start])

        while current != end:
            neighbors = [n for n in G.neighbors(current) if n not in visited]
            if not neighbors:
                break

            # Probabilities inversely proportional to weight and biased by k-shortest paths
            weights = []
            for n in neighbors:
                path_bias = sum(1.0 for p in k_shortest_paths if current in p and n in p and p.index(n) > p.index(current))
                weights.append(1.0 / (G[current][n]['weight'] + (1.0 / (path_bias + 1e-6))))

            total = sum(weights)
            probs = [w / total for w in weights]

            next_node = random.choices(neighbors, probs)[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path
    
    def BiasedRandomWalk(self,edge,id=1):
        if self.params.bias_walk == 'floyd_warshall':
            path = self.floyd_warshall_random_walk(edge, id)
        elif self.params.bias_walk == 'bellman_ford':
            path = self.bellman_ford_random_walk(edge, id)
        elif self.params.bias_walk == 'johnson':
            path = self.johnson_random_walk(edge, id)
        elif self.params.bias_walk == 'dijkstra':
            path = self.GetRandomWalkWDijkstra(edge, id)
        elif self.params.bias_walk == 'yen':
            path = self.yen_random_walk(edge, id)

        return [len(path)]
    


class MolecularAStar(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        

    def atomic_features(self, atom_idx):
        atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(atom_idx)
        return {
            "mass": self.atomic_masses.GetAtomicWeight(atom.GetAtomicNum()),
            "valence": atom.GetTotalValence(),
            "charge": atom.GetFormalCharge(),
            "hybridization": int(atom.GetHybridization()),  # RDKit enum
        }

    def bond_order(self, atom1, atom2):
        bond = self.matrixdescriptors.new_mol.GetBondBetweenAtoms(atom1, atom2)
        return bond.GetBondTypeAsDouble() if bond else 1.0

    def coulomb_interaction(self, i, j,id=1):
        Zi = self.matrixdescriptors.new_mol.GetAtomWithIdx(i).GetAtomicNum()
        Zj = self.matrixdescriptors.new_mol.GetAtomWithIdx(j).GetAtomicNum()
        self.conf = self.matrixdescriptors.new_mol.GetConformer(id)
        if self.conf:
            dist = self.conf.GetAtomPosition(i).Distance(self.conf.GetAtomPosition(j))
            return Zi * Zj / (dist if dist != 0 else 1)
        return 0  # if no coordinates available

    def heuristic(self, current_idx, goal_idx,id=1):
        # Graph distance approximation using atomic features
        feat_curr = self.atomic_features(current_idx)
        feat_goal = self.atomic_features(goal_idx)
        
        diff = (
            abs(feat_curr["mass"] - feat_goal["mass"]) +
            abs(feat_curr["valence"] - feat_goal["valence"]) +
            abs(feat_curr["charge"] - feat_goal["charge"]) +
            abs(feat_curr["hybridization"] - feat_goal["hybridization"])
        )
        coulomb = self.coulomb_interaction(current_idx, goal_idx,id=1)
        return diff - 0.1 * coulomb  # you can tune this

    def cost(self, from_idx, to_idx):
        bond = self.matrixdescriptors.new_mol.GetBondBetweenAtoms(from_idx, to_idx)
        bond_order = bond.GetBondTypeAsDouble() if bond else 1.0
        feat = self.atomic_features(to_idx)
        atomic_valence = feat["valence"]
        mass = feat["mass"]
        charge = abs(feat["charge"])
        return 1.0 / bond_order + 0.1 * mass + 0.5 * charge + 0.05 * atomic_valence

    def search(self, start_idx, goal_idx,id=1):
        frontier = [(0, start_idx)]
        came_from = {start_idx: None}
        cost_so_far = {start_idx: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal_idx:
                break

            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(current)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                new_cost = cost_so_far[current] + self.cost(current, neighbor_idx)
                if neighbor_idx not in cost_so_far or new_cost < cost_so_far[neighbor_idx]:
                    cost_so_far[neighbor_idx] = new_cost
                    priority = new_cost + self.heuristic(neighbor_idx, goal_idx,id=id)
                    heapq.heappush(frontier, (priority, neighbor_idx))
                    came_from[neighbor_idx] = current

        # Reconstruct path
        path = []
        curr = goal_idx
        while curr is not None:
            path.append(curr)
            curr = came_from[curr]
        path.reverse()
        return path

    def GetAStarPathLength(self,edge,id=1):
        if self.params.bias_walk == 'astar':
            start, end = edge
            path = self.search(start, end,id=id)
            return [len(path)]
        else:
            return []
    