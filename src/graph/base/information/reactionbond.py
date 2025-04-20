from ..reaction import BaseReactionFeaturizer
from rdkit import Chem
import networkx as nx
from rdkit.Chem import BRICS


class ReactiveBondInformation(BaseReactionFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    
    def BondChangeInfo(self,edge):
        BO_R = self.reactant.matrixdescriptors.bond_mat[edge[0],edge[1]]
        BO_P = self.product.matrixdescriptors.bond_mat[edge[0],edge[1]]
        if BO_R == BO_P:
            RBtype = self.reactant.properties.bond_encode['T1']
            PBtype = self.reactant.properties.bond_encode['T1']
        elif BO_R == 0.0:
            RBtype = self.reactant.properties.bond_encode['T4']
            PBtype = self.reactant.properties.bond_encode['T3']
        elif BO_P == 0.0:
            RBtype = self.reactant.properties.bond_encode['T3']
            PBtype = self.reactant.properties.bond_encode['T4']
        elif BO_R < BO_P and BO_R > 0:
            RBtype = self.reactant.properties.bond_encode['T2']
            PBtype = self.reactant.properties.bond_encode['T2']
        elif BO_R > BO_P and BO_R > 0:
            RBtype = self.reactant.properties.bond_encode['T5']
            PBtype = self.reactant.properties.bond_encode['T5']
        return RBtype,PBtype

    def OldBondChangeInfo(self,edge):
        BO_R = self.reactant.matrixdescriptors.bond_mat[edge[0],edge[1]]
        BO_P = self.product.matrixdescriptors.bond_mat[edge[0],edge[1]]
        if BO_R == BO_P:
            RBtype = self.reactant.properties.old_bond_encode['T1']
            PBtype = self.reactant.properties.old_bond_encode['T1']
        elif BO_R == 0.0:
            RBtype = self.reactant.properties.old_bond_encode['T4']
            PBtype = self.reactant.properties.old_bond_encode['T3']
        elif BO_P == 0.0:
            RBtype = self.reactant.properties.old_bond_encode['T3']
            PBtype = self.reactant.properties.old_bond_encode['T4']
        elif BO_R != BO_P and BO_R > 0:
            RBtype = self.reactant.properties.old_bond_encode['T2']
            PBtype = self.reactant.properties.old_bond_encode['T2']
        return RBtype,PBtype
    
    def DistanceFromReactingBond(self,edge):
        if self.params.adddisttoreactingbonds:
            if len(self.reactionpropeties.bond.reactive) > 0:
                dis = min([self.reactant.gs[edge[0]][rb[0]] + self.reactant.gs[edge[1]][rb[1]] for rb in self.reactionpropeties.bond.reactive])
            else:
                dis = 0
            return [dis]
        else:
            return []
                    
    
    def BondOrderChange(self, edge):
        BO_R = self.reactant.matrixdescriptors.bond_mat[edge[0], edge[1]]
        BO_P = self.product.matrixdescriptors.bond_mat[edge[0], edge[1]]
        return [BO_P - BO_R]

    def BondNeighborhoodChange(self, edge):
        neighbors_R = set(self.reactant.matrixdescriptors.adj_mat[edge[0]].nonzero()[0]).union(
            set(self.reactant.matrixdescriptors.adj_mat[edge[1]].nonzero()[0]))
        neighbors_P = set(self.product.matrixdescriptors.adj_mat[edge[0]].nonzero()[0]).union(
            set(self.product.matrixdescriptors.adj_mat[edge[1]].nonzero()[0]))
        common_neighbors = neighbors_R.intersection(neighbors_P)
        total_neighbors = neighbors_R.union(neighbors_P)
        if len(total_neighbors) > 0:
            return [len(common_neighbors) / len(total_neighbors)]
        else:
            return [0]

    def ShortestPathChangeAcrossBond(self, edge):
        reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
        product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
        sp_R = nx.shortest_path_length(reactant_graph, source=edge[0], target=edge[1])
        sp_P = nx.shortest_path_length(product_graph, source=edge[0], target=edge[1])
        return [sp_P - sp_R]

    def BondParticipationDegree(self, edge):
        reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
        product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
        degree_R = reactant_graph.degree(edge[0]) + reactant_graph.degree(edge[1])
        degree_P = product_graph.degree(edge[0]) + product_graph.degree(edge[1])
        return [degree_P - degree_R]

    def BondConnectivityPathDifference(self, edge):
        reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
        product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
        connectivity_R = nx.all_pairs_shortest_path_length(reactant_graph)
        connectivity_P = nx.all_pairs_shortest_path_length(product_graph)
        diff = 0
        for node in connectivity_R:
            for target, length in connectivity_R[node].items():
                if target in connectivity_P[node]:
                    diff += abs(length - connectivity_P[node][target])
                else:
                    diff += length
        return [diff]

    def BRICSBondRoleChange(self, edge):
        if self.params.getbrics:
            role_R = self.reactant.brics[edge[0]] if edge[0] in self.reactant.brics else None
            role_P = self.product.brics[edge[0]] if edge[0] in self.product.brics else None
            if role_R == role_P:
                return [0]
            else:
                return [1]
        else:
            return []

    def BRICSBondFormationLikelihood(self, edge):
        if self.params.getbrics:
            mol = Chem.MolFromSmiles(self.smiles)
            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            bond_breaks = [(bond[0][0], bond[0][1]) for bond in brics_bonds]
            if edge in bond_breaks or (edge[1], edge[0]) in bond_breaks:
                return [1]
            else:
                return [0]
        else:
            return []

    def BRICSFragmentationConsistency(self, edge):
        if self.params.getbrics:
            mol = Chem.MolFromSmiles(self.smiles)
            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            bond_breaks = [(bond[0][0], bond[0][1]) for bond in brics_bonds]
            reactant_bonds = set(self.reactant.brics)
            product_bonds = set(self.product.brics)
            common_bonds = reactant_bonds.intersection(product_bonds)
            total_bonds = reactant_bonds.union(product_bonds)
            if len(total_bonds) > 0:
                return [len(common_bonds) / len(total_bonds)]
            else:
                return [0]
        else:
            return []
    
