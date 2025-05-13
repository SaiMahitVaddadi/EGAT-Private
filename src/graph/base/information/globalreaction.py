from ..reaction import BaseReactionFeaturizer
from ....utils.descriptors.egat.functional import FunctionalGroups

import networkx as nx
from dataclasses import dataclass

@dataclass
class ReactionParams:
    getshortestpathchange: bool = False
    getcommonneighborcountchange: bool = False
    getrandomwalkchange: bool = False
    getbondpathorderchange: bool = False
    getsharedfunctionalgroupchange: bool = False
    getconnectivitypathdifference: bool = False
    getreactivitydistance: bool = False
    getelectronflowcorrelation: bool = False


class GlobalReactionBondInformation(BaseReactionFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    def ShortestPathChange(self, edge):
        if self.params.getshortestpathchange:
            reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
            product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
            sp_R = nx.shortest_path_length(reactant_graph, source=edge[0], target=edge[1])
            sp_P = nx.shortest_path_length(product_graph, source=edge[0], target=edge[1])
            return [sp_P - sp_R]
        else:
            return []

    def CommonNeighborCountChange(self, edge):
        if self.params.getcommonneighborcountchange:
            neighbors_R = set(self.reactant.matrixdescriptors.adj_mat[edge[0]].nonzero()[0]).union(
                set(self.reactant.matrixdescriptors.adj_mat[edge[1]].nonzero()[0]))
            neighbors_P = set(self.product.matrixdescriptors.adj_mat[edge[0]].nonzero()[0]).union(
                set(self.product.matrixdescriptors.adj_mat[edge[1]].nonzero()[0]))
            common_neighbors_R = neighbors_R.intersection(neighbors_P)
            common_neighbors_P = neighbors_P.intersection(neighbors_R)
            return [len(common_neighbors_P) - len(common_neighbors_R)]
        else:
            return []

    def RandomWalkChange(self, edge):
        if self.params.getrandomwalkchange:
            rwct_R = nx.algorithms.approximation.rwct(self.reactant.matrixdescriptors.adj_mat, edge[0], edge[1])
            rwct_P = nx.algorithms.approximation.rwct(self.product.matrixdescriptors.adj_mat, edge[0], edge[1])
            return [rwct_P - rwct_R]
        else:
            return []

    def BondPathOrderChange(self, edge):
        if self.params.getbondpathorderchange:
            bo_R = self.reactant.matrixdescriptors.bond_mat[edge[0], edge[1]]
            bo_P = self.product.matrixdescriptors.bond_mat[edge[0], edge[1]]
            return [bo_P - bo_R]
        else:
            return []

    def SharedFunctionalGroupChange(self, edge):
        if self.params.getsharedfunctionalgroupchange:
            fg = FunctionalGroups()
            res_R = fg.are_atoms_in_same_functional_group(self.reactant.new_mol, edge[0], edge[1])
            res_P = fg.are_atoms_in_same_functional_group(self.product.new_mol, edge[0], edge[1])
            return [int(res_P) - int(res_R)]
        else:
            return []


    def ReactivityDistance(self, edge):
        if self.params.getreactivitydistance:
            reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
            product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
            rd_R = nx.shortest_path_length(reactant_graph, source=edge[0], target=edge[1])
            rd_P = nx.shortest_path_length(product_graph, source=edge[0], target=edge[1])
            return [rd_P - rd_R]
        else:
            return []

    def ElectronFlowCorrelation(self, edge):
        if self.params.getelectronflowcorrelation:
            reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
            product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
            efc_R = nx.degree_centrality(reactant_graph)[edge[0]] * nx.degree_centrality(reactant_graph)[edge[1]]
            efc_P = nx.degree_centrality(product_graph)[edge[0]] * nx.degree_centrality(product_graph)[edge[1]]
            return [efc_P - efc_R]
        else:
            return []