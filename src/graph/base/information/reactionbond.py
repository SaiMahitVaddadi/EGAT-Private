from ..reaction import BaseReactionFeaturizer
from rdkit import Chem
import networkx as nx
from rdkit.Chem import BRICS
from dataclasses import dataclass

@dataclass
class ReactiveBondParams:
    removebondchangeinfo: bool = False
    oldbondencode: bool = False
    adddisttoreactingbonds: bool = False
    getbricsbondrolechange: bool = False

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
    
    def BondChangeInformation(self,edge,prod=False):
        if not self.params.removebondchangeinfo:
            if self.params.oldbondencode: RBtype,PBtype = self.OldBondChangeInfo(edge)
            else: RBtype,PBtype = self.BondChangeInfo(edge)
            if not prod: return RBtype
            else: return PBtype

    def DistanceFromReactingBond(self,edge,prod=False):
        if self.params.adddisttoreactingbonds:
            if len(self.reactive_atoms) > 0:
                if not prod:
                    dis = min([self.Rgs[edge[0]][indr]  + self.Rgs[edge[1]][indr] for indr in self.reactive_atoms])
                else:
                    dis = min([self.Pgs[edge[0]][indr]  + self.Pgs[edge[1]][indr] for indr in self.reactive_atoms])
            else:
                dis = 0 
            return [dis]
        else:
            return []
                    
    
    def BondOrderChange(self, edge,prod=False):
        BO_R = self.reactant.matrixdescriptors.bond_mat[edge[0], edge[1]]
        BO_P = self.product.matrixdescriptors.bond_mat[edge[0], edge[1]]
        if not prod: return [BO_P - BO_R]
        else: return [BO_R - BO_P]

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

    def ShortestPathChangeAcrossBond(self, edge,prod=False):
        sp_R = nx.shortest_path_length(self.reactant_graph, source=edge[0], target=edge[1])
        sp_P = nx.shortest_path_length(self.product_graph, source=edge[0], target=edge[1])
        if not prod: return [sp_P - sp_R]
        else: return [sp_R - sp_P]

    def BondParticipationDegree(self, edge,prod=False):
        degree_R = self.reactant_graph.degree(edge[0]) + self.reactant_graph.degree(edge[1])
        degree_P = self.product_graph.degree(edge[0]) + self.product_graph.degree(edge[1])
        if not prod: return [degree_P - degree_R]
        else: return [degree_R - degree_P]

    def BRICSBondRoleChange(self, edge,prod=False):
        if self.params.getbricsbondrolechange:
            role_R = (self.reactant.brics[edge[0]], self.reactant.brics[edge[1]]) if edge[0] in self.reactant.brics and edge[1] in self.reactant.brics else None
            role_P = (self.product.brics[edge[0]], self.product.brics[edge[1]]) if edge[0] in self.product.brics and edge[1] in self.product.brics else None
            if role_R == role_P:
                return [0]
            else:
                return [1]
        else:
            return []

    
