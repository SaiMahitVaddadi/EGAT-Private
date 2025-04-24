from ..base import BaseFeaturizer
from rdkit import Chem
from dataclasses import dataclass
import networkx as nx

@dataclass
class BondParams:
    """
    Parameters
    ----------
    removebondorderinfo : bool, optional
        A flag indicating whether to remove bond order information. 
        Default is False.
    """
    removebondorderinfo: bool = False


class BondInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()

    def EncodeBondOrder(self,edge):
        BO = self.matrixdescriptors.bond_mat[edge[0],edge[1]]
        return BO
    
    def EncodeBondOrderRDKit(self,edge):
        bond = self.new_mol.GetBondBetweenAtoms(edge[0], edge[1])
        if bond is not None:
            bond_type = bond.GetBondType()
            return bond_type
        else:
            return None


    def BondOrder(self,edge,bo):
        if not self.params.removebondorderinfo:
            if bo == 0:            
                return [0,0,0,0,1]
            else:
                if tuple(edge) in self.stereo.bond_aromatic and self.stereo.bond_aromatic[tuple(edge)]: return self.properties.bond_order_encode['BA']
                else: return self.properties.bond_order_encode['B{}'.format(int(bo))]
        else:
            return []
    
    def BondOrderFull(self,edge,bo):
        if not self.params.removebondorderinfo:
            if bo == 0:            
                return [0,0,0,0,0]
            else:
                if tuple(edge) in self.stereo.bond_aromatic and self.stereo.bond_aromatic[tuple(edge)] and bo == Chem.rdchem.BondType.AROMATIC : return self.properties.bond_order_encode_new['BA']
                elif self.stereo.bond_aromatic[tuple(edge)] or bo == Chem.rdchem.BondType.AROMATIC : return self.properties.bond_order_encode_new['BHA']
                elif bo == Chem.rdchem.BondType.SINGLE: return self.properties.bond_order_encode_new['B{}'.format(int(1))]
                elif bo == Chem.rdchem.BondType.DOUBLE: return self.properties.bond_order_encode_new['B{}'.format(int(2))]
                elif bo == Chem.rdchem.BondType.TRIPLE: return self.properties.bond_order_encode_new['B{}'.format(int(3))]
                elif bo == Chem.rdchem.BondType.ONEANDAHALF: return self.properties.bond_order_encode_new['B{}'.format(1.5)]
                elif bo == Chem.rdchem.BondType.DATIVE: return self.properties.bond_order_encode_new['DATIVE']
                elif bo == Chem.rdchem.BondType.DATIVEL: return self.properties.bond_order_encode_new['DATIVEL']
                elif bo == Chem.rdchem.BondType.DATIVEONE: return self.properties.bond_order_encode_new['DATIVEONE']
                elif bo == Chem.rdchem.BondType.DATIVER: return self.properties.bond_order_encode_new['DATIVER']
                elif bo == Chem.rdchem.BondType.FIVEANDAHALF: return self.properties.bond_order_encode_new['FIVEANDAHALF']
                elif bo == Chem.rdchem.BondType.FOURANDAHALF: return self.properties.bond_order_encode_new['FOURANDAHALF']
                elif bo == Chem.rdchem.BondType.HEXTUPLE: return self.properties.bond_order_encode_new['HEXTUPLE']
                elif bo == Chem.rdchem.BondType.HYDROGEN: return self.properties.bond_order_encode_new['HYDROGEN']
                elif bo == Chem.rdchem.BondType.IONIC: return self.properties.bond_order_encode_new['IONIC']
                elif bo == Chem.rdchem.BondType.ONEANDAHALF: return self.properties.bond_order_encode_new['ONEANDAHALF']
                elif bo == Chem.rdchem.BondType.OTHER: return self.properties.bond_order_encode_new['OTHER']
                elif bo == Chem.rdchem.BondType.QUADRUPLE: return self.properties.bond_order_encode_new['QUADRUPLE']
                elif bo == Chem.rdchem.BondType.QUINTUPLE: return self.properties.bond_order_encode_new['QUINTUPLE']
                elif bo == Chem.rdchem.BondType.THREEANDAHALF: return self.properties.bond_order_encode_new['THREEANDAHALF']
                elif bo == Chem.rdchem.BondType.THREECENTER: return self.properties.bond_order_encode_new['THREECENTER']
                elif bo == Chem.rdchem.BondType.TWOANDAHALF: return self.properties.bond_order_encode_new['TWOANDAHALF']
                elif bo == Chem.rdchem.BondType.UNSPECIFIED: return self.properties.bond_order_encode_new['UNSPECIFIED']
                elif bo == Chem.rdchem.BondType.ZERO: return self.properties.bond_order_encode_new['ZERO']
                




                if tuple(edge) in self.stereo.bond_aromatic and self.stereo.bond_aromatic[tuple(edge)]: return self.properties.bond_order_encode['BA']
                else: return self.properties.bond_order_encode['B{}'.format(int(bo))]
        else:
            return []
    
    
    