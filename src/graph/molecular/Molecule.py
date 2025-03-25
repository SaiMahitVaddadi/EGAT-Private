from ..base.base import BaseFeaturizer
from ..base.information import NeighborInformation,BondInformation,ElectronInformation,ChargeInformation,AcidBaseInformation,RDInformation,StereoInformation,RingInformation,BRICSInformation
from dataclasses import dataclass
from typing import List


@dataclass
class MoleculeFeaturizerParams:
    removeelementinfo: bool = False

# Example usage:
# params = FeaturizerParams(removeelementinfo=True, element_encode=[1, 2, 3])


class MoleculeFeaturizer(NeighborInformation,BondInformation,ElectronInformation,ChargeInformation,AcidBaseInformation,StereoInformation,RingInformation,RDInformation,BRICSInformation):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    
    def EncodeElement(self,ind):
        if not self.params.removeelementinfo:
            return self.properties.element_encode[self.matrixdescriptors.element[ind]]
        else:
            return []
        

    def AtomFeatureVector(self,ind,debug=False):
        atom_map_number = self.matrixdescriptors.new_mol.GetAtoms()[ind].GetAtomMapNum()
        if atom_map_number == 0 : atom_map_number = ind


        atom_feature = []
        atom_feature += self.EncodeElement(ind)
        atom_feature += self.GetNeighbors(ind)
        atom_feature += self.RingCheck(atom_map_number)
        atom_feature += self.FormalCharge(ind)

        atom_feature += self.AromaticityCheck(atom_map_number,ind)
        atom_feature += self.HybridizationCheck(atom_map_number,ind)
        atom_feature += self.ChiralityCheck(atom_map_number)
        atom_feature += self.RadicalCheck(atom_map_number)
        atom_feature += self.SpiroCheck(atom_map_number)
        atom_feature += self.BridgeHeadCheck(atom_map_number)
        atom_feature += self.ElectronegativityCheck(atom_map_number)
        atom_feature += self.ChargeCheck(ind)
        atom_feature += self.AcidBaseCheck(atom_map_number)
        atom_feature += self.IsPartOfBRICSBond(ind)


        if debug and ind == 0:
            print("Encoded Element:", self.EncodeElement(ind))
            print("Neighbors:", self.GetNeighbors(ind))
            print("Ring Check:", self.RingCheck(atom_map_number))
            print("Formal Charge:", self.FormalCharge(ind))

            print("Aromaticity Check:", self.AromaticityCheck(atom_map_number, ind))
            print("Hybridization Check:", self.HybridizationCheck(atom_map_number, ind))
            print("Chirality Check:", self.ChiralityCheck(atom_map_number))
            print("Radical Check:", self.RadicalCheck(atom_map_number))
            print("Spiro Check:", self.SpiroCheck(atom_map_number))
            print("Bridge Head Check:", self.BridgeHeadCheck(atom_map_number))
            print("Electronegativity Check:", self.ElectronegativityCheck(atom_map_number))
            print("Charge Check:", self.ChargeCheck(ind))
            print("Acid Base Check:", self.AcidBaseCheck(atom_map_number))


        return atom_feature

    def BondFeatureVector(self,ind):
        bond_feature = []
        edge = sorted([self.edges_u[ind],self.edges_v[ind]])
        bo = self.EncodeBondOrder(edge)
        bond_feature += self.BondinRing(edge)
        bond_feature += self.BondOrder(edge,bo)
        bond_feature += self.BondConjugation(edge,bo)
        bond_feature += self.BondStereochemistry(edge,bo)
        bond_feature += self.BondRotation(edge,bo)
        bond_feature += self.IsBRICSBond(edge)
        return bond_feature

    def GenerateAtomFeatureVector(self):
        self.atom_features = []
        self.node_vector_length = 0 
        for ind in range(len(self.matrixdescriptors.element)):
            atom_feature = self.AtomFeatureVector(ind)
            self.atom_features.append(atom_feature)
            self.node_vector_length = len(atom_feature)

    def GenerateBondFeatureVector(self):
        self.GenerateEdges()
        self.bond_features = []
        for ind in range(len(self.edges_u)):
            bond_feature = self.BondFeatureVector(ind)
            self.bond_features.append(bond_feature)
            self.bond_feature_length = len(bond_feature)
    
    def run(self):
        self.GenerateAtomFeatureVector()
        self.GenerateBondFeatureVector()

