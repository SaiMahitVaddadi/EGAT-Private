from ..base.information import NeighborInformation,BondInformation,ElectronInformation,ChargeInformation,AcidBaseInformation,RDInformation,StereoInformation,RingInformation,BRICSInformation,FusedInformation
from dataclasses import dataclass
from typing import List


@dataclass
class MoleculeFeaturizerParams:
    removeelementinfo: bool = False

# Example usage:
# params = FeaturizerParams(removeelementinfo=True, element_encode=[1, 2, 3])


# For this, I need to set up YARP to obtain the mol files. This will only change the bond order. 

class MoleculeFeaturizer(NeighborInformation,BondInformation,ElectronInformation,ChargeInformation,AcidBaseInformation,StereoInformation,RingInformation,RDInformation,BRICSInformation,FusedInformation):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    
    def EncodeElement(self,ind):
        if not self.params.removeelementinfo:
            return self.properties.element_encode[self.matrixdescriptors.element[ind]]
        else:
            return []
        

    def AtomFeatureVector(self,ind,yarpid=0):
        atom_map_number = self.matrixdescriptors.new_mol.GetAtoms()[ind].GetAtomMapNum()
        if atom_map_number == 0 : atom_map_number = ind


        atom_feature = []
        atom_feature += self.EncodeElement(ind) # Works
        atom_feature += self.GetNeighbors(ind) # Works
        atom_feature += self.RingCheck(atom_map_number) # Works
        atom_feature += self.FormalCharge(ind) # Works
        atom_feature += self.AromaticityCheck(atom_map_number,ind) # Works
        atom_feature += self.HybridizationCheck(atom_map_number,ind) # Works
        atom_feature += self.ChiralityCheck(atom_map_number) # Works
        atom_feature += self.RadicalCheck(ind,yarpid) # Works
        atom_feature += self.SpiroCheck(ind) # Works
        atom_feature += self.BridgeHeadCheck(ind) # Works
        atom_feature += self.ElectronegativityCheck(ind) 
        atom_feature += self.ChargeCheck(ind)
        atom_feature += self.AcidBaseCheck(atom_map_number) # Works
        atom_feature += self.IsPartOfBRICSBond(ind)
        atom_feature += self.AtominFusedRing(ind) # Works
        atom_feature += self.AtominXRings(ind) # Works
        return atom_feature
    

    def GetAtomFeatureVectorDetails(self, ind,yarpid=0):
        details = {}
        atom_map_number = self.matrixdescriptors.new_mol.GetAtoms()[ind].GetAtomMapNum()
        if atom_map_number == 0:
            atom_map_number = ind

        details['element_encoding'] = self.EncodeElement(ind)
        details['neighbors'] = self.GetNeighbors(ind)
        details['ring_check'] = self.RingCheck(atom_map_number)
        details['formal_charge'] = self.FormalCharge(ind)
        details['aromaticity'] = self.AromaticityCheck(atom_map_number, ind)
        details['hybridization'] = self.HybridizationCheck(atom_map_number, ind)
        details['chirality'] = self.ChiralityCheck(atom_map_number)
        details['radical'] = self.RadicalCheck(ind, yarpid)
        details['spiro'] = self.SpiroCheck(ind)
        details['bridge_head'] = self.BridgeHeadCheck(ind)
        details['electronegativity'] = self.ElectronegativityCheck(ind)
        details['charge'] = self.ChargeCheck(ind)
        details['acid_base'] = self.AcidBaseCheck(atom_map_number)
        details['brics_bond'] = self.IsPartOfBRICSBond(ind)
        details['fused_ring'] = self.AtominFusedRing(ind)
        details['x_rings'] = self.AtominXRings(ind)

        return details

    def DipoleEncoder(self,edge):
        if self.params.getdipole:
            return [self.matrixdescriptors.bond_dipole_moments[edge[0],edge[1]]]
        else:
            return []

    def BondFeatureVector(self,ind):
        bond_feature = []
        edge = sorted([self.edges_u[ind],self.edges_v[ind]])
        bo = self.EncodeBondOrder(edge)
        bond_feature += self.BondinRing(edge) # Works
        bond_feature += self.BondOrder(edge,bo) # Works
        bond_feature += self.BondConjugation(edge,bo) # Works
        bond_feature += self.BondStereochemistry(edge,bo) # Works
        bond_feature += self.BondRotation(edge,bo)
        bond_feature += self.IsBRICSBond(edge)
        bond_feature += self.BondinFusedRing(edge) # Works
        bond_feature += self.DipoleEncoder(edge)
        return bond_feature
    
    def GetBondFeatureVectorDetails(self, ind):
        details = {}
        edge = sorted([self.edges_u[ind], self.edges_v[ind]])
        bo = self.EncodeBondOrder(edge)

        details['bond_in_ring'] = self.BondinRing(edge)
        details['bond_order'] = self.BondOrder(edge, bo)
        details['bond_conjugation'] = self.BondConjugation(edge, bo)
        details['bond_stereochemistry'] = self.BondStereochemistry(edge, bo)
        details['bond_rotation'] = self.BondRotation(edge, bo)
        details['brics_bond'] = self.IsBRICSBond(edge)
        details['bond_in_fused_ring'] = self.BondinFusedRing(edge)
        details['dipole'] = self.DipoleEncoder(edge)

        return details

    def GenerateAtomFeatureVector(self,yarpid=0):
        self.atom_features = []
        self.node_vector_length = 0 
        for ind in range(len(self.matrixdescriptors.element)):
            atom_feature = self.AtomFeatureVector(ind,yarpid)
            self.atom_features.append(atom_feature)
            self.node_vector_length = len(atom_feature)

    def GenerateBondFeatureVector(self):
        self.GenerateEdges()
        self.bond_features = []
        for ind in range(len(self.edges_u)):
            bond_feature = self.BondFeatureVector(ind)
            self.bond_features.append(bond_feature)
            self.bond_feature_length = len(bond_feature)
    
    def _runusingyarp(self):
        if len(self.electroninfo.yarpecule.bond_mats) == 1:
            self.GenerateAtomFeatureVector()
            self.GenerateBondFeatureVector()
        else:
            self.atom_features_dict = {}
            self.bond_features_dict = {}
            for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                self.GenerateAtomFeatureVector(i)
                self.GenerateBondFeatureVector()
                self.atom_features_dict[i] = self.atom_features
                self.bond_features_dict[i] = self.bond_features

    def run(self):
        self.ObtainFusedRingInfo()
        if self.params.getradical == 'YARP':
            self._runusingyarp()
        else:
            self.GenerateAtomFeatureVector()
            self.GenerateBondFeatureVector()

