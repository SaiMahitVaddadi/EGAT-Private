from ..utils.descriptors.egat.encodings import Encodings
from ..utils.descriptors.egat.reactive import Reactive
from ..utils.descriptors.egat.molmatdesc import MolMatDesc
from ..utils.descriptors.egat.radicals import RDKElectronInfo,YARPElectronInfo
from ..utils.descriptors.egat.stereo import StereoChemistry
from ..utils.matrices.graph_seps import graph_seps
from ..utils.misc.taffi_functions import return_rings,adjmat_to_adjlist

from rdkit import Chem
from dataclasses import dataclass
from typing import Literal



class MoleculeFeaturizer:
    def __init__(self, smiles, arguments):
        self.smiles = smiles
        self.params = arguments
        self.properties = Encodings()
        self.CreateAtomMapping()
        self.MatrixDescriptors()
        self.Rings()
    
    def CreateAtomMapping(self):
        molecule = Chem.MolFromSmiles(self.smiles)
        molecule = Chem.AddHs(molecule)
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        self.am_smiles = Chem.MolToSmiles(molecule)
    
    def MatrixDescriptors(self):
        self.matrixdescriptors = MolMatDesc(self.am_smiles)
        self.matrixdescriptors.run()
        
    def DistanceMatrix(self):
        self.gs = graph_seps(self.matrixdescriptors.adj_mat)
        self.gs[self.gs < 0] = 100

    def Rings(self):
        self.ring_atoms = return_rings(adjmat_to_adjlist(self.adj),max_size=20,remove_fused=True)
    
    def Radicals(self):
        if self.params.getradical == 'RDKit':
            self.electroninfo = RDKElectronInfo(self.matrixdescriptors)
        elif self.params.getradical == 'YARP':
            self.electroninfo = YARPElectronInfo(self.matrixdescriptors)

    def Rotatability(self):
        self.matrixdescriptors.BridgeHead()
        self.matrixdescriptors.Spiro()
        self.matrixdescriptors.RotatableBondCount()
    
    def Polarity(self):
        self.matrixdescriptors.BondPolarityPauling()
        self.matrixdescriptors.Electronegativity()
    
    def Charges(self):
        self.matrixdescriptors.Gasteiger()
        self.matrixdescriptors.BondDipoleMoments()

    def Stereochem(self):
        self.stereo = StereoChemistry(self.matrixdescriptors)
        self.stereo.run()


    def CheckMapping(self,ind):
        atom_mappings = min([atom.GetAtomMapNum() for atom in self.new_mol.GetAtoms()])
        #print(f"ind: {ind}, Rsmiles: {Rsmiles}, molecule: {molecule}, Generated Atom Mapping: {atom_mappings}\n")
        if atom_mappings == 0:
            ind_in_mol = ind
        else:
            ind_in_mol = ind+1
        return ind

    def EncodeElement(self,ind):
        if not self.params.removeelementinfo:
            return self.properties.element_encode[self.matrixdescriptors.element[ind]]
        else:
            return []

    def GetNeighbors(self,ind):
        neighbors = [self.matrixdescriptors.element[counti] for counti,i in enumerate(self.matrixdescriptors.adj_mat[ind,:]) if i != 0]
        if self.params.neighbor == 'onlyH':
            NE_count = [neighbors.count('H')]
        elif self.params.neighbor == 'onlyCHNO':
            NE_count = [neighbors.count('H'), neighbors.count('C'), neighbors.count('N'), neighbors.count('O')]
        elif self.params.neighbor == 'onlyorganic':
            NE_count = [neighbors.count('H'), neighbors.count('C'), neighbors.count('N'), neighbors.count('O'), 
                        neighbors.count('P'), neighbors.count('S'), neighbors.count('F'), neighbors.count('Cl'), 
                        neighbors.count('Br'), neighbors.count('I')]
        elif self.params.neighbor == 'all':
            NE_count = [neighbors.count(element) for element in self.properties.element_encode.keys()]
        else:
            NE_count = []
        return NE_count

    def DistanceFromReactingAtom(self,ind):
        if not self.params.removereactiveinfo:
            if len(self.reactive_atoms) > 0:
                dis = min([self.gs[ind][indr] for indr in self.reactive_atoms])
            else:
                dis = 0 
            return [dis]
        else:
            return []

    def RingCheck(self,ind):
        if not self.params.removeringinfo:
            if True in [ind in ra_list for ra_list in self.ring_atoms]: inR = 1
            else: inR = 0
            return [inR]
        else:
            return []
    
    def FormalCharge(self,ind):
        if not self.params.removeformalchargeinfo:
            fc = self.matrixdescriptor.fc[ind]
            return fc
        else:
            return []

    def AromaticityCheck(self,ind):
        if not self.args.removearomaticity:
            ###### GET AROMATICITY
            if self.self.matrixdescriptors.element[ind] == 'H': aromaticity = 0
            elif self.stereo.atom_aromatic[ind]: aromaticity = 1
            else: aromaticity = 0
            return [aromaticity]
        else:
            return []
       
    def HybridizationCheck(self,ind):
        ###### GET HYBRIDIZATION
        if not self.args.removehybridinfo:
            if self.self.matrixdescriptors.element[ind] == 'H':
                if not self.params.useFullHyb: hybrid = [0,0,0,1]
                else: hybrid = [0,0,0,1,0,0,0,0,0]

            else:
                hybrid = self.stereo.Hybridization[ind] 
            return hybrid
        else:
            return []
    
    def ChiralityCheck(self,ind):
        if not self.args.removechiralinfo:
            ###### CHECK IF IT IS IN A CHIRAL CENTER
            if ind in self.stereo.chiral_centers: chiral = self.properties.atom_chiral_encode[self.stereo.chiral_centers[ind]]
            else: chiral = [0,0,1]
            return [chiral]
        else:
            return []
    
    def RadicalCheck(self,ind):
        if self.args.getradical:
            return [self.electroninfo.rads[ind],self.electroninfo.lps[ind]]
        else:
            return []

    def SpiroCheck(self,ind):
        if self.args.getspiro:
            return [self.matrixdescriptors.spiro[ind]]
        else:
            return []

    def BridgeHeadCheck(self,ind):
        if self.args.getbridgehead:
            return [self.matrixdescriptors.bridgehead[ind]]
        else:
            return []

    def ElectronegativityCheck(self,ind):
        if self.args.getelectronegativity:
            return [self.matrixdescriptors.electronegativity[ind]]
        else:
            return []

    def ChargeCheck(self,ind):
        if self.args.charge == 'Gasteiger':
            return [self.matrixdescriptors.gasteiger_charges[ind]]
        elif self.args.charge == 'psi4':
            return []
        elif self.args.charge == 'pyscf':
            return []
        elif self.args.charge == 'gaussian':
            return []
        elif self.args.charge == 'orca':
            return []
        elif self.args.charge == 'ase':
            return []
        elif self.args.charge == 'gemnet':
            return []
        elif self.args.charge == 'chemprop':
            return []
        elif self.args.charge == 'unimol':
            return []
        else:
            return []

    def GenerateAtomFeatureVector(self):
        self.atom_features = []
        self.node_vector_length = 0 
        for ind in range(len(self.matrixdescriptors.element)):
            atom_feature = []
            atom_feature += self.EncodeElement(ind)
            atom_feature += self.GetNeighbors(ind)
            atom_feature += self.RingCheck(ind)
            atom_feature += self.FormalCharge(ind)
            atom_feature += self.AromaticityCheck(ind)
            atom_feature += self.HybridizationCheck(ind)
            atom_feature += self.ChiralityCheck(ind)
            atom_feature += self.RadicalCheck(ind)
            atom_feature += self.SpiroCheck(ind)
            atom_feature += self.BridgeHeadCheck(ind)
            atom_feature += self.ElectronegativityCheck(ind)
            atom_feature += self.ChargeCheck(ind)
            atom_feature += self.AcidBaseCheck(ind)
            self.atom_features.append(atom_feature)
            self.node_vector_length = len(atom_feature)


    def InitializeAddons(self):
        self.Radicals()
        self.Rotatability()
        self.Charges()
        self.Polarity()
        self.Stereochem()
        if self.params.acidbase == 'Lewis':
            self.AcidBaseSitesLewis()
        elif self.params.acidbase == 'BL':
            self.AcidBaseSitesBO()
    
    def GenerateEdges(self):
        ###### GENERATE THE RP-ADJACENCY MATRIX SO THAT AT LEAST ONE SIDE IS CONNECTED 
        self.edges_u,self.edges_v  = [],[]
        for i in range(len(self.matrixdescriptors.element)):
            for j in range(len(self.matrixdescriptors.element)):
                # if reaction, also check if P_adj > 0
                if self.matrixdescriptors.adj_mat[i][j] > 0:
                    self.edges_u.append(i)
                    self.edges_v.append(j)
        
    
    def EncodeBondOrder(self,edge):
        BO = self.matrixdescriptors.bond_mat[edge[0],edge[1]]
        return BO


    def BondinRing(self,edge):
        if not self.params.removeringinfo:
            if True in [(edge[0] in ra_list and edge[1] in ra_list) for ra_list in self.ring_atoms]: inR = 1
            else: inR = 0
            return [inR]
        else:
            return []
    
    def BondOrder(self,edge,bo):
        if not self.params.removebondorderinfo:
            if bo == 0:            
                return [0,0,0,0,1]
            else:
                if tuple(edge) in self.stereo.bond_aromatic and self.stereo.bond_aromatic[tuple(edge)]: return self.properties.bond_order_encode['BA']
                else: return self.properties.bond_order_encode['B{}'.format(int(bo))]
        else:
            return []

    def BondConjugation(self,edge,bo):
        # Add Bond Conjugation info if not stated that you want to remove it from training. 
        if not self.params.removeconjinfo:
            if bo == 0:
                return [0]
            else:
                if tuple(edge) in self.stereo.conjugation and self.stereo.conjugation[tuple(edge)]: return [1]
                else: return [0]
        else:
            return []
    
    def BondStereochemistry(self,edge,bo):
        # Add Bond Conjugation info if not stated that you want to remove it from training. 
        if not self.params.removestereoinfo:
            if bo == 0:
                return [0,0,0]
            else:
                if tuple(edge) in self.stereo.bond_stereo and self.stereo.bond_stereo[tuple(edge)]: return self.properties.bond_stereo_encode[self.stereo.bond_stereo[tuple(edge)]]
                else: return [0,0,0]
        else:
            return []
    
    def BondRotation(self,edge,bo):
        # Add Bond Conjugation info if not stated that you want to remove it from training. 
        if self.params.getrotatablebonds:
            if bo == 0:
                return [0,0]
            else:
                if tuple(edge) in self.matrixdescriptors.rotationalbond:
                    return self.properties.bond_rotat_encode['TRUE']
                else:
                    return self.properties.bond_rotat_encode['FALSE']
        else:
            return []
        
    def AcidBaseSitesBO(self):
        self.acid_sites = []
        self.base_sites = []

        for atom in self.new_mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            idx = atom.GetIdx()

            # Check for acidic hydrogens (O, S, N with hydrogen attached)
            if atomic_num in [8, 16, 7]:  # Oxygen, Sulfur, Nitrogen
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:  # Hydrogen
                        self.acid_sites.append(idx)

            # Check for basic lone pairs (N, O, S that can accept H+)
            if atomic_num in [7, 8, 16]:  # Nitrogen, Oxygen, Sulfur
                if atom.GetTotalDegree() < atom.GetExplicitValence():
                    self.base_sites.append(idx)


    def AcidBaseSitesLewis(self):
        self.acid_sites = []
        self.base_sites = []

        for atom in self.new_mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            idx = atom.GetIdx()

            # Lewis Acid (electron pair acceptor)
            if atom.GetFormalCharge() > 0:  # Positively charged atoms
                self.acid_sites.append(idx)
            elif atomic_num == 6 and atom.GetDegree() == 3:  # Carbocations
                self.acid_sites.append(idx)
            elif atomic_num == 5 and atom.GetDegree() == 3:  # Boron (BF3-type acids)
                self.acid_sites.append(idx)
            elif atomic_num == 6 and any(bond.GetBondType() == rdchem.BondType.DOUBLE for bond in atom.GetBonds()):
                # Carbonyl carbon (C=O) can act as a Lewis acid
                self.acid_sites.append(idx)

            # Lewis Base (electron pair donor)
            if atomic_num in [7, 8, 16, 17]:  # N, O, S, Halides
                if atom.GetTotalDegree() < atom.GetExplicitValence():  # Available lone pairs
                    self.base_sites.append(idx)

    def AcidBaseCheck(self, ind):
        if self.params.getacidbaseinfo:
            if ind in self.acid_sites:
                return [1, 0]
            elif ind in self.base_sites:
                return [0, 1]
            else:
                return [0, 0]
        else:
            return []


    def GenerateBondFeatureVector(self):
        self.GenerateEdges()
        self.bond_features = []
        for ind in range(len(self.edges_u)):
            bond_feature = []
            edge = sorted([self.edges_u[ind],self.edges_v[ind]])
            bo = self.EncodeBondOrder()
            bond_feature += self.BondinRing(edge)
            bond_feature += self.BondOrder(edge,bo)
            bond_feature += self.BondConjugation(edge,bo)
            bond_feature += self.BondStereochemistry(edge,bo)
            bond_feature += self.BondRotation(edge,bo)
            self.bond_features.append(bond_feature)
            self.bond_feature_length = len(bond_feature)
    
    def run(self):
        self.InitializeAddons()
        self.GenerateAtomFeatureVector()
        self.GenerateBondFeatureVector()

