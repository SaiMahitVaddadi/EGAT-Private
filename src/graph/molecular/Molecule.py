from ..base.information import NeighborInformation,BondInformation,ElectronInformation,ChargeInformation,AcidBaseInformation,RDInformation,StereoInformation,RingInformation,BRICSInformation,FusedInformation,InteractionInformation,LocantInformation,MordredInformation
from ...utils.descriptors.egat.encodings import Encodings
from ...utils.descriptors.egat.properties import Properties
from ...utils.descriptors.egat.radicals import YARPElectronInfo
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from icecream import ic
@dataclass
class MoleculeFeaturizerParams:
    removeelementinfo: bool = False

# Example usage:
# params = FeaturizerParams(removeelementinfo=True, element_encode=[1, 2, 3])


# For this, I need to set up YARP to obtain the mol files. This will only change the bond order. 

class MoleculeFeaturizer(NeighborInformation,BondInformation,ElectronInformation,ChargeInformation,AcidBaseInformation,StereoInformation,
                         RingInformation,RDInformation,BRICSInformation,FusedInformation,InteractionInformation,LocantInformation,
                         MordredInformation):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.smiles = smiles
        self.params = arguments
        self.SetupStep()
    
    def SetupStep(self):
        self.properties = Encodings()
        self.props = Properties()
        self.CreateAtomMapping()
        self.MatrixDescriptors()
        self.Rings()
        self.InitializeAddons()
        self.Eneg()
        self.Stereochem()
        self.Radicals()
        self.ObtainFusedRingInfo()
        self.RunKallistoBases()
        self.RunMordredBases()
        self.LoadYARP()
    
    def EncodeElement(self,ind):
        if not self.params.removeelementinfo:
            return self.properties.element_encode[self.matrixdescriptors.element[ind]]
        else:
            return []
        
    def _runfunc(self,func,ind,atom_map_number,yarpid):
        if func in self.reqs_amap:
            return func(atom_map_number)
        elif func in self.reqs_indplusmapping:
            return func(atom_map_number,ind)
        elif func == self.reqs_yarpid:
            return func(ind,yarpid)
        elif func in self.reqs_nothing:
            return func()
        else:
            return func(ind)
        
    def _iterateatomfeatures(self, features, vec, ind, atom_map_number, yarpid):
        for func in tqdm(features, desc="Processing atom features"):
            try:
                vec += self._runfunc(func, ind, atom_map_number, yarpid)
                debug = False
                if debug: print(f"Function {func.__name__} executed successfully for atom index {ind}, with result: {self._runfunc(func, ind, atom_map_number, yarpid)}")
            except Exception as e:
                print('==========================ERROR ALERT==========================')
                print(f"Error in function {func.__name__} for atom index {ind}: {e}")
                print('===============================================================')
                import traceback
                print('==========================TRACEBACK============================')
                print('===============================================================')
                traceback.print_exc()
                print('===============================================================')
                print('===============================================================')
        return vec
    
    def LoadYARP(self):
        self.yarpinfo = YARPElectronInfo(self.matrixdescriptors,canon=True,mapping=True)
        self.atom_resonance,self.bond_resonance = self.yarpinfo.GetResonanceInfo()
        
    def AtomResonance(self,ind):
        if self.params.getresonance:
            return self.atom_resonance[ind]
        else:
            return []
        
    def AtomFeatureVector(self,ind,yarpid=0):
        atom_map_number = self.matrixdescriptors.new_mol.GetAtoms()[ind].GetAtomMapNum()
        if atom_map_number == 0 : atom_map_number = ind

        self.reqs_amap = [self.RingCheck,self.ChiralityCheck,self.AcidBaseCheck]
        self.reqs_indplusmapping = [self.AromaticityCheck,self.HybridizationCheck]
        self.reqs_yarpid = self.RadicalCheck
        self.RunAtomVector(ind)
        self.reqs_nothing = [self.BaryszAtom,self.ATSAtom,self.PropsRelativetoCarbon,self.VertexDistanceDegreeAtom,self.EtaPsi,self.HDSA,
                             self.VertexAdjacency]


        self.atomfuncs = [
            self.EncodeElement,
            self.GetNeighbors,
            self.RingCheck,
            self.FormalCharge,
            self.AromaticityCheck,
            self.HybridizationCheck,
            self.ChiralityCheck,
            self.RadicalCheck,
            self.SpiroCheck,
            self.BridgeHeadCheck,
            self.ElectronegativityCheck,
            self.ChargeCheck,
            self.AcidBaseCheck,
            self.IsPartOfBRICSBond,
            self.AtominFusedRing,
            self.AtominXRings,
            self.PiElectrons,
            self.SigmaElectrons,
            self.CoreElectrons,
            self.RamificationNumber,
            self.IonizationPotential,
            self.SurroundingIPFeatures,
            self.IntrinsicState,
            self.EtaBeta,
            self.LocantCountForAtom,
            self.DeltaInteraction,
            self.FreeEnergies,
            self.GetAP,
            self.GetAPC,
            self.GetCN,
            self.GetPS,
            self.GetVdW,
            self.BaryszAtom,
            self.ATSAtom,
            self.PropsRelativetoCarbon,
            self.VertexDistanceDegreeAtom,
            self.SuperdenticIndex,
            self.Eccentricity,
            self.Schultz,
            self.Xui,
            self.CoreCount,
            self.VEMAtom,
            self.EtaPsi,
            self.ZagrebAtom,
            self.HarmonicAtom,
            self.SomborAtom,
            self.RandicAtom,
            self.NirmalaAtom,
            self.ESOSAtom,
            self.AugmentedGraphAttributeAtom,
            self.HyperbolicAtom,
            self.AugZagrebAtom,
            self.KleinAtom,
            self.HyperDegreeAtom,
            self.VEWIAtom,
            self.VEWIAtomByOrder,
            self.HDSA,
            self.ETSAtom,
            self.InformationContent,
            self.WeightedInformationContent,
            self.VertexAdjacency,
            self.TPSA,
            self.LabuteASA,
            self.LogS,
            self.EState,
            self.TopoChargeAtom,
            self.SMRandSLogP,
            self.ChiAtom,
            self.ChiAtomValence,
            self.AtomResonance
        ]

        atom_feature = []
        atom_feature = self._iterateatomfeatures(self.atomfuncs,atom_feature,ind,atom_map_number,yarpid)
        atom_feature = self._checkforbadvalues(atom_feature)
        return atom_feature
    
    def DipoleEncoder(self,edge):
        if self.params.getdipole:
            return [self.matrixdescriptors.bond_dipole_moments[edge[0],edge[1]]]
        else:
            return []

    def PolarityEncoder(self,edge):
        if self.params.getpolarity:
            return [self.matrixdescriptors.polarity_pauling[edge[0],edge[1]]]
        else:
            return []
        
    def _runbondfunc(self,func,edge,bo):
        if func in self.reqs_bo:
            return func(edge,bo)
        elif func in self.reqs_nothing_bond:
            return func()
        elif func == self.reqs_bo_only:
            return func(bo)
        else:
            return func(edge)
            
    def _iteratebondfeatures(self,features,vec,edge,bo):
        for func in tqdm(features, desc="Processing bond features"):
            try:
                vec += self._runbondfunc(func, edge, bo)
            except Exception as e:
                print('==========================ERROR ALERT==========================')
                print(f"Error in function {func.__name__} for atom index {edge}: {e}")
                print('===============================================================')
                import traceback
                print('==========================TRACEBACK============================')
                print('===============================================================')
                traceback.print_exc()
                print('===============================================================')
                print('===============================================================')
        return vec
    
    def BondResonance(self,edge):  
        if self.params.getresonance:
            return self.bond_resonance[tuple(edge)]
        else:
            return []

    def _checkforbadvalues(self,vector):
        for i,x in enumerate(vector):
            if x is None or x == np.nan:
                vector[i] = 0
            elif x == float('inf') or x == np.inf:
                vector[i] = 0
        return vector

    def BondFeatureVector(self,ind,yarpid=0):
        bond_feature = []
        edge = sorted([self.edges_u[ind],self.edges_v[ind]])
        bo = self.EncodeBondOrder(edge,yarpid=yarpid)
        self.RunBondVector(edge)
        self.reqs_bo = [self.BondOrder,self.BondConjugation,self.BondStereochemistry,self.BondRotation]
        self.reqs_bo_only = self.BaryszBond
        self.reqs_nothing_bond = [self.ABCIndex,self.ATSBond,self.VertexDistanceDegreeBond,self.BalabanBondFactor,self.EdgeWiener]
        self.bondfuncs = [
            self.BondinRing,
            self.BondOrder,
            self.BondConjugation,
            self.BondStereochemistry,
            self.BondRotation,
            self.IsBRICSBond,
            self.BondinFusedRing,
            self.DipoleEncoder,
            self.PolarityEncoder,
            self.EtaBond,
            self.ABCIndex,
            self.BaryszBond,
            self.ATSBond,
            self.DetourMatrix,
            self.BurdenValue,
            self.VertexDistanceDegreeBond,
            self.BalabanBondFactor,
            self.Gutman,
            self.Horary,
            self.Mohar,
            self.HPValue,
            self.VEMBond,
            self.EtaComposite,
            self.Gravity,
            self.ZagrebBond,
            self.Harmonic,
            self.Sombor,
            self.Randic,
            self.Nirmala,
            self.ESOS,
            self.AugmentedGraphAttribute,
            self.Hyperbolic,
            self.AugZagreb,
            self.Klein,
            self.HyperDegree,
            self.HyperWiener,
            self.Wiener,
            self.ETSBond,
            self.EdgeWiener,
            self.EdgeWienerByOrder,
            self.VEWIBond,
            self.VEWIBondByOrder,
            self.MolecularDistanceEdge,
            self.TopoChargeBond,
            self.ChiBond,
            self.ChiBondValence,
            self.BondResonance]
        bond_feature = self._iteratebondfeatures(self.bondfuncs,bond_feature,edge,bo)
        bond_feature = self._checkforbadvalues(bond_feature)
        return bond_feature
    
    def GenerateAtomFeatureVector(self,yarpid=0):
        self.atom_features = []
        self.node_vector_length = 0 
        for ind in range(len(self.matrixdescriptors.element)):
            atom_feature = self.AtomFeatureVector(ind,yarpid)
            self.atom_features.append(atom_feature)
            self.node_vector_length = len(atom_feature)

    def GenerateBondFeatureVector(self,yarpid=0):
        self.GenerateEdges()
        self.bond_features = []
        for ind in range(len(self.edges_u)):
            bond_feature = self.BondFeatureVector(ind,yarpid)
            self.bond_features.append(bond_feature)
            self.bond_feature_length = len(bond_feature)
    
    def _runusingyarp(self):
        if len(self.electroninfo.yarpecule.bond_mats) == 1:
            self.GenerateAtomFeatureVector()
            self.GenerateBondFeatureVector()
            self.bond_mats = 1
        else:
            self.atom_features_dict = {}
            self.bond_features_dict = {}
            for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                self.GenerateAtomFeatureVector(i)
                self.GenerateBondFeatureVector(i)
                self.atom_features_dict[i] = self.atom_features
                self.bond_features_dict[i] = self.bond_features
            self.bond_mats = len(self.electroninfo.yarpecule.bond_mats)

    def run(self):
        if self.params.getradical == 'YARP':
            self._runusingyarp()
        else:
            self.GenerateAtomFeatureVector()
            self.GenerateBondFeatureVector()
            self.bond_mats = 1

