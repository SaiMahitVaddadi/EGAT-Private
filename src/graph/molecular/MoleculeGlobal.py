from .MoleculeGeometry import MoleculeFeaturizerwithGeometry
from ..base.information import GlobalBondInformation,NonBondedInformation,HydrogenBondInformation
from dataclasses import dataclass
from typing import Literal



@dataclass
class MoleculeGlobalParams:
    addcho: bool = False
    adddihydrogenbonds: bool = False
    addcationpi: bool = False
    addpipistack: bool = False
    addhalogenbonds: bool = False
    addmetallophilic: bool = False
    addelectrostatic: bool = False
    addeneg: bool = False
    addmissingbonds: Literal['none', 'global', 'hbonds', 'all'] = 'none'

class MoleculeFeaturizerwithPadding(MoleculeFeaturizerwithGeometry,NonBondedInformation,GlobalBondInformation,HydrogenBondInformation):
    def __init__(self, smiles, arguments):
        super(MoleculeFeaturizerwithGeometry).__init__(smiles, arguments)
        
    def FindNBI(self):
        self.FindHBonds()
        self.FindCH_OInteractions()
        self.FindDihydrogenBonds()
        self.FindCationPiInteractions()
        self.FindHalogenBonds()
        self.FindElectrostaticInteractions()
        self.FindMetallophilicInteractions()
        self.FindPiPiStacking()

    def GetEdge(self,ind):
        edge = sorted([self.edges_u[ind],self.edges_v[ind]])
        return edge
    
    def BondFeatureVector(self, ind):
        edge = self.GetEdge(ind)
        bond_feature = super().BondFeatureVector(ind)
        bond_feature += self.DefaultNBInteraction(ind)
        return bond_feature,edge
    
    def BondFeatureVectorFromEdge(self,edge):
        bond_feature = []
        bo = self.EncodeBondOrder()
        bond_feature += self.BondinRing(edge)
        bond_feature += self.BondOrder(edge,bo)
        bond_feature += self.BondConjugation(edge,bo)
        bond_feature += self.BondStereochemistry(edge,bo)
        bond_feature += self.BondRotation(edge,bo)
        return bond_feature

    def GlobalBondFeatures(self,edge):
        bond_feature = []
        bond_feature += self.ShortestPathDistance(edge)
        bond_feature += self.ShortestPathDistanceWithWeights(edge)
        bond_feature += self.ShortestPathDistanceWithPBC(edge)
        bond_feature += self.RandomWalkCommuteTime(edge)
        bond_feature += self.CommuteTimeMetrics(edge)
        bond_feature += self.ShortestPathCount(edge)
        bond_feature += self.EffectiveResistance(edge)
        bond_feature += self.CommonNeighbors(edge)
        bond_feature += self.PercentCommonNeighbors(edge)
        bond_feature += self.JaccardIndex(edge)
        bond_feature += self.AdamicAdarIndex(edge)
        bond_feature += self.PreferentialAttachmentIndex(edge)
        bond_feature += self.KatzCentralitySimilarity(edge)
        bond_feature += self.EigenvectorCentralityDifference(edge)
        bond_feature += self.BetweennessCentralityCorrelation(edge)
        bond_feature += self.MinimumCutValue(edge)
        bond_feature += self.MaximumFlow(edge)
        bond_feature += self.LaplacianEigenvectorSimilarity(edge)
        bond_feature += self.FiedlerVectorSimilarity(edge)
        bond_feature += self.GraphDistanceWeightedByBondOrder(edge)
        bond_feature += self.BetweennessCentralityOfPathways(edge)
        bond_feature += self.RingsInSharedPath(edge)
        bond_feature += self.LocalAtomicEnvironmentSimilarity(edge)
        return bond_feature

    def NonBondedAtomFeatureVector(self, ind):
        atom_feature = self.AtomFeatureVector(ind)
        atom_feature += self.HydrogenBondCheck(ind)
        return atom_feature
    
    def GenerateAtomFeatureVector(self):
        self.atom_features = []
        self.node_vector_length = 0 
        for ind in range(len(self.matrixdescriptors.element)):
            atom_feature = self.NonBondedAtomFeatureVector(ind)
            self.atom_features.append(atom_feature)
            self.node_vector_length = len(atom_feature)
    

    def GlobalBondFeatureVector(self):
        for i in range(len(self.matrixdescriptors.element)):
            for j in range(len(self.matrixdescriptors.element)):
                if i > j:
                    if [i,j] not in self.all_edges:
                        self.edges_u.append(i)
                        self.edges_v.append(j)
                        bond_feature = self.BondFeatureVectorFromEdge([i,j])
                        bond_feature += self.GlobalBondFeatures([i,j])
                        self.bond_features.append(bond_feature)

    def NonBondedFeatureVector(self):
        self.FindNBI()
        for i in range(len(self.matrixdescriptors.element)):
            for j in range(len(self.matrixdescriptors.element)):
                if i > j:
                    if [i,j] not in self.all_edges:
                        self.edges_u.append(i)
                        self.edges_v.append(j)
                        bond_feature = self.BondFeatureVectorFromEdge([i,j])
                        bond_feature += self.NBIInteraction(i,j)
                        self.bond_features.append(bond_feature)
    



    def GlobalNonBondFeatures(self):
        for i in range(len(self.matrixdescriptors.element)):
            for j in range(len(self.matrixdescriptors.element)):
                if i > j:
                    if [i,j] not in self.all_edges:
                        self.edges_u.append(i)
                        self.edges_v.append(j)
                        bond_feature = self.BondFeatureVectorFromEdge([i,j])
                        bond_feature += self.GlobalNonBondFeatures([i,j])
                        bond_feature += self.NBIInteraction(i,j)
                        self.bond_features.append(bond_feature)
                        
    def NBIInteraction(self,i,j):
        bond_feature = []
        if [i,j] in self.hbond_edges:
            bond_feature += [1]
        else:
            bond_feature += [0]

        if self.params.addcho:
            if [i,j] in self.ch_o_edges:
                bond_feature += [1]
            else:
                bond_feature += [0]

        if self.params.adddihydrogenbonds:
            if [i,j] in self.dihydrogen_edges:
                bond_feature += [1]
            else:
                bond_feature += [0]

        if self.params.addcationpi:
            if [i,j] in self.cation_pi_edges:
                bond_feature += [1]
            else:
                bond_feature += [0]

        if self.params.addpipistack:
            if [i,j] in self.pi_pi_edges:
                bond_feature += [1]
            else:
                bond_feature += [0]

        if self.params.addhalogenbonds:
            if [i,j] in self.halogen_bond_edges:
                bond_feature += [1]
            else:
                bond_feature += [0]
        
        if self.params.addmetallophilic:
            if [i,j] in self.metallophilic_edges:
                bond_feature += [1]
            else:
                bond_feature += [0]

        if self.params.addelectrostatic:
            if [i,j] in self.electrostatic_edges['attract']:
                bond_feature += [0,1]
            elif [i,j] in self.electrostatic_edges['repel']:
                bond_feature += [1,0]
            else:
                bond_feature += [0,0]

        if self.params.addeneg:
            bond_feature += [self.ElectronegativityDifference([i,j])]
        return bond_feature

    def GenerateBondFeatureVector(self):
        self.GenerateEdges()
        self.bond_features = []
        self.all_edges = [] 
        for ind in range(len(self.edges_u)):
            bond_feature,edge = self.BondFeatureVector(ind)
            self.all_edges.append(edge)
            self.bond_features.append(bond_feature)
            self.bond_feature_length = len(bond_feature)
        if self.params.addmissingbonds == 'global':  
            self.GlobalBondFeatureVector()
        elif self.params.addmissingbonds == 'hbonds':
            self.NonBondedFeatureVector()
        elif self.params.addmissingbonds == 'all':
            self.GlobalNonBondFeatures()

    def run(self):
        self.InitializeAddons()
        self.GenerateAtomFeatureVector()
        self.GenerateBondFeatureVector()

