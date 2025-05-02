from .MoleculeGeometry import MoleculeFeaturizerwithGeometry
from ..base.information import GlobalBondInformation,NonBondedInformation,HydrogenBondInformation,GlobalAtomInformation
from dataclasses import dataclass
from typing import Literal
from tqdm import tqdm
from icecream import ic
        
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

class MoleculePaddingCommands(MoleculeFeaturizerwithGeometry,NonBondedInformation,GlobalBondInformation,HydrogenBondInformation,GlobalAtomInformation):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

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

    def __exceptionfunction(self,fcn,ind,e):
        print('==========================ERROR ALERT==========================')
        print(f"Error in function {fcn.__name__} for atom index {ind}: {e}")
        print('===============================================================')
        import traceback
        print('==========================TRACEBACK============================')
        print('===============================================================')
        traceback.print_exc()
        print('===============================================================')
        print('===============================================================')

    def SetupPadding(self):
        self.SetupGeometryStep()
        self.FindNBI()
        self.paddingaddonfcns_atom = [self.GetClusteringCoefficient,self.GetClosenessCentrality,self.GetDegreeCentrality,
                                      self.GetDensity,self.GetAvgDegreeNeighbors,self.GetAssortativity,self.GetNodeClustering,self.GetSpectralRadius,self.GetDegreeEntropy,self.GetLocalBertzCT,
                                      self.HydrogenBondCheck,self.GetCoulombValue,self.GetPageRank]
        self.paddingaddonfcns_bond = [self.RandomWalkStatistics,self.RandomWalkCommuteTime,self.ShortestPathDistance,self.ShortestPathCount,self.EffectiveResistance,self.CommonNeighbors,
                                      self.JaccardIndex,self.AdamicAdarIndex,self.PreferentialAttachmentIndex,self.KatzCentralitySimilarity,self.EigenvectorCentralityDifference,
                                      self.BetweennessCentralityCorrelation,self.MinimumCutValue,self.MaximumFlow,self.LaplacianEigenvectorSimilarity,self.FiedlerVectorSimilarity,
                                      self.BetweennessCentralityOfPathways,self.RingsInSharedPath,self.LocalAtomicEnvironmentSimilarity,self.FirstPassageTime,
                                      self.AromaticSequence,self.GetTopoOverlap,self.GetEdgeClustering,self.GetFormanCurve,self.IsSameFG,self.GetCoulombValueForBond]
        self.all_edges = []

    def _runatompaddingfcn(self,atom_feature,ind,conf=None):
        for fcn in tqdm(self.paddingaddonfcns_atom, desc="Processing atom functions"):
            try:
                if conf is not None:
                    try: 
                        atom_feature += fcn(ind,conf)
                    except Exception as e:
                        atom_feature += fcn(ind)
                else: atom_feature += fcn(ind)
            except Exception as e:
                self.__exceptionfunction(fcn,ind,e)      
        return atom_feature

    def _atompaddingaddons(self,atom_feature,ind):
        return self._runatompaddingfcn(atom_feature,ind)
    
    def _atompaddingaddonswithconf(self,atom_feature,ind,conf=0):
        return self._runatompaddingfcn(atom_feature,ind,conf)
    
    def _onegeomcasewpadding(self,atom_feature,ind):
        if self.params.getradical == 'YARP':
            if len(self.electroninfo.yarpecule.bond_mats) == 1: 
                atom_feature = self._atomaddons(atom_feature,ind)
                atom_feature = self._atompaddingaddons(atom_feature,ind)
                
            else:
                for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                    self.atom_features_dict[i] = self._atomaddons(self.atom_features_dict[i],ind) 
                    self.atom_features_dict[i] = self._atompaddingaddons(self.atom_features_dict[i],ind)
                    self.atom_feature_dict[i] = self._checkforbadvalues(self.atom_features_dict[i])
        else:
            atom_feature = self._atomaddons(atom_feature,ind)
            atom_feature = self._atompaddingaddons(atom_feature,ind)
        atom_feature = self._checkforbadvalues(atom_feature)    
        return atom_feature
    
    def _iteratemultiplegeomswpadding(self,atom_feature,ind):
        conf_ids = [conf.GetId() for conf in self.matrixdescriptors.new_mol.GetConformers()]
        j = 0
        for conf_id in conf_ids:
            self.ConformerCalcs(conf_id)
            self.StermiolMatrices(conf_id)
            if not self._yarpcase():
                atom_feature = self._atomaddonswithconf(self.AtomFeatureVector(ind), ind, conf_id)
                atom_feature = self._atompaddingaddonswithconf(atom_feature, ind, conf_id)
                self.atom_feature_confs[conf_id] = atom_feature
                self.atom_feature_confs[conf_id] = self._checkforbadvalues(self.atom_feature_confs[conf_id])
            else:
                for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                    self.atom_features_dict[j] = self._atomaddonswithconf(self.atom_features_dict[i],ind,conf_id)
                    self.atom_feature_confs[j] = self._atompaddingaddonswithconf(self.atom_features_dict[j], ind, conf_id)
                    self.atom_feature_confs[j] = self._checkforbadvalues(self.atom_feature_confs[j])
                    j += 1
        return atom_feature

    def _runbondpaddingfcn(self,bond_feature,ind,conf=None,edge=None):
        if self.Edge(ind) is None:
            edge = edge
        else:
            edge = self.Edge(ind)
        ic(edge)
        for fcn in tqdm(self.paddingaddonfcns_bond,total=len(self.paddingaddonfcns_bond), desc="Processing bond functions"):
            try:
                if conf is None: bond_feature += fcn(edge)
                else: 
                    try:
                        bond_feature += fcn(edge,conf)
                    except:
                        bond_feature += fcn(edge)
            except Exception as e:
                self.__exceptionfunction(fcn,edge,e)
        return bond_feature
    
    def _bondpaddingaddons(self,bond_feature,ind,edge=None):
        if edge == None:
            bond_feature = self._runbondpaddingfcn(bond_feature,ind,edge=edge)
            bond_feature += self.DefaultNBInteraction(ind)
        else:
            bond_feature = self._runbondpaddingfcn(bond_feature,ind,edge=edge)
            bond_feature += self.DefaultNBInteraction(ind)
        return bond_feature
    
    def _bondpaddingaddonswithconf(self,bond_feature,ind,conf=0,edge=None):
        if edge == None:
            bond_feature = self._runbondpaddingfcn(bond_feature,ind,conf,edge=edge)
            bond_feature += self.DefaultNBInteraction(ind)
        else:
            bond_feature = self._runbondpaddingfcn(bond_feature,ind,conf,edge=edge)
            bond_feature += self.DefaultNBInteraction(ind)
        return bond_feature
    
    def _bondonegeomcasewpadding(self,bond_feature,ind,edge=None):
        if self.params.getradical == 'YARP':
            if len(self.electroninfo.yarpecule.bond_mats) == 1:
                bond_feature = self._bondaddons(bond_feature,ind,edge=edge)
                bond_feature = self._bondpaddingaddons(bond_feature,ind,edge=edge)
                bond_feature = self._checkforbadvalues(bond_feature)
            else:
                for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                    self.bond_features_dict[i] = self._bondaddons(self.bond_features_dict[i],ind,edge=edge)
                    self.bond_features_dict[i] = self._bondpaddingaddons(self.bond_features_dict[i],ind,edge=edge)
                    self.bond_features_dict[i] = self._checkforbadvalues(self.bond_features_dict[i])
        else:
            bond_feature = self._bondpaddingaddons(bond_feature,ind,edge=edge)
            bond_feature = self._bondpaddingaddons(bond_feature,ind,edge=edge)
            bond_feature = self._checkforbadvalues(bond_feature)
        return bond_feature
    
    def _bondmultigeomcasewpadding(self,bond_feature,ind,edge=None):
        conf_ids = [conf.GetId() for conf in self.matrixdescriptors.new_mol.GetConformers()]
        j = 0 
        for conf_id in conf_ids:
            self.ConformerCalcs(conf_id)
            self.StermiolMatrices(conf_id)
            if not self._yarpcase():
                bond_feature = self._bondaddonswithconf(bond_feature, ind, conf_id,edge=edge)
                bond_feature = self._bondpaddingaddonswithconf(bond_feature, ind, conf_id,edge=edge)
                self.bond_feature_confs[conf_id] = bond_feature
                self.bond_feature_confs[conf_id] = self._checkforbadvalues(self.bond_feature_confs[conf_id])
            else:
                for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                    self.bond_features_dict[i] = self._bondaddonswithconf(self.bond_features_dict[i],ind,conf_id,edge=edge)
                    self.bond_feature_confs[j] = self._bondpaddingaddonswithconf(self.bond_features_dict[i], ind, conf_id,edge=edge)
                    self.bond_feature_confs[j] = self._checkforbadvalues(self.bond_feature_confs[j])
                    j += 1
        return bond_feature
    
    def _handleyarpcasesforatomspadding(self,ind):
        atom_feature = self.AtomFeatureVector(ind)
        if self.params.conformer.nconfs == 1:
            atom_feature = self._onegeomcasewpadding(atom_feature,ind)
            if self._yarpcase():
                self.atom_feature_confs_list.append(self.atom_features_dict)
        else: 
            atom_feature = self._iteratemultiplegeomswpadding(atom_feature,ind)
            self.atom_feature_confs_list.append(self.atom_features_dict)    
        self.atom_features.append(atom_feature)
        self.node_vector_length = len(atom_feature)

    def _handleyarpcasesforbondspadding(self,ind):
        bond_feature = self.BondFeatureVector(ind)
        edge = self.GetEdge(ind)
        self.bond_features.append(bond_feature)
        if self.params.conformer.nconfs > 1:
            self.bond_feature_confs_list.append(self.bond_feature_confs)
        else:
            if self.params.getradical == 'YARP':
                if len(self.electroninfo.yarpecule.bond_mats) > 1:
                    self.bond_feature_confs_list.append(self.bond_features_dict)
        if self.params.conformer.nconfs == 1:
            bond_feature = self._bondonegeomcasewpadding(bond_feature,ind)
            if self._yarpcase():
                self.bond_feature_confs_list.append(self.bond_features_dict)
        else: 
            bond_feature = self._bondmultigeomcasewpadding(bond_feature,ind)
            self.bond_feature_confs_list.append(self.bond_features_dict)    
        self.bond_features.append(bond_feature)
        self.bond_vector_length = len(bond_feature)
        self.all_edges.append(edge)

    def _bondvectorfromedge(self,edge,conf=None):
        self.RunBondVector(edge)
        bond_feature = []
        bond_feature = self._iteratebondfeatures(self.bondfuncs,bond_feature,edge,0)
        if conf is None:
            bond_feature = self._bondaddons(bond_feature,edge,edge=edge)
            bond_feature = self._bondpaddingaddons(bond_feature,edge,edge=edge)
        else:
            bond_feature = self._bondaddonswithconf(bond_feature,edge,conf,edge=edge)
            bond_feature = self._bondpaddingaddonswithconf(bond_feature,edge,conf,edge=edge)
        bond_feature += self.NBIInteraction(edge[0],edge[1])
        bond_feature = self._checkforbadvalues(bond_feature)
        return bond_feature
    
    def _paddedbondonegeomcase(self,edge):
        if self._yarpcase() == False:
            ic(edge)
            bond_feature = self._bondvectorfromedge(edge)
            bond_feature = self._checkforbadvalues(bond_feature)
        else:
            for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                self.bond_features_dict[i] = self._bondvectorfromedge(edge)
                self.bond_features_dict[i] = self._checkforbadvalues(self.bond_features_dict[i])
            bond_feature = self.bond_features_dict[i]
        return bond_feature
    
    def _paddedbondmultigeomcase(self,edge):
        conf_ids = [conf.GetId() for conf in self.matrixdescriptors.new_mol.GetConformers()]
        j= 0 
        for conf_id in conf_ids:
            self.ConformerCalcs(conf_id)
            self.StermiolMatrices(conf_id)
            if not self._yarpcase():
                bond_feature = self._bondvectorfromedge(edge,conf_id)
                bond_feature = [0 if x is None or x is float('inf') or x != x else x for x in bond_feature]
                self.bond_feature_confs[conf_id] = self._checkforbadvalues(bond_feature)
            else:
                for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                    self.bond_feature_confs[j] = bond_feature = self._bondvectorfromedge(edge,conf_id)
                    self.bond_feature_confs[j] = self._checkforbadvalues(self.bond_feature_confs[j])
                    j += 1
        return bond_feature
    
    def _globalbondvector(self,edge):
        if self.params.conformer.nconfs == 1:
            ic(edge)
            bond_feature = self._paddedbondonegeomcase(edge)
        else:
            bond_feature = self._paddedbondmultigeomcase(edge)
        return bond_feature

    def _addglobalbonds(self):
        for i in tqdm(range(len(self.matrixdescriptors.element)), desc="Processing global bonds (outer loop)"):
            for j in tqdm(range(len(self.matrixdescriptors.element)), desc="Processing global bonds (inner loop)", leave=False):
                if i < j:
                    if [i,j] not in self.all_edges:
                        self.edges_u.append(i)
                        self.edges_v.append(j)
                        bond_feature = self._globalbondvector([i,j])
                        self.bond_features.append(bond_feature)
                        self.all_edges.append([i,j])

class MoleculeFeaturizerwithPadding(MoleculePaddingCommands):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.SetupPadding()
        
    def GenerateAtomFeatureVector(self):
        self.atom_features = []
        self.node_vector_length = 0 
        for ind in range(len(self.matrixdescriptors.element)):
            self._handleyarpcasesforatomspadding(ind)
        
    def GenerateBondFeatureVector(self):
        self.GenerateEdges()
        self.bond_features = []
        self.all_edges = [] 
        self.bond_vector_length = 0
        for ind in range(len(self.edges_u)):
            self._handleyarpcasesforbondspadding(ind)
        self._addglobalbonds()
        
    def run(self):
        self.GenerateAtomFeatureVector()
        self.GenerateBondFeatureVector()
        self.GenerateAtomGeometryVector()

