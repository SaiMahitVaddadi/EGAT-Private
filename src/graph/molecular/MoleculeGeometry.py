from ..base.geom import GeomFeaturizer
from .Molecule import MoleculeFeaturizer
from ..base.information import BondGeometryInformation,AtomGeometryInformation,SterimolFeaturizer,KallistoInformation,InteractionInformation

class MoleculeFeaturizerwithGeometry(GeomFeaturizer,MoleculeFeaturizer,BondGeometryInformation,AtomGeometryInformation,
                                     SterimolFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        #super(MoleculeFeaturizer).__init__(smiles, arguments)
        self.GenerateConformers()
        

    def GrabAtomFeatureVector(self,ind,yarpid=0):
        if self.params.getradical == 'YARP':
            if len(self.electroninfo.yarpecule.bond_mats) == 1:
                atom_feature = self.AtomFeatureVector(ind,yarpid)
            else:
                self.atom_features_dict = {}
                for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                    self.atom_features_dict[i] = self.AtomFeatureVector(ind,yarpid)
        else:
            atom_feature = self.AtomFeatureVector(ind)
        return atom_feature

    def _onegeomcase(self,atom_feature,ind):
        if self.params.getradical == 'YARP':
            if len(self.electroninfo.yarpecule.bond_mats) == 1:
                atom_feature = self._atomaddons(atom_feature,ind)
            else:
                for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                    self.atom_features_dict[i] = self._atomaddons(self.atom_features_dict[i],ind)
        else:
            atom_feature = self._atomaddons(atom_feature,ind)
        return atom_feature
    
    def _multigeomcase(self,atom_feature,ind):
        self.atom_feature_confs = dict()
        conf_ids = [conf.GetId() for conf in self.matrixdescriptors.new_mol.GetConformers()]
        if self.params.getradical == 'YARP':
            if len(self.electroninfo.yarpecule.bond_mats) == 1:
                for conf_id in conf_ids:
                    atom_feature = self._atomaddonswithconf(atom_feature, ind, conf_id)
                    self.atom_feature_confs[conf_id] = atom_feature
            else:
                j = 0
                for conf_id in conf_ids:
                    for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                        self.atom_feature_confs[j] = self._atomaddonswithconf(self.atom_features_dict[i], ind, conf_id)
                        j += 1
        else:
            for conf_id in conf_ids:
                atom_feature = self._atomaddonswithconf(atom_feature, ind, conf_id)
                self.atom_feature_confs[conf_id] = atom_feature
        return atom_feature
        
    def AtomFeatureVectorAddons(self,ind,yarpid=0):
        self.atom_feature_confs = dict()
        atom_feature = self.GrabAtomFeatureVector(ind,yarpid)

        if self.params.conformer.nconfs == 1:
            atom_feature = self._onegeomcase(atom_feature,ind)
        else:
            atom_feature = self._multigeomcase(atom_feature,ind)

        return atom_feature

    def _atomaddons(self,atom_feature,ind):
        atom_feature += self.AtomCoordination(ind) # Works
        atom_feature += self.DistanceToCenterOfMass(ind) # Works
        atom_feature += self.StericHindrance(ind) 
        atom_feature += self.VdWStrain(ind)
        atom_feature += self.AtomicSolventAccessibility(ind)
        atom_feature += self.GaussianCurvature(ind)
        atom_feature += self.MolecularShapeIndex(ind)
        atom_feature += self.DistanceToConvexHull(ind)
        atom_feature += self.GetVanDerWaalsRadii(ind)
        atom_feature += self.GetSterimolFeatures(ind) # Works
        atom_feature += self.GetBuriedVolume(ind)
        return atom_feature
    
    def _atomaddonswithconf(self,atom_feature,ind,conf=0):
        atom_feature += self.AtomCoordination(ind,conf)
        atom_feature += self.DistanceToCenterOfMass(ind,conf)
        atom_feature += self.StericHindrance(ind,conf)
        atom_feature += self.VdWStrain(ind,conf)
        atom_feature += self.AtomicSolventAccessibility(ind,conf)
        atom_feature += self.GaussianCurvature(ind,conf)
        atom_feature += self.MolecularShapeIndex(ind,conf)
        atom_feature += self.DistanceToConvexHull(ind,conf)
        atom_feature += self.GetVanDerWaalsRadii(ind,conf)
        atom_feature += self.GetSterimolFeatures(ind,conf)  # Works
        atom_feature += self.GetBuriedVolume(ind,conf)
        return atom_feature
    
    def _bondaddons(self,bond_feature,ind):
        edge = self.Edge(ind)
        bond_feature += self.BondLength(edge)
        bond_feature += self.BondAngle(edge)
        bond_feature += self.DihedralAngle(edge)
        return bond_feature
    
    def _bondaddonswithconf(self,bond_feature,ind,conf=0):
        edge = self.Edge(ind)
        bond_feature += self.BondLength(edge,conf)
        bond_feature += self.BondAngle(edge,conf)
        bond_feature += self.DihedralAngle(edge,conf)
        return bond_feature


    def Edge(self,ind):
        bond_feature = []
        edge = sorted([self.edges_u[ind],self.edges_v[ind]])
        return edge


    def _bondonegeomcase(self,ind):
        if self.params.getradical == 'YARP':
            if len(self.electroninfo.yarpecule.bond_mats) == 1:
                bond_feature = self.BondFeatureVector(ind)
                bond_feature = self._bondaddons(bond_feature,ind)
            else:
                for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                    self.bond_features_dict[i] = self._bondaddons(self.bond_features_dict[i],ind)
        else:
            bond_feature = self.BondFeatureVector(ind)
            bond_feature = self._bondaddons(bond_feature,ind)
        return bond_feature
    

    def _bondmultigeomcase(self,ind):
        self.bond_feature_confs = dict()
        conf_ids = [conf.GetId() for conf in self.matrixdescriptors.new_mol.GetConformers()]
        if self.params.getradical == 'YARP':
            if len(self.electroninfo.yarpecule.bond_mats) == 1:
                for conf_id in conf_ids:
                    bond_feature = self._bondaddonswithconf(self.bond_features_dict[0], ind, conf_id)
                    self.bond_feature_confs[conf_id] = bond_feature
            else:
                j = 0
                for conf_id in conf_ids:
                    for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                        self.bond_feature_confs[j] = self._bondaddonswithconf(self.bond_features_dict[i], ind, conf_id)
                        j += 1
        else:
            for conf_id in conf_ids:
                bond_feature = self._bondaddonswithconf(bond_feature, ind, conf_id)
                self.bond_feature_confs[conf_id] = bond_feature

    def BondFeatureVectorAddons(self,ind):
        self.bond_feature_confs = dict()
        if self.params.conformer.nconfs == 1:
            bond_feature = self._bondonegeomcase(ind)
        else:
            bond_feature = self._bondmultigeomcase(ind)
        return bond_feature
    

    def _handleyarpcasesforatoms(self,ind):
        atom_feature = self.AtomFeatureVectorAddons(ind)
        self.atom_features.append(atom_feature)
        if self.params.conformer.nconfs > 1:
            self.atom_feature_confs_list.append(self.atom_feature_confs)
        else:
            if self.params.getradical == 'YARP':
                if len(self.electroninfo.yarpecule.bond_mats) > 1:
                    self.atom_feature_confs_list.append(self.atom_features_dict)
        self.node_vector_length = len(atom_feature)

    def _handleyarpcasesforbonds(self,ind):
        bond_feature = self.BondFeatureVectorAddons(ind)
        self.bond_features.append(bond_feature)
        if self.params.conformer.nconfs > 1:
            self.bond_feature_confs_list.append(self.bond_feature_confs)
        else:
            if self.params.getradical == 'YARP':
                if len(self.electroninfo.yarpecule.bond_mats) > 1:
                    self.bond_feature_confs_list.append(self.bond_features_dict)
        self.bond_vector_length = len(bond_feature)

    def GenerateAtomFeatureVector(self):
        self.atom_features = []
        self.atom_feature_confs_list = []
        self.node_vector_length = 0 
        for ind in range(len(self.matrixdescriptors.element)):
            self._handleyarpcasesforatoms(ind)

    def GenerateBondFeatureVector(self):
        self.GenerateEdges()
        self.bond_features = []
        self.bond_feature_confs_list = []
        for ind in range(len(self.edges_u)):
            self._handleyarpcasesforbonds(ind)
    
    def run(self):
        self.ObtainFusedRingInfo()
        self.GenerateAtomFeatureVector()
        self.GenerateBondFeatureVector()
