from ..base.geom import GeomFeaturizer
from .Molecule import MoleculeFeaturizer
from ..base.information import BondGeometryInformation,AtomGeometryInformation,SterimolFeaturizer
from tqdm import tqdm

# Write a function that grabs the geometry as a vector for each atom. 
class MoleculeGeometryCommands(GeomFeaturizer,MoleculeFeaturizer,BondGeometryInformation,AtomGeometryInformation,SterimolFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        #super(MoleculeFeaturizer).__init__(smiles, arguments)
        
    def SetupGeometryStep(self):
        self.GenerateConformers()

        self.vdw_radii = {
            'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.92, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
            'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88, 'K': 2.75, 'Ca': 2.31,
            'Sc': 2.11, 'Ti': 2.00, 'V': 2.00, 'Cr': 2.00, 'Mn': 2.00, 'Fe': 2.00, 'Co': 2.00, 'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39,
            'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02, 'Rb': 3.03, 'Sr': 2.49, 'Y': 2.00, 'Zr': 2.00,
            'Nb': 2.00, 'Mo': 2.00, 'Tc': 2.00, 'Ru': 2.00, 'Rh': 2.00, 'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58, 'In': 1.93, 'Sn': 2.17,
            'Sb': 2.00, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16, 'Cs': 3.43, 'Ba': 2.68, 'La': 2.00, 'Ce': 2.00, 'Pr': 2.00, 'Nd': 2.00,
            'Pm': 2.00, 'Sm': 2.00, 'Eu': 2.00, 'Gd': 2.00, 'Tb': 2.00, 'Dy': 2.00, 'Ho': 2.00, 'Er': 2.00, 'Tm': 2.00, 'Yb': 2.00,
            'Lu': 2.00, 'Hf': 2.00, 'Ta': 2.00, 'W': 2.00, 'Re': 2.00, 'Os': 2.00, 'Ir': 2.00, 'Pt': 1.75, 'Au': 1.66, 'Hg': 1.55,
            'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Po': 2.00, 'At': 2.00, 'Rn': 2.00, 'Fr': 2.00, 'Ra': 2.00, 'Ac': 2.00, 'Th': 2.00,
            'Pa': 2.00, 'U': 1.86, 'Np': 2.00, 'Pu': 2.00, 'Am': 2.00, 'Cm': 2.00, 'Bk': 2.00, 'Cf': 2.00, 'Es': 2.00, 'Fm': 2.00,
            'Md': 2.00, 'No': 2.00, 'Lr': 2.00, 'Rf': 2.00, 'Db': 2.00, 'Sg': 2.00, 'Bh': 2.00, 'Hs': 2.00, 'Mt': 2.00, 'Ds': 2.00,
            'Rg': 2.00, 'Cn': 2.00, 'Nh': 2.00, 'Fl': 2.00, 'Mc': 2.00, 'Lv': 2.00, 'Ts': 2.00, 'Og': 2.00
        }
        if self.params.conformer.nconfs == 1: 
            self.ConformerCalcs()
            self.StermiolMatrices()
            self.RunMordred3D()
        self.SetupStep()

        self.geomaddonfcns_atom = [self.AtomCoordination,self.DistanceToCenterOfMass,self.StericHindrance,self.AtomicSolventAccessibility,
                     self.GaussianCurvature,self.MolecularShapeIndex,self.DistanceToConvexHull,self.GetVanDerWaalsRadii,self.VdWStrain,self.SurroundingVdWMetrics,
                     self.AtomSterimolFeatures,self.GetBuriedVolume,self.GeomMomentDescriptors,self.SterimolMomentDescriptors,self.SterGeomMomentDescriptors,
                     self.SolidAngleCoverage]
        
        self.geomaddonfcns_bond = [self.BondLength,self.BondAngle,self.DihedralAngle,self.BondSterimolFeatures,self.Morse]

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

    def _runatomfcn(self,atom_feature,ind,conf=None):
        for fcn in tqdm(self.geomaddonfcns_atom, desc="Processing atom functions"):
            try:
                if conf is not None: atom_feature += fcn(ind,conf)
                else: atom_feature += fcn(ind)
            except Exception as e:
                self.__exceptionfunction(fcn,ind,e)      
        return atom_feature

    def _atomaddons(self,atom_feature,ind):
        return self._runatomfcn(atom_feature,ind)
    
    def _atomaddonswithconf(self,atom_feature,ind,conf=0):
        return self._runatomfcn(atom_feature,ind,conf)
    
    def _onegeomcase(self,atom_feature,ind):
        if self.params.getradical == 'YARP':
            if len(self.electroninfo.yarpecule.bond_mats) == 1: 
                atom_feature = self._atomaddons(atom_feature,ind)
                atom_feature = self._checkforbadvalues(atom_feature)
            else:
                for i in range(len(self.electroninfo.yarpecule.bond_mats)): 
                    self.atom_features_dict[i] = self._atomaddons(self.atom_features_dict[i],ind)
                    self.atom_features_dict[i] = self._checkforbadvalues(self.atom_features_dict[i])
        else:
            atom_feature = self._atomaddons(atom_feature,ind)
        return atom_feature
    
    def _yarpcase(self):
        if self.params.getradical == 'YARP':
            if len(self.electroninfo.yarpecule.bond_mats) == 1:
                return False
            else:
                return True
        else:
            return False

    def _iteratemultiplegeoms(self,atom_feature,ind):
        conf_ids = [conf.GetId() for conf in self.matrixdescriptors.new_mol.GetConformers()]
        j = 0
        for conf_id in conf_ids:
            self.ConformerCalcs(conf_id)
            self.StermiolMatrices(conf_id)
            self.RunMordred3D(conf_id)
            if not self._yarpcase():
                atom_feature = self._atomaddonswithconf(atom_feature, ind, conf_id)
                self.atom_feature_confs[conf_id] = self._checkforbadvalues(atom_feature)
            else:
                for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                    self.atom_feature_confs[j] = self._atomaddonswithconf(self.atom_features_dict[i], ind, conf_id)
                    self.atom_feature_confs[j] = self._checkforbadvalues(self.atom_feature_confs[j])
                    j += 1
        return atom_feature

    def Edge(self,ind):
        try:
            edge = sorted([self.edges_u[ind],self.edges_v[ind]])
            return edge
        except:
            return None
    
    def _runbondfcn(self,bond_feature,ind,conf=None,edge=None):
        if self.Edge(ind) == None:
            edge = edge
        else:
            edge = self.Edge(ind)
        for fcn in tqdm(self.geomaddonfcns_bond,total=len(self.geomaddonfcns_bond), desc="Processing bond functions"):
            try:
                if conf is None: bond_feature += fcn(edge)
                else: bond_feature += fcn(edge,conf)
            except Exception as e:
                self.__exceptionfunction(fcn,edge,e)
        return bond_feature
    
    def _bondaddons(self,bond_feature,ind,edge=None):
        print(f"Processing bond {ind} with edge {edge} in this function")
        if edge == None:
            return self._runbondfcn(bond_feature,ind)
        else:
            return self._runbondfcn(bond_feature,ind,edge=edge)
    
    def _bondaddonswithconf(self,bond_feature,ind,conf=0,edge=None):
        if edge == None:
            return self._runbondfcn(bond_feature,ind,conf)
        else:
            return self._runbondfcn(bond_feature,ind,conf,edge)
    
    def _bondonegeomcase(self,ind,edge=None):
        if self.params.getradical == 'YARP':
            if len(self.electroninfo.yarpecule.bond_mats) == 1:
                bond_feature = self.BondFeatureVector(ind)
                bond_feature = self._bondaddons(bond_feature,ind,edge=edge)
                bond_feature = self._checkforbadvalues(bond_feature)
            else:
                for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                    self.bond_features_dict[i] = self._bondaddons(self.bond_features_dict[i],ind,edge=edge)
                    self.bond_features_dict[i] = self._checkforbadvalues(self.bond_features_dict[i])
        else:
            bond_feature = self.BondFeatureVector(ind)
            bond_feature = self._bondaddons(bond_feature,ind)
            bond_feature = self._checkforbadvalues(bond_feature)
        return bond_feature
    
    def _bondmultigeomcase(self,ind,edge=None):
        conf_ids = [conf.GetId() for conf in self.matrixdescriptors.new_mol.GetConformers()]
        j= 0 
        for conf_id in conf_ids:
            self.ConformerCalcs(conf_id)
            self.StermiolMatrices(conf_id)
            if not self._yarpcase():
                bond_feature = self._bondaddonswithconf(self.BondFeatureVector(ind), ind, conf_id,edge=edge)
                bond_feature = self._checkforbadvalues(bond_feature)
                self.bond_feature_confs[conf_id] = bond_feature
            else:
                for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                    self.bond_feature_confs[j] = self._bondaddonswithconf(self.bond_features_dict[i], ind, conf_id,edge=edge)
                    self.bond_feature_confs[j] = self._checkforbadvalues(self.bond_feature_confs[j])
                    j += 1
        return bond_feature
    
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

    def AtomFeatureVectorAddons(self,ind,yarpid=0):
        self.atom_feature_confs = dict()
        atom_feature = self.GrabAtomFeatureVector(ind,yarpid)

        if self.params.conformer.nconfs == 1:
            atom_feature = self._onegeomcase(atom_feature,ind)
        else:
            self.ConformerCalcs()
            atom_feature = self._iteratemultiplegeoms(atom_feature,ind)

        return atom_feature

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
class MoleculeFeaturizerwithGeometry(MoleculeGeometryCommands):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.SetupGeometryStep()

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

    def ObtainAtomGeometry(self,ind,id=0):
        self.GrabGeometry(id)
        return self.positions[ind]
    
    def ObtainAtomGeometryVector(self,id=0):
        atom_geom = [] 
        for ind in range(len(self.matrixdescriptors.element)):
            atom_geom.append(self.ObtainAtomGeometry(ind,id))
        return atom_geom
    

    def GenerateAtomGeometryVector(self):
        self.atom_geometry_features = []
        self.atom_geometry_feature_confs_list = []
        if self._yarpcase():
            for i in range(len(self.electroninfo.yarpecule.bond_mats)):
                for j in range(self.params.conformer.nconfs):
                    self.atom_geometry_feature_confs_list.append(self.ObtainAtomGeometryVector(j))
        else:
            self.atom_geometry_features = self.ObtainAtomGeometryVector()



    def run(self):
        self.GenerateAtomFeatureVector()
        self.GenerateBondFeatureVector()
        self.GenerateAtomGeometryVector()
