from ..base.geom import GeomFeaturizer
from .Molecule import MoleculeFeaturizer
from ..base.information import BondGeometryInformation,AtomGeometryInformation



# To-do: Blow this up for x number of conformers

class MoleculeFeaturizerwithGeometry(GeomFeaturizer,MoleculeFeaturizer,BondGeometryInformation,AtomGeometryInformation):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        #super(MoleculeFeaturizer).__init__(smiles, arguments)
        self.GenerateConformers()
        
    
    def AtomFeatureVectorAddons(self,ind):
        atom_feature_confs = dict()
        atom_feature = self.AtomFeatureVector(ind)
        if self.params.conformer.nconfs == 1:
            atom_feature = self._atomaddons(atom_feature,ind)
        else:
            for conf in self.matrixdescriptors.new_mol.GetConformer():
                atom_feature_confs[conf] = atom_feature



        return atom_feature

    def _atomaddons(self,atom_feature,ind):
        atom_feature += self.AtomCoordination(ind)
        atom_feature += self.DistanceToCenterOfMass(ind)
        atom_feature += self.StericHindrance(ind)
        atom_feature += self.AtomicSolventAccessibility(ind)
        atom_feature += self.GaussianCurvature(ind)
        atom_feature += self.MolecularShapeIndex(ind)
        atom_feature += self.DistanceToConvexHull(ind)
        atom_feature += self.GetVanDerWaalsRadii(ind)
        return atom_feature
    
    def _atomaddonswithconf(self,atom_feature,ind,conf=0):
        atom_feature += self.AtomCoordination(ind,conf)
        atom_feature += self.DistanceToCenterOfMass(ind,conf)
        atom_feature += self.StericHindrance(ind,conf)
        atom_feature += self.AtomicSolventAccessibility(ind,conf)
        atom_feature += self.GaussianCurvature(ind,conf)
        atom_feature += self.MolecularShapeIndex(ind,conf)
        atom_feature += self.DistanceToConvexHull(ind,conf)
        atom_feature += self.GetVanDerWaalsRadii(ind,conf)
        return atom_feature
    

    def Edge(self,ind):
        bond_feature = []
        edge = sorted([self.edges_u[ind],self.edges_v[ind]])
        return edge


    def BondFeatureVectorAddons(self,ind):
        bond_feature = self.BondFeatureVector(ind)
        edge = self.Edge(ind)
        bond_feature += self.BondLength(edge)
        bond_feature += self.BondAngle(edge)
        bond_feature += self.DihedralAngle(edge)
        return bond_feature
    

    def GenerateAtomFeatureVector(self):
        self.atom_features = []
        self.node_vector_length = 0 
        for ind in range(len(self.matrixdescriptors.element)):
            atom_feature = self.AtomFeatureVectorAddons(ind)
            self.atom_features.append(atom_feature)
            self.node_vector_length = len(atom_feature)

    def GenerateBondFeatureVector(self):
        self.GenerateEdges()
        self.bond_features = []
        for ind in range(len(self.edges_u)):
            bond_feature = self.BondFeatureVectorAddons(ind)
            self.bond_features.append(bond_feature)
            self.bond_feature_length = len(bond_feature)
    
    def run(self):
        self.GenerateAtomFeatureVector()
        self.GenerateBondFeatureVector()
