from .base import BaseFeaturizer
from .reaction import BaseReactionFeaturizer
from rdkit import Chem
from rdkit.Chem import DataStructs
import networkx as nx
import numpy as np
from rdkit.Chem import BRICS
from ...utils.descriptors.egat.functional import FunctionalGroups
class NeighborInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
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
    
    def GetNeighborsChemprop(self,ind):
        pass


class RingInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    def RingCheck(self,ind):
        if not self.params.removeringinfo:
            if True in [ind in ra_list for ra_list in self.ring_atoms]: inR = 1
            else: inR = 0
            return [inR]
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
    
    def BondinRing(self,edge):
        if not self.params.removeringinfo:
            if True in [(edge[0] in ra_list and edge[1] in ra_list) for ra_list in self.ring_atoms]: inR = 1
            else: inR = 0
            return [inR]
        else:
            return []


class ElectronInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    
    def RadicalCheck(self,ind):
        if self.args.getradical:
            return [self.electroninfo.rads[ind],self.electroninfo.lps[ind]]
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
        
class ChargeInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    
    def ElectronegativityCheck(self,ind):
        if self.args.getelectronegativity:
            return [self.matrixdescriptors.electronegativity[ind]]
        else:
            return []
        
    def FormalCharge(self,ind):
        if not self.params.removeformalchargeinfo:
            fc = self.matrixdescriptor.fc[ind]
            return fc
        else:
            return []

    def ChargeCheck(self,ind):
        if self.params.charge == 'Gasteiger':
            return [self.matrixdescriptors.gasteiger_charges[ind]]
        elif self.params.charge == 'psi4':
            return []
        elif self.params.charge == 'pyscf':
            return []
        elif self.params.charge == 'gaussian':
            return []
        elif self.params.charge == 'orca':
            return []
        elif self.params.charge == 'ase':
            return []
        elif self.params.charge == 'gemnet':
            return []
        elif self.params.charge == 'chemprop':
            return []
        elif self.params.charge == 'unimol':
            return []
        else:
            return []


class AcidBaseInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    
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
        

class RDInformation(BaseFeaturizer):    
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
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

class StereoInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def ChiralityCheck(self,ind):
        if not self.args.removechiralinfo:
            ###### CHECK IF IT IS IN A CHIRAL CENTER
            if ind in self.stereo.chiral_centers: chiral = self.properties.atom_chiral_encode[self.stereo.chiral_centers[ind]]
            else: chiral = [0,0,1]
            return [chiral]
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

class BondInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()

    def EncodeBondOrder(self,edge):
        BO = self.matrixdescriptors.bond_mat[edge[0],edge[1]]
        return BO


    def BondOrder(self,edge,bo):
        if not self.params.removebondorderinfo:
            if bo == 0:            
                return [0,0,0,0,1]
            else:
                if tuple(edge) in self.stereo.bond_aromatic and self.stereo.bond_aromatic[tuple(edge)]: return self.properties.bond_order_encode['BA']
                else: return self.properties.bond_order_encode['B{}'.format(int(bo))]
        else:
            return []
        

class BondGeometryInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        

    def BondLength(self,edge):
        if self.params.getbondlength:
            bond = self.new_mol.GetBondBetweenAtoms(edge[0], edge[1])
            bond_length = bond.GetBondLength()
            return [bond_length]
        else:
            return []

    def AtomDistance(self, edge):
        if self.params.getatomdistance:
            
            atom1 = edge[0]
            atom2 = edge[1]
            pos1 = self.new_mol.GetConformer().GetAtomPosition(atom1)
            pos2 = self.new_mol.GetConformer().GetAtomPosition(atom2)
            distance = pos1.Distance(pos2)
            return [distance]
        else:
            return []

    def GetAngle(self,atom1, atom2):
        angles = []
        for neighbor in self.GetNeighbor(atom2, atom1):
            angle = self.new_mol.GetBondBetweenAtoms(atom1, atom2).GetAngle(self.new_mol.GetConformer(), atom1, atom2, neighbor)
            angles.append(angle)
        return angles 

    def BondAngle(self,edge):
        if self.params.getbondangle == 'first': 
            atom1 = edge[0]
            atom2 = edge[1]
            atom3 = self.GetNeighbor(atom2,atom1)[0]
            angle = self.new_mol.GetBondBetweenAtoms(atom1, atom2).GetAngle(self.new_mol.GetConformer(), atom1, atom2, atom3)
            return [angle]
        elif self.params.getbondangle == 'smallest':
            angles = self.GetAngle(atom1,atom2)
            return [min(angles)]
        elif self.params.getbondangle == 'largest':
            angles = self.GetAngle(atom1,atom2)
            return [max(angles)]
        elif self.params.getbondangle == 'average':
            angles = self.GetAngle(atom1,atom2)
            return [sum(angles)/len(angles)]
        elif self.params.getbondangle == 'all':
            angles = self.GetAngle(atom1,atom2)
            return self.FlattenList(angles)
        elif self.params.getbondangle == 'main':
            angles = self.GetAngle(atom1,atom2)
            return [min(angles),max(angles),sum(angles)/len(angles)]
        else:
            return []

    def GetDihedral(self, edge):
        atom1 = edge[0]
        atom2 = edge[1]
        dihedrals = []
        for neighbor1 in self.GetNeighbor(atom2, atom1):
            row = []
            for neighbor2 in self.GetNeighbor(neighbor1, atom2):
                dihedral = self.new_mol.GetBondBetweenAtoms(atom1, atom2).GetDihedral(self.new_mol.GetConformer(), atom1, atom2, neighbor1, neighbor2)
                row.append(dihedral)
            dihedrals.append(row)
        return dihedrals
    
    def FlattenList(self, nested_list):
        return [item for sublist in nested_list for item in sublist]

    def DihedralAngle(self,edge):
        if self.params.getdihedral == 'first':
            atom1 = edge[0]
            atom2 = edge[1]
            atom3 = self.GetNeighbor(atom2,atom1)[0]
            atom4 = self.GetNeighbor(atom3,atom2)[0]
            dihedral = self.new_mol.GetBondBetweenAtoms(atom1, atom2).GetDihedral(self.new_mol.GetConformer(), atom1, atom2, atom3, atom4)
            return [dihedral]
        elif self.params.getdihedral == 'firstsmallest':
            dihedrals = self.GetDihedral(edge)
            return [min(dihedrals[0])]
        elif self.params.getdihedral == 'firstlargest':
            dihedrals = self.GetDihedral(edge)
            return [max(dihedrals[0])]
        elif self.params.getdihedral == 'firstaverage': 
            dihedrals = self.GetDihedral(edge)
            return [sum(dihedrals[0])/len(dihedrals[0])]
        elif self.params.getdihedral == 'firstall':
            dihedrals = self.GetDihedral(edge)
            return dihedrals[0]
        elif self.params.getdihedral == 'firstmain':
            dihedrals = self.GetDihedral(edge)
            return [min(dihedrals[0]),max(dihedrals[0]),sum(dihedrals[0])/len(dihedrals[0])]
        elif self.params.getdihedral == 'smallest':
            dihedrals = self.GetDihedral(edge)
            return [min(self.FlattenList(dihedrals))]
        elif self.params.getdihedral == 'largest':
            dihedrals = self.GetDihedral(edge)
            return [max(self.FlattenList(dihedrals))]
        elif self.params.getdihedral == 'average': 
            dihedrals = self.GetDihedral(edge)
            return [sum(self.FlattenList(dihedrals))/len(self.FlattenList(dihedrals))]
        elif self.params.getdihedral == 'all':
            dihedrals = self.GetDihedral(edge)
            return self.FlattenList(dihedrals)
        elif self.params.getdihedral == 'main':
            dihedrals = self.GetDihedral(edge)
            return [min(self.FlattenList(dihedrals)),max(self.FlattenList(dihedrals)),sum(self.FlattenList(dihedrals))/len(self.FlattenList(dihedrals))]
        else:
            return []
    
    def BondMidpoint(self, edge):
        if self.params.getbondmidpoint:
            atom1 = edge[0]
            atom2 = edge[1]
            pos1 = self.new_mol.GetConformer().GetAtomPosition(atom1)
            pos2 = self.new_mol.GetConformer().GetAtomPosition(atom2)
            midpoint = (pos1 + pos2) / 2
            return [midpoint.x, midpoint.y, midpoint.z]
        else:
            return []
        


class BRICSInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.brics = self.GetBRICSDecomposition()
    
    def GetBRICSDecomposition(self):
        mol = Chem.MolFromSmiles(self.smiles)
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        bond_breaks = [(bond[0][0], bond[0][1]) for bond in brics_bonds]
        return bond_breaks

    def BRICSPattern(self,ind):
        if self.params.getbrics:
            return [self.brics[ind]]
        else:
            return []
    
    def IsPartOfBRICSBond(self, ind):
        if self.params.checkbricsbond:
            for bond in self.brics:
                if ind in bond:
                    return [1]
            return [0]
        else:
            return []
    


class AtomGeometryInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    
    def AtomCoordination(self,ind):
        if not self.params.removecoordinationinfo:
            return [self.matrixdescriptors.coord[ind]]
        else:
            return []

    def DistanceToCenterOfMass(self, ind):
        if self.params.getdistancetocenterofmass:
            mol = Chem.MolFromSmiles(self.smiles)
            conformer = mol.GetConformer()
            atom_positions = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
            center_of_mass = np.mean(atom_positions, axis=0)
            atom_position = conformer.GetAtomPosition(ind)
            distance = np.linalg.norm(atom_position - center_of_mass)
            return [distance]
        else:
            return []

    def StericHindrance(self, ind):
        if self.params.getsterichindrance:
            mol = Chem.MolFromSmiles(self.smiles)
            conformer = mol.GetConformer()
            atom_position = conformer.GetAtomPosition(ind)
            hindrance = 0
            for neighbor in range(mol.GetNumAtoms()):
                if neighbor != ind:
                    neighbor_position = conformer.GetAtomPosition(neighbor)
                    distance = atom_position.Distance(neighbor_position)
                    if distance < 3.5:  # typical steric hindrance distance cutoff
                        hindrance += 1 / distance
            return [hindrance]
        else:
            return []
    
    def AtomicSolventAccessibility(self, ind):
        if self.params.getasa:
            mol = Chem.MolFromSmiles(self.smiles)
            conformer = mol.GetConformer()
            atom_position = conformer.GetAtomPosition(ind)
            
            # Calculate solvent accessible surface area using RDKit
            radii = Chem.rdFreeSASA.classifyAtoms(mol)
            asa = Chem.rdFreeSASA.CalcSASA(mol, radii)
            
            return [asa[ind]]
        else:
            return []
    
    def GaussianCurvature(self, ind):
        if self.params.getgaussiancurvature:
            mol = Chem.MolFromSmiles(self.smiles)
            conformer = mol.GetConformer()
            atom_position = conformer.GetAtomPosition(ind)
            
            # Calculate Gaussian curvature using neighboring atoms
            neighbors = self.GetNeighbor(ind)
            if len(neighbors) < 3:
                return [0]  # Not enough neighbors to define a surface
            angles = []
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    pos1 = conformer.GetAtomPosition(neighbors[i])
                    pos2 = conformer.GetAtomPosition(neighbors[j])
                    angle = pos1.Angle(atom_position, pos2)
                    angles.append(angle)
            gaussian_curvature = 2 * np.pi - sum(angles)
            return [gaussian_curvature]
        else:
            return []

    def MolecularShapeIndex(self, ind):
        if self.params.getmolecularshapeindex:
            mol = Chem.MolFromSmiles(self.smiles)
            conformer = mol.GetConformer()
            atom_position = conformer.GetAtomPosition(ind)
            
            # Calculate molecular shape index using neighboring atoms
            neighbors = self.GetNeighbor(ind)
            if len(neighbors) < 3:
                return [0]  # Not enough neighbors to define a surface
            distances = []
            for neighbor in neighbors:
                pos = conformer.GetAtomPosition(neighbor)
                distance = atom_position.Distance(pos)
                distances.append(distance)
            shape_index = sum(distances) / len(distances)
            return [shape_index]
        else:
            return []

    def DistanceToConvexHull(self, ind):
        if self.params.getdistancetoconvexhull:
            mol = Chem.MolFromSmiles(self.smiles)
            conformer = mol.GetConformer()
            atom_position = conformer.GetAtomPosition(ind)
            
            # Calculate convex hull of the molecule
            atom_positions = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
            points = np.array([[pos.x, pos.y, pos.z] for pos in atom_positions])
            hull = ConvexHull(points)
            
            # Calculate distance from atom to convex hull
            distances = []
            for simplex in hull.simplices:
                vertices = points[simplex]
                distance = np.min(np.linalg.norm(vertices - np.array([atom_position.x, atom_position.y, atom_position.z]), axis=1))
                distances.append(distance)
            return [min(distances)]
        else:
            return []

    def GetVanDerWaalsRadii(self, ind):
        vdw_radii = {
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
        element = self.matrixdescriptors.element[ind]
        return [vdw_radii.get(element, 2.00)]  # Default to 2.00 if element not found


class ReactiveAtomInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    
    def DistanceFromReactingAtom(self,ind):
        if not self.params.removereactiveinfo:
            if len(self.reactive_atoms) > 0:
                dis = min([self.gs[ind][indr] for indr in self.reactive_atoms])
            else:
                dis = 0 
            return [dis]
        else:
            return []
        
    def NeighboringReactives(self,adj_mat,ind):
        if self.params.addneighboringreactives: 
            reactive_neighbors = 0
            for neighbor in adj_mat[ind]:
                if neighbor in self.reactionpropeties.atom.reactive:
                    reactive_neighbors += 1
            return [reactive_neighbors]
        else:
            return []



class ReactiveAtomChangeInformation(BaseReactionFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def ChangeInAtomicHybridization(self, atom):
        if self.params.gethybridizationchange:
            initial_hybridization = self.reactant.stereo.Hybridization[atom]
            final_hybridization = self.product.stereo.Hybridization[atom]
            return final_hybridization - initial_hybridization
        else:
            return []

    def DegreeCentralityOfChangingAtoms(self, atom):
        if self.params.getdegreecentrality:
            reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
            product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
            reactant_centrality = nx.degree_centrality(reactant_graph)[atom]
            product_centrality = nx.degree_centrality(product_graph)[atom]
            return [reactant_centrality - product_centrality]
        else:
            return []

    def AtomicValencyChange(self, atom):
        if self.params.getvalencychange:
            initial_valency = self.reactant.matrixdescriptors.valency[atom]
            final_valency = self.product.matrixdescriptors.valency[atom]
            return [final_valency - initial_valency]
        else:
            return []

    def OxidationOrReduction(self, atom):
        if self.params.getoxidationreduction:
            initial_oxidation_state = self.reactant.matrixdescriptors.oxidation_state[atom]
            final_oxidation_state = self.product.matrixdescriptors.oxidation_state[atom]
            return [initial_oxidation_state - final_oxidation_state]
        else:
            return []

    def LocalBondOrderSumChange(self, atom):
        if self.params.getbondordersumchange:
            initial_bond_order_sum = sum(self.reactant.matrixdescriptors.bond_mat[atom])
            final_bond_order_sum = sum(self.product.matrixdescriptors.bond_mat[atom])
            return [initial_bond_order_sum - final_bond_order_sum]
        else:
            return []

    def AtomicNeighborhoodChangeRatio(self, atom):
        if self.params.getneighborhoodchangeratio:
            initial_neighbors = set(self.reactant.matrixdescriptors.adj_mat[atom].nonzero()[0])
            final_neighbors = set(self.product.matrixdescriptors.adj_mat[atom].nonzero()[0])
            common_neighbors = initial_neighbors.intersection(final_neighbors)
            total_neighbors = initial_neighbors.union(final_neighbors)
            if len(total_neighbors) > 0:
                return [len(common_neighbors) / len(total_neighbors)]
            else:
                return [0]
        else:
            return []
    
    def LocalAtomicEnvironmentSimilarity(self, atom):
        if self.params.getLocalAtomicEnvironmentSimilarity:
            try:
                if self.params.global_fingerprint_type == 'Morgan':
                    fp1 = Chem.GetMorganFingerprintAsBitVect(self.reactant.new_mol, 2, nBits=2048, fromAtoms=[atom])
                    fp2 = Chem.GetMorganFingerprintAsBitVect(self.product.new_mol, 2, nBits=2048, fromAtoms=[atom])
                elif self.params.global_fingerprint_type == 'MACCS':
                    fp1 = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(self.reactant.new_mol, fromAtoms=[atom])
                    fp2 = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(self.product.new_mol, fromAtoms=[atom])
                elif self.params.global_fingerprint_type == 'RDK':
                    fp1 = Chem.RDKFingerprint(self.reactant.new_mol, fromAtoms=[atom])
                    fp2 = Chem.RDKFingerprint(self.product.new_mol, fromAtoms=[atom])
                elif self.params.global_fingerprint_type == 'AtomPair':
                    fp1 = Chem.GetAtomPairFingerprint(self.reactant.new_mol, fromAtoms=[atom])
                    fp2 = Chem.GetAtomPairFingerprint(self.product.new_mol, fromAtoms=[atom])
                elif self.params.global_fingerprint_type == 'TopologicalTorsion':
                    fp1 = Chem.GetTopologicalTorsionFingerprintAsIntVect(self.reactant.new_mol, fromAtoms=[atom])
                    fp2 = Chem.GetTopologicalTorsionFingerprintAsIntVect(self.product.new_mol, fromAtoms=[atom])
                elif self.params.global_fingerprint_type == 'Avalon':
                    fp1 = Chem.rdMolDescriptors.GetAvalonFP(self.reactant.new_mol, fromAtoms=[atom])
                    fp2 = Chem.rdMolDescriptors.GetAvalonFP(self.product.new_mol, fromAtoms=[atom])
                elif self.params.global_fingerprint_type == 'Estate':
                    fp1 = Chem.rdMolDescriptors.GetEstateFingerprint(self.reactant.new_mol, fromAtoms=[atom])
                    fp2 = Chem.rdMolDescriptors.GetEstateFingerprint(self.product.new_mol, fromAtoms=[atom])
                elif self.params.global_fingerprint_type == 'Layered':
                    fp1 = Chem.LayeredFingerprint(self.reactant.new_mol, fromAtoms=[atom])
                    fp2 = Chem.LayeredFingerprint(self.product.new_mol, fromAtoms=[atom])
                else:
                    return [0]

                if self.params.global_similarity_metric == 'Tanimoto':
                    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Dice':
                    similarity = DataStructs.DiceSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Cosine':
                    self.params.global_similarity = DataStructs.CosineSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Sokal':
                    similarity = DataStructs.SokalSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Russel':
                    similarity = DataStructs.RusselSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Kulczynski':
                    similarity = DataStructs.KulczynskiSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'McConnaughey':
                    similarity = DataStructs.McConnaugheySimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Asymmetric':
                    similarity = DataStructs.AsymmetricSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'BraunBlanquet':
                    similarity = DataStructs.BraunBlanquetSimilarity(fp1, fp2)
                else:
                    return [0]

                return [similarity]
            except:
                return [0]
        else:
            return []

        
    

class ReactiveBondInformation(BaseReactionFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    
    def BondChangeInfo(self,edge):
        BO_R = self.reactant.matrixdescriptors.bond_mat[edge[0],edge[1]]
        BO_P = self.product.matrixdescriptors.bond_mat[edge[0],edge[1]]
        if BO_R == BO_P:
            RBtype = self.reactant.properties.bond_encode['T1']
            PBtype = self.reactant.properties.bond_encode['T1']
        elif BO_R == 0.0:
            RBtype = self.reactant.properties.bond_encode['T4']
            PBtype = self.reactant.properties.bond_encode['T3']
        elif BO_P == 0.0:
            RBtype = self.reactant.properties.bond_encode['T3']
            PBtype = self.reactant.properties.bond_encode['T4']
        elif BO_R < BO_P and BO_R > 0:
            RBtype = self.reactant.properties.bond_encode['T2']
            PBtype = self.reactant.properties.bond_encode['T2']
        elif BO_R > BO_P and BO_R > 0:
            RBtype = self.reactant.properties.bond_encode['T5']
            PBtype = self.reactant.properties.bond_encode['T5']
        return RBtype,PBtype

    def OldBondChangeInfo(self,edge):
        BO_R = self.reactant.matrixdescriptors.bond_mat[edge[0],edge[1]]
        BO_P = self.product.matrixdescriptors.bond_mat[edge[0],edge[1]]
        if BO_R == BO_P:
            RBtype = self.reactant.properties.old_bond_encode['T1']
            PBtype = self.reactant.properties.old_bond_encode['T1']
        elif BO_R == 0.0:
            RBtype = self.reactant.properties.old_bond_encode['T4']
            PBtype = self.reactant.properties.old_bond_encode['T3']
        elif BO_P == 0.0:
            RBtype = self.reactant.properties.old_bond_encode['T3']
            PBtype = self.reactant.properties.old_bond_encode['T4']
        elif BO_R != BO_P and BO_R > 0:
            RBtype = self.reactant.properties.old_bond_encode['T2']
            PBtype = self.reactant.properties.old_bond_encode['T2']
        return RBtype,PBtype
    
    def DistanceFromReactingBond(self,edge):
        if self.params.adddisttoreactingbonds:
            if len(self.reactionpropeties.bond.reactive) > 0:
                dis = min([self.reactant.gs[edge[0]][rb[0]] + self.reactant.gs[edge[1]][rb[1]] for rb in self.reactionpropeties.bond.reactive])
            else:
                dis = 0
            return [dis]
        else:
            return []
                    
    
    def BondOrderChange(self, edge):
        BO_R = self.reactant.matrixdescriptors.bond_mat[edge[0], edge[1]]
        BO_P = self.product.matrixdescriptors.bond_mat[edge[0], edge[1]]
        return [BO_P - BO_R]

    def BondNeighborhoodChange(self, edge):
        neighbors_R = set(self.reactant.matrixdescriptors.adj_mat[edge[0]].nonzero()[0]).union(
            set(self.reactant.matrixdescriptors.adj_mat[edge[1]].nonzero()[0]))
        neighbors_P = set(self.product.matrixdescriptors.adj_mat[edge[0]].nonzero()[0]).union(
            set(self.product.matrixdescriptors.adj_mat[edge[1]].nonzero()[0]))
        common_neighbors = neighbors_R.intersection(neighbors_P)
        total_neighbors = neighbors_R.union(neighbors_P)
        if len(total_neighbors) > 0:
            return [len(common_neighbors) / len(total_neighbors)]
        else:
            return [0]

    def ShortestPathChangeAcrossBond(self, edge):
        reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
        product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
        sp_R = nx.shortest_path_length(reactant_graph, source=edge[0], target=edge[1])
        sp_P = nx.shortest_path_length(product_graph, source=edge[0], target=edge[1])
        return [sp_P - sp_R]

    def BondParticipationDegree(self, edge):
        reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
        product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
        degree_R = reactant_graph.degree(edge[0]) + reactant_graph.degree(edge[1])
        degree_P = product_graph.degree(edge[0]) + product_graph.degree(edge[1])
        return [degree_P - degree_R]

    def BondConnectivityPathDifference(self, edge):
        reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
        product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
        connectivity_R = nx.all_pairs_shortest_path_length(reactant_graph)
        connectivity_P = nx.all_pairs_shortest_path_length(product_graph)
        diff = 0
        for node in connectivity_R:
            for target, length in connectivity_R[node].items():
                if target in connectivity_P[node]:
                    diff += abs(length - connectivity_P[node][target])
                else:
                    diff += length
        return [diff]

    def BRICSBondRoleChange(self, edge):
        if self.params.getbrics:
            role_R = self.reactant.brics[edge[0]] if edge[0] in self.reactant.brics else None
            role_P = self.product.brics[edge[0]] if edge[0] in self.product.brics else None
            if role_R == role_P:
                return [0]
            else:
                return [1]
        else:
            return []

    def BRICSBondFormationLikelihood(self, edge):
        if self.params.getbrics:
            mol = Chem.MolFromSmiles(self.smiles)
            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            bond_breaks = [(bond[0][0], bond[0][1]) for bond in brics_bonds]
            if edge in bond_breaks or (edge[1], edge[0]) in bond_breaks:
                return [1]
            else:
                return [0]
        else:
            return []

    def BRICSFragmentationConsistency(self, edge):
        if self.params.getbrics:
            mol = Chem.MolFromSmiles(self.smiles)
            brics_bonds = list(BRICS.FindBRICSBonds(mol))
            bond_breaks = [(bond[0][0], bond[0][1]) for bond in brics_bonds]
            reactant_bonds = set(self.reactant.brics)
            product_bonds = set(self.product.brics)
            common_bonds = reactant_bonds.intersection(product_bonds)
            total_bonds = reactant_bonds.union(product_bonds)
            if len(total_bonds) > 0:
                return [len(common_bonds) / len(total_bonds)]
            else:
                return [0]
        else:
            return []
    

class ReactiveAtomGeometryInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

class ReactiveBondGeometryInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    
    


class NonBondedInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    def FindHBonds(self):
        self.hbond_edges = []
        for donor in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.element[donor] in ['O', 'N', 'F', 'Cl', 'Br', 'I']:
                for hydrogen in range(len(self.matrixdescriptors.element)):
                    if self.matrixdescriptors.element[hydrogen] == 'H' and self.matrixdescriptors.adj_mat[donor][hydrogen] > 0:
                        for acceptor in range(len(self.matrixdescriptors.element)):
                            if self.matrixdescriptors.element[acceptor] in ['O', 'N', 'F', 'Cl', 'Br', 'I'] and donor != acceptor:
                                distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(hydrogen).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(acceptor))
                                if distance > 0 and distance <= 3:  # typical H-bond distance cutoff
                                    self.hbond_edges.append([hydrogen, acceptor])
    
    def FindElectrostaticInteractions(self):
        self.electrostatic_edges = dict()
        self.electrostatic_edges['attract'] = []
        self.electrostatic_edges['repel'] = []
        for atom1 in range(len(self.matrixdescriptors.element)):
            for atom2 in range(atom1 + 1, len(self.matrixdescriptors.element)):
                if self.matrixdescriptors.fc[atom1] != 0 and self.matrixdescriptors.fc[atom2] != 0:
                    distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(atom1).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(atom2))
                    if distance > 0 and distance <= 5:  # typical electrostatic interaction distance cutoff
                        if self.matrixdescriptors.fc[atom1] > 0 and self.matrixdescriptors.fc[atom2] < 0 or self.matrixdescriptors.fc[atom1] < 0 and self.matrixdescriptors.fc[atom2] > 0:
                            self.electrostatic_edges['attract'].append([atom1, atom2])
                        elif self.matrixdescriptors.fc[atom1] > 0 and self.matrixdescriptors.fc[atom2] > 0 or self.matrixdescriptors.fc[atom1] < 0 and self.matrixdescriptors.fc[atom2] < 0:
                            self.electrostatic_edges['repel'].append([atom1, atom2])

    def FindCH_OInteractions(self):
        self.ch_o_edges = []
        for carbon in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.element[carbon] == 'C':
                for hydrogen in range(len(self.matrixdescriptors.element)):
                    if self.matrixdescriptors.element[hydrogen] == 'H' and self.matrixdescriptors.adj_mat[carbon][hydrogen] > 0:
                        for oxygen in range(len(self.matrixdescriptors.element)):
                            if self.matrixdescriptors.element[oxygen] == 'O' and carbon != oxygen:
                                distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(hydrogen).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(oxygen))
                                if distance > 0 and distance <= 3:  # typical C–H···O interaction distance cutoff
                                    self.ch_o_edges.append([hydrogen, oxygen])

    def FindDihydrogenBonds(self):
        self.dihydrogen_edges = []
        for hydrogen1 in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.element[hydrogen1] == 'H':
                for hydrogen2 in range(hydrogen1 + 1, len(self.matrixdescriptors.element)):
                    if self.matrixdescriptors.element[hydrogen2] == 'H':
                        distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(hydrogen1).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(hydrogen2))
                        if distance > 0 and distance <= 2.5:  # typical dihydrogen bond distance cutoff
                            self.dihydrogen_edges.append([hydrogen1, hydrogen2])
    
    def FindCationPiInteractions(self):
        self.cation_pi_edges = []
        for cation in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.fc[cation] > 0:
                for ring in self.ring_atoms:
                    if all(self.matrixdescriptors.element[atom] in ['C', 'N', 'O', 'S'] for atom in ring):  # check if all atoms in the ring are aromatic
                        for atom in ring:
                            distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(cation).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(atom))
                            if distance > 0 and distance <= 5:  # typical cation–π interaction distance cutoff
                                self.cation_pi_edges.append([cation, atom])

    def FindHalogenBonds(self):
        self.halogen_bond_edges = []
        halogens = ['F', 'Cl', 'Br', 'I']
        for halogen in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.element[halogen] in halogens:
                for acceptor in range(len(self.matrixdescriptors.element)):
                    if self.matrixdescriptors.element[acceptor] in ['O', 'N', 'S'] and halogen != acceptor:
                        distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(halogen).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(acceptor))
                        if distance > 0 and distance <= 3.5:  # typical halogen bond distance cutoff
                            self.halogen_bond_edges.append([halogen, acceptor])

    def FindMetallophilicInteractions(self):
        self.metallophilic_edges = []
        metals = ['Cu', 'Ag', 'Au', 'Zn', 'Cd', 'Hg', 'Pt', 'Pd']
        for metal1 in range(len(self.matrixdescriptors.element)):
            if self.matrixdescriptors.element[metal1] in metals:
                for metal2 in range(metal1 + 1, len(self.matrixdescriptors.element)):
                    if self.matrixdescriptors.element[metal2] in metals:
                        distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(metal1).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(metal2))
                        if distance > 0 and distance <= 3.5:  # typical metallophilic interaction distance cutoff
                            self.metallophilic_edges.append([metal1, metal2])

    def FindPiPiStacking(self):
        self.pi_pi_edges = []
        for ring1 in self.ring_atoms:
            if all(self.matrixdescriptors.element[atom] in ['C', 'N', 'O', 'S'] for atom in ring1):  # check if all atoms in the ring are aromatic
                for ring2 in self.ring_atoms:
                    if ring1 != ring2 and all(self.matrixdescriptors.element[atom] in ['C', 'N', 'O', 'S'] for atom in ring2):
                        distances = []
                        for atom1 in ring1:
                            for atom2 in ring2:
                                distance = self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(atom1).Distance(self.matrixdescriptors.new_mol.GetConformer().GetAtomPosition(atom2))
                                distances.append(distance)
                        min_distance = min(distances)
                        if min_distance > 0 and min_distance <= 5:  # typical π-π stacking distance cutoff
                            self.pi_pi_edges.append((ring1, ring2, min_distance))
        
        # Sort by distance and take the two most vertical and nearest ones
        self.pi_pi_edges.sort(key=lambda x: x[2])
        self.pi_pi_edges = self.pi_pi_edges[:2]


    def ElectronegativityDifference(self, edge):
        en_atom1 = self.pauling_dict.get(self.matrixdescriptors.element[edge[0]], 0)
        en_atom2 = self.pauling_dict.get(self.matrixdescriptors.element[edge[1]], 0)
        return abs(en_atom1 - en_atom2)
    
    
    
    def DefaultNBInteraction(self,edge):
        bond_feature = []
        if self.params.addmissingbonds == 'hbonds':
            bond_feature += [0]
        
        if self.params.addcho:            
            bond_feature += [0]

        if self.params.adddihydrogenbonds:
            bond_feature += [0]

        if self.params.addcationpi:
            bond_feature += [0]

        if self.params.addpipistack:
            bond_feature += [0]

        if self.params.addhalogenbonds:
            bond_feature += [0]
        
        if self.params.addmetallophilic:
            bond_feature += [0]

        if self.params.addelectrostatic:
            bond_feature += [0,0]
        
        if self.params.addeneg:
            bond_feature += [self.ElectronegativityDifference(edge)]
        return bond_feature
    



class GlobalBondInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    def EdgeCheck(self,edge):
        if self.matrixdescriptors.adj_mat[edge[0]][edge[1]] > 0:
            return True
        elif self.matrixdescriptors.adj_mat[edge[1]][edge[0]] == 0:
            return False
        else:
            return None
        
    
    def IsGlobal(self,edge):
        if self.params.getglobal:
            if self.EdgeCheck(edge):
                return self.properties.global_bond_encode['TRUE']
            else:
                return self.properties.global_bond_encode['FALSE']
        else:
            return []

    
    def ShortestPathDistance(self, edge):
        if self.params.getshortestpath:
            if self.EdgeCheck(edge):
                return [1]
            else:
                mol = Chem.MolFromSmiles(self.smiles)
                try:
                    path_length = Chem.GetDistanceMatrix(mol)[edge[0], edge[1]]
                    return [path_length]
                except:
                    return [100]  # Return a large value if the edge does not exist or an error occurs
        else:
            return []
    
    def ShortestPathDistanceWithWeights(self, edge, weights):
        if self.params.getshortestpathweighted:
            if self.EdgeCheck(edge):
                return [1]
            else:
                mol = Chem.MolFromSmiles(self.smiles)
                try:
                    distance_matrix = Chem.GetDistanceMatrix(mol)
                    atomic_masses = [atom.GetMass() for atom in mol.GetAtoms()]
                    distance = distance_matrix[edge[0], edge[1]]
                    weighted_distance = distance * (atomic_masses[edge[0]] + atomic_masses[edge[1]]) / 2
                    return [weighted_distance]
                except:
                    return [100]  # Return a large value if the edge does not exist or an error occurs
        else:
            return []
        
    def RandomWalkCommuteTime(self, edge):
        if self.params.getrandomwalk:
            try:
                rwct = nx.algorithms.approximation.rwct(self.matrixdescriptors.adj_mat, edge[0], edge[1])
                return [rwct]
            except:
                return [100]  # Return a large value if an error occurs
        else:
            return []
    
    def CommuteTimeMetrics(self, edge):
        if self.params.getcommutetimes:
            try:
                # Calculate commute time using different metrics
                rwct = nx.algorithms.approximation.rwct(self.matrixdescriptors.adj_mat, edge[0], edge[1])
                hitting_time = nx.algorithms.approximation.hitting_time(self.matrixdescriptors.adj_mat, edge[0], edge[1])
                mean_first_passage_time = nx.algorithms.approximation.mean_first_passage_time(self.matrixdescriptors.adj_mat, edge[0], edge[1])
                resistance_distance = nx.algorithms.approximation.resistance_distance(self.matrixdescriptors.adj_mat, edge[0], edge[1])
                clustering_coefficient = nx.clustering(self.matrixdescriptors.adj_mat, edge[0])
                closeness_centrality = nx.closeness_centrality(self.matrixdescriptors.adj_mat, edge[0])
                degree_centrality = nx.degree_centrality(self.matrixdescriptors.adj_mat)[edge[0]]
                return [rwct, hitting_time, mean_first_passage_time, resistance_distance,clustering_coefficient,closeness_centrality, degree_centrality]
            except:
                return [100] * 7  # Return an empty list if an error occurs
        else:
            return []  # Return an empty list if the edge does not exist

    def ShortestPathCount(self, edge):
        if self.params.getshortestpathcount:
            if self.EdgeCheck(edge):
                return [1]
            else:
                try:
                    paths = list(nx.all_shortest_paths(self.matrixdescriptors.adj_mat, source=edge[0], target=edge[1]))
                    return [len(paths)]
                except:
                    return [0]
        else:
            return []

    def EffectiveResistance(self, edge):
        if self.params.geteffectiveresistance:
            try:
                resistance = nx.resistance_distance(self.matrixdescriptors.adj_mat, edge[0], edge[1])
                return [resistance]
            except:
                return [0]
        else:
            return []

    def CommonNeighbors(self, edge):
        if self.params.getcommonneighbors:
            neighbors1 = set(self.matrixdescriptors.adj_mat[edge[0]].nonzero()[0])
            neighbors2 = set(self.matrixdescriptors.adj_mat[edge[1]].nonzero()[0])
            common_neighbors = neighbors1.intersection(neighbors2)
            return [len(common_neighbors)]
        else:
            return []

    def PercentCommonNeighbors(self, edge):
        if self.params.getpctcommonneighbors:
            neighbors1 = set(self.matrixdescriptors.adj_mat[edge[0]].nonzero()[0])
            neighbors2 = set(self.matrixdescriptors.adj_mat[edge[1]].nonzero()[0])
            common_neighbors = neighbors1.intersection(neighbors2)
            total_neighbors = neighbors1.union(neighbors2)
            if len(total_neighbors) > 0:
                return [len(common_neighbors) / len(total_neighbors)]
            else:
                return [0]
        else:
            return []

    def JaccardIndex(self, edge):
        if self.params.getglobaljaccard:
            neighbors1 = set(self.matrixdescriptors.adj_mat[edge[0]].nonzero()[0])
            neighbors2 = set(self.matrixdescriptors.adj_mat[edge[1]].nonzero()[0])
            common_neighbors = neighbors1.intersection(neighbors2)
            total_neighbors = neighbors1.union(neighbors2)
            if len(total_neighbors) > 0:
                return [len(common_neighbors) / len(total_neighbors)]
            else:
                return [0]
        else:
            return []

    def AdamicAdarIndex(self, edge):
        if self.params.getglobaladamicadar:
            neighbors1 = set(self.matrixdescriptors.adj_mat[edge[0]].nonzero()[0])
            neighbors2 = set(self.matrixdescriptors.adj_mat[edge[1]].nonzero()[0])
            common_neighbors = neighbors1.intersection(neighbors2)
            adamic_adar = sum(1 / np.log(len(self.matrixdescriptors.adj_mat[neighbor].nonzero()[0])) for neighbor in common_neighbors)
            return [adamic_adar]
        else:
            return []

    def PreferentialAttachmentIndex(self, edge):
        if self.params.getprefattachment:
            neighbors1 = set(self.matrixdescriptors.adj_mat[edge[0]].nonzero()[0])
            neighbors2 = set(self.matrixdescriptors.adj_mat[edge[1]].nonzero()[0])
            return [len(neighbors1) * len(neighbors2)]
        else:
            return []

    def ShortestPathDistanceWithPBC(self, edge):
        if self.EdgeCheck(edge) and self.params.getshortestpathpbc:
            return [1]
        elif not self.EdgeCheck(edge) and self.params.getshortestpathpbc:
            mol = Chem.MolFromSmiles(self.smiles)
            try:
                pos1 = mol.GetConformer().GetAtomPosition(edge[0])
                pos2 = mol.GetConformer().GetAtomPosition(edge[1])
                delta = pos2 - pos1
                delta -= self.params.sp_box_size * np.round(delta / self.params.sp_box_size)  # Apply periodic boundary conditions
                distance = np.linalg.norm(delta)
                return [distance]
            except:
                return [100]  # Return a large value if the edge does not exist or an error occurs
        else:
            return []
    def KatzCentralitySimilarity(self, edge, beta=0.1):
        if self.params.getkatz:
            try:
                katz_centrality = nx.katz_centrality_numpy(self.matrixdescriptors.adj_mat, beta=beta)
                return [katz_centrality[edge[0]] * katz_centrality[edge[1]]]
            except:
                return [0]
        else:
            return []

    def EigenvectorCentralityDifference(self, edge):
        if self.params.getEigenvectorCentrality:
            try:
                eigenvector_centrality = nx.eigenvector_centrality_numpy(self.matrixdescriptors.adj_mat)
                return [abs(eigenvector_centrality[edge[0]] - eigenvector_centrality[edge[1]])]
            except:
                return [0]
        else:
            return []

    def BetweennessCentralityCorrelation(self, edge):
        if self.params.getBetweennessCentralityCorrelation:
            try:
                betweenness_centrality = nx.betweenness_centrality(self.matrixdescriptors.adj_mat)
                return [betweenness_centrality[edge[0]] * betweenness_centrality[edge[1]]]
            except:
                return [0]
        else:
            return []

    def MinimumCutValue(self, edge):
        if self.params.getMinCutValue:
            try:
                cut_value, partition = nx.minimum_cut(self.matrixdescriptors.adj_mat, edge[0], edge[1])
                return [cut_value]
            except:
                return [0]
        else:
            return []

    def MaximumFlow(self, edge):
        if self.params.getMaximumFlow:
            try:
                flow_value, flow_dict = nx.maximum_flow(self.matrixdescriptors.adj_mat, edge[0], edge[1])
                return [flow_value]
            except:
                return [0]
        else:
            return []
    
    def LaplacianEigenvectorSimilarity(self, edge):
        if self.params.getLaplacianEigenvectorSimilarity:
            try:
                laplacian = nx.laplacian_matrix(self.matrixdescriptors.adj_mat).todense()
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                similarity = np.dot(eigenvectors[:, 1], eigenvectors[:, 1])
                return [similarity]
            except:
                return [0]
        else:
            return []

    def FiedlerVectorSimilarity(self, edge):
        if self.params.getFiedlerVectorSimilarity:
            try:
                laplacian = nx.laplacian_matrix(self.matrixdescriptors.adj_mat).todense()
                eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
                fiedler_vector = eigenvectors[:, 1]
                similarity = np.dot(fiedler_vector[edge[0]], fiedler_vector[edge[1]])
                return [similarity]
            except:
                return [0]
        else:
            return []

    def GraphDistanceWeightedByBondOrder(self, edge):
        if self.params.getGraphDistanceWeightedByBondOrder:
            try:
                bond_order = self.matrixdescriptors.bond_mat[edge[0], edge[1]]
                distance = self.gs[edge[0], edge[1]]
                weighted_distance = distance / bond_order
                return [weighted_distance]
            except:
                return [100]
        else:
            return []

    def BetweennessCentralityOfPathways(self, edge):
        if self.params.getBetweennessCentralityOfPathways:
            try:
                betweenness_centrality = nx.edge_betweenness_centrality(self.matrixdescriptors.adj_mat)
                return [betweenness_centrality[edge]]
            except:
                return [0]
        else:
            return []

    def RingsInSharedPath(self, edge):
        if self.params.getRingsInSharedPath:
            try:
                rings_in_path = 0
                for ring in self.ring_atoms:
                    if edge[0] in ring and edge[1] in ring:
                        rings_in_path += 1
                return [rings_in_path]
            except:
                return [0]
        else:
            return []

    def LocalAtomicEnvironmentSimilarity(self, edge):
        if self.params.getLocalAtomicEnvironmentSimilarity:
            try:
                mol = Chem.MolFromSmiles(self.smiles)
                if self.params.global_fingerprint_type == 'Morgan':
                    fp1 = Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, fromAtoms=[edge[0]])
                    fp2 = Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, fromAtoms=[edge[1]])
                elif self.params.global_fingerprint_type == 'MACCS':
                    fp1 = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol, fromAtoms=[edge[0]])
                    fp2 = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol, fromAtoms=[edge[1]])
                elif self.params.global_fingerprint_type == 'RDK':
                    fp1 = Chem.RDKFingerprint(mol, fromAtoms=[edge[0]])
                    fp2 = Chem.RDKFingerprint(mol, fromAtoms=[edge[1]])
                elif self.params.global_fingerprint_type == 'AtomPair':
                    fp1 = Chem.GetAtomPairFingerprint(mol, fromAtoms=[edge[0]])
                    fp2 = Chem.GetAtomPairFingerprint(mol, fromAtoms=[edge[1]])
                elif self.params.global_fingerprint_type == 'TopologicalTorsion':
                    fp1 = Chem.GetTopologicalTorsionFingerprintAsIntVect(mol, fromAtoms=[edge[0]])
                    fp2 = Chem.GetTopologicalTorsionFingerprintAsIntVect(mol, fromAtoms=[edge[1]])
                elif self.params.global_fingerprint_type == 'Avalon':
                    fp1 = Chem.rdMolDescriptors.GetAvalonFP(mol, fromAtoms=[edge[0]])
                    fp2 = Chem.rdMolDescriptors.GetAvalonFP(mol, fromAtoms=[edge[1]])
                elif self.params.global_fingerprint_type == 'Estate':
                    fp1 = Chem.rdMolDescriptors.GetEstateFingerprint(mol, fromAtoms=[edge[0]])
                    fp2 = Chem.rdMolDescriptors.GetEstateFingerprint(mol, fromAtoms=[edge[1]])
                elif self.params.global_fingerprint_type == 'Layered':
                    fp1 = Chem.LayeredFingerprint(mol, fromAtoms=[edge[0]])
                    fp2 = Chem.LayeredFingerprint(mol, fromAtoms=[edge[1]])
                else:
                    return [0]

                if self.params.global_similarity_metric == 'Tanimoto':
                    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Dice':
                    similarity = DataStructs.DiceSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Cosine':
                    self.params.global_similarity = DataStructs.CosineSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Sokal':
                    similarity = DataStructs.SokalSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Russel':
                    similarity = DataStructs.RusselSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Kulczynski':
                    similarity = DataStructs.KulczynskiSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'McConnaughey':
                    similarity = DataStructs.McConnaugheySimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'Asymmetric':
                    similarity = DataStructs.AsymmetricSimilarity(fp1, fp2)
                elif self.params.global_similarity_metric == 'BraunBlanquet':
                    similarity = DataStructs.BraunBlanquetSimilarity(fp1, fp2)
                else:
                    return [0]

                return [similarity]
            except:
                return [0]
        else:
            return []

    def WeightedRandomWalkAll(self, start_atom, steps):
        mol = Chem.MolFromSmiles(self.smiles)
        atomic_masses = [atom.GetMass() for atom in mol.GetAtoms()]
        current_atom = start_atom
        path = [current_atom]
        
        for _ in range(steps):
            neighbors = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(current_atom).GetNeighbors()]
            if not neighbors:
                break
            
            weights = []
            for neighbor in neighbors:
                bond = mol.GetBondBetweenAtoms(current_atom, neighbor)
                bond_order = bond.GetBondTypeAsDouble()
                weight = atomic_masses[neighbor] * bond_order
                weights.append(weight)
            
            total_weight = sum(weights)
            probabilities = [weight / total_weight for weight in weights]
            next_atom = np.random.choice(neighbors, p=probabilities)
            path.append(next_atom)
            current_atom = next_atom
        
        return path
    
    def WeightedRandomWalk(self, start_atom, steps, weight_by='atomic_mass'):
        mol = Chem.MolFromSmiles(self.smiles)
        current_atom = start_atom
        path = [current_atom]
        
        for _ in range(steps):
            neighbors = [neighbor.GetIdx() for neighbor in mol.GetAtomWithIdx(current_atom).GetNeighbors()]
            if not neighbors:
                break
            
            if weight_by == 'atomic_mass':
                atomic_masses = [atom.GetMass() for atom in mol.GetAtoms()]
                weights = [atomic_masses[neighbor] for neighbor in neighbors]
            elif weight_by == 'bond_order':
                weights = [mol.GetBondBetweenAtoms(current_atom, neighbor).GetBondTypeAsDouble() for neighbor in neighbors]
            else:
                raise ValueError("weight_by must be 'atomic_mass' or 'bond_order'")
            
            total_weight = sum(weights)
            probabilities = [weight / total_weight for weight in weights]
            next_atom = np.random.choice(neighbors, p=probabilities)
            path.append(next_atom)
            current_atom = next_atom
        
        return path
    
    def IsSameFG(self,edge):
        if self.params.getsamefg:
            fg = FunctionalGroups()
            res = fg.are_atoms_in_same_functional_group(self.new_mol, edge[0], edge[1])
            if res: 
                return self.properties.functional_group_encode['TRUE']
            else:
                return self.properties.functional_group_encode['FALSE']
        else:
            return []


class HydrogenBondInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    def HydrogenBondCheck(self,ind):
        hbond_info = []
        if self.matrixdescriptors.element[ind] in ['O', 'N', 'F', 'Cl', 'Br', 'I']:
            hbond_info.append([1, 0])  # Donor
        elif self.matrixdescriptors.element[ind] == 'H':
            for neighbor in range(len(self.matrixdescriptors.element)):
                if self.matrixdescriptors.adj_mat[ind][neighbor] > 0 and self.matrixdescriptors.element[neighbor] in ['O', 'N', 'F', 'Cl', 'Br', 'I']:
                    hbond_info.append([0, 1])  # Acceptor
                break
        else:
            hbond_info.append([0, 0])  # Neither
        return hbond_info


class GlobalReactionBondInformation(BaseReactionFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
    
    def ShortestPathChange(self, edge):
        reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
        product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
        sp_R = nx.shortest_path_length(reactant_graph, source=edge[0], target=edge[1])
        sp_P = nx.shortest_path_length(product_graph, source=edge[0], target=edge[1])
        return [sp_P - sp_R]

    def CommonNeighborCountChange(self, edge):
        neighbors_R = set(self.reactant.matrixdescriptors.adj_mat[edge[0]].nonzero()[0]).union(
            set(self.reactant.matrixdescriptors.adj_mat[edge[1]].nonzero()[0]))
        neighbors_P = set(self.product.matrixdescriptors.adj_mat[edge[0]].nonzero()[0]).union(
            set(self.product.matrixdescriptors.adj_mat[edge[1]].nonzero()[0]))
        common_neighbors_R = neighbors_R.intersection(neighbors_P)
        common_neighbors_P = neighbors_P.intersection(neighbors_R)
        return [len(common_neighbors_P) - len(common_neighbors_R)]

    def RandomWalkChange(self, edge):
        rwct_R = nx.algorithms.approximation.rwct(self.reactant.matrixdescriptors.adj_mat, edge[0], edge[1])
        rwct_P = nx.algorithms.approximation.rwct(self.product.matrixdescriptors.adj_mat, edge[0], edge[1])
        return [rwct_P - rwct_R]

    def BondPathOrderChange(self, edge):
        bo_R = self.reactant.matrixdescriptors.bond_mat[edge[0], edge[1]]
        bo_P = self.product.matrixdescriptors.bond_mat[edge[0], edge[1]]
        return [bo_P - bo_R]

    def SharedFunctionalGroupChange(self, edge):
        fg = FunctionalGroups()
        res_R = fg.are_atoms_in_same_functional_group(self.reactant.new_mol, edge[0], edge[1])
        res_P = fg.are_atoms_in_same_functional_group(self.product.new_mol, edge[0], edge[1])
        return [int(res_P) - int(res_R)]

    def ConnectivityPathDifference(self, edge):
        reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
        product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
        connectivity_R = nx.all_pairs_shortest_path_length(reactant_graph)
        connectivity_P = nx.all_pairs_shortest_path_length(product_graph)
        diff = 0
        for node in connectivity_R:
            for target, length in connectivity_R[node].items():
                if target in connectivity_P[node]:
                    diff += abs(length - connectivity_P[node][target])
                else:
                    diff += length
        return [diff]

    def ReactivityDistance(self, edge):
        reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
        product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
        rd_R = nx.shortest_path_length(reactant_graph, source=edge[0], target=edge[1])
        rd_P = nx.shortest_path_length(product_graph, source=edge[0], target=edge[1])
        return [rd_P - rd_R]

    def ElectronFlowCorrelation(self, edge):
        reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
        product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
        efc_R = nx.degree_centrality(reactant_graph)[edge[0]] * nx.degree_centrality(reactant_graph)[edge[1]]
        efc_P = nx.degree_centrality(product_graph)[edge[0]] * nx.degree_centrality(product_graph)[edge[1]]
        return [efc_P - efc_R]