from ...utils.descriptors.egat.encodings import Encodings
from ...utils.descriptors.egat.properties import Properties
from ...utils.descriptors.egat.molmatdesc import MolMatDesc
from ...utils.descriptors.egat.radicals import RDKElectronInfo,YARPElectronInfo
from ...utils.descriptors.egat.stereo import StereoChemistry
from ...utils.matrices.graph_seps import graph_seps
from ...utils.misc.taffi_functions import return_rings,adjmat_to_adjlist
from rdkit import Chem
from dataclasses import dataclass
from typing import Optional
from rdkit.Chem import rdchem


@dataclass
class BaseReactionParams:
    getradical: Optional[str] = None  # Options: 'RDKit', 'YARP' - Determines the method for calculating radicals
    stereo_full: Optional[bool] = False  # Options: True, False - Whether to use full stereochemistry in calculations
    acidbase: Optional[str] = None  # Options: 'Lewis', 'BL' - Specifies the method for identifying acid and base sites
    

class BaseFeaturizer(object): 
    def __init__(self, smiles, arguments):
        self.smiles = smiles
        self.params = arguments
        self.properties = Encodings()
        self.props = Properties()
        self.CreateAtomMapping()
        self.MatrixDescriptors()
        self.Rings()
        self.InitializeAddons()
        self.Eneg()
        self.Stereochem()
        self.Radicals()


    def Eneg(self):
        self.pauling_dict = {
            'H': 2.2, 'He': None, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': None,
            'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.9, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': None, 'K': 0.82, 'Ca': 1.0,
            'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.9, 'Zn': 1.65,
            'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 3.0, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33,
            'Nb': 1.6, 'Mo': 2.16, 'Tc': 1.9, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.2, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96,
            'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 2.6, 'Cs': 0.79, 'Ba': 0.89, 'La': 1.1, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14,
            'Pm': None, 'Sm': 1.17, 'Eu': None, 'Gd': 1.2, 'Tb': None, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': None,
            'Lu': 1.27, 'Hf': 1.3, 'Ta': 1.5, 'W': 2.36, 'Re': 1.9, 'Os': 2.2, 'Ir': 2.2, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.0,
            'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02, 'Po': 2.0, 'At': 2.2, 'Rn': None, 'Fr': 0.7, 'Ra': 0.9, 'Ac': 1.1, 'Th': 1.3,
            'Pa': 1.5, 'U': 1.38, 'Np': 1.36, 'Pu': 1.28, 'Am': 1.3, 'Cm': 1.3, 'Bk': 1.3, 'Cf': 1.3, 'Es': 1.3, 'Fm': 1.3,
            'Md': 1.3, 'No': 1.3
        }

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
        self.ring_atoms = return_rings(adjmat_to_adjlist(self.matrixdescriptors.adj_mat),max_size=20,remove_fused=True) # add an argument here for max_size and reomoving fused rings


    def Radicals(self):
        if self.params.getradical == 'RDKit':
            self.electroninfo = RDKElectronInfo(self.matrixdescriptors)
            self.electroninfo.run()
        elif self.params.getradical == 'YARP':
            self.electroninfo = YARPElectronInfo(self.matrixdescriptors)
            self.electroninfo.run()
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
        self.stereo = StereoChemistry(self.matrixdescriptors,v2=self.params.stereo_full)
        self.stereo.run()


    def CheckMapping(self,ind):
        atom_mappings = min([atom.GetAtomMapNum() for atom in self.matrixdescriptors.new_mol.GetAtoms()])
        #print(f"ind: {ind}, Rsmiles: {Rsmiles}, molecule: {molecule}, Generated Atom Mapping: {atom_mappings}\n")
        if atom_mappings == 0:
            ind_in_mol = ind
        else:
            ind_in_mol = ind+1
        return ind
    
    def AcidBaseSitesBO(self):
        self.acid_sites = []
        self.base_sites = []

        for atom in self.matrixdescriptors.new_mol.GetAtoms():
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

        for atom in self.matrixdescriptors.new_mol.GetAtoms():
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


    def GrabConformer(self,id=1):
        if id == 1: 
            self.conformer = self.matrixdescriptors.new_mol
        else:
            self.conformer = self.matrixdescriptors.new_mol.GetConformer(conf_id=id)
        
            
    

    def GrabGeometry(self,id=1):
        mol = self.conformer
        self.positions = {}
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            ind = atom.GetIdx()
            self.positions[ind] = [pos.x, pos.y, pos.z]
        
        