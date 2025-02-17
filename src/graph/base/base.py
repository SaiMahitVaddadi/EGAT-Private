from ...utils.descriptors.egat.encodings import Encodings
from ...utils.descriptors.egat.molmatdesc import MolMatDesc
from ...utils.descriptors.egat.radicals import RDKElectronInfo,YARPElectronInfo
from ...utils.descriptors.egat.stereo import StereoChemistry
from ...utils.matrices.graph_seps import graph_seps
from ...utils.misc.taffi_functions import return_rings,adjmat_to_adjlist
from rdkit import Chem

class BaseFeaturizer: 
    def __init__(self, smiles, arguments):
        self.smiles = smiles
        self.params = arguments
        self.properties = Encodings()
        self.CreateAtomMapping()
        self.MatrixDescriptors()
        self.Rings()
        self.InitializeAddons()
        self.Eneg()

    def Eneg(self):
        with open('../utils/database/pauling.txt', 'r') as file:
            pauling_data = file.readlines()
        
        self.pauling_dict = {}
        for line in pauling_data:
            element, electronegativity = line.strip().split()
            self.pauling_dict[element] = float(electronegativity)

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
        self.stereo = StereoChemistry(self.matrixdescriptors,v2=self.params.stereo_full)
        self.stereo.run()


    def CheckMapping(self,ind):
        atom_mappings = min([atom.GetAtomMapNum() for atom in self.new_mol.GetAtoms()])
        #print(f"ind: {ind}, Rsmiles: {Rsmiles}, molecule: {molecule}, Generated Atom Mapping: {atom_mappings}\n")
        if atom_mappings == 0:
            ind_in_mol = ind
        else:
            ind_in_mol = ind+1
        return ind
    
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

