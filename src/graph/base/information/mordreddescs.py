from ..base import BaseFeaturizer
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from dataclasses import dataclass,field
from rdkit.Chem.EState import EState
import networkx as nx
import numpy as np 
from mordred import _atomic_property
from .helpers.detourmatrix import detour_matrix,build_connectivity_matrix_B,build_modified_adjacency_matrix,get_pendent_matrix,compute_laplacians,get_distance_path_count_matrix
from .helpers.edgewiener import edge_wiener_index, hyper_wiener_index,edge_wiener_index_byorder,hyper_wiener_index_byorder,vertex_edge_wiener_for_vertex,vertex_edge_wiener_for_edge,vertex_edge_distance_histogram_for_vertices,vertex_edge_distance_histogram_for_edges
from .helpers.bfstree import BFSTree
from .helpers.psa import get_tpsa_contributions,compute_atomwise_logs,get_slogp_smr_contributions
from .helpers.estate import match_all_smarts
from .helpers.morse import get_morse_descriptors,get_morse_sinc_descriptors
from itertools import groupby
from .helpers.mordredtable import PeriodicTable

@dataclass
class MordredParams:
    useabc: str = None  # Options: 'reg', 'gg', None
    useabs: str = None  # Options: 'gg', None
    getbarysz: bool = False  # Whether to calculate Barysz descriptors
    getats: str = None  # Options: 'ATS', 'AATS', 'ATSD', 'AATSD', 'AATSM', 'AATSG', None
    getdetour: bool = False  # Whether to calculate detour matrix
    getburdenmat: str = None  # Options: 'reg', 'modified', None
    burdenweight: str = None  # Options: 'reg', 'atom', 'partialcharge', 'polarizability', 'ionizationpotential', 'pauling', 'sanderson', 'allred', None
    getpropsrelativetocarbon: bool = False  # Whether to calculate properties relative to carbon
    getvertexdistancedegree: bool = False  # Whether to calculate vertex distance degree
    getbalabanbondfactor: bool = False  # Whether to calculate Balaban bond factor
    getsuperdentic: bool = False  # Whether to calculate superdentic index
    geteccentricity: bool = False  # Whether to calculate eccentricity
    getschultz: bool = False  # Whether to calculate Schultz index
    getgutman: bool = False  # Whether to calculate Gutman index
    getxui: bool = False  # Whether to calculate Xui index
    gethorary: bool = False  # Whether to calculate Horary index
    getmohar: str = None  # Options: 'laplacian', None
    gethp: bool = False  # Whether to calculate HP value
    getcorecount: bool = False  # Whether to calculate core count
    getvem: bool = False  # Whether to calculate VEM descriptors
    getetacomposite: bool = False  # Whether to calculate eta composite
    getetapsi: bool = False  # Whether to calculate eta psi
    getgravity: bool = False  # Whether to calculate gravity index
    getzagreb: int = None  # Options: 2, 3, ..., None
    getharmonic: bool = False  # Whether to calculate harmonic index
    getsombor: bool = False  # Whether to calculate Sombor index
    getrandic: bool = False  # Whether to calculate Randic index
    getnirmala: bool = False  # Whether to calculate Nirmala index
    getsoss: bool = False  # Whether to calculate SOS index
    getaugmentedgraphattributes: bool = False  # Whether to calculate augmented graph attributes
    gethyperbolic: bool = False  # Whether to calculate hyperbolic index
    getaugzagreb: bool = False  # Whether to calculate augmented Zagreb index
    getklein: bool = False  # Whether to calculate Klein index
    gethyperwiener: bool = False  # Whether to calculate hyper Wiener index
    getwiener: bool = False  # Whether to calculate Wiener index
    gethdsa: bool = False  # Whether to calculate HDSA index
    getets: bool = False  # Whether to calculate ETS descriptors
    getinformationcontent: int = None  # Options: 0, 1, 2, ..., None
    getmoleculardistanceedge: bool = False  # Whether to calculate molecular distance edge
    gettopocharge: int = None  # Options: 1, 2, ..., None
    getSMR: bool = False  # Whether to calculate SMR descriptors
    SLogP: list = None  # Options: ['Li', 'logPregion'], None
    SMR: list = None  # Options: ['Ri', 'SMRregion'], None
    getMoRSE: str = None  # Options: 'cos', 'sinc', None
    getchi: int = None  # Options: 0, 1, 2, ..., None
    getedgewiener: bool = False  # Whether to calculate edge Wiener index
    getedgewienerbyorder: int = None
    getvertexadjacency: bool = False  # Whether to calculate vertex adjacency
    getTPSA: bool = False  # Whether to calculate TPSA
    getASA: bool = False  # Whether to calculate ASA
    LogS: bool = False  # Whether to calculate LogS
    getEstate: str = None  # Options: 'all', 'vectoronly', 'indicesonly', None
    getchivalence: int = None  # Options: 0, 1, 2, ..., None
    MDEvalences: list = field(default_factory=lambda: [1, 1])  # Default valences for molecular distance edge
    MDEreference: str = 'C'  # Default reference atom for molecular distance edge
    mohar: str = None
    getvewi: bool = False  # Whether to calculate vertex-edge Wiener index
    getvewibyorder: int = None  # Options: 2, 3, ..., None

    getMoRSEweights: str = 'mass'  # Weights for MoRSE descriptors
    getMoRSEbins: int = 10  # Number of bins for MoRSE descriptors



class BaseMordredFunctions(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()

    def mol_to_nx_graph(self,mol):
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), element=atom.GetSymbol(), degree=atom.GetDegree())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())
        return G

    def _atomicpropertyvector(self,ind):
        atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
        return [float(self.matrixdescriptors.gasteiger_charges[ind]),
                    _atomic_property.get_valence_electrons(atom),
                    _atomic_property.get_sigma_electrons(atom),
                    _atomic_property.get_intrinsic_state(atom),
                    _atomic_property.get_atomic_number(atom),
                    _atomic_property.get_mass(atom),
                    _atomic_property.get_sanderson_en(atom),
                    _atomic_property.get_pauling_en(atom),
                    _atomic_property.get_allred_rocow_en(atom),
                    _atomic_property.get_polarizability(atom),
                    _atomic_property.get_ionization_potential(atom),]

    def carbon_properties(self,atom):
        N = atom.GetAtomicNum()
        if N == 1:
            return 0
        _table = Chem.GetPeriodicTable()
        Zv = _table.GetNOuterElecs(N) - atom.GetFormalCharge()
        Z = atom.GetAtomicNum() - atom.GetFormalCharge()
        hi = 0
        try:
            he = sum(1 for a in atom.GetNeighbors() if a.GetAtomicNum() == 0)
        except:
            he = 0
        h = hi + he

        ve = (Zv - h) / (Z - Zv - 1)
        return ve,he,0


    def _atomicpropertyvectorwrtcarbon(self,ind):
        pv = np.array(self._atomicpropertyvector(ind))
        atom = Chem.Atom(6)
        ve,he,i = self.carbon_properties(atom)

        
        carbon = [_atomic_property.get_gasteiger_charge(atom),
                    ve,he,i,
                    _atomic_property.get_atomic_number(atom),
                    _atomic_property.get_mass(atom),
                    _atomic_property.get_sanderson_en(atom),
                    _atomic_property.get_pauling_en(atom),
                    _atomic_property.get_allred_rocow_en(atom),
                    _atomic_property.get_polarizability(atom),
                    _atomic_property.get_ionization_potential(atom),]

        carbon = np.array(carbon)
        return pv/carbon
    
    def _graphdistmatrix(self):
        mol = self.matrixdescriptors.new_mol
        num_atoms = mol.GetNumAtoms()
        dist_matrix = np.zeros((num_atoms, num_atoms), dtype=float)
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    path_length = Chem.rdmolops.GetShortestPath(mol, i, j)
                    dist_matrix[i, j] = abs(len(path_length))
                else:
                    dist_matrix[i, j] = 0
        return dist_matrix 
    
    def _getcounts(self,dist,num_atoms=None,dist_matrix=None):
        counts = 0
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    if dist_matrix[i, j] == dist:
                        counts += 1
        return counts/2
    def _getallvectors(self):
        mol = self.matrixdescriptors.new_mol
        num_atoms = mol.GetNumAtoms()
        atom_vectors = []
        for i in range(num_atoms):
            atom_vectors.append(self._atomicpropertyvector(i))
        return atom_vectors
    
    def _avgpropvector(self):
        atom_vectors = self._getallvectors()
        avg_vector = np.mean(atom_vectors, axis=0)
        return avg_vector
    
    def _morandenom(self):
        w_hat = self._avgpropvector()
        all_w = self._getallvectors()
        total_w = 0
        for w in all_w:
            w_diff = np.array(w) - np.array(w_hat)
            w_res = w_diff**2
            total_w += w_res 
        
        total_w = total_w/len(all_w)
        return total_w
    
    def _gearydenom(self):
        w_hat = self._avgpropvector()
        all_w = self._getallvectors()
        total_w = 0
        for w in all_w:
            w_diff = np.array(w) - np.array(w_hat)
            w_res = w_diff**2
            total_w += w_res 
        
        total_w = total_w/(len(all_w)-1)
        return total_w
    
    def _detourmat(self):
        G = self.mol_to_nx_graph(self.matrixdescriptors.new_mol)
        D = detour_matrix(G)
        return D
    
    def _atomicwtmat(self):
        atom_weights = []
        for atom in self.matrixdescriptors.new_mol.GetAtoms():
            atom_weights.append(atom.GetMass())
        return np.array(atom_weights)
    
    def _partialchargemat(self):
        partial_charges = []
        for atom in self.matrixdescriptors.new_mol.GetAtoms():
            partial_charges.append(atom.GetProp('_GasteigerCharge'))
        return np.array(partial_charges)
    
    def _polarizabilitymat(self):
        polarizabilities = []
        for atom in self.matrixdescriptors.new_mol.GetAtoms():
            polarizabilities.append(_atomic_property.get_polarizability(atom))
        return np.array(polarizabilities)
    
    def _ipmat(self):
        ionization_potentials = []
        for atom in self.matrixdescriptors.new_mol.GetAtoms():
            ionization_potentials.append(_atomic_property.get_ionization_potential(atom))
        return np.array(ionization_potentials)
    
    def _enmat(self,entype='pauling'):
        electronegativities = []
        for atom in self.matrixdescriptors.new_mol.GetAtoms():
            if entype == 'pauling':
                electronegativities.append(_atomic_property.get_pauling_en(atom))
            elif entype == 'sanderson':
                electronegativities.append(_atomic_property.get_sanderson_en(atom))
            elif entype == 'allred':
                electronegativities.append(_atomic_property.get_allred_rowcow_en(atom))
            else:
                electronegativities.append(0)
        return np.array(electronegativities)

    def _burdenmat(self):
        if self.params.getburdenmat == 'reg':
            B = build_connectivity_matrix_B(self.matrixdescriptors.new_mol)
        else:
            B = build_modified_adjacency_matrix(self.matrixdescriptors.new_mol)
        return B
    
    def _vertexdegree(self,ind):
        rowsum = np.sum(self.D[ind, :])
        return rowsum
    
    def _pendent(self):
        pendent_matrix = get_pendent_matrix(self.G, self.D)
        return np.array(pendent_matrix)
    
    def _eccentricity(self,ind):
        return np.max(self.D[ind,:])

    def _avgeccentricity(self):
        eccentricities = [self._eccentricity(i) for i in range(self.Natoms)]
        return np.mean(eccentricities)
    
    def _valence(self,ind):
        atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
        return atom.GetTotalValence()
    
    def _degreevector(self):
        degvector = [] 
        for atom in self.matrixdescriptors.new_mol.GetAtoms():
            degvector.append(_atomic_property.get_sigma_electrons(atom))
        return np.array(degvector)

    def _laplacian(self):
        laplacians = compute_laplacians(self.G)
        return laplacians

    def _pdmatrix(self):
        return get_distance_path_count_matrix(self.G)

    def _getbondcounts(self,edge):
        mol = self.matrixdescriptors.new_mol
        path = Chem.rdmolops.GetShortestPath(mol, edge[0], edge[1])
        sigma_bonds = 0
        pi_bonds = 0

        for i in range(len(path) - 1):
            bond = mol.GetBondBetweenAtoms(path[i], path[i + 1])
            if bond is not None:
                bond_order = bond.GetBondTypeAsDouble()
                sigma_bonds += 1  # Every bond has one sigma bond
                if bond_order > 1:
                    pi_bonds += int(bond_order - 1)  # Additional bonds are pi bonds

        return sigma_bonds, pi_bonds

    def _bondepsigma(self,atom1,atom2):
        e1 = _atomic_property.get_eta_epsilon(atom1)
        e2 = _atomic_property.get_eta_epsilon(atom2)

        sigmacheck = np.abs(e1-e2) <= .3

        if sigmacheck:
            return .5
        else:
            return .75

    def _bondeppi(self,atom1,atom2,bond):
        e1 = _atomic_property.get_eta_epsilon(atom1)
        e2 = _atomic_property.get_eta_epsilon(atom2)

        aromatic = bond.GetIsAromatic() if bond else False

        sigmacheck = np.abs(e1-e2) <= .3

        if aromatic: return 2
        else:
            if sigmacheck:
                return 1
            else:
                return 1.5

    def _getwt(self,wt,ind):
        if wt == 'mass':
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return _atomic_property.get_mass(atom)
        elif wt == 'partialcharge':
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return _atomic_property.get_gasteiger_charge(atom)
        elif wt == 'polarizability':
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return _atomic_property.get_polarizability(atom)
        elif wt == 'ionizationpotential':
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return _atomic_property.get_ionization_potential(atom)
        elif wt == 'pauling':
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return _atomic_property.get_pauling_en(atom)
        elif wt == 'sanderson':
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return _atomic_property.get_sanderson_en(atom)
        elif wt == 'allred':
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return _atomic_property.get_allred_rowcow_en(atom)
        elif wt == 'z':
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return atom.GetAtomicNum()

    def _refcheck(self,edge,V,valences=[1,1],reference='C'):
        try:
            refatom = self.properties.element_encode[reference]
        except:
            refatom = self.properties.element_encode[reference.lower()]
        if (V[edge[0]] == valences[0] and V[edge[1]] == valences[1]) or (V[edge[1]] == valences[0] and V[edge[0]] == valences[1]):
            masses = [self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[0]).GetMass(),self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[1]).GetMass()]
            if masses[0] == masses[1]:
                if masses[0] == refatom:
                    return True
        return False
    
    def _topochargemats(self):
        self.A = self.matrixdescriptors.adj_mat
        D = np.array(self.D)
        try:
            self.D_inv_square = np.linalg.inv(D**2)
            self.M = np.matmul(self.A, self.D_inv_square)
        except: 
            self.D_inv_square = np.linalg.pinv(D**2)
            self.M = np.matmul(self.A, self.D_inv_square)
    
    def _chiterm(self,atom,valence=False):
        if valence:
            return _atomic_property.get_valence_electrons(atom)
        else:
            return _atomic_property.get_sigma_electrons(atom) 
        
    def _chivector(self,valence=False):
        chi = []
        for atom in self.matrixdescriptors.new_mol.GetAtoms():
            chi.append(self._chiterm(atom,valence=valence))
        return chi 
    
    def _getgdistlocs(self,ind,dist):
        locations = np.argwhere(self.D[ind, :] == dist)[0]
        return locations
    
class MordredInformation(BaseMordredFunctions):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        self.InitializeAddons()

    def RunMordredBases(self):
        self.G = self.mol_to_nx_graph(self.matrixdescriptors.new_mol)
        self.D = self._graphdistmatrix()
        self.P = self._getallvectors()
        self.w_hat = np.array(self._avgpropvector())
        self.geary = self._gearydenom()
        self.moran = self._morandenom()
        self.Detour = self._detourmat()
        self.deg = self._degreevector()
        self.Natoms = len(self.matrixdescriptors.new_mol.GetAtoms())
        self.wtmat = self._atomicwtmat()
        self.enmat = self._enmat(self.params.burdenweight)
        self.ipmat = self._ipmat()
        self.chargemat = self._partialchargemat()
        self.polmat = self._polarizabilitymat()
        self.B = self._burdenmat()
        self.num_bonds = self.matrixdescriptors.new_mol.GetNumBonds()
        self.num_rings = self.matrixdescriptors.new_mol.GetRingInfo().NumRings()
        self.pendent = self._pendent()
        if self.params.mohar is not None and isinstance(self.params.mohar, str): 
            self.laplacian = self._laplacian()[self.params.mohar] #if mohar is a list, get a set of them. 
        elif self.params.mohar is not None and isinstance(self.params.mohar, list):
            self.laplacian = [self._laplacian()[self.params.mohar[i]] for i in range(len(self.params.mohar))]
        elif self.params.mohar is None:
            self.laplacian = None
        self.pdmat = self._pdmatrix()
        self.tpsas = get_tpsa_contributions(self.matrixdescriptors.new_mol)
        self.vsa_contribs    = list(rdMolDescriptors._CalcLabuteASAContribs(self.matrixdescriptors.new_mol)[0])
        self.logS = compute_atomwise_logs(self.matrixdescriptors.new_mol)
        self.estate_indices  = EState.EStateIndices(self.matrixdescriptors.new_mol)
        self.atom_vector = match_all_smarts(self.matrixdescriptors.new_mol)
        self._topochargemats()
        self.Li,self.Ri,self.logP,self.SMR = get_slogp_smr_contributions(self.matrixdescriptors.new_mol)
        
        self.Bmat = Chem.GetAdjacencyMatrix(self.matrixdescriptors.new_mol, useBO=True, force=True)
        self.V = self.Bmat.sum(axis=0)
        self.Vlen = len(self.V)
        self.chis = self._chivector()
        self.chivalences = self._chivector(valence=False)
        self.vewi_edges = vertex_edge_distance_histogram_for_edges(self.G)
        self.vewi_nodes = vertex_edge_distance_histogram_for_vertices(self.G)
        
    def RunMordred3D(self,id=0):
        if self.params.getMoRSE == 'cos': 
            self.morse_vector, self.contribs = get_morse_descriptors(self.matrixdescriptors.new_mol,id=id,weights=self.params.getMoRSEweights,num_bins=self.params.getMoRSEbins)
        elif self.params.getMoRSE == 'sinc':
            self.morse_vector, self.contribs = get_morse_sinc_descriptors(self.matrixdescriptors.new_mol,weights=self.params.getMoRSEweights,num_bins=self.params.getMoRSEbins)
        

    def RunAtomVector(self,ind):
        self.pv = np.array(self._atomicpropertyvector(ind))
        self.pvc = np.array(self._atomicpropertyvectorwrtcarbon(ind))
        self.atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
        self.vdeg = self._vertexdegree(ind)
        self.valence = self.pv[1]

    def RunBondVector(self,edge):
        self.du = self.deg[edge[0]]
        self.dv = self.deg[edge[1]]
        self.u = edge[0]
        self.v = edge[1]
        self.atom1 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[0])
        self.atom2 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[1])
        self.path_length = Chem.rdmolops.GetShortestPath(self.matrixdescriptors.new_mol, edge[0], edge[1])
        self.graph_distance = len(self.path_length) - 1 if self.path_length else 0
        self.totalints = self._getcounts(self.graph_distance,self.Natoms,self.D)
        self.pv1 = np.array(self._atomicpropertyvector(edge[0]))
        self.pv2 = np.array(self._atomicpropertyvector(edge[1]))
        self.wpv1 = self.pv1 - self.w_hat
        self.wpv2 = self.pv2 - self.w_hat
        self.Bval = self.B[edge[0], edge[1]]
        self.vdeg1 = self._vertexdegree(edge[0])
        self.vdeg2 = self._vertexdegree(edge[1])
        self.bond = self.matrixdescriptors.new_mol.GetBondBetweenAtoms(edge[0], edge[1])
        self.ew = edge_wiener_index(self.G, edge)
        self.hw = hyper_wiener_index(self.G, edge)

    def ABCIndex(self):
        if self.params.useabc == 'reg':
            return [np.sqrt((self.du+self.dv-2)/(self.du*self.dv))]
        elif self.params.useabs == 'gg':
            # Convert RDKit molecule to NetworkX graph
            if self.G.has_edge(self.u, self.v):
                nodes = self.G.nodes()
                du = sum(1 for x in nodes if nx.shortest_path_length(self.G, source=self.u, target=x) < nx.shortest_path_length(self.G, source=self.v, target=x))
                dv = sum(1 for x in nodes if nx.shortest_path_length(self.G, source=self.v, target=x) < nx.shortest_path_length(self.G, source=self.u, target=x))
                # To avoid division by zero
                if du == 0 or dv == 0:
                    return [0.0]
                return [np.sqrt((du+dv-2)/(du*dv))]
            else:
                return [0.0]
        else:
            return []
        
    def BaryszBond(self,bo):
        if self.params.getbarysz:
            if bo == 0:
                return [0]
            else:
                Zc = 6
                return [(1/bo)*(Zc**2)/(self.atom1.GetAtomicNum() + self.atom2.GetAtomicNum())]
        else:
            return []

    def BaryszAtom(self):
        if self.params.getbarysz:
            Zc = 6
            return [1 - Zc/self.atom.GetAtomicNum()]
        else:
            return []

    def ATSAtom(self):
        if self.params.getats == 'ATS':
            return [self.pv.dot(self.pv)]
        elif self.params.getats == 'AATS':
            return [self.pv.dot(self.pv)/self.Natoms]
        elif self.params.getats == 'ATSD':
            return [(self.pv-self.w_hat).dot((self.pv-self.w_hat))]
        elif self.params.getats == 'AATSD':
            return [(self.pv-self.w_hat).dot((self.pv-self.w_hat))/self.Natoms]
        elif self.params.getats == 'AATSM':
            return [(self.pv-self.w_hat).dot((self.pv-self.w_hat))/(self.Natoms * self.moran)]
        elif self.params.getats == 'AATSG':
            return [(self.pv-self.w_hat).dot((self.pv-self.w_hat))/(self.Natoms * self.geary * 2)]
        else: 
            return []

    def ATSBond(self):
        if self.params.getats == 'ATS':
            return [.5*self.pv1.dot(self.pv2)]
        elif self.params.getats == 'AATS':
            return [.5*(self.pv1+self.pv2)/self.totalints]
        elif self.params.getats == 'ATSD': 
            return [.5*self.wpv1.dot(self.wpv2)]
        elif self.params.getats == 'AATSD':
            return [.5*self.wpv1.dot(self.wpv2)/self.totalints]
        elif self.params.getats == 'AATSM':
            return [.5*self.pv1.dot(self.pv2)/(self.totalints * self.moran)]
        elif self.params.getats == 'AATSG':  
            return [.5*self.wpv1.dot(self.wpv2)/(2*self.totalints * self.geary)]
        else:
            return []
    
    def DetourMatrix(self,edge):
        if self.params.getdetour:
            return [self.Detour[edge[0], edge[1]]]
        else:
            return []
    
    def BurdenValue(self,edge):
        if self.params.getburdenmat is not None and self.params.burdenweight is not None:
            output = self.Bval
            wtterm = 0
            if self.params.burdenweight == 'reg':
                return [output]
            elif 'atom' in self.params.burdenweight:
                wtterm += self.wtmat[edge[0]] * self.wtmat[edge[1]]
                return [output * np.sqrt(wtterm)]
            elif 'partialcharge' in self.params.burdenweight:
                wtterm += self.chargemat[edge[0]] * self.chargemat[edge[1]]
                return [output * np.sqrt(wtterm)]
            elif 'polarizability' in self.params.burdenweight:
                wtterm += self.polmat[edge[0]] * self.polmat[edge[1]]
                return [output * np.sqrt(wtterm)]
            elif 'ionizationpotential' in self.params.burdenweight:
                wtterm += self.ipmat[edge[0]] * self.ipmat[edge[1]]
                return [output * np.sqrt(wtterm)]
            elif 'pauling' in self.params.burdenweight or 'sanderson' in self.params.burdenweight or 'allred' in self.params.burdenweight:
                wtterm += self.enmat[edge[0]] * self.enmat[edge[1]]
                return [output * np.sqrt(wtterm)]
        else:
            return []
        
    def PropsRelativetoCarbon(self):
        if self.params.getpropsrelativetocarbon:
            print('pvc',self.pvc,type(self.pvc))
            return self.pvc
        else:
            return []
 
    def VertexDistanceDegreeAtom(self):
        if self.params.getvertexdistancedegree:
            return [self.vdeg]
        else:
            return []
        
    def VertexDistanceDegreeBond(self):
        if self.params.getvertexdistancedegree:
            return [self.vdeg1*self.vdeg2]
        else:
            return []
    
    def BalabanBondFactor(self):
        if self.params.getbalabanbondfactor:
            return [1/np.sqrt(self.vdeg1*self.vdeg2)*(self.num_bonds/(self.num_rings+1))]
        else:
            return []

    def SuperdenticIndex(self,ind):
        if self.params.getsuperdentic:
            pendentrow = self.pendent[ind,:]
            nonzeros = pendentrow[pendentrow != 0]
            return [np.prod(nonzeros)]
        else:
            return [] 
        
    def Eccentricity(self,ind):
        if self.params.geteccentricity:
            pv = self._eccentricity(ind)
            return [pv,(1/self.Natoms) * np.abs(pv - self._avgeccentricity()),pv*self._valence(ind)]
        else:
            return []
         
    def Schultz(self,ind):
        if self.params.getschultz:
            mat = self.matrixdescriptors.adj_mat + self.D
            result = np.matmul(mat,self.deg)
            return [result[ind]]
        else:
            return []

    def Gutman(self,edge):
        if self.params.getgutman:
            return [self.D[edge[0], edge[1]] * self.deg[edge[0]] * self.deg[edge[1]]]
        else:
            return []
    
    def Xui(self,node):
        if self.params.getxui:
            return [self.deg[node]* self.valence,
                    self.deg[node] * self.valence**2]
        else:
            return []
    
    def Horary(self,edge):
        if self.params.gethorary:
            return [self.D[edge[0], edge[1]]/2]
        else:
            return []
        
    def Mohar(self,edge):
        if self.params.getmohar is not None and isinstance(self.params.getmohar, str):
            return [self.laplacian[edge[0], edge[1]]]
        elif self.params.getmohar is not None and isinstance(self.params.getmohar, list):
            return [self.laplacian[i][edge[0], edge[1]] for i in range(len(self.params.getmohar))]
        else:
            return []

    def HPValue(self,edge):
        if self.params.gethp:
            return [self.pdmat[edge[0], edge[1]]/2]
        else:
            return []
    
    def CoreCount(self,node):
        if self.params.getcorecount:
            return [_atomic_property.get_core_count(self.matrixdescriptors.new_mol.GetAtomWithIdx(node))]
        else:
            return []
        
    def VEMAtom(self,ind):
        if self.params.getvem:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            return [_atomic_property.get_eta_beta_sigma(atom)/2,
                    _atomic_property.get_eta_beta_non_sigma(atom)/2,
                    _atomic_property.get_eta_beta_sigma(atom) + _atomic_property.get_eta_beta_non_sigma(atom),
                    _atomic_property.get_eta_beta_sigma(atom) - _atomic_property.get_eta_beta_non_sigma(atom),
                    _atomic_property.get_eta_beta_delta(atom)]
        else:
            return []
        
    def VEMBond(self,edge):
        if self.params.getvem:
            sigma, pi = self._getbondcounts(edge)
            # Check if the bond between atom1 and atom2 is aromatic
            xij = self._bondepsigma(self.atom1, self.atom2)
            yij = self._bondeppi(self.atom1, self.atom2, self.bond)
            return [xij * sigma,yij*pi,xij*sigma - yij*pi]
        else:
            return []
    
    def EtaComposite(self,edge):
        if self.params.getetacomposite:
            g1 = _atomic_property.get_eta_gamma(self.atom1)
            g2 = _atomic_property.get_eta_gamma(self.atom2)

            
            return [g1*g2/self.D[edge[0], edge[1]]]
        else:
            return []
    
    def EtaPsi(self):
        if self.params.getetapsi:
            psi = _atomic_property.get_core_count(self.atom)/_atomic_property.get_eta_epsilon(self.atom)
            return [psi]
        else:
            return []
    
    def Gravity(self,edge):
        if self.params.getgravity:
            m1 = _atomic_property.get_mass(self.atom1)
            m2 = _atomic_property.get_mass(self.atom2)
            return [m1*m2/self.D[edge[0], edge[1]]]    
        else:
            return []
    
    def ZagrebAtom(self,ind):
        if self.params.getzagreb is not None:
            return [self.deg[ind] ** 2]
        elif isinstance(self.params.getzagreb,int) and self.params.getzagreb > 2:
            return [self.deg[ind] ** self.params.getzagreb]
        else:
            return []
    
    def ZagrebBond(self,edge):
        if self.params.getzagreb:
            return [self.deg[edge[0]] * self.deg[edge[1]],self.deg[edge[0]] + self.deg[edge[1]]]
        elif isinstance(self.params.getzagreb,int) and self.params.getzagreb > 2:
            deg = self._degreevector()
            return [(self.deg[edge[0]] * self.deg[edge[1]])**(self.params.zagreb-1),(self.deg[edge[0]] + self.deg[edge[1]])**(self.params.zagreb-1)]
        else:
            return []
        
    def Harmonic(self,edge):
        if self.params.getharmonic:
            return [2/self.D[edge[0], edge[1]],2/(self.deg[edge[0]] + self.deg[edge[1]])]
        else:
            return []
        
    def HarmonicAtom(self,ind):
        if self.params.getharmonic:
            return [2/(self.deg[ind]**2),1/(self.deg[ind]**2)]
        else:
            return []
    
    def Sombor(self,edge):
        if self.params.getsombor:
            return [np.sqrt(self.D[edge[0], edge[1]]**2 + self.deg[edge[0]]**2 + self.deg[edge[1]]**2),
                    np.sqrt(self.deg[edge[0]]**2 + self.deg[edge[1]]**2)]
        else:
            return []
    
    def SomborAtom(self,ind):
        if self.params.getsombor:
            return [np.sqrt(self.deg[ind]**2)]
        else:
            return []
    
    def Randic(self,edge):
        if self.params.getrandic:
            return [1/(self.D[edge[0], edge[1]]**(1/2) * self.deg[edge[0]]**(1/2) * self.deg[edge[1]]**(1/2)),
                    self.deg[edge[0]]**(1/2) * self.deg[edge[1]]**(1/2)]
        else:
            return []
    
    def RandicAtom(self,ind):
        if self.params.getrandic:
            return [1/(self.deg[ind]**(1/2)),1/(self.deg[ind])]

    def Nirmala(self,edge):
        if self.params.getnirmala:
            return [np.exp(np.sqrt(self.D[edge[0], edge[1]] * (self.deg[edge[0]] + self.deg[edge[1]]))),
                    np.exp(np.sqrt(self.deg[edge[0]] + self.deg[edge[1]]))]
        else:
            return []
        
    def NirmalaAtom(self,ind):
        if self.params.getnirmala:
            return [np.exp(np.sqrt(self.deg[ind]))]
        else:
            return []
    
    def ESOS(self,edge):
        if self.params.getsoss:
            return [(self.deg[edge[0]] + self.deg[edge[1]])* np.sqrt(self.deg[edge[0]]**2 + self.deg[edge[1]]**2)]
        else:
            return []
    
    def ESOSAtom(self,ind):
        if self.params.getsoss:
            return [self.deg[ind] * np.sqrt(self.deg[ind]**2)]
        else:
            return []

    def AugmentedGraphAttribute(self,edge):
        if self.params.getaugmentedgraphattributes:
            a = 2*np.sqrt(self.deg[edge[0]] * self.deg[edge[1]])/(self.deg[edge[0]] + self.deg[edge[1]])
            b = self.deg[edge[0]]/self.deg[edge[1]] + self.deg[edge[1]]/self.deg[edge[0]]
            return [a,1/a,b]
        else:
            return []
    
    def AugmentedGraphAttributeAtom(self,ind):
        if self.params.getaugmentedgraphattributes:
            a = np.sqrt(self.deg[ind])/(self.deg[ind])
            b = self.deg[ind]
            return [a,1/a]
        else:
            return []
    
    def Hyperbolic(self,edge):
        if self.params.gethyperbolic:
            return [np.exp(self.D[edge[0], edge[1]]/(self.deg[edge[0]] + self.deg[edge[1]])),
                    np.exp(self.deg[edge[0]]/(self.deg[edge[0]] + self.deg[edge[1]])),
                    np.exp(self.deg[edge[1]]/(self.deg[edge[0]] + self.deg[edge[1]])),
                    4*(self.deg[edge[0]] * self.deg[edge[1]])/(self.deg[edge[0]] + self.deg[edge[1]])**2]
        else:
            return []
    
    def HyperbolicAtom(self,ind):
        if self.params.gethyperbolic:
            return [np.exp(self.deg[ind]/self.deg[ind])]
        else:
            return []

    def AugZagreb(self,edge):
        if self.params.getaugzagreb:
            return [(self.deg[edge[0]] * self.deg[edge[1]])/(self.deg[edge[0]] + self.deg[edge[1]]-2)]
        else:
            return []
    
    def AugZagrebAtom(self,ind):
        if self.params.getaugzagreb:
            return [self.deg[ind]/(self.deg[ind]-2)]
        else:
            return []

    def Klein(self,edge):
        if self.params.getklein:
            return [.5*(self.deg[edge[0]]**2 + self.deg[edge[1]]**2) + .5*(self.deg[edge[0]] + self.deg[edge[1]])]
        else:
            return []

    def KleinAtom(self,ind):
        if self.params.getklein:
            return [.5*(self.deg[ind]**2) + .5*self.deg[ind]]
        else:
            return []  
        
    def HyperDegree(self,edge):
        if self.params.gethyperwiener:
            return [(self.deg[edge[0]] + self.deg[edge[1]])/(2**(self.deg[edge[0]] + self.deg[edge[1]])),
                    (self.deg[edge[0]] * self.deg[edge[1]])/(2**(self.deg[edge[0]] + self.deg[edge[1]]))]
        else:
            return []
    
    def HyperDegreeAtom(self,ind):
        if self.params.gethyperwiener:
            return [self.deg[ind]/(2**self.deg[ind])]
        else:
            return []
        
    def HyperWiener(self,edge):
        if self.params.gethyperwiener:
            return [(self.D[edge[0], edge[1]])/(2**(self.D[edge[0], edge[1]])),
                    (self.D[edge[0], edge[1]])/(2**(self.D[edge[0], edge[1]]))]
        else:
            return []
    
    def Wiener(self,edge):
        if self.params.getwiener:
            return [self.D[edge[0], edge[1]]]
        else:
            return []
    
    def HDSA(self):
        if self.params.gethdsa:
            charge = _atomic_property.get_gasteiger_charge(self.atom)
            Sd = _atomic_property.get_intrinsic_state(self.atom)
            Stot = 0
            for atom in self.matrixdescriptors.new_mol.GetAtoms():
                Stot += _atomic_property.get_intrinsic_state(self.atom)
            return [charge*Sd**.5/Stot]
        else:
            return []

    def ETSBond(self,edge):
        if self.params.getets:
            atom1 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[0])
            atom2 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[1])
            Ii = _atomic_property.get_intrinsic_state(atom1)
            Ij = _atomic_property.get_intrinsic_state(atom2)
            return [np.abs(Ii-Ij)/self.D[edge[0], edge[1]]]
        else:
            return []
    
    def ETSAtom(self,ind):
        if self.params.getets:
            Ii = _atomic_property.get_intrinsic_state(self.atom)

            bonds = []
            for bond in self.matrixdescriptors.new_mol.GetBonds():
                if bond.GetBeginAtomIdx() == ind or bond.GetEndAtomIdx() == ind:
                    bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

            if len(bonds) == 0:
                return [Ii]
            else:
                Ib = 0
                for b in bonds:
                    Ib += self.ETSBond(b)[0]
                return [Ii+Ib]
        else:
            return []
    
    def InformationContent(self,ind):
        total = sum(self.deg)
        if self.params.getinformationcontent is not None and self.params.getinformationcontent == 0:
            pi = self.deg[ind]/total
            A = self.matrixdescriptors.new_mol.GetNumAtoms()
            B = sum(b.GetBondTypeAsDouble() for b in self.matrixdescriptors.new_mol.GetBonds())
            se = np.log2(pi) * pi
            return [pi, se, A*se,se/np.log2(A),se/np.log2(B)]
        elif self.params.getinformationcontent > 0:
            tree = BFSTree(self.matrixdescriptors.new_mol)
            atoms = [
                tree.get_code(i, self._order) for i in range(self.matrixdescriptors.new_mol.GetNumAtoms())
            ]
            ad = {a: i for i, a in enumerate(atoms)}
            Ags = [(k, sum(1 for _ in g)) for k, g in groupby(sorted(atoms))]
            Nags = len(Ags)
            pi = np.fromiter((ag for _, ag in Ags), "float", Nags)
            A = self.matrixdescriptors.new_mol.GetNumAtoms()
            B = sum(b.GetBondTypeAsDouble() for b in self.matrixdescriptors.new_mol.GetBonds())
            se = np.log2(pi) * pi
            return [pi, se, A*se,se/np.log2(A),se/np.log2(B)]
        else:
            return []

    def WeightedInformationContent(self,ind):
        w = [self._getwt(self.params.IC_weight, i) for i in range(len(self.deg))]
        total = np.array(w).dot(np.array(self.deg))
        if self.params.getweightedinformationcontent is not None and self.params.getweightedinformationcontent == 0:
            pi = self.deg[ind]/total
            A = self.matrixdescriptors.new_mol.GetNumAtoms()
            B = sum(b.GetBondTypeAsDouble() for b in self.matrixdescriptors.new_mol.GetBonds())
            se = np.log2(pi) * pi
            return [pi, se, A*se,se/np.log2(A),se/np.log2(B)]
        elif self.params.getweightedinformationcontent > 0: #adapt this for weight
            tree = BFSTree(self.matrixdescriptors.new_mol)
            atoms = [
                tree.get_code(i, self._order) for i in range(self.matrixdescriptors.new_mol.GetNumAtoms())
            ]
            Ags = [(k, sum(1 for _ in g)) for k, g in groupby(sorted(atoms))]
            Nags = len(Ags)
            pi = np.fromiter((ag for _, ag in Ags), "float", Nags)
            A = self.matrixdescriptors.new_mol.GetNumAtoms()
            B = sum(b.GetBondTypeAsDouble() for b in self.matrixdescriptors.new_mol.GetBonds())
            se = np.log2(pi) * pi
            return [pi, se, A*se,se/np.log2(A),se/np.log2(B)]
        else:
            return []

    def EdgeWiener(self):
        if self.params.getedgewiener:
            return [self.ew,self.hw]
    
    def EdgeWienerByOrder(self,edge):
        if self.params.getedgewienerbyorder > 1:
            output = []
            for ord in range(1,self.params.getedgewienerbyorder+1):
                try: 
                    output += [edge_wiener_index_byorder(self.G, edge, order=ord)/self.ew,hyper_wiener_index_byorder(self.G, edge, order=ord)/self.hw]
                except:
                    output += [0,0]
            return output
        else:
            return []

    def VEWIAtom(self,ind):
        if self.params.getvewi:
            return [vertex_edge_wiener_for_vertex(self.G,ind)]
        else:
            return [] 
    def VEWIBond(self,edge):
        if self.params.getvewi:
            return [vertex_edge_wiener_for_edge(self.G,edge)]
        else:
            return []

    def VEWIAtomByOrder(self,ind):
        if self.params.getvewibyorder > 1:
            output = []
            for order in range(1,self.params.getvewibyorder+1):
                try:
                    res = self.vewi_nodes[ind][order-1]/vertex_edge_wiener_for_vertex(self.G,ind)
                except:
                    res = 0
                output += [res]
            return output
        else:
            return []

    def VEWIBondByOrder(self,edge):
        if self.params.getvewibyorder > 1:
            output = []
            for ord in range(1,self.params.getvewibyorder+1):
                try:
                    output += [self.vewi_edges[tuple(edge)][ord-1]/vertex_edge_wiener_for_edge(self.G,edge)]
                except:
                    output += [1]
            return output
        else:
            return []

    def VertexAdjacency(self):
        if self.params.getvertexadjacency:
            heavy_bond_count = 0
            for bond in self.atom.GetBonds():
                if bond.GetBeginAtom().GetAtomicNum() > 1 and bond.GetEndAtom().GetAtomicNum() > 1:
                    heavy_bond_count += 1
            return [1+ np.log2(heavy_bond_count)]
        else:
            return []
    
    def TPSA(self,ind):
        if self.params.getTPSA: 
            return [self.tpsas[0][ind]]
        else:
            return []
        
    def LabuteASA(self,ind):
        if self.params.getASA:
            return [self.vsa_contribs[ind]]
        else:
            return []
            
    def LogS(self,ind):
        if self.params.LogS:
            
            return [self.logS[ind]]
        else:
            return []

    def EState(self,ind):
        if self.params.getEstate == 'all': 
            return [self.estate_indices[ind]] + self.atom_vector[ind]
        elif self.params.getEstate == 'vectoronly':
            return self.atom_vector[ind]
        elif self.params.getEstate == 'indicesonly':
            return [self.estate_indices[ind]]
        else:
            return [] 
    
    def MolecularDistanceEdge(self,edge):
        if self.params.getmoleculardistanceedge:
            refcheck = self._refcheck(edge,self.V,valences=self.params.MDEvalences,reference=self.params.MDEreference)
            if refcheck: 
                return [self.D[edge[0], edge[1]]**(.5/self.Vlen)]
            else:
                return [0]
        else:
            return []

    def TopoChargeBond(self,edge):
        if self.params.gettopocharge:
            CT = self.M[edge[0], edge[1]] - self.M[edge[1], edge[0]]
            return [self.D_inv_square[edge[0], edge[1]],self.M[edge[0], edge[1]],self.M[edge[1], edge[0]],CT]
        else:
            return []

    def _TopoChargeAtomFunc(self,ind,dist=1):
        D = np.array(self.D)
        locations = np.argwhere(D[ind, :] == dist)[0]
        result = dict()
        result['Dinv'] = 0
        result['CT'] = 0
        for location in locations:
            result['Dinv'] += self.D_inv_square[ind, location]
            result['CT'] += self.M[ind, location] - self.M[location, ind]
        return result
                
    def TopoChargeAtom(self,ind):
        if self.params.gettopocharge == 1:
            try:
                result = self._TopoChargeAtomFunc(ind)
                return [result['Dinv'],result['CT']]
            except:
                return [0,0]
        elif self.params.gettopocharge > 1:
            output = [] 
            for i in range(1,self.params.gettopocharge):
                try: 
                    result = self._TopoChargeAtomFunc(ind,dist=i)
                    output += [result['Dinv'],result['CT']]
                except:
                    output += [0,0]
            return output
        else:
            return []

    def SMRandSLogP(self,ind):
        if self.params.getSMR:
            output = [] 
            if 'Li' in self.params.getSMR:
                output.append(self.Li[ind])
            if 'Ri' in self.params.getSMR:
                output.append(self.Ri[ind])
            if 'logPregion' in self.params.getSMR:
                output += self.logP[ind]
            if 'SMRregion' in self.params.getSMR:
                output += self.SMR[ind]    
            return output        
        else:
            return []

    def Morse(self,edge):
        if self.params.getMoRSE == 'cos':
            return list(self.contribs[:,edge[0],edge[1]])
        elif self.params.getMoRSE == 'sin':
            return list(self.contribs[:,edge[0],edge[1]])

    def ChiAtom(self,ind):
        if self.params.getchi == 0:
            try:
                return [1/np.sqrt(self.chis[ind])]
            except:
                return [0]
        elif self.params.getchi >= 1:
            try:
                chi1 = 1/np.sqrt(self.chis[ind])
                output = [chi1]
                for ord in range(1,self.params.getchi):
                    try:
                        locations = self._getgdistlocs(ind,ord)
                        chipaths = 0
                        for loc in locations:
                            try:
                                path = Chem.rdmolops.GetShortestPath(self.matrixdescriptors.new_mol, ind, int(loc))
                                intermediate_atoms = path[1:-1]  # Exclude ind and loc
                                chipath = chi1
                                for atom in intermediate_atoms:
                                    deltaj = 1/np.sqrt(self.chis[atom])
                                    chipath = chipath * deltaj
                            except:
                                chipath = 0
                            chipaths += chipath
                        output.append(chipaths)    
                    except:
                        chipaths = 0
                return output
            except:
                return [0] * self.params.getchi
        else:
            return [] 
    
    def ChiAtomValence(self,ind):
        if self.params.getchivalence == 0:
            try:
                return [1/np.sqrt(self.chivalences[ind])]
            except:
                return [0]
        elif self.params.getchivalence >= 1:
            try:
                chi1 = 1/np.sqrt(self.chivalences[ind])
                output = [chi1]
                for ord in range(1,self.params.getchi):
                    try:
                        locations = self._getgdistlocs(ind,ord)
                        chipaths = 0
                        for loc in locations:
                            try:
                                path = Chem.rdmolops.GetShortestPath(self.matrixdescriptors.new_mol, ind, int(loc))
                                intermediate_atoms = path[1:-1]  # Exclude ind and loc
                                chipath = chi1
                                for atom in intermediate_atoms:
                                    deltaj = 1/np.sqrt(self.chis[atom])
                                    chipath = chipath * deltaj
                            except:
                                chipath = 0
                            chipaths += chipath
                    except:
                        chipaths = 0
                    output.append(chipaths)    

                return output
            except:
                return [0] * self.params.getchivalence
        else:
            return [] 
        
    def ChiBond(self,edge):
        if self.params.getchi is not None:
            deltai = self.chis[edge[0]]
            deltaj = self.chis[edge[1]]
            chiij = 1/np.sqrt(deltai*deltaj)
            chi1 = 1/np.sqrt(deltai)
            if self.bond is None:
                path = Chem.rdmolops.GetShortestPath(self.matrixdescriptors.new_mol, edge[0], edge[1])
                intermediate_atoms = path[1:-1]  # Exclude edge[0] and edge[1]
            else:
                intermediate_atoms = []
            chipath = chi1
            for atom in intermediate_atoms:
                deltaj = 1/np.sqrt(self.chis[atom])
                chipath = chipath * deltaj
            return [chiij,chipath]
        else:
            return []
        
    def ChiBondValence(self,edge): 
        if self.params.getchivalence is not None:
            deltai = self.chivalences[edge[0]]
            deltaj = self.chivalences[edge[1]]
            chiij = 1/np.sqrt(deltai*deltaj)
            chi1 = 1/np.sqrt(deltai)
            if self.bond is None:
                path = Chem.rdmolops.GetShortestPath(self.matrixdescriptors.new_mol, edge[0], edge[1])
                intermediate_atoms = path[1:-1]  # Exclude edge[0] and edge[1]
            else:
                intermediate_atoms = []
            chipath = chi1
            for atom in intermediate_atoms:
                deltaj = 1/np.sqrt(self.chivalences[atom])
                chipath = chipath * deltaj
            return [chiij,chipath]
        else:
            return []