from ..base import BaseFeaturizer
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from dataclasses import dataclass
from rdkit.Chem.EState import EState
import networkx as nx
import numpy as np 
from mordreddescs import _atomic_property
from .helpers.detourmatrix import detour_matrix,build_connectivity_matrix_B,build_modified_adjacency_matrix,get_pendent_matrix,compute_laplacians,get_distance_path_count_matrix
from .helpers.edgewiener import edge_wiener_index, hyper_wiener_index
from .helpers.bfstree import BFSTree
from .helpers.psa import get_tpsa_contributions,compute_atomwise_logs,get_slogp_smr_contributions
from .helpers.estate import match_all_smarts
from .helpers.morse import get_morse_descriptors,get_morse_sinc_descriptors
from itertools import groupby

class MordredInformation(BaseFeaturizer):
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

    def ABCIndex(self,edge):
        if self.params.useabc == 'reg':
            du = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[0]).GetDegree()
            dv = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[1]).GetDegree()
            return [np.sqrt((du+dv-2)/(du*dv))]
        elif self.params.useabs == 'gg':
            # Convert RDKit molecule to NetworkX graph
            G = self.mol_to_nx_graph(self.matrixdescriptors.new_mol)
            u = edge[0]
            v = edge[v]
            if G.has_edge(u, v):
                nodes = G.nodes()
                du = sum(1 for x in nodes if nx.shortest_path_length(G, source=u, target=x) < nx.shortest_path_length(G, source=v, target=x))
                dv = sum(1 for x in nodes if nx.shortest_path_length(G, source=v, target=x) < nx.shortest_path_length(G, source=u, target=x))
    
                # To avoid division by zero
                if du == 0 or dv == 0:
                    return [0.0]
                return [np.sqrt((du+dv-2)/(du*dv))]
            else:
                return [0.0]
        else:
            return []
        
    
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
                    _atomic_property.get_allred_rowcow_en(atom),
                    _atomic_property.get_polarizability(atom),
                    _atomic_property.get_ioniziation_potential(atom),]

    def _atomicpropertyvectorwrtcarbon(self,ind):
        pv = np.array(self._atomicpropertyvector(ind))
        atom = Chem.Atom(6)
        carbon = [_atomic_property.get_gasteiger_charge(atom),
                    _atomic_property.get_valence_electrons(atom),
                    _atomic_property.get_sigma_electrons(atom),
                    _atomic_property.get_intrinsic_state(atom),
                    _atomic_property.get_atomic_number(atom),
                    _atomic_property.get_mass(atom),
                    _atomic_property.get_sanderson_en(atom),
                    _atomic_property.get_pauling_en(atom),
                    _atomic_property.get_allred_rowcow_en(atom),
                    _atomic_property.get_polarizability(atom),
                    _atomic_property.get_ioniziation_potential(atom),]

        carbon = np.array(carbon)
        return pv/carbon
    

    def BaryszBond(self,edge,bo):
        if self.params.getbarysz:
            if bo == 0:
                return [0]
            else:
                Zc = 6
                atom1 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[0])
                atom2 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[1])
                return [(1/bo)*(Zc**2)/(atom1.GetAtomicNum() + atom2.GetAtomicNum())]
        else:
            return []

    def BaryszAtom(self,ind):
        if self.params.getbarysz:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            Zc = 6
            return [1 - Zc/atom.GetAtomicNum()]
        else:
            return []

    def _graphdistmatrix(self,dist=2):
        mol = self.matrixdescriptors.new_mol
        num_atoms = mol.GetNumAtoms()
        dist_matrix = np.zeros((num_atoms, num_atoms), dtype=float)
        counts = 0 
        for i in range(num_atoms):
            for j in range(num_atoms):
                if i != j:
                    path_length = Chem.rdmolops.GetShortestPath(mol, i, j)
                    dist_matrix[i, j] = len(path_length) - 1
                else:
                    dist_matrix[i, j] = 0
                if dist_matrix[i, j] == dist:
                    counts += 1
        return counts,dist_matrix 
    
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
    
    def ATSAtom(self,ind):
        if self.getats == 'ATS':
            pv = np.array(self._atomicpropertyvector(ind))
            return [pv**2]
        elif self.getats == 'AATS':
            pv = np.array(self._atomicpropertyvector(ind))
            return [pv/len(self.matrixdescriptors.new_mol.GetAtoms())]
        elif self.getats == 'ATSD':
            pv = np.array(self._atomicpropertyvector(ind))
            w_hat = np.array(self._avgpropvector())    
            return [(pv-w_hat)**2]
        elif self.getats == 'AATSD':
            pv = np.array(self._atomicpropertyvector(ind))
            w_hat = np.array(self._avgpropvector())    
            return [(pv-w_hat)**2/len(self.matrixdescriptors.new_mol.GetAtoms())]
        elif self.getats == 'AATSM':
            pv = np.array(self._atomicpropertyvector(ind))
            w_hat = np.array(self._avgpropvector())  
            denom = len(self.matrixdescriptors.new_mol.GetAtoms()) * self._morandenom()  
            return [(pv-w_hat)**2/denom]
        elif self.getats == 'AATSG':
            pv = np.array(self._atomicpropertyvector(ind))
            w_hat = np.array(self._avgpropvector())  
            denom = 2 * len(self.matrixdescriptors.new_mol.GetAtoms()) * self._gearydenom()  
            return [(pv-w_hat)**2/denom]
        else:
            return []

    def ATSBond(self,edge):
        if self.getats == 'ATS':
            pv1 = np.array(self._atomicpropertyvector(edge[0]))
            pv2 = np.array(self._atomicpropertyvector(edge[1]))
            return [.5*pv1.dot(pv2)]
        elif self.getats == 'AATS':
            pv1 = np.array(self._atomicpropertyvector(edge[0]))
            pv2 = np.array(self._atomicpropertyvector(edge[1]))
            mol = self.matrixdescriptors.new_mol
            path_length = Chem.rdmolops.GetShortestPath(mol, edge[0], edge[1])
            graph_distance = len(path_length) - 1 if path_length else 0
            totalints,_ = self._graphdistmatrix(graph_distance)
            return [.5*(pv1+pv2)/totalints]
        elif self.getats == 'ATSD':
            pv1 = np.array(self._atomicpropertyvector(edge[0]))
            pv2 = np.array(self._atomicpropertyvector(edge[1]))
            w_hat = np.array(self._avgpropvector())
            pv1 = pv1 - w_hat
            pv2 = pv2 - w_hat    
            return [.5*pv1.dot(pv2)]
        elif self.getats == 'AATSD':
            pv1 = np.array(self._atomicpropertyvector(edge[0]))
            pv2 = np.array(self._atomicpropertyvector(edge[1]))
            w_hat = np.array(self._avgpropvector())
            pv1 = pv1 - w_hat
            pv2 = pv2 - w_hat    
            mol = self.matrixdescriptors.new_mol
            path_length = Chem.rdmolops.GetShortestPath(mol, edge[0], edge[1])
            graph_distance = len(path_length) - 1 if path_length else 0
            totalints,_ = self._graphdistmatrix(graph_distance)
            return [.5*pv1.dot(pv2)/totalints]
        elif self.getats == 'AATSM':
            pv1 = np.array(self._atomicpropertyvector(edge[0]))
            pv2 = np.array(self._atomicpropertyvector(edge[1]))
            w_hat = np.array(self._avgpropvector())
            totalints,_ = self._graphdistmatrix(graph_distance)
            denom = totalints * self._morandenom()
            pv1 = pv1 - w_hat
            pv2 = pv2 - w_hat    
            return [.5*pv1.dot(pv2)/denom]
        elif self.getats == 'AATSG':
            pv1 = np.array(self._atomicpropertyvector(edge[0]))
            pv2 = np.array(self._atomicpropertyvector(edge[1]))
            w_hat = np.array(self._avgpropvector())
            totalints,_ = self._graphdistmatrix(graph_distance)
            denom = 2 * totalints * self._gearydenom()
            pv1 = pv1 - w_hat
            pv2 = pv2 - w_hat    
            return [.5*pv1.dot(pv2)/denom]
        else:
            return []
    
    def _detourmat(self):
        G = self.mol_to_nx_graph(self.matrixdescriptors.new_mol)
        D = detour_matrix(G)
        return D
    
    def DetourMatrix(self,edge):
        if self.getdetour:
            D = self._detourmat()
            return [D[edge[0], edge[1]]]
        else:
            return []
    

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
            ionization_potentials.append(_atomic_property.get_ioniziation_potential(atom))
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
        return np.array(electronegativities)


    def _burdenmat(self):
        if self.params.getburdenmat == 'reg':
            B = build_connectivity_matrix_B(self.matrixdescriptors.new_mol)
        else:
            B = build_modified_adjacency_matrix(self.matrixdescriptors.new_mol)
        return B

    def BurdenValue(self,edge):
        if self.params.getburdenmat is not None and self.params.burdenweight is not None:
            B = self._burdenmat()
            Bval = B[edge[0], edge[1]]
            if self.params.burdenweight == 'reg':
                return [Bval]
            elif self.params.burdenweight == 'atom':
                atom_weights = self._atomicwtmat()
                return [Bval * atom_weights[edge[0]] * atom_weights[edge[1]]]
            elif self.params.burdenweight == 'partialcharge':
                partial_charges = self._partialchargemat()
                return [Bval * partial_charges[edge[0]] * partial_charges[edge[1]]]
            elif self.params.burdenweight == 'polarizability':
                polarizabilities = self._polarizabilitymat()
                return [Bval * polarizabilities[edge[0]] * polarizabilities[edge[1]]]
            elif self.params.burdenweight == 'ionizationpotential':
                ionization_potentials = self._ipmat()
                return [Bval * ionization_potentials[edge[0]] * ionization_potentials[edge[1]]]
            elif self.params.burdenweight == 'pauling':
                en = self._enmat(entype='pauling')
                return [Bval * en[edge[0]] * en[edge[1]]]
            elif self.params.burdenweight == 'sanderson':
                en = self._enmat(entype='sanderson')
                return [Bval * en[edge[0]] * en[edge[1]]]
            elif self.params.burdenweight == 'allred':
                en = self._enmat(entype='allred')
                return [Bval * en[edge[0]] * en[edge[1]]]
            elif self.params.burdenweight == 'atomandpartialcharge':
                atom_weights = self._atomicwtmat()
                partial_charges = self._partialchargemat()
                return [Bval * atom_weights[edge[0]] * atom_weights[edge[1]] * partial_charges[edge[0]] * partial_charges[edge[1]]]
            elif self.params.burdenweight == 'atomandpolarizability':
                atom_weights = self._atomicwtmat()
                polarizabilities = self._polarizabilitymat()
                return [Bval * atom_weights[edge[0]] * atom_weights[edge[1]] * polarizabilities[edge[0]] * polarizabilities[edge[1]]]
            elif self.params.burdenweight == 'partialchargeandpolarizability':
                partial_charges = self._partialchargemat()
                polarizabilities = self._polarizabilitymat()
                return [Bval * partial_charges[edge[0]] * partial_charges[edge[1]] * polarizabilities[edge[0]] * polarizabilities[edge[1]]]
            elif self.params.burdenweight == 'atomandionizationpotential':
                atom_weights = self._atomicwtmat()
                ionization_potentials = self._ipmat()
                return [Bval * atom_weights[edge[0]] * atom_weights[edge[1]] * ionization_potentials[edge[0]] * ionization_potentials[edge[1]]]
            elif self.params.burdenweight == 'partialchargeandionizationpotential':
                partial_charges = self._partialchargemat()
                ionization_potentials = self._ipmat()
                return [Bval * partial_charges[edge[0]] * partial_charges[edge[1]] * ionization_potentials[edge[0]] * ionization_potentials[edge[1]]]
            elif self.params.burdenweight == 'polarizabilityandionizationpotential':
                polarizabilities = self._polarizabilitymat()
                ionization_potentials = self._ipmat()
                return [Bval * polarizabilities[edge[0]] * polarizabilities[edge[1]] * ionization_potentials[edge[0]] * ionization_potentials[edge[1]]]
            elif self.params.burdenweight == 'all':
                atom_weights = self._atomicwtmat()
                partial_charges = self._partialchargemat()
                polarizabilities = self._polarizabilitymat()
                ionization_potentials = self._ipmat()
                en = self._enmat(entype='pauling')
                return [Bval * atom_weights[edge[0]] * atom_weights[edge[1]] * partial_charges[edge[0]] * partial_charges[edge[1]] * polarizabilities[edge[0]] * polarizabilities[edge[1]] * ionization_potentials[edge[0]] * ionization_potentials[edge[1]] * en[edge[0]] * en[edge[1]]]
        else:
            return []
        


    def PropsRelativetoCarbon(self,ind):
        if self.params.getpropsrelativetocarbon:
            pv = self._atomicpropertyvectorwrtcarbon(ind)
            return pv
        else:
            return []
    

    def _vertexdegree(self,ind):
        _,distmat = self._graphdistmatrix(dist=1)
        rowsum = np.sum(distmat[ind, :])
        return rowsum
    
    def VertexDistanceDegreeAtom(self,ind):
        if self.params.getvertexdistancedegree:
            pv = self._vertexdegree(ind)
            return [pv]
        else:
            return []
        
    def VertexDistanceDegreeBond(self,edge):
        if self.params.getvertexdistancedegree:
            pv1 = self._vertexdegree(edge[0])
            pv2 = self._vertexdegree(edge[1])
            return [pv1*pv2]
        else:
            return []
    
    def BalabanBondFactor(self,edge):
        if self.params.getbalabanbondfactor:
            pv1 = self._vertexdegree(edge[0])
            pv2 = self._vertexdegree(edge[1])
            num_bonds = self.matrixdescriptors.new_mol.GetNumBonds()
            num_rings = self.matrixdescriptors.new_mol.GetRingInfo().NumRings()
            return [1/np.sqrt(pv1*pv2)*(num_bonds/(num_rings+1))]
        else:
            return []

    def _pendent(self):
        G = self.mol_to_nx_graph(self.matrixdescriptors.new_mol)
        _,distmat = self._graphdistmatrix(dist=1)
        pendent_matrix = get_pendent_matrix(G, distmat)
        return np.array(pendent_matrix)

    def SuperdenticIndex(self,ind):
        if self.params.getsuperdentic:
            pendent = self._pendent()
            pendentrow = pendent[ind,:]
            nonzeros = pendentrow[pendentrow != 0]
            return [np.prod(nonzeros)]
        else:
            return [] 
        
    def _eccentricity(self,ind):
        _,distmat = self._graphdistmatrix(dist=1)
        return np.max(distmat[ind,:])

    def _avgeccentricity(self):
        mol = self.matrixdescriptors.new_mol
        num_atoms = mol.GetNumAtoms()
        eccentricities = [self._eccentricity(i) for i in range(num_atoms)]
        return np.mean(eccentricities)
    
    def _valence(self,ind):
        atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
        return atom.GetTotalValence()

    def Eccentricity(self,ind):
        if self.params.geteccentricity:
            pv = self._eccentricity(ind)
            num_atoms = self.matrixdescriptors.new_mol.GetNumAtoms()
            return [pv,(1/num_atoms) * np.abs(pv - self._avgeccentricity()),pv*self._valence(ind)]
        else:
            return []
        
    def _degreevector(self):
        degvector = [] 
        for atom in self.matrixdescriptors.new_mol.GetAtoms():
            degvector.append(_atomic_property.get_sigma_electrons(atom))
        return np.array(degvector)
    
    def Schultz(self,ind):
        if self.params.getschultz:
            _,distmat = self._graphdistmatrix(dist=1)
            mat = self.matrixdescriptors.adj_mat + distmat
            deg = self._degreevector()
            result = mat @ deg
            return [result[ind]]
        else:
            return []

    def Gutman(self,edge):
        if self.params.getgutman:
            _,distmat = self._graphdistmatrix(dist=1)
            deg = self._degreevector()
            return [distmat[edge[0], edge[1]] * deg[edge[0]] * deg[edge[1]]]
        else:
            return []
    
    def Xui(self,node):
        if self.params.getxui:
            deg = self._degreevector()
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(node)
            return [deg[node]* _atomic_property.get_valence_electrons(atom),
                    deg[node] * _atomic_property.get_valence_electrons(atom)**2]
        else:
            return []
    
    def Horary(self,edge):
        if self.params.gethorary:
            _,distmat = self._graphdistmatrix(dist=1)
            return [distmat[edge[0], edge[1]]/2]
        else:
            return []
        
    def _laplacian(self):
        G = self.mol_to_nx_graph(self.matrixdescriptors.new_mol)
        laplacians = compute_laplacians(G)
        return laplacians
    
    def Mohar(self,edge):
        if self.params.getmohar is not None:
            laplacians = self._laplacian()[self.params.mohar]
            return [laplacians[edge[0], edge[1]]]
        else:
            return []

    def _pdmatrix(self):
        G = self.mol_to_nx_graph(self.matrixdescriptors.new_mol)
        return get_distance_path_count_matrix(G)

    
    def HPValue(self,edge):
        if self.params.gethp:
            distmat = self._pdmatrix()
            return [distmat[edge[0], edge[1]]/2]
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

        aromatic = bond.IsAromatic() if bond else False

        sigmacheck = np.abs(e1-e2) <= .3

        if aromatic: return 2
        else:
            if sigmacheck:
                return 1
            else:
                return 1.5


        
    def VEMBond(self,edge):
        if self.params.getvem:
            atom1 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[0])
            atom2 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[1])
            sigma, pi = self._getbondcounts(edge)
            
            # Check if the bond between atom1 and atom2 is aromatic
            bond = self.matrixdescriptors.new_mol.GetBondBetweenAtoms(edge[0], edge[1])
            
            xij = self._bondepsigma(atom1, atom2)
            yij = self._bondeppi(atom1, atom2, bond)

            
            return [xij * sigma,yij*pi,xij*sigma - yij*pi]
        else:
            return []
    
    def EtaComposite(self,edge):
        if self.params.getetacomposite:
            atom1 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[0])
            atom2 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[1])
            _,distmat = self._graphdistmatrix()
            g1 = _atomic_property.get_eta_gamma(atom1)
            g2 = _atomic_property.get_eta_gamma(atom2)

            
            return [g1*g2/distmat[edge[0], edge[1]]]
        else:
            return []
    
    def EtaPsi(self,ind):
        if self.params.getetapsi:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            psi = _atomic_property.get_core_count(atom)/_atomic_property.get_eta_epsilon(atom)
            return [psi]
        else:
            return []
    
    def Gravity(self,edge):
        if self.params.getgravity:
            atom1 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[0])
            atom2 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[1])
            _,distmat = self._graphdistmatrix()
            m1 = _atomic_property.get_mass(atom1)
            m2 = _atomic_property.get_mass(atom2)
            return [m1*m2/distmat[edge[0], edge[1]]]    
        else:
            return []
    

    def ZagrebAtom(self,ind):
        if self.params.getzagreb is not None:
            deg = self._degreevector()
            return [deg[ind] ** 2]
        elif isinstance(self.params.getzagreb,int) and self.params.getzagreb > 2:
            deg = self._degreevector()
            return [deg[ind] ** self.params.getzagreb]
        else:
            return []
    
    def ZagrebBond(self,edge):
        if self.params.getzagreb:
            deg = self._degreevector()
            return [deg[edge[0]] * deg[edge[1]],deg[edge[0]] + deg[edge[1]]]
        elif isinstance(self.params.getzagreb,int) and self.params.getzagreb > 2:
            deg = self._degreevector()
            return [(deg[edge[0]] * deg[edge[1]])**(self.params.zagreb-1),(deg[edge[0]] + deg[edge[1]])**(self.params.zagreb-1)]
        else:
            return []
        
    def Harmonic(self,edge):
        if self.params.getharmonic:
            _,distmat = self._graphdistmatrix()
            deg = self._degreevector()
            return [2/distmat[edge[0], edge[1]],2/(deg[edge[0]] + deg[edge[1]])]
        else:
            return []
    

    def Sombor(self,edge):
        if self.params.getsombor:
            _,distmat = self._graphdistmatrix()
            deg = self._degreevector()
            return [np.sqrt(distmat[edge[0], edge[1]]**2 + deg[edge[0]]**2 + deg[edge[1]]**2),
                    np.sqrt(deg[edge[0]]**2 + deg[edge[1]]**2)]
        else:
            return []
    

    def Randic(self,edge):
        if self.params.getrandic:
            _,distmat = self._graphdistmatrix()
            deg = self._degreevector()
            return [1/(distmat[edge[0], edge[1]]**(1/2) * deg[edge[0]]**(1/2) * deg[edge[1]]**(1/2)),
                    deg[edge[0]]**(1/2) * deg[edge[1]]**(1/2)]
        else:
            return []
    

    def Nirmala(self,edge):
        if self.params.getnirmala:
            _,distmat = self._graphdistmatrix()
            deg = self._degreevector()
            return [np.exp(np.sqrt(distmat[edge[0], edge[1]] * (deg[edge[0]] + deg[edge[1]]))),
                    np.exp(np.sqrt(deg[edge[0]] + deg[edge[1]]))]
        else:
            return []
    
    def ESOS(self,edge):
        if self.params.getsoss:
            deg = self._degreevector()
            return [(deg[edge[0]] + deg[edge[1]])* np.sqrt(deg[edge[0]]**2 + deg[edge[1]]**2)]
        else:
            return []
    

    def AugmentedGraphAttribute(self,edge):
        if self.params.getaugmentedgraphattributes:
            deg = self._degreevector()
            a = 2*np.sqrt(deg[edge[0]] * deg[edge[1]])/(deg[edge[0]] + deg[edge[1]])
            b = edge[0]/edge[1] + edge[1]/edge[0]
            return [a,1/a,b]
        else:
            return []
    

    def Hyperbolic(self,edge):
        if self.params.gethyperbolic:
            _,distmat = self._graphdistmatrix()
            deg = self._degreevector()
            return [np.exp(distmat[edge[0], edge[1]]/(deg[edge[0]] + deg[edge[1]])),
                    np.exp(deg[edge[0]]/(deg[edge[0]] + deg[edge[1]])),
                    np.exp(deg[edge[1]]/(deg[edge[0]] + deg[edge[1]])),
                    4*(deg[edge[0]] * deg[edge[1]])/(deg[edge[0]] + deg[edge[1]])**2]
        else:
            return []
    
    def AugZagreb(self,edge):
        if self.params.getaugzagreb:
            deg = self._degreevector()
            return [(deg[edge[0]] * deg[edge[1]])/(deg[edge[0]] + deg[edge[1]]-2)]
        else:
            return []
    

    
        
    def Klein(self,edge):
        if self.params.getklein:
            deg = self._degreevector()
            return [.5*(deg[edge[0]]**2 + deg[edge[1]]**2) + .5*(deg[edge[0]] + deg[edge[1]])]
        else:
            return []
        
    def HyperWienerDegree(self,edge):
        if self.params.gethyperwiener:
            deg = self._degreevector()
            return [(deg[edge[0]] + deg[edge[1]])/(2**(deg[edge[0]] + deg[edge[1]])),
                    (deg[edge[0]] * deg[edge[1]])/(2**(deg[edge[0]] + deg[edge[1]]))]
        else:
            return []
    
    def Wiener(self,edge):
        if self.params.getwiener:
            _,distmat = self._graphdistmatrix()
            return [distmat[edge[0], edge[1]]]
        else:
            return []
    
    def HDSA(self,ind):
        if self.params.gethdsa:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            charge = _atomic_property.get_gasteiger_charge(atom)
            Sd = _atomic_property.get_intrinsic_state(atom)
            Stot = 0
            for atom in self.matrixdescriptors.new_mol.GetAtoms():
                Stot += _atomic_property.get_intrinsic_state(atom)
            return [charge*Sd**.5/Stot]
        else:
            return []

    def ETSBond(self,edge):
        if self.params.getets:
            _,distmat = self._graphdistmatrix()
            atom1 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[0])
            atom2 = self.matrixdescriptors.new_mol.GetAtomWithIdx(edge[1])
            Ii = _atomic_property.get_intrinsic_state(atom1)
            Ij = _atomic_property.get_intrinsic_state(atom2)
            return [np.abs(Ii-Ij)/distmat[edge[0], edge[1]]]
        else:
            return []
    
    def ETSAtom(self,ind):
        if self.params.getets:
            atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
            Ii = _atomic_property.get_intrinsic_state(atom)

            bonds = []
            for bond in self.matrixdescriptors.new_mol.GetBonds():
                if bond.GetBeginAtomIdx() == ind or bond.GetEndAtomIdx() == ind:
                    bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

            if len(bonds) == 0:
                return [Ii]
            else:
                Ib = 0
                for b in bond:
                    Ib += self.ETSBond(b)[0]
                return [Ii+Ib]
        else:
            return []
    
    def InformationContent(self,ind):
        deg = self._degreevector()
        total = sum(deg)
        if self.params.getinformationcontent is not None and self.params.getinformationcontent == 0:
            pi = deg[ind]/total
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
            return _atomic_property.get_ioniziation_potential(atom)
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

    def WeightedInformationContent(self,ind,wt='mass'):
        deg = self._degreevector()
        w = [self._getwt(wt, i) for i in range(len(deg))]
        total = sum(list(np.array(w).dot(np.array(deg))))
        if self.params.getinformationcontent is not None and self.params.getinformationcontent == 0:
            pi = deg[ind]/total
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


    def EdgeWiener(self,edge):
        G = self.mol_to_nx_graph(self.matrixdescriptors.new_mol)
        if self.params.getedgewiener:
            return [edge_wiener_index(G, edge),hyper_wiener_index(G, edge)]
        pass

    def VertexAdjacency(self,ind):
        if self.params.getvertexadjacency:
            mol = self.matrixdescriptors.new_mol
            atom = mol.GetAtomWithIdx(ind)
            heavy_bond_count = 0
            for bond in atom.GetBonds():
                if bond.GetBeginAtom().GetAtomicNum() > 1 and bond.GetEndAtom().GetAtomicNum() > 1:
                    heavy_bond_count += 1
            return [1+ np.log2(heavy_bond_count)]
        else:
            return []
    
    def TPSA(self,ind):
        if self.params.getTPSA: 
            tpsas = get_tpsa_contributions(self.matrixdescriptors.new_mol)
            return [tpsas[ind]]
        else:
            return []
        
    def LabuteASA(self,ind):
        if self.params.getASA:
            vsa_contribs    = list(rdMolDescriptors._CalcLabuteASAContribs(self.matrixdescriptors.new_mol)[0])
            return [vsa_contribs[ind]]
        else:
            return []
            
    def LogS(self,ind):
        if self.params.LogS:
            logS = compute_atomwise_logs(self.matrixdescriptors.new_mol)
            return [logS[ind]]
        else:
            return []

    def EState(self,ind):
        if self.params.getEstate == 'all': 
            estate_indices  = EState.EStateIndices(self.matrixdescriptors.new_mol)
            atom_vector = match_all_smarts(self.matrixdescriptors.new_mol)
            return [estate_indices[ind]] + atom_vector
        elif self.params.getEstate == 'vectoronly':
            atom_vector = match_all_smarts(self.matrixdescriptors.new_mol)
            return atom_vector
        elif self.params.getEstate == 'indicesonly':
            estate_indices  = EState.EStateIndices(self.matrixdescriptors.new_mol)
            return [estate_indices[ind]]
        else:
            return [] 

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

    def MolecularDistanceEdge(self,edge,valences=[1,1],reference='C'):
        if self.params.getmoleculardistanceedge:
            D = Chem.GetAdjacencyMatrix(self.matrixdescriptors.new_mol, useBO=True, force=True)
            V = D.sum(axis=0)
            N = len(V)
            _,distmat = self._graphdistmatrix()
            refcheck = self._refcheck(edge,V,valences=valences,reference=reference)
            if refcheck: 
                return [distmat[edge[0], edge[1]]**(.5/N)]
            else:
                return [0]
        else:
            return []

    def _topochargemats(self,dist=1):
        A = self.matrixdescriptors.adj_mat
        ords,D = self._graphdistmatrix(dist=dist)
        D = np.array(D)
        D_inv_square = np.linalg.inv(D**2)
        M = np.matmul(A, D_inv_square)
        return M,D_inv_square,ords,D

    def TopoChargeBond(self,edge,dist=1):
        if self.params.gettopocharge:
            M,D_inv_square,_,_ = self._topochargemats(dist=dist)
            CT = M[edge[0], edge[1]] - M[edge[1], edge[0]]
            return [D_inv_square[edge[0], edge[1]],M[edge[0], edge[1]],M[edge[1], edge[0]],CT]
        else:
            return []

    def _TopoChargeAtomFunc(self,ind,dist=1):
        M,D_inv_square,_,D = self._topochargemats(dist=dist)
        locations = np.argwhere(D[ind, :] == dist)[0]
        result = dict()
        result['Dinv'] = 0
        result['CT'] = 0
        for location in locations:
            result['Dinv'] += D_inv_square[ind, location]
            result['CT'] += M[ind, location] - M[location, ind]
        return result
                
    def TopoChargeAtom(self,ind):
        if self.params.gettopocharge == 1:
            result = self._TopoChargeAtomFunc(ind)
            return [result['Dinv'],result['CT']]
        elif self.params.gettopocharge > 1:
            output = [] 
            for i in range(1,self.params.gettopocharge):
                result = self._TopoChargeAtomFunc(ind,dist=i)
                output += [result['Dinv'],result['CT']]
            return output
        else:
            return []

    def SMRandSLogP(self,ind):
        
        if self.params.getSMR:
            Li,Ri,logP,SMR = get_slogp_smr_contributions(self.matrixdescriptors.new_mol)
            output = [] 
            if 'Li' in self.params.SLogP:
                output.append(Li[ind])
            if 'Ri' in self.params.SMR:
                output.append(Ri[ind])
            if 'logPregion' in self.params.SLogP:
                output += logP[ind]
            if 'SMRregion' in self.params.SMR:
                output += SMR[ind]            
        else:
            return []


    def Morse(self,edge):
        if self.params.getMoRSE == 'cos':
            morse_vector, contribs = get_morse_descriptors(self.matrixdescriptors.new_mol)
            return contribs[:,edge[0],edge[1]]
        elif self.params.getMoRSE == 'sin':
            morse_vector, contribs = get_morse_sinc_descriptors(self.matrixdescriptors.new_mol)
            return contribs[:,edge[0],edge[1]]


    def _chiterm(self,ind,valence=False):
        atom = self.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
        if valence:
            return _atomic_property.get_valence_electrons(atom)
        else:
            return _atomic_property.get_sigma_electrons(atom) 
        
    def _chivector(self,valence=False):
        chi = []
        for atom in self.matrixdescriptors.new_mol.GetAtoms():
            chi.append(self._chiterm.get_chi(atom.GetIdx(),valence=valence))
        return chi 
    
    def _getgdistlocs(self,ind,dist):
        ords,D = self._graphdistmatrix(dist=ord)
        locations = np.argwhere(D[ind, :] == ord)[0]
        return locations
            



    def ChiAtom(self,ind,valence=False):
        if self.getchi == 0:
            deltai = self._chiterm(ind,valence=valence)
            return [1/np.sqrt(deltai)]
        elif self.getchi >= 1:
            deltai = self._chiterm(ind,valence=valence)
            chi1 = 1/np.sqrt(deltai)
            output = [chi1]
            chivector = self._chivector(valence=valence)
            for ord in range(1,self.getchi):
                locations = self._getgdistlocs(ind,ord)
                chipaths = 0
                for loc in locations:
                    path = Chem.rdmolops.GetShortestPath(self.matrixdescriptors.new_mol, ind, loc)
                    intermediate_atoms = path[1:-1]  # Exclude ind and loc
                    chipath = chi1
                    for atom in intermediate_atoms:
                        deltaj = 1/np.sqrt(chivector[atom])
                        chipath = chipath * deltaj
                    chipaths += chipath
                output.append(chipaths)    

            return output
        else:
            return [] 
        
    def ChiBond(self,edge,order=1,valence=False):
        if self.params.getchi is not None:
            deltai = self._chiterm(edge[0],valence=valence)
            deltaj = self._chiterm(edge[1],valence=valence)
            chiij = 1/np.sqrt(deltai*deltaj)
            chi1 = 1/np.sqrt(deltai)
            chivector = self._chivector(valence=valence)
            bond = self.matrixdescriptors.new_mol.GetBondBetweenAtoms(edge[0], edge[1])
            if bond is None:
                path = Chem.rdmolops.GetShortestPath(self.matrixdescriptors.new_mol, edge[0], edge[1])
                intermediate_atoms = path[1:-1]  # Exclude edge[0] and edge[1]
            else:
                intermediate_atoms = []
            chipath = chi1
            for atom in intermediate_atoms:
                deltaj = 1/np.sqrt(chivector[atom])
                chipath = chipath * deltaj
            return [chiij,chipath]
        else:
            return []