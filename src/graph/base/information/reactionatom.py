from ..base import BaseFeaturizer
from ..reaction import BaseReactionFeaturizer
from rdkit import Chem
from rdkit.Chem import DataStructs
import networkx as nx
import numpy as np 
from scipy.linalg import eigvals
from scipy.stats import entropy as shannon_entropy
from collections import defaultdict
from mordred import _atomic_property
from ....utils.misc.taffi_functions import graph_seps
from dataclasses import dataclass


@dataclass
class ReactiveAtomParams:
    removereactiveinfo: bool = False
    addneighboringreactives: bool = False
    gethybridizationchange: bool = False
    getdegreecentralitydiff: bool = False
    getclosenesscentralitydiff: bool = False
    getbetweennesscentralitydiff: bool = False
    geteigenvectorcentralitydiff: bool = False
    getkatzcentralitydiff: bool = False
    getpagerankcentralitydiff: bool = False
    getkcorenumberdiff: bool = False
    getharmoniccentralitydiff: bool = False
    getlocalbridgingdiff: bool = False
    gettrianglecountdiff: bool = False
    getlocalatomfeaturesdiff: bool = False
    getvalencychangediff: bool = False
    getoxidationreductiondiff: bool = False
    getbondordersumchangediff: bool = False
    getneighborhoodchangeratio: bool = False
    getconnectivitypathdifference: bool = False
    k_iter: int = 2

class ReactiveAtomInformation(BaseFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)
        
    def GrabReactiveAtomInformation(self):
        # Get the locations of 1 and -1 in bondmat_change
        self.bondmat_change_ones = np.where(self.reactionpropeties.bondmat_change == 1)
        self.bondmat_change_neg_ones = np.where(self.reactionpropeties.bondmat_change == -1)
        # Combine the indices of 1 and -1 into a single list and get unique indices
        combined_indices = np.concatenate((self.bondmat_change_ones[0], self.bondmat_change_neg_ones[0]))
        unique_indices = np.unique(combined_indices)
        self.reactive_atoms = unique_indices
        self.Rgs = graph_seps(self.reactant.matrixdescriptors.adj_mat)
        self.Rgs[self.Rgs < 0] = 100
        self.Pgs = graph_seps(self.product.matrixdescriptors.adj_mat)
        self.Pgs[self.Pgs < 0] = 100
        
    def DistanceFromReactingAtom(self,ind,prod=False):
        if not self.params.removereactiveinfo:
            if len(self.reactive_atoms) > 0:
                if not prod:
                    dis = min([self.Rgs[ind][indr] for indr in self.reactive_atoms])
                else:
                    dis = min([self.Pgs[ind][indr] for indr in self.reactive_atoms])
            else:
                dis = 0 
            return [dis]
        else:
            return []
             
    def NeighboringReactives(self,adj_mat,ind,prod=False):
        if self.params.addneighboringreactives: 
            reactive_neighbors = 0
            for neighbor in adj_mat[ind]:
                if neighbor in self.reactive_atoms and adj_mat[ind][neighbor] > 0:
                    reactive_neighbors += 1
            return [reactive_neighbors]
        else:
            return []
        
    

    
class ReactiveAtomChangeInformation(BaseReactionFeaturizer):
    def __init__(self, smiles, arguments):
        super().__init__(smiles, arguments)

    def GrabNetworkXFunctions(self):
        self.reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
        self.product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
        self.reactant_centrality = dict()
        self.product_centrality = dict()
        self.reactant_centrality['degree'] = nx.degree_centrality(self.reactant_graph)
        self.product_centrality['degree'] = nx.degree_centrality(self.product_graph)
        self.reactant_centrality['closeness'] = nx.closeness_centrality(self.reactant_graph)
        self.product_centrality['closeness'] = nx.closeness_centrality(self.product_graph)
        self.reactant_centrality['betweenness'] = nx.betweenness_centrality(self.reactant_graph)
        self.product_centrality['betweenness'] = nx.betweenness_centrality(self.product_graph)
        self.reactant_centrality['eigenvector'] = nx.eigenvector_centrality(self.reactant_graph)
        self.product_centrality['eigenvector'] = nx.eigenvector_centrality(self.product_graph)
        self.reactant_centrality['katz'] = nx.katz_centrality(self.reactant_graph)
        self.product_centrality['katz'] = nx.katz_centrality(self.product_graph)
        self.reactant_centrality['pagerank'] = nx.pagerank(self.reactant_graph)
        self.product_centrality['pagerank'] = nx.pagerank(self.product_graph)

        # K-core number
        self.reactant_centrality['k_core_number'] = nx.core_number(self.reactant_graph)
        self.product_centrality['k_core_number'] = nx.core_number(self.product_graph)

        # Harmonic centrality
        self.reactant_centrality['harmonic_centrality'] = nx.harmonic_centrality(self.reactant_graph)
        self.product_centrality['harmonic_centrality'] = nx.harmonic_centrality(self.product_graph)
        # Local bridging
        self.reactant_centrality['local_bridge'] = self.localbridge(self.reactant_graph)
        self.product_centrality['local_bridge'] = self.localbridge(self.product_graph)
        # Graphlet degree vector (example: count of triangles)
        triangles = nx.triangles(self.reactant_graph)  # number of triangles each node is in for reactant
        self.reactant_centrality["triangle_count"] = triangles
        triangles = nx.triangles(self.product_graph)  # number of triangles each node is in for product
        self.product_centrality["triangle_count"] = triangles
        self.reactant_centrality['local_atom_features'] = self.compute_atom_features(self.reactant_graph,self.params.k_iter)
        self.product_centrality['local_atom_features'] = self.compute_atom_features(self.product_graph,self.params.k_iter)
        self.local_atom_keys = list(self.reactant_centrality['local_atom_features'][0].keys())

    def localbridge(self,G):
        bridging = {}
        for node in G.nodes:
            neighbors = set(G.neighbors(node))
            deg = G.degree(node)
            if deg < 2:
                bridging[node] = 0.0
                continue
            actual_edges = 0
            for u in neighbors:
                for v in neighbors:
                    if u != v and G.has_edge(u, v):
                        actual_edges += 1
            max_edges = deg * (deg - 1)
            clustering = actual_edges / max_edges if max_edges > 0 else 0
            bridging[node] = 1 - clustering
        return bridging
    
    def compute_atom_features(self,G, k_iter=2):
        features = defaultdict(dict)

        for node in G.nodes:
            # 1. Ego Network Metrics (1-hop neighborhood)
            ego_net = nx.ego_graph(G, node, radius=1)

            num_edges = ego_net.number_of_edges()
            density = nx.density(ego_net)
            clustering = nx.clustering(G, node)
            triangles = nx.triangles(G, node)

            features[node]["ego_num_edges"] = num_edges
            features[node]["ego_density"] = density
            features[node]["ego_clustering"] = clustering
            features[node]["ego_triangles"] = triangles

            # 2. Local Spectral Radius (largest eigenvalue of adjacency of ego network)
            A = nx.to_numpy_array(ego_net)
            if A.shape[0] > 1:
                spectral_radius = np.max(np.abs(eigvals(A)))
            else:
                spectral_radius = 0.0
            features[node]["local_spectral_radius"] = float(np.real(spectral_radius))

            # 3. Local Fiedler Value (2nd-smallest Laplacian eigenvalue of ego net)
            if A.shape[0] > 2:
                L = nx.laplacian_matrix(ego_net).toarray()
                eigs = np.sort(np.real(eigvals(L)))
                fiedler = eigs[1] if len(eigs) > 1 else 0.0
            else:
                fiedler = 0.0
            features[node]["local_fiedler_value"] = float(fiedler)

            # 4. Entropy of Degree Distribution in 1-hop neighborhood
            neighbor_degrees = [G.degree(nbr) for nbr in G.neighbors(node)]
            if neighbor_degrees:
                values, counts = np.unique(neighbor_degrees, return_counts=True)
                probs = counts / counts.sum()
                degree_entropy = shannon_entropy(probs)
            else:
                degree_entropy = 0.0
            features[node]["neighbor_degree_entropy"] = degree_entropy

        # 5. Weisfeiler-Lehman Colors (Iterated Degree)
        '''
        wl_labels = {n: str(G.degree(n)) for n in G.nodes()}
        print("Initial WL labels:", wl_labels)
        for i in range(k_iter):
            new_labels = {}
            for n in G.nodes:
                neighborhood = sorted([str(wl_labels[nb]) for nb in G.neighbors(n)])
                new_label = wl_labels[n] + "_" + "_".join(neighborhood)
                new_labels[n] = hash(new_label)  # hashed for compactness
            wl_labels = new_labels

        for n in G.nodes:
            features[n][f"wl_color_k"] = wl_labels[n]
        '''
        return features


    def ChangeInAtomicHybridization(self, atom,prod=False):
        if self.params.gethybridizationchange:
            initial_hybridization = self.reactant.stereo.Hybridization[atom]
            final_hybridization = self.product.stereo.Hybridization[atom]
            initial_hybridization = self.reactant.properties.hybridization_encode[initial_hybridization]
            final_hybridization = self.product.properties.hybridization_encode[final_hybridization]
            if not prod: return list(np.array(final_hybridization) - np.array(initial_hybridization))
            else: return list(np.array(final_hybridization) - np.array(initial_hybridization))
        else:
            return []
        
    def basenetworkxchangingatomsfcn(self,atom,key='degree',prod=False,getparams=None):
        if getattr(self.params,getparams):
            if key in self.local_atom_keys:
                reactant_centrality = self.reactant_centrality['local_atom_features'][atom][key]
                product_centrality = self.product_centrality['local_atom_features'][atom][key]
            else:
                reactant_centrality = self.reactant_centrality[key][atom]
                product_centrality = self.product_centrality[key][atom]
            if not prod: return [reactant_centrality - product_centrality]
            else: return [product_centrality - reactant_centrality]
        else:
            return []

    def DegreeCentralityOfChangingAtoms(self, atom,prod=False):
        return self.basenetworkxchangingatomsfcn(atom,key='degree',getparams='getdegreecentralitydiff',prod=prod)
    
    def ClosenessCentralityOfChangingAtoms(self, atom,prod=False):
        return self.basenetworkxchangingatomsfcn(atom,key='closeness',getparams='getclosenesscentralitydiff',prod=prod)
    
    def BetweennessCentralityOfChangingAtoms(self, atom,prod=False):
        return self.basenetworkxchangingatomsfcn(atom,key='betweenness',getparams='getbetweennesscentralitydiff',prod=prod)
    
    def EigenvectorCentralityOfChangingAtoms(self, atom,prod=False):
        return self.basenetworkxchangingatomsfcn(atom,key='eigenvector',getparams='geteigenvectorcentralitydiff',prod=prod)
    
    def KatzCentralityOfChangingAtoms(self, atom,prod=False):
        return self.basenetworkxchangingatomsfcn(atom,key='katz',getparams='getkatzcentralitydiff',prod=prod)
    
    def PageRankCentralityOfChangingAtoms(self, atom,prod=False):
        return self.basenetworkxchangingatomsfcn(atom,key='pagerank',getparams='getpagerankcentralitydiff',prod=prod)
    
    def KCoreNumberOfChangingAtoms(self, atom,prod=False):
        return self.basenetworkxchangingatomsfcn(atom,key='k_core_number',getparams='getkcorenumberdiff',prod=prod)
    
    def HarmonicCentralityOfChangingAtoms(self, atom,prod=False):
        return self.basenetworkxchangingatomsfcn(atom,key='harmonic_centrality',getparams='getharmoniccentralitydiff',prod=prod)
    
    def LocalBridgingOfChangingAtoms(self, atom,prod=False):
        return self.basenetworkxchangingatomsfcn(atom,key='local_bridge',getparams='getlocalbridgingdiff',prod=prod)
    
    def TriangleCountOfChangingAtoms(self, atom,prod=False):
        return self.basenetworkxchangingatomsfcn(atom,key='triangle_count',getparams='gettrianglecountdiff',prod=prod)
    
    def LocalAtomFeaturesOfChangingAtoms(self, atom, prod=False):
        results = []
        if self.params.getlocalatomfeatures:
            for key in self.local_atom_keys:
                results.extend(self.basenetworkxchangingatomsfcn(atom, key=key, getparams='getlocalatomfeatures', prod=prod))
        return results

    def lonepairfcn(self,ind,feat):
        atom = feat.matrixdescriptors.new_mol.GetAtomWithIdx(ind)
        pibonds = 0 
        sigmabonds = 0 
        bonds = 0
        for b in atom.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE:
                pibonds += 1
            elif b.GetBondType() == Chem.BondType.TRIPLE:
                pibonds += 2
            if b.GetIsAromatic():
                pibonds += 1

            if b.GetBondType() == Chem.BondType.SINGLE:
                sigmabonds += 1
            bonds += 1
        
        ve = _atomic_property.get_valence_electrons(atom)

        lp = (ve - pibonds - sigmabonds - atom.GetFormalCharge() - atom.GetNumRadicalElectrons())/2
        return lp,bonds

    def AtomicValencyChange(self, atom,prod=False):
        if self.params.getvalencychange:
            initial_lp,init_bonds = self.lonepairfcn(atom,self.reactant)
            final_lp,final_bonds = self.lonepairfcn(atom,self.product)
            if prod:
                return [final_lp - initial_lp,final_bonds - init_bonds]
            else:
                return [initial_lp - final_lp,init_bonds - final_bonds]
        else:
            return []

    def OxidationOrReduction(self, atom,prod=False):
        if self.params.getoxidationreduction:
            initial_lp,init_bonds = self.lonepairfcn(atom,self.reactant)
            final_lp,final_bonds = self.lonepairfcn(atom,self.product)
            initial_oxidation_state = atom.GetFormalCharge()
            final_oxidation_state = self.product.matrixdescriptors.new_mol.GetAtomWithIdx(atom).GetFormalCharge()

            if final_lp > initial_lp or final_oxidation_state > initial_oxidation_state:
                return [1,0]
            elif final_lp < initial_lp or final_oxidation_state < initial_oxidation_state:
                return [0,1]
            else:
                return [0,0]
        else:
            return []

    def LocalBondOrderSumChange(self, atom,prod=False):
        if self.params.getbondordersumchange:
            initial_bond_order_sum = sum(self.reactant.matrixdescriptors.bond_mat[atom])
            final_bond_order_sum = sum(self.product.matrixdescriptors.bond_mat[atom])
            if not prod: return [initial_bond_order_sum - final_bond_order_sum]
            else: return [final_bond_order_sum - initial_bond_order_sum]
        else:
            return []

    def AtomicNeighborhoodChangeRatio(self, atom,prod=False):
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
    
    def ConnectivityPathDifference(self, node):
        if self.params.getconnectivitypathdifference:
            reactant_graph = nx.Graph(self.reactant.matrixdescriptors.adj_mat)
            product_graph = nx.Graph(self.product.matrixdescriptors.adj_mat)
            connectivity_R = nx.single_source_shortest_path_length(reactant_graph, source=node)
            connectivity_P = nx.single_source_shortest_path_length(product_graph, source=node)
            diff = 0
            for target, length in connectivity_R.items():
                if target in connectivity_P:
                    diff += abs(length - connectivity_P[target])
                else:
                    diff += length
            for target, length in connectivity_P.items():
                if target not in connectivity_R:
                    diff += length
            return [diff]
        else:
            return []
    