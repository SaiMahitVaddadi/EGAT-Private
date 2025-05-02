from .randomwalk import RandomWalk
from .dijkstra import DijkstraFeaturizer,MolecularAStar
from rdkit import Chem
from rdkit.Chem import DataStructs
import networkx as nx
import numpy as np
from ....utils.descriptors.egat.functional import FunctionalGroups
from scipy.stats import skew, kurtosis
from scipy.linalg import pinv
import random
from dataclasses import dataclass


@dataclass
class GlobalBondParams:
    getglobal: bool = False
    getshortestpath: str = None
    bias_walk: str = 'random'
    num_walks: int = 10
    getuniquewalks: bool = False
    getrandomwalk: bool = False
    getshortestpathcount: bool = False
    getcommonneighbors: str = None
    getglobaljaccard: bool = False
    getglobaladamicadar: bool = False
    getprefattachment: bool = False
    getkatz: bool = False
    getEigenvectorCentrality: bool = False
    getBetweennessCentralityCorrelation: bool = False
    getMinCutValue: bool = False
    getMaximumFlow: bool = False
    getLaplacianEigenvectorSimilarity: bool = False
    getFiedlerVectorSimilarity: bool = False
    getBetweennessCentralityOfPathways: bool = False
    getRingsInSharedPath: bool = False
    getfirstpassagetime: bool = False
    geteffectiveresistance: bool = False
    getSameAromaticSequence: bool = False
    getTopoOverlap: bool = False
    getEdgeClustering: bool = False
    getFormanCurve: bool = False
    global_fingerprint_type: str = 'Morgan'
    global_similarity_metric: str = 'Tanimoto'
    getLocalAtomicEnvironmentSimilarity: bool = False
    getsamefg: bool = False
    ct_method: str = 'spectral'

class GlobalBondInformation(RandomWalk,DijkstraFeaturizer,MolecularAStar):
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
        if self.params.getshortestpath is not None:
            if self.EdgeCheck(edge):
                return [1]
            else:
                if self.params.getshortestpath == 'unwt':
                    return self.SPDFunc(edge)
                elif self.params.getshortestpath == 'wtbymass':
                    return self.WtSPDByMass(edge)
                elif self.params.getshortestpath == 'wtbybo':
                    return self.WtSPDByBO(edge)
                elif self.params.getshortestpath == 'wtbyvalence':
                    return self.WTSPDByVE(edge)
                elif self.params.getshortestpath == 'wtbyhyb':
                    return self.WTSPDByHYB(edge)
                elif self.params.getshortestpath == 'wtbyhybve':
                    return self.WTSPDByHYBVE(edge)
                elif self.params.getshortestpath == 'wtbycoulomb':
                    return self.WTSPDbyCM(edge)
                elif self.params.getshortestpath == 'wtbyall':
                    return self.WtSPDByAll(edge)
                else:
                    return self.SPDFunc(edge)
        else:
            return []
        
    def RandomWalkStatistics(self, edge,id=1):
        if self.params.bias_walk == 'random':
            try:
                walk_lengths = []
                paths = []
                for _ in range(self.params.num_walks):
                    path = self.RWFunc(edge)
                    paths.append(path)
                    walk_lengths.append(len(path))
                
                unique_paths = list(set(tuple(p) for p in paths))
                length_unique_paths = [len(p) for p in unique_paths]

                if self.params.getuniquewalks:
                    mean_length = np.mean(length_unique_paths)
                    std_length = np.std(length_unique_paths)
                    median_length = np.median(length_unique_paths)
                    skewness = skew(length_unique_paths)
                    kurt = kurtosis(length_unique_paths)
                    min_length = np.min(length_unique_paths)
                    max_length = np.max(length_unique_paths)
                    return [mean_length, std_length, median_length, skewness, kurt, min_length, max_length,len(unique_paths)]
                else:
                    mean_length = np.mean(walk_lengths)
                    std_length = np.std(walk_lengths)
                    median_length = np.median(walk_lengths)
                    skewness = skew(walk_lengths)
                    kurt = kurtosis(walk_lengths)
                    min_length = np.min(walk_lengths)
                    max_length = np.max(walk_lengths)
                    return [mean_length, std_length, median_length, skewness, kurt, min_length, max_length]
            except Exception as e:
                if self.params.getuniquewalks:
                    return [-1] * 8
                else:
                    return [-1] * 7
        elif self.params.bias_walk in ['dijkstra','johnson','yen','bellman-ford','floyd-warshall']:
            return self.BiasedRandomWalk(edge,id=id)
        elif self.params.bias_walk == 'astar':
            return self.GetAStarPathLength(edge,id=id)
        else:
            return []

    def RandomWalkCommuteTime(self, edge):
        if self.params.getrandomwalkcommutetime:
            try:
                rwct = self.CommuteTime(edge[0], edge[1],method=self.params.ct_method)
                return [rwct]
            except Exception as e:
                print(e)
                return [-1]
        else:
            return []
    
    def ShortestPathCount(self, edge):
        if self.params.getshortestpathcount:
            if self.EdgeCheck(edge):
                return [1]
            else:
                try:
                    paths = list(set(tuple(path) for path in nx.all_shortest_paths(self.matrixdescriptors.adj_mat, source=edge[0], target=edge[1])))
                    return [len(paths)]
                except Exception as e:
                    return [0]
        else:
            return []

    def CommonNeighbors(self, edge):
        if self.params.getcommonneighbors == 'number':
            neighbors1 = set(self.matrixdescriptors.adj_mat[edge[0]].nonzero()[0])
            neighbors2 = set(self.matrixdescriptors.adj_mat[edge[1]].nonzero()[0])
            common_neighbors = neighbors1.intersection(neighbors2)
            return [len(common_neighbors)]
        elif self.params.getcommonneighbors == 'percent':
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
        
    def FirstPassageTime(self, edge):
        if self.params.getfirstpassagetime:
            try:
                fpt = self.FirstPassageTime(edge[0], edge[1])
                return [fpt]
            except:
                return [-1]
            
    def EffectiveResistance(self, edge):
        if self.params.geteffectiveresistance:
            try:
                effective_resistance = self.ResistanceDistance(edge)
                return [effective_resistance]
            except:
                return [-1]
        else:
            return []

    def AromaticSequence(self, edge):
        if self.params.getSameAromaticSequence:
            seq = self.SameAromaticSequence(edge)
            if seq:
                return [1]
            else:
                return [0]
        else:
            return []
        
    def GetTopoOverlap(self,edge):
        if self.params.getTopoOverlap:
            try:
                topo_overlap = self.TopologicalOverlap(edge)
                return [topo_overlap]
            except:
                return [-1]
        else:
            return []
    
    def GetEdgeClustering(self,edge):
        if self.params.getEdgeClustering:
            try:
                edge_clustering = self.EdgeClusteringCoefficient(edge)
                return [edge_clustering]
            except:
                return [-1]
        else:
            return []
    
    def GetFormanCurve(self,edge):
        if self.params.getFormanCurve:
            try:
                forman_curve = self.FormanRicciCurvature(edge)
                return [forman_curve]
            except:
                return [-1]
        else:
            return []

    def getfp(self,mol,edge):
        if self.params.global_fingerprint_type == 'Morgan':
            fp1 = Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, fromAtoms=[edge[0]])
            fp2 = Chem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, fromAtoms=[edge[1]])
            return fp1, fp2
        elif self.params.global_fingerprint_type == 'MACCS':
            fp1 = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol, fromAtoms=[edge[0]])
            fp2 = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(mol, fromAtoms=[edge[1]])
            return fp1, fp2
        elif self.params.global_fingerprint_type == 'RDK':
            fp1 = Chem.RDKFingerprint(mol, fromAtoms=[edge[0]])
            fp2 = Chem.RDKFingerprint(mol, fromAtoms=[edge[1]])
            return fp1, fp2
        elif self.params.global_fingerprint_type == 'AtomPair':
            fp1 = Chem.GetAtomPairFingerprint(mol, fromAtoms=[edge[0]])
            fp2 = Chem.GetAtomPairFingerprint(mol, fromAtoms=[edge[1]])
            return fp1, fp2
        elif self.params.global_fingerprint_type == 'TopologicalTorsion':
            fp1 = Chem.GetTopologicalTorsionFingerprintAsIntVect(mol, fromAtoms=[edge[0]])
            fp2 = Chem.GetTopologicalTorsionFingerprintAsIntVect(mol, fromAtoms=[edge[1]])
            return fp1, fp2
        elif self.params.global_fingerprint_type == 'Avalon':
            fp1 = Chem.rdMolDescriptors.GetAvalonFP(mol, fromAtoms=[edge[0]])
            fp2 = Chem.rdMolDescriptors.GetAvalonFP(mol, fromAtoms=[edge[1]])
            return fp1, fp2
        elif self.params.global_fingerprint_type == 'Estate':
            fp1 = Chem.rdMolDescriptors.GetEstateFingerprint(mol, fromAtoms=[edge[0]])
            fp2 = Chem.rdMolDescriptors.GetEstateFingerprint(mol, fromAtoms=[edge[1]])
            return fp1, fp2
        elif self.params.global_fingerprint_type == 'Layered':
            fp1 = Chem.LayeredFingerprint(mol, fromAtoms=[edge[0]])
            fp2 = Chem.LayeredFingerprint(mol, fromAtoms=[edge[1]])
            return fp1, fp2
        else:
            return [0]

    def getsim(self,fp1,fp2):
        if self.params.global_similarity_metric == 'Tanimoto':
            return DataStructs.TanimotoSimilarity(fp1, fp2)
        elif self.params.global_similarity_metric == 'Dice':
            return DataStructs.DiceSimilarity(fp1, fp2)
        elif self.params.global_similarity_metric == 'Cosine':
            self.params.global_similarity = DataStructs.CosineSimilarity(fp1, fp2)
        elif self.params.global_similarity_metric == 'Sokal':
            return DataStructs.SokalSimilarity(fp1, fp2)
        elif self.params.global_similarity_metric == 'Russel':
            return DataStructs.RusselSimilarity(fp1, fp2)
        elif self.params.global_similarity_metric == 'Kulczynski':
            return DataStructs.KulczynskiSimilarity(fp1, fp2)
        elif self.params.global_similarity_metric == 'McConnaughey':
            return DataStructs.McConnaugheySimilarity(fp1, fp2)
        elif self.params.global_similarity_metric == 'Asymmetric':
            return DataStructs.AsymmetricSimilarity(fp1, fp2)
        elif self.params.global_similarity_metric == 'BraunBlanquet':
            return DataStructs.BraunBlanquetSimilarity(fp1, fp2)
        else:
            return 0
    
    def LocalAtomicEnvironmentSimilarity(self, edge):
        if self.params.getLocalAtomicEnvironmentSimilarity:
            try:
                mol =self.matrixdescriptors.new_mol
                
                try:
                    fp1, fp2 = self.getfp(mol, edge)
                    return [self.getsim(fp1, fp2)]
                except:
                    return [0]
            except:
                return [0]
        else:
            return []

    def IsSameFG(self,edge):
        if self.params.getsamefg:
            fg = FunctionalGroups()
            res = fg.are_atoms_in_same_functional_group(self.matrixdescriptors.new_mol, edge[0], edge[1])
            if res: 
                return self.properties.functional_group_encode['TRUE']
            else:
                return self.properties.functional_group_encode['FALSE']
        else:
            return []
