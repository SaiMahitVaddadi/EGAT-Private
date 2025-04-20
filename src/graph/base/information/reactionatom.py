from ..base import BaseFeaturizer
from ..reaction import BaseReactionFeaturizer
from rdkit import Chem
from rdkit.Chem import DataStructs


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

  