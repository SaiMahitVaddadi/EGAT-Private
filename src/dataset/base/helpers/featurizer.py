
from ....graph.molecular.Molecule import MoleculeFeaturizer
from ....graph.molecular.MoleculeGeometry import MoleculeFeaturizerwithGeometry
from ....graph.reaction.ReactionGeometry import ReactionFeaturizerwithGeometry
from ....graph.reaction.ReactionGlobalGeometry import ReactionFeaturizerwithPaddingandGeometry
from ....graph.molecular.MoleculeGlobal import MoleculeFeaturizerwithPadding
from ....graph.reaction.ReactionGlobal import ReactionFeaturizerwithPadding
from ....graph.reaction.Reaction import ReactionFeaturizer



    

class FeauritizerFunctions:
    def __init__(self,params):
        self.params = params 

    def _featurization(self, smiles, reaction=False):
        """
        This function initializes a featurizer object based on the type of input (reaction or molecule),
        the dimensionality (2D or 3D), and whether global mode is enabled.

        Parameters:
        - smiles: The input SMILES string(s) to be featurized.
        - reaction: A boolean indicating whether the input represents a reaction (default is False).

        Logic:
        1. Determine the prefix string (`prestring`) based on whether the input is a reaction or molecule:
            a. Use 'Reaction' for reactions.
            b. Use 'Molecule' for molecules.
        2. Check the dimensionality (`params.dimension`):
            a. If 2D:
                i. Use 'FeaturizerwithPadding' if global mode is enabled.
                ii. Use 'Featurizer' otherwise.
            b. If 3D:
                i. Use 'FeaturizerwithPadding' if global mode is enabled.
                ii. Use 'FeaturizerwithGeometry' otherwise.
        3. Concatenate the prefix string and featurizer type to form the featurizer class name.
        4. Dynamically initialize the featurizer object using the class name and input parameters.

        Returns:
        - featurizer: The initialized featurizer object.
        """
        if reaction: 
            prestring = 'Reaction'
        else: 
            prestring = 'Molecule'
        if self.params.dimension == '2d':
            if self.params.globalmode:
                featurizer_str = 'FeaturizerwithPadding'
            else:
                featurizer_str = 'Featurizer'
        elif self.params.dimension == '3d':
            if self.params.globalmode:
                featurizer_str = 'FeaturizerwithPadding'
            else:
                featurizer_str = 'FeaturizerwithGeometry'
        featurizer_str = prestring + featurizer_str
        featurizer = globals()[featurizer_str](smiles, self.params)
        return featurizer
    
    def ChooseFeaturizerStringCase(self,rxn):
        if 'molecular' in self.params.graph:
            self.featurizer = self._featurization(rxn[self.params.smiles])
        if 'reaction' in self.params.graph:
            self.featurizer = self._featurization(rxn[self.params.smiles],reaction=True)
        self.featurizer.run()

    def ChooseFeaturizerListCase(self,rxn):
        self.featurizer = []
        for smi in self.params.smiles:
            if 'molecular' in self.params.graph:
                featurizer = self._featurization(rxn[smi],reaction=False)
            if 'reaction' in self.params.graph:
                featurizer = self._featurization(rxn[smi],reaction=True)
            featurizer.run()
            self.featurizer.append(featurizer)
    
    def ChooseFeaturizer(self,rxn):
        if isinstance(self.params.smiles,str):
            self.ChooseFeaturizerStringCase(rxn)
        elif isinstance(self.params.smiles,list):
            self.ChooseFeaturizerListCase(rxn)
            