from ....utils.misc.RDKHelpers import getInchifromSMILES,RemoveMapping
from rdkit import Chem

class InchisSMILESandTypes:
    """
    This class is responsible for handling SMILES strings, InChI strings, and reaction types.

    Attributes:
        - smiles: A list of SMILES strings.
        - inchis: A list of InChI strings.
        - types: A list of reaction types.
    """
    def __init__(self, arguments):
        self.params = arguments
    

    def _createinchisandsmiles(self,smiles,info,prequel='R',sequel=None):
        """
        This function generates InChI and canonical SMILES representations for a given SMILES string
        and stores them in the `info` dictionary with appropriate keys.

        Parameters:
        - smiles: The input SMILES string.
        - info: The dictionary where the generated InChI and SMILES will be stored.
        - prequel: A prefix for the keys in the `info` dictionary (default is 'R').
        - sequel: An optional suffix for the keys in the `info` dictionary.

        Logic:
        1. Determine the key names for InChI and SMILES based on the `prequel` and `sequel` parameters:
            a. If `sequel` is provided, append it to the key names.
            b. Otherwise, use only the `prequel` as the prefix.
        2. Generate the InChI representation from the input SMILES using `getInchifromSMILES`.
        3. Remove atom mapping from the input SMILES using `RemoveMapping`.
        4. Generate the canonical SMILES representation from the modified SMILES using RDKit's `MolToSmiles`.
        5. Store the generated InChI and SMILES in the `info` dictionary under the determined keys.
        6. Return the updated `info` dictionary.

        Returns:
        - info: The updated dictionary containing the InChI and SMILES representations.
        """
        if sequel is not None:
            inchikeyname = f'{prequel}inchi' + '_' + sequel
            smileskeyname = f'{prequel}smiles' + '_' + sequel
        else:
            inchikeyname = f'{prequel}inchi'
            smileskeyname = f'{prequel}smiles'
        info[inchikeyname] = getInchifromSMILES(smiles)
        Nsmiles = RemoveMapping(smiles)
        info[smileskeyname] = Chem.MolToSmiles(Nsmiles)
        return info
    
    def _addrxntypestoinfo(self, rxn):
        """
        This function extracts reaction type columns from a reaction dataframe (`rxn`) 
        and stores their values in the `info` dictionary.

        Parameters:
        - rxn: The reaction dataframe containing the data.

        Logic:
        1. Identify columns in the dataframe that contain 'rxntype' in their names.
        2. Check if the identified columns are a list or list-like object:
            a. Iterate through each column in the list.
            b. Store the column values in the `info` dictionary.
        3. If the identified columns are not a list:
            a. Store the column values in the `info` dictionary.
        4. Store the identified column names in the `rxntype_columns` attribute.

        Returns:
        - None
        """
        columns = [col for col in rxn.columns if 'rxntype' in col]
        if isinstance(columns, list):
            for outputs in columns:
                self.info[outputs] = rxn[outputs]
        else:
            self.info[columns] = rxn[columns]
        self.rxntype_columns = columns

    def _reactioninchiandsmilessetup(self, rxn, info, smiles):
        """
        This function processes a reaction SMILES string to generate InChI and canonical SMILES 
        representations for reactants and products, and updates the `info` dictionary.

        Parameters:
        - rxn: The reaction dataframe containing the data.
        - info: The dictionary where the generated InChI and SMILES will be stored.
        - smiles: The column name in the dataframe containing the reaction SMILES string.

        Logic:
        1. Split the reaction SMILES string into reactant and product SMILES using '>>' as the delimiter.
        2. Extract the reactant SMILES (`Rsmiles`) and product SMILES (`Psmiles`).
        3. Call `_addrxntypestoinfo` to update the `info` dictionary with reaction type information.
        4. Generate InChI and canonical SMILES for the reactants and products:
            a. Call `_createinchisandsmiles` for the reactants and update the `info` dictionary.
            b. Call `_createinchisandsmiles` for the products and update the `info` dictionary.
        5. Return the updated `info` dictionary.

        Returns:
        - info: The updated dictionary containing InChI and SMILES representations for reactants and products.
        """
        RPsmiles = rxn[smiles].split('>>')
        Rsmiles = RPsmiles[0]
        Psmiles = RPsmiles[1]
        self._addrxntypestoinfo(rxn)        
        info, _, _ = self._createinchisandsmiles(Rsmiles, info, 'R')
        info, _, _ = self._createinchisandsmiles(Psmiles, info, 'P')
        return info
    
    def _multireactioninchiandsmilessetup(self, rxn, info, smiles):
        """
        This function processes multiple reaction SMILES strings to generate InChI and canonical SMILES 
        representations for reactants and products, and updates the `info` dictionary.

        Parameters:
        - rxn: The reaction dataframe containing the data.
        - info: The dictionary where the generated InChI and SMILES will be stored.
        - smiles: A list of SMILES strings representing reactions.

        Logic:
        1. Initialize empty lists for storing reactant and product InChIs and SMILES.
        2. Iterate through each SMILES string in the `smiles` list:
            a. Split the reaction SMILES string into reactant and product SMILES using '>>' as the delimiter.
            b. Extract the reactant SMILES (`Rsmiles`) and product SMILES (`Psmiles`).
            c. Call `_addrxntypestoinfo` to update the `info` dictionary with reaction type information.
            d. Generate InChI and canonical SMILES for the reactants and products:
                i. Call `_createinchisandsmiles` for the reactants and update the `info` dictionary.
                ii. Call `_createinchisandsmiles` for the products and update the `info` dictionary.
            e. Append the generated InChIs and SMILES to their respective lists.
        3. Return the updated `info` dictionary.

        Returns:
        - info: The updated dictionary containing InChI and SMILES representations for reactants and products.
        """
        for smi in smiles:
            RPsmiles = rxn[smiles].split('>>')
            Rsmiles = RPsmiles[0]
            Psmiles = RPsmiles[1]    
            self._addrxntypestoinfo(rxn)    
            info = self._createinchisandsmiles(Rsmiles, info, 'R', smi)
            info = self._createinchisandsmiles(Psmiles, info, 'P', smi)
        return info
    

    def _setupreactionsmiles(self, rxn, info):
        if isinstance(self.params.smiles,list):
            info = self._multireactioninchiandsmilessetup(rxn, info, self.params.smiles)
        else:
            info = self._reactioninchiandsmilessetup(rxn, info, self.params.smiles)
    
    def _setupmolecularsmiles(self, rxn, info):
        if isinstance(self.params.smiles, list):
            for smi in self.params.smiles:
                info = self._createinchisandsmiles(rxn[smi], info, prequel='R', sequel=smi)
        else:
            info = self._createinchisandsmiles(rxn[self.params.smiles], info)
        return info



    def AddSMILESandInchIsandTypes(self, rxn):
        """
        This function adds SMILES strings, InChI strings, and reaction types to the `info` dictionary.

        Parameters:
        - rxn: The reaction dataframe containing the data.
        - info: The dictionary where the generated InChI and SMILES will be stored.

        Logic:
        1. Check if 'rxn' is in the parameters:
            a. If true, call `_reactioninchiandsmilessetup` to process the reaction SMILES string.
            b. If false, call `_multireactioninchiandsmilessetup` to process multiple reaction SMILES strings.
        2. Return the updated `info` dictionary.

        Returns:
        - info: The updated dictionary containing InChI and SMILES representations for reactants and products.
        """
        if 'reaction' in self.params.graph:
            self.info = self._setupreactionsmiles(rxn, self.info)
        else:
            self.info = self._setupmolecularsmiles(rxn, self.info)