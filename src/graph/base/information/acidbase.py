from ..base import BaseFeaturizer
from dataclasses import dataclass
"""
class AcidBaseInformation(smiles, arguments)
    A class that extends `BaseFeaturizer` to provide information about acid and base sites
    in a molecular structure represented by SMILES notation. This class is designed to 
    check whether a given site in the molecule is an acid site, a base site, or neither.
Parameters
----------
smiles : str
    The SMILES (Simplified Molecular Input Line Entry System) representation of the molecule.
arguments : dict
    A dictionary of additional arguments required for the featurizer.
Methods
-------
AcidBaseCheck(ind)
    Checks if the given site index corresponds to an acid site, a base site, or neither.
    Parameters
    ----------
    ind : int
        The index of the site to check.
    Returns
    -------
    list
        A list indicating the type of site:
        - [1, 0] if the site is an acid site.
        - [0, 1] if the site is a base site.
        - [0, 0] if the site is neither an acid nor a base site.
        - [] if the `getacidbaseinfo` parameter is not set.
"""

@dataclass
class AcidBaseParams:
    """
    Parameters
    ----------
    getacidbaseinfo : bool, optional
        A flag indicating whether to retrieve acid/base site information. 
        Default is False.
    """
    getacidbaseinfo: bool = False


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
        
