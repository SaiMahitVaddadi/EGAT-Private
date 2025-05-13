from icecream import ic
import numpy as np
from dataclasses import dataclass
from typing import Union, List



class FeaturestoInfo:
    def __init__(self,params):
        self.params = params 

    def __edgedictnames(self, sequel=None):
        """
        This function generates edge dictionary names (`u` and `v`) for graph representation.

        Parameters:
        - sequel: An optional suffix to append to the edge names.

        Logic:
        1. Initialize `uname` as 'u' and `vname` as 'v'.
        2. If `sequel` is provided:
            a. Append the `sequel` to `uname` and `vname`.

        Returns:
        - uname: The name for the source nodes in the edge dictionary.
        - vname: The name for the destination nodes in the edge dictionary.
        """
        uname = 'u'
        vname = 'v'
        if sequel is not None:
            uname = f'u_{sequel}'
            vname = f'v_{sequel}'
        return uname, vname

    def __atomfromdictnames(self, reaction=False, sequel=None):
        """
        This function generates atom feature dictionary names for reactants and products.

        Parameters:
        - reaction: A boolean indicating whether the names are for a reaction (default is False).
        - sequel: An optional suffix to append to the atom feature names.

        Logic:
        1. Initialize `atomFname` as 'atom_F'.
        2. If `reaction` is True:
            a. If `sequel` is provided, append it to the reactant and product atom feature names.
            b. Otherwise, use default reactant and product atom feature names.
        3. If `reaction` is False:
            a. If `sequel` is provided, append it to the reactant atom feature names.
            b. Otherwise, use default reactant atom feature names.

        Returns:
        - atomFRname: The name for the reactant atom features.
        - atomFPname: The name for the product atom features (if applicable).
        """
        atomFname = 'atom_F'
        if reaction:
            if sequel is not None:
                atomFRname = f'{atomFname}_R_{sequel}'
                atomFPname = f'{atomFname}_P_{sequel}'
            else:
                atomFRname = f'{atomFname}_R'
                atomFPname = f'{atomFname}_P'
        else:
            if sequel is not None:
                atomFRname = f'{atomFname}_R_{sequel}'
                atomFPname = f'{atomFname}_R_{sequel}'
            else:
                atomFRname = f'{atomFname}_R'
                atomFPname = f'{atomFname}_R'
        return atomFRname, atomFPname
    
    def __bondfromdictnames(self, reaction=False, sequel=None):
        """
        This function generates bond feature dictionary names for reactants and products.

        Parameters:
        - reaction: A boolean indicating whether the names are for a reaction (default is False).
        - sequel: An optional suffix to append to the bond feature names.

        Logic:
        1. Initialize `bondFname` as 'bond_F'.
        2. If `reaction` is True:
            a. If `sequel` is provided, append it to the reactant and product bond feature names.
            b. Otherwise, use default reactant and product bond feature names.
        3. If `reaction` is False:
            a. If `sequel` is provided, append it to the reactant bond feature names.
            b. Otherwise, use default reactant bond feature names.

        Returns:
        - bondFRname: The name for the reactant bond features.
        - bondFPname: The name for the product bond features (if applicable).
        """
        bondFname = 'bond_F'
        if reaction:
            if sequel is not None:
                bondFRname = f'{bondFname}_R_{sequel}'
                bondFPname = f'{bondFname}_P_{sequel}'
            else:
                bondFRname = f'{bondFname}_R'
                bondFPname = f'{bondFname}_P'
        else:
            if sequel is not None:
                bondFRname = f'{bondFname}_R_{sequel}'
                bondFPname = f'{bondFname}_R_{sequel}'
            else:
                bondFRname = f'{bondFname}_R'
                bondFPname = f'{bondFname}_R'
        return bondFRname, bondFPname

    def __geomfromdictnames(self, reaction=False, sequel=None):
        """
        This function generates geometry feature dictionary names for reactants and products.

        Parameters:
        - reaction: A boolean indicating whether the names are for a reaction (default is False).
        - sequel: An optional suffix to append to the geometry feature names.

        Logic:
        1. Initialize `geomFname` as 'Geom'.
        2. If `reaction` is True:
            a. If `sequel` is provided, append it to the reactant and product geometry feature names.
            b. Otherwise, use default reactant and product geometry feature names.
        3. If `reaction` is False:
            a. If `sequel` is provided, append it to the reactant geometry feature names.
            b. Otherwise, use default reactant geometry feature names.

        Returns:
        - geomFRname: The name for the reactant geometry features.
        - geomFPname: The name for the product geometry features (if applicable).
        """
        geomFname = 'Geom'
        if reaction:
            if sequel is not None:
                geomFRname = f'{geomFname}_R_{sequel}'
                geomFPname = f'{geomFname}_P_{sequel}'
            else:
                geomFRname = f'{geomFname}_R'
                geomFPname = f'{geomFname}_P'
        else:
            if sequel is not None:
                geomFRname = f'{geomFname}_R_{sequel}'
                geomFPname = f'{geomFname}_R_{sequel}'
            else:
                geomFRname = f'{geomFname}_R'
                geomFPname = f'{geomFname}_R'
        return geomFRname, geomFPname

    def __usegeometryvector(self):
        """
        Determines whether to use geometry vector features and returns the corresponding feature names.

        Logic:
        1. Check if `getradical` is set to 'YARP':
            a. If `dimension` is '2d', return False with an empty string.
            b. If `dimension` is '3d', return True with 'atom_geometry_feature_confs_list'.
        2. If `conformer.nconfs` is greater than 1 and `dimension` is '3d':
            a. Return True with 'atom_geometry_feature_confs_list'.
        3. If `conformer.nconfs` is equal to 1 and `dimension` is '3d':
            a. Return True with 'atom_geometry_features'.
        4. Otherwise, return False with an empty string.

        Returns:
        - usecase: A boolean indicating whether geometry vector features are used.
        - geomfeat: The name of the geometry feature attribute.
        """
        if self.params.getradical == 'YARP':
            if self.params.dimension == '2d':   
                return False, ''
            elif self.params.dimension == '3d':
                return True, 'atom_geometry_feature_confs_list'
        if self.params.conformer.nconfs > 1 and self.params.dimension == '3d':
            return True, 'atom_geometry_feature_confs_list'
        elif self.params.conformer.nconfs == 1 and self.params.dimension == '3d':
            return True, 'atom_geometry_features'
        return False, ''

    def __singleormulticonf(self, featurizer, features='atom'):
        """
        Determines the appropriate feature attribute name based on the featurizer's configuration.

        Parameters:
        - featurizer: The featurizer object containing molecular or reaction features.
        - features: A string indicating the type of features ('atom' by default).

        Logic:
        1. If the dimension is '2d':
            a. Check if `getradical` is 'YARP' and `bond_mats` is greater than 1:
                i. Return the feature name with '_features_dict' suffix.
            b. Otherwise:
                i. Return the feature name with '_features' suffix.
        2. If the dimension is not '2d':
            a. Check if `getradical` is 'YARP' and `bond_mats` is greater than 1:
                i. Return the feature name with '_features_dict' suffix.
            b. Check if `conformer.nconfs` is greater than 1:
                i. Return the feature name with '_features_confs' suffix.
            c. Otherwise:
                i. Return the feature name with '_features' suffix.

        Returns:
        - A string representing the appropriate feature attribute name.
        """
        if self.params.dimension == '2d':
            if self.params.getradical == 'YARP' and featurizer.bond_mats > 1:
                return f'{features}_features_dict'
            else:
                return f'{features}_features'
        else:
            if self.params.getradical == 'YARP' and featurizer.bond_mats > 1:
                return f'{features}_features_dict'
            elif self.params.conformer.nconfs > 1:
                return f'{features}_features_confs'
            else:
                return f'{features}_features'

    def __molecularinfosingleconf(self, info, featurizer, uname, vname, atomFRname, bondFRname):
        """
        This function extracts molecular-related features from the featurizer and updates the `info` dictionary.

        Parameters:
        - info: The dictionary to store extracted features.
        - featurizer: The featurizer object containing molecular features.
        - uname: The key for storing source node indices of edges.
        - vname: The key for storing destination node indices of edges.
        - atomFRname: The key for storing atom features.
        - bondFRname: The key for storing bond features.

        Logic:
        1. Extract source and destination node indices for edges and store them in `info`.
        2. Extract atom and bond features and store them in `info`.

        Returns:
        - info: The updated dictionary containing molecular-related features.
        """
        info[uname] = featurizer.edges_u
        info[vname] = featurizer.edges_v
        info[atomFRname] = featurizer.atom_features
        info[bondFRname] = featurizer.bond_features
        return info
    
    def __molecularinfomulticonf(self, info, featurizer, uname, vname, atomFRname, bondFRname):
        atomconf = self.__singleormulticonf(featurizer, 'atom')
        bondconf = self.__singleormulticonf(featurizer, 'bond')
        info[uname] = featurizer.edges_u
        info[vname] = featurizer.edges_v
        info[atomFRname] = list(getattr(featurizer, atomconf).values())
        info[bondFRname] = list(getattr(featurizer, bondconf).values())
        return info

    def __reactioninfosingleconf(self, info, featurizer, uname, vname, atomFRname, atomFPname, bondFRname, bondFPname):
        """
        This function extracts reaction-related features from the featurizer and updates the `info` dictionary.

        Parameters:
        - info: The dictionary to store extracted features.
        - featurizer: The featurizer object containing reaction features.
        - uname: The key for storing source node indices of reactant edges.
        - vname: The key for storing destination node indices of reactant edges.
        - atomFRname: The key for storing reactant atom features.
        - atomFPname: The key for storing product atom features.
        - bondFRname: The key for storing reactant bond features.
        - bondFPname: The key for storing product bond features.

        Logic:
        1. Extract source and destination node indices for reactant edges and store them in `info`.
        2. Extract reactant atom and bond features and store them in `info`.
        3. Extract product atom and bond features and store them in `info`.

        Returns:
        - info: The updated dictionary containing reaction-related features.
        """
        info = self.__molecularinfo(info, featurizer.reactant, uname, vname, atomFRname, bondFRname)
        info[atomFPname] = featurizer.product.atom_features
        info[bondFPname] = featurizer.product.bond_features
        return info
    
    def __reactioninfomulticonf(self, info, featurizer, uname, vname, atomFRname, atomFPname, bondFRname, bondFPname):
        """
        This function extracts reaction-related features from the featurizer and updates the `info` dictionary for multiple conformations.

        Parameters:
        - info: The dictionary to store extracted features.
        - featurizer: The featurizer object containing reaction features.
        - uname: The key for storing source node indices of reactant edges.
        - vname: The key for storing destination node indices of reactant edges.
        - atomFRname: The key for storing reactant atom features.
        - atomFPname: The key for storing product atom features.
        - bondFRname: The key for storing reactant bond features.
        - bondFPname: The key for storing product bond features.

        Logic:
        1. Extract source and destination node indices for reactant edges and store them in `info`.
        2. Extract reactant atom and bond features and store them in `info`.
        3. Extract product atom and bond features and store them in `info`.

        Returns:
        - info: The updated dictionary containing reaction-related features.
        """
        atomconf = self.__singleormulticonf(featurizer.reactant, 'atom')
        bondconf = self.__singleormulticonf(featurizer.reactant, 'bond')
        info = self.__molecularinfomulticonf(info, featurizer.reactant, uname, vname, atomFRname, bondFRname)
        info[atomFPname] = getattr(featurizer.product, atomconf).values()
        info[bondFPname] = getattr(featurizer.product, bondconf).values()
        return info
    
    def __reactioninfo(self, info, featurizer, uname, vname, atomFRname, atomFPname, bondFRname, bondFPname):
        if 'features_' in self.__singleormulticonf(featurizer.reactant, 'atom'):
            info = self.__reactioninfomulticonf(info, featurizer, uname, vname, atomFRname, atomFPname, bondFRname, bondFPname)
        else:
            info = self.__reactioninfosingleconf(info, featurizer, uname, vname, atomFRname, atomFPname, bondFRname, bondFPname)
        return info
    
    def __molecularinfo(self, info, featurizer, uname, vname, atomFRname, bondFRname):
        """
        This function extracts molecular-related features from the featurizer and updates the `info` dictionary.

        Parameters:
        - info: The dictionary to store extracted features.
        - featurizer: The featurizer object containing molecular features.
        - uname: The key for storing source node indices of edges.
        - vname: The key for storing destination node indices of edges.
        - atomFRname: The key for storing atom features.
        - bondFRname: The key for storing bond features.

        Logic:
        1. Check if `featurizer` is an instance of `ReactionFeaturizerwithPaddingandGeometry`.
            a. If True, call `__molecularinfomulticonf` to extract multi-conformation features.
            b. If False, call `__molecularinfosingleconf` to extract single-conformation features.

        Returns:
        - info: The updated dictionary containing molecular-related features.
        """
        if 'features_' in self.__singleormulticonf(featurizer, 'atom'):
            info = self.__molecularinfomulticonf(info, featurizer, uname, vname, atomFRname, bondFRname)
        else:
            info = self.__molecularinfosingleconf(info, featurizer, uname, vname, atomFRname, bondFRname)
        return info

    def __singleormulticonfgeometry(self,featurizer,features='atom'):
        if self.params.dimension == '2d':
            if self.params.getradical == 'YARP' and featurizer.bond_mats > 1:
                return f'{features}_geometry_features_dict'
            else:
                return f'{features}_geometry_features'
        else:
            if self.params.getradical == 'YARP' and featurizer.bond_mats > 1:
                return f'{features}_geometry_features_dict'
            elif self.params.conformer.nconfs > 1:
                return f'{features}_geometry_features_confs'
            else:
                return f'{features}_geometry_features'

    def __moleculargeominfosingleconf(self, info, featurizer, atomFRname, atomFPname):
        """
        This function extracts geometry-related features from the featurizer and updates the `info` dictionary.

        Parameters:
        - info: The dictionary to store extracted features.
        - featurizer: The featurizer object containing geometry features.
        - atomFRname: The key for storing reactant geometry features.
        - atomFPname: The key for storing product geometry features.

        Logic:
        1. Extract reactant geometry features and store them in `info`.
        2. Extract product geometry features and store them in `info`.

        Returns:
        - info: The updated dictionary containing geometry-related features.
        """
        info[atomFRname] = featurizer.atom_geometry_features
        return info
    

    def __moleculargeominfomulticonf(self, info, featurizer, atomFRname):
        atomconf = self.__singleormulticonfgeometry(featurizer, 'atom')
        info[atomFRname] = getattr(featurizer, atomconf).values()
        return info
    
    def __moleculargeominfo(self, info, featurizer, atomFRname, atomFPname):    
        """
        This function extracts geometry-related features from the featurizer and updates the `info` dictionary.

        Parameters:
        - info: The dictionary to store extracted features.
        - featurizer: The featurizer object containing geometry features.
        - atomFRname: The key for storing reactant geometry features.
        - atomFPname: The key for storing product geometry features.

        Logic:
        1. Check if `featurizer` is an instance of `ReactionFeaturizerwithPaddingandGeometry`.
            a. If True, call `__moleculargeominfomulticonf` to extract multi-conformation features.
            b. If False, call `__moleculargeominfosingleconf` to extract single-conformation features.

        Returns:
        - info: The updated dictionary containing geometry-related features.
        """
        if 'features_' in self.__singleormulticonfgeometry(featurizer, 'atom'):
            info = self.__moleculargeominfomulticonf(info, featurizer, atomFRname)
        else:
            info = self.__moleculargeominfosingleconf(info, featurizer, atomFRname, atomFPname)
        return info
    
    def __reactiongeominfosingleconf(self, info, featurizer, atomFRname, atomFPname):
        """
        This function extracts geometry-related features from the featurizer and updates the `info` dictionary.

        Parameters:
        - info: The dictionary to store extracted features.
        - featurizer: The featurizer object containing geometry features.
        - atomFRname: The key for storing reactant geometry features.
        - atomFPname: The key for storing product geometry features.

        Logic:
        1. Extract reactant geometry features and store them in `info`.
        2. Extract product geometry features and store them in `info`.

        Returns:
        - info: The updated dictionary containing geometry-related features.
        """
        info[atomFRname] = featurizer.reactant.atom_geometry_features
        info[atomFPname] = featurizer.product.atom_geometry_features
        return info
    def __reactiongeominfomulticonf(self, info, featurizer, atomFRname, atomFPname):
        atomconf = self.__singleormulticonfgeometry(featurizer.reactant, 'atom')
        info[atomFRname] = getattr(featurizer.reactant, atomconf).values()
        info[atomFPname] = getattr(featurizer.product, atomconf).values()
        return info
    def __reactiongeominfo(self, info, featurizer, atomFRname, atomFPname):
        """
        This function extracts geometry-related features from the featurizer and updates the `info` dictionary.

        Parameters:
        - info: The dictionary to store extracted features.
        - featurizer: The featurizer object containing geometry features.
        - atomFRname: The key for storing reactant geometry features.
        - atomFPname: The key for storing product geometry features.

        Logic:
        1. Check if `featurizer` is an instance of `ReactionFeaturizerwithPaddingandGeometry`.
            a. If True, call `__reactiongeominfomulticonf` to extract multi-conformation features.
            b. If False, call `__reactiongeominfosingleconf` to extract single-conformation features.

        Returns:
        - info: The updated dictionary containing geometry-related features.
        """
        if 'features_' in self.__singleormulticonfgeometry(featurizer.reactant, 'atom'):
            info = self.__reactiongeominfomulticonf(info, featurizer, atomFRname, atomFPname)
        else:
            info = self.__reactiongeominfosingleconf(info, featurizer, atomFRname, atomFPname)
        return info




    def __geominfo(self, info, featurizer, atomFRname, atomFPname, reaction=False):
        """
        This function extracts geometry-related features from the featurizer and updates the `info` dictionary.

        Parameters:
        - info: The dictionary to store extracted features.
        - featurizer: The featurizer object containing geometry features.
        - atomFRname: The key for storing reactant geometry features.
        - atomFPname: The key for storing product geometry features (if applicable).
        - reaction: A boolean indicating whether the input represents a reaction (default is False).

        Logic:
        1. Check if geometry features are enabled using `__usegeometryvector`.
        2. If `reaction` is True:
            a. Extract reactant geometry features and store them in `info`.
            b. Extract product geometry features and store them in `info`.
        3. If `reaction` is False:
            a. Extract geometry features and store them in `info`.

        Returns:
        - info: The updated dictionary containing geometry-related features.
        """
        geomusecase, _ = self.__usegeometryvector()
        if reaction: 
            if geomusecase: info = self.__reactiongeominfo(info, featurizer, atomFRname, atomFPname)  
        else: 
            if geomusecase: info = self.__moleculargeominfo(info, featurizer, atomFRname, atomFPname)
        return info
    
    def _addfeatstoinfosinglecomp(self, info, featurizer, reaction=False, sequel=None,i=None):
        """
            This function adds features to the `info` dictionary based on the featurizer object and the type of input (reaction or molecule).

            Parameters:
            - info: The dictionary where features will be stored.
            - featurizer: The featurizer object containing features.
            - reaction: A boolean indicating whether the input represents a reaction (default is False).
            - sequel: An optional suffix to append to feature names.
            - i: An optional index for handling multiple components.

            Logic:
            1. Generate edge, atom, bond, and geometry feature names using helper functions:
                a. Call `__edgedictnames` to get edge names (`uname`, `vname`).
                b. Call `__atomfromdictnames` to get atom feature names (`atomFRname`, `atomFPname`).
                c. Call `__bondfromdictnames` to get bond feature names (`bondFRname`, `bondFPname`).
                d. Call `__geomfromdictnames` to get geometry feature names (`atomgeomnameR`, `atomgeomnameP`).
            2. Check if `i` is provided:
                a. If `i` is None:
                i. Check the type of input (`reaction` or `molecular`):
                    - If `reaction`, call `__reactioninfo` to update `info` with reaction-related features.
                    - If `molecular`, call `__molecularinfo` to update `info` with molecular-related features.
                ii. Call `__geominfo` to update `info` with geometry-related features.
                b. If `i` is provided:
                i. Check the type of input (`reaction` or `molecular`):
                    - If `reaction`, call `__reactioninfo` with indexed parameters to update `info`.
                    - If `molecular`, call `__molecularinfo` with indexed parameters to update `info`.
                ii. Call `__geominfo` with indexed parameters to update `info` with geometry-related features.
            3. Return the updated `info` dictionary.

            Returns:
            - info: The updated dictionary containing added features.
            """
        uname, vname = self.__edgedictnames(sequel)
        atomFRname, atomFPname = self.__atomfromdictnames(reaction, sequel)
        bondFRname, bondFPname = self.__bondfromdictnames(reaction, sequel)
        atomgeomnameR, atomgeomnameP = self.__geomfromdictnames(reaction, sequel)
        if i == None: 
            if 'reaction' in self.params.graph:
                info = self.__reactioninfo(info, featurizer, uname, vname, atomFRname, atomFPname, bondFRname, bondFPname)
            elif 'molecular' in self.params.graph:
                info = self.__molecularinfo(info, featurizer, uname, vname, atomFRname, bondFRname)
            info = self.__geominfo(info, featurizer, atomgeomnameR, atomgeomnameP, reaction)
        else:
            if 'reaction' in self.params.graph:
                info = self.__reactioninfo(info, featurizer[i], uname, vname, atomFRname, atomFPname, bondFRname, bondFPname)
            elif 'molecular' in self.params.graph:
                info = self.__molecularinfo(info, featurizer[i], uname, vname, atomFRname, bondFRname)
            info = self.__geominfo(info, featurizer[i], atomgeomnameR, atomgeomnameP, reaction)
        return info
    
    def _addfeatstoinfomulticomp(self,info,featurizer,reaction=False):
        if reaction: reaction = True 
        else: reaction = False
        for i,smi in enumerate(self.params.smiles):
            info = self._addfeatstoinfosinglecomp(info, featurizer, reaction=reaction, sequel=smi,i=i)
        return info

    
    def _addfeatstoinfo(self, info, featurizer, reaction=False):
        """
        This function adds features to the `info` dictionary based on the featurizer object and the type of input (reaction or molecule).

        Parameters:
        - info: The dictionary where features will be stored.
        - featurizer: The featurizer object containing features.
        - reaction: A boolean indicating whether the input represents a reaction (default is False).
        - sequel: An optional suffix to append to feature names.

        Logic:
        1. Generate edge, atom, bond, and geometry feature names using helper functions:
            a. Call `__edgedictnames` to get edge names (`uname`, `vname`).
            b. Call `__atomfromdictnames` to get atom feature names (`atomFRname`, `atomFPname`).
            c. Call `__bondfromdictnames` to get bond feature names (`bondFRname`, `bondFPname`).
            d. Call `__geomfromdictnames` to get geometry feature names (`atomgeomnameR`, `atomgeomnameP`).
        2. Check the type of input (`reaction` or `molecular`):
            a. If `reaction`, call `__reactioninfo` to update `info` with reaction-related features.
            b. If `molecular`, call `__molecularinfo` to update `info` with molecular-related features.
        3. Call `__geominfo` to update `info` with geometry-related features.
        4. Return the updated `info` dictionary.

        Returns:
        - info: The updated dictionary containing added features.
        """
        if isinstance(self.params.smiles,str):
            info = self._addfeatstoinfosinglecomp(info, featurizer, reaction=reaction)
        else:
            info = self._addfeatstoinfomulticomp(info, featurizer, reaction=reaction)
        return info
    
    def AddFeatures(self):
        """
        This function adds features to the `info` dictionary based on the featurizer object and the type of input (reaction or molecule).

        Logic:
        1. Check if `featurizer` is an instance of `ReactionFeaturizerwithPaddingandGeometry`.
            a. If True, call `_addfeatstoinfosinglecomp` to add features for multiple conformations.
            b. If False, call `_addfeatstoinfomulticomp` to add features for single conformation.

        Returns:
        - None: The function updates the `info` dictionary in place.
        """
        if 'reaction' in self.params.graph: reaction = True
        else: reaction = False

        self.info = self._addfeatstoinfo(self.info, self.featurizer, reaction=reaction)
        self.GrabFeatureLengths()
    
    def GrabFeatureLengths(self):
        """
        This function retrieves the lengths of various features from the `info` dictionary.

        Logic:
        1. Check if `featurizer` is an instance of `ReactionFeaturizerwithPaddingandGeometry`.
            a. If True, retrieve lengths for reactant and product features.
            b. If False, retrieve lengths for molecular features.

        Returns:
        - None: The function updates the `info` dictionary in place.
        """
        
        if isinstance(self.params.smiles,str):
            self.node_feature_length = self.info['atom_F_R'].shape[-1]
            self.bond_feature_length = self.info['bond_F_R'].shape[-1]
        else:
            self.node_feature_length = []
            self.bond_feature_length = []
            for smi in self.params.smiles:
                self.node_feature_length.append(np.array(self.info[f'atom_F_R_{smi}']).shape[-1])
                self.bond_feature_length.append(np.array(self.info[f'bond_F_R_{smi}']).shape[-1])
            self.node_feature_length = max(self.node_feature_length)
            self.bond_feature_length = max(self.bond_feature_length)
        ic(self.node_feature_length,self.bond_feature_length)

