from ....tools.jepa.tools import mask_uv_vectors,mask_node_features,mask_node_features_v2,mask_edge_features,mask_edge_features_v2,mask_node_and_edges,mask_node_and_edges_v2,mask_node_and_edges_v3,mask_node_and_neighbors
import torch,dgl
import numpy as np
from icecream import ic

class DGLGraphCreation:
    """
    This class is responsible for creating DGL graphs from molecular and reaction data.
    
    Attributes:
        params: Configuration parameters for the graph creation process.
        info: A dictionary to store various information related to molecules and reactions.
        featurizer: A dictionary to store featurizers for different components (e.g., molecules, reactions).
        infolist: A list to store additional information or features.
        rxntype_columns: A list to store columns related to reaction types.
    """
    
    def __init__(self, params,info=None):
        self.params = params
        self.info = info
        self.__grabmatkeys(info)
        self._convmolecularatomandbondfeatstomat(info)
    
    def DGLSetup(self):
        self.__grabmatkeys(self.info)
        self._convmolecularatomandbondfeatstomat(self.info)

    def __grabmatkeys(self,info):
        self.matkeys = []
        for key in info.keys():
            if key.startswith('atom_F_'):
                self.matkeys.append(key)
            elif key.startswith('bond_F_'):
                self.matkeys.append(key)
            elif key.startswith('Geom_'):
                self.matkeys.append(key)
    
    def __grabuv(self,info,uname,vname):
        u, v = torch.Tensor(info[uname]).int(),torch.Tensor(info[vname]).int()
        if self.params.jepa: u,v = mask_uv_vectors(u,v)
        return u,v
    
    def __convfeatstomat(self,info,key):
        """
        This function converts the features stored in the info dictionary into a array format.
        
        Parameters:
            info: A dictionary containing information about the molecular or reaction data.
            key: The key in the info dictionary to retrieve the features.
        
        Returns:
            atomFR: A tensor of atom features.
            bondFR: A tensor of bond features.
            atomgeom: A tensor of atomic geometry features.
        """
        return np.array(info[key])
    
    def _convmolecularatomandbondfeatstomat(self,info):
        """
        This function retrieves the atomic and bond features from the provided information dictionary.
        
        Parameters:
            info: A dictionary containing information about the molecular or reaction data.
        
        Returns:
            atomFR: A tensor of atomic features.
            bondFR: A tensor of bond features.
            atomgeom: A tensor of atomic geometry features.
        """
        self.matkeyshape = dict()
        for keys in self.matkeys:
            info[keys] = self.__convfeatstomat(info,keys)
            self.matkeyshape[keys] = len(info[keys].shape)
        return info

    def _getuv(self,info,smiles=None):
        """
        This function retrieves the unique node and edge features from the provided information dictionary.
        
        Parameters:
            info: A dictionary containing information about the molecular or reaction data.
            smiles: Optional; a SMILES string for additional processing (default is None).
        
        Returns:
            ulist: A list of unique node features.
            vlist: A list of unique edge features.
        """
        if smiles is not None:
            uname = f'u_{smiles}'
            vname = f'v_{smiles}'
            u,v = self.__grabuv(info,uname,vname)
        else:
            u,v = self.__grabuv(info,'u','v')
        return u,v
    
    def _dglgraphcreation(self,u,v):
        gR   = dgl.graph((u,v))
        gP   = dgl.graph((u,v))
        return gR,gP
    
    def _dglmoleculargraphfcn(self,gR,atomFR,bondFR,atomgeom):
        try:
            gR.ndata['x'] = torch.Tensor(atomFR)
            gR.edata['x'] = torch.Tensor(bondFR)
        except:
            gR.ndata['x'] = torch.Tensor(atomFR[0])
            gR.edata['x'] = torch.Tensor(bondFR[0])
        gR = self._addgeomstodglgraph(gR,atomgeom)
        if self.params.jepa: chosen_index = self.RunJEPA(gR)
        return gR

    def _dglreactiongraphfcn(self,gR,gP,atomFR,atomFP,bondFR,bondFP,atomgeomR,atomgeomP):
        gR = self._dglmoleculargraphfcn(gR,atomFR,bondFR,atomgeomR)
        gP = self._dglmoleculargraphfcn(gP,atomFP,bondFP,atomgeomP)
        return gR,gP
        
    def _addgeomstodglgraph(self,g,atomgeom):
        if self.params.dimension == '3d':
            if atomgeom != None: g.ndata['g'] = torch.Tensor(atomgeom)
        return g
    
    def _grabwhichkeyhassmiles(self,smiles=None):
        """
        This function retrieves the key in the info dictionary that contains the SMILES string.
        
        Parameters:
            smiles: Optional; a SMILES string for additional processing (default is None).
        
        Returns:
            key: The key in the info dictionary that contains the SMILES string.
        """
        if smiles is not None:
            return [key for key in self.matkeys if smiles in key]
        else:
            return self.matkeys

    def _dglmoleculargraphsinglecomp(self,info,smiles=None):
        """
        This function creates a DGL graph from the provided information dictionary for a single molecular conformation.
        
        Parameters:
            info: A dictionary containing information about the molecular or reaction data.
        
        Returns:
            gR: A DGL graph representing the reactant.
            gP: A DGL graph representing the product.
        """
        u,v = self._getuv(info,smiles)
        gR,gP = self._dglgraphcreation(u,v)
        relevantkeys = self._grabwhichkeyhassmiles(smiles)
        atomFR,bondFR,atomgeom = None,None,None
        gRs = []
        
        for key in relevantkeys:
            if self.matkeyshape[key] == 3:
                for i in range(info[key].shape[0]):
                    atomFR,bondFR,atomgeom = self._grabrelevantmatsmolecule(info,relevantkeys,i)
                    gR = self._dglmoleculargraphfcn(gR,atomFR,bondFR,atomgeom)
                    gRs.append(gR)
                return gRs
            else:
                atomFR,bondFR,atomgeom = self._grabrelevantmatsmolecule(info,relevantkeys,None)
                gR = self._dglmoleculargraphfcn(gR,atomFR,bondFR,atomgeom)
                return gR
    
    def _grabkeyname(self,relevantkeys,prefix='Geom_'):
        if any(prefix in key for key in relevantkeys):
            key_ = [key for key in relevantkeys if prefix in key][0]
        else:
            key_ = None
        return key_
    
    def _grabrelevantmatsmolecule(self,info,relevantkeys,i=0):
        atomkey = self._grabkeyname(relevantkeys,prefix='atom_F_')
        bondkey = self._grabkeyname(relevantkeys,prefix='bond_F_')
        geomkey = self._grabkeyname(relevantkeys,prefix='Geom_')

        if atomkey == None: atomFR = None
        else: 
            if i != None: atomFR = info[atomkey][i]
            else: atomFR = info[atomkey]
        if bondkey == None: bondFR = None
        else:
            if i != None: bondFR = info[bondkey][i]
            else: bondFR = info[bondkey]
        if geomkey == None: atomgeom = None
        else:
            if i != None: atomgeom = info[geomkey][i]
            else: atomgeom = info[geomkey]
        
        return atomFR,bondFR,atomgeom

    def _grabrelevantmatsreaction(self,info,relevantkeys,i=0):
        atomkeyR = self._grabkeyname(relevantkeys,prefix='atom_F_R')
        bondkeyR = self._grabkeyname(relevantkeys,prefix='bond_F_R')
        geomkeyR = self._grabkeyname(relevantkeys,prefix='Geom_R')
        atomkeyP = self._grabkeyname(relevantkeys,prefix='atom_F_P')
        bondkeyP = self._grabkeyname(relevantkeys,prefix='bond_F_P')
        geomkeyP = self._grabkeyname(relevantkeys,prefix='Geom_P')

        if atomkeyR == None: atomFR = None
        else: 
            if i != None: atomFR = info[atomkeyR][i]
            else: atomFR = info[atomkeyR]
        if bondkeyR == None: bondFR = None
        else:
            if i != None: bondFR = info[bondkeyR][i]
            else: bondFR = info[bondkeyR]
        if geomkeyR == None: atomgeomR = None
        else:
            if i != None: atomgeomR = info[geomkeyR][i]
            else: atomgeomR = info[geomkeyR]

        if atomkeyP == None: atomFR = None
        else: 
            if i != None: atomFP = info[atomkeyP][i]
            else: atomFP = info[atomkeyP]
        if bondkeyP == None: bondFP = None
        else:
            if i != None: bondFP = info[bondkeyP][i]
            else: bondFP = info[bondkeyP]
        if geomkeyP == None: atomgeomP = None
        else:
            if i != None: atomgeomP = info[geomkeyP][i]
            else: atomgeomP = info[geomkeyP]
        
        return atomFR,bondFR,atomgeomR,atomFP,bondFP,atomgeomP

    def _dglreactiongraphsinglecomp(self,info,smiles=None):
        """
        This function creates a DGL graph from the provided information dictionary for a single molecular conformation.
        
        Parameters:
            info: A dictionary containing information about the molecular or reaction data.
        
        Returns:
            gR: A DGL graph representing the reactant.
            gP: A DGL graph representing the product.
        """
        u,v = self._getuv(info)
        gR,gP = self._dglgraphcreation(u,v)
        relevantkeys = self._grabwhichkeyhassmiles(smiles)
        atomFR,bondFR,atomgeomR = None,None,None
        atomFP,bondFP,atomgeomP = None,None,None
        gRs = []
        for key in relevantkeys:
            if self.matkeyshape[key] == 3:
                for i in range(info[key].shape[0]):
                    atomFR,bondFR,atomgeomR,atomFP,bondFP,atomgeomP = self._grabrelevantmatsreaction(info,relevantkeys,i)
                    gR = self._dglreactiongraphfcn(gR,atomFR,bondFR,atomgeomR,atomFP,bondFP,atomgeomP)
                    gRs.append(gR)
                return gRs
            else:
                atomFR,bondFR,atomgeomR,atomFP,bondFP,atomgeomP = self._grabrelevantmatsreaction(info,relevantkeys)
                gR = self._dglmoleculargraphfcn(gR,atomFR,bondFR,atomgeomR,atomFP,bondFP,atomgeomP)
                return gR

    def addGraphsSingleComp(self,info,smiles=None):
        if smiles is not None:
            Rgraphname = 'Graph_R_' + smiles
            Pgraphname = 'Graph_P_' + smiles
        else:
            Rgraphname = 'Graph_R'
            Pgraphname = 'Graph_P'
        if 'reaction' in self.params.graph:
            info[Rgraphname],info[Pgraphname] = self._dglreactiongraphsinglecomp(info,smiles) 
        else:
            info[Rgraphname] = self._dglmoleculargraphsinglecomp(info,smiles)
        return info

    

    def addGraphsMultiComp(self,info):
        for smiles in self.params.smiles:
            info = self.addGraphsSingleComp(info,smiles)
        return info
                
    def AddGraphs(self):
        """
        This function adds DGL graphs to the provided information dictionary.
        
        Parameters:
            info: A dictionary containing information about the molecular or reaction data.
        
        Returns:
            info: The updated dictionary with added DGL graphs.
        """
        info = self.info
        if isinstance(self.params.smiles, str):
            info = self.addGraphsSingleComp(info)
        else:
            info = self.addGraphsMultiComp(info)
        self.info = info

    def RunJEPA(self,graph):
        if self.params.jepa:
            if self.params.jepa_masking == 'node':
                mask_node_features(graph)
            elif self.params.jepa_masking == 'edge':
                mask_edge_features(graph)
            elif self.params.jepa_masking == 'node+edge':
                mask_node_and_edges(graph)
            elif self.params.jepa_masking == 'node+edge_v2':
                mask_node_and_edges_v2(graph,self.params.jepa_num_nodes)
            elif self.params.jepa_masking == 'node+edge_v3':
                mask_node_and_edges_v3(graph,self.params.jepa_num_nodes,self.params.jepa_num_edges)
            elif self.params.jepa_masking == 'node+neighbors':
                mask_node_and_neighbors(graph,self.params.jepa_neighbors)
            elif self.params.jepa_masking == 'node_v2':
                mask_node_features_v2(graph, self.params.jepa_num_nodes)
            elif self.params.jepa_masking == 'edge_v2':
                mask_edge_features_v2(graph, self.params.jepa_num_edges)

    

