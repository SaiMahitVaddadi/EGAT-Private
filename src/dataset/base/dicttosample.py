import pandas as pd
import itertools,torch
from icecream import ic
from dataclasses import dataclass
from typing import List, Union, Optional

@dataclass
class SampleParams:
    graph: str  # 'reaction' or 'molecular'
    addons: Optional[str,List[str]] = None
    addonmixing: Optional[bool] = False
    fingerprint: Optional[bool] = False

class InfoToSample:
    def __init__(self, params,info):
        self.info = info
        self.params = params
        self.sample = []
        self._grabkeysfrominfo()
        self._checkgraphkeys()

    def _grabkeysfrominfo(self): # write a special function for the addons. 
        self.infoRinchikeys = []
        self.infoRsmikeys = []
        self.infoRgraphkeys = [] 
        self.infoPinchikeys = []
        self.infoPsmikeys = []
        self.infoPgraphkeys = []
        self.rxntypekeys = []
        for each_key in self.info.keys():
            if 'Rinchi' in each_key:
                self.infoRinchikeys += [each_key]
            elif 'Rsmile' in each_key:
                self.infoRsmikeys += [each_key]
            elif 'Graph_R' in each_key:
                self.infoRgraphkeys += [each_key]
            elif 'Pinchi' in each_key:
                self.infoPinchikeys += [each_key]
            elif 'Psmile' in each_key:
                self.infoPsmikeys += [each_key]
            elif 'Paddon' in each_key:
                self.infoPaddonkeys += [each_key]
            elif 'Graph_P' in each_key:
                self.infoPgraphkeys += [each_key]
            elif 'rxntype' in each_key:
                self.rxntypekeys += [each_key]

        aseries = pd.Series(list(self.info.keys()))
        if isinstance(self.params.smiles,list)and self.params.addonmixing: self.infoRaddonkeys = aseries[aseries.str.contains('R_Addon') & aseries.str.contains('|'.join(self.params.smiles))].tolist()
        else: self.infoRaddonkeys = aseries[aseries == 'R_Addon'].tolist()
        
        if isinstance(self.params.smiles,list) and self.params.addonmixing: self.infoPaddonkeys = aseries[aseries.str.contains('P_Addon') & aseries.str.contains('|'.join(self.params.smiles))].tolist()
        else: self.infoPaddonkeys = aseries[aseries == 'P_Addon'].tolist()
    
    def _checkifkeyisgraphorlist(self,key):
        if isinstance(self.info[key], list):
            return ['list',len(self.info[key])]
        else:
            self.info[key] = [self.info[key]]
            return ['graph',1]
        
    def _checkgraphkeys(self):
        if len(self.infoRgraphkeys) > 0:
            for each_key in self.infoRgraphkeys:
                self._checkifkeyisgraphorlist(each_key)
        if len(self.infoPgraphkeys) > 0:
            for each_key in self.infoPgraphkeys:
                self._checkifkeyisgraphorlist(each_key)
            

    def handleindex(self):
        self.sample += [[self.info['Indices']]]

    def handleinchikeysmolecular(self):
        if len(self.infoRinchikeys) > 1:
            self.sample += [[[self.info[each_key] for each_key in self.infoRsmikeys],
                             [self.info[each_key] for each_key in self.infoRinchikeys]]]
        else:
            self.sample += [[self.info[self.infoRsmikeys[0]],
                             self.info[self.infoRinchikeys[0]]]]
    
    def handleinchikeysreaction(self):
        if len(self.infoPinchikeys) > 1:
            self.sample += [[[self.info[each_key] for each_key in self.infoRsmikeys],
                             [self.info[each_key] for each_key in self.infoPsmikeys],
                             [self.info[each_key] for each_key in self.infoRinchikeys],
                             [self.info[each_key] for each_key in self.infoPinchikeys]]]
        else:
            self.sample += [[self.info[self.infoRsmikeys[0]],
                             self.info[self.infoPsmikeys[0]],
                             self.info[self.infoRinchikeys[0]],
                             self.info[self.infoPinchikeys[0]]]]
            
    def handleinchikeys(self):
        if self.params.graph == 'reaction':
            self.handleinchikeysreaction()
        elif self.params.graph == 'molecular':
            self.handleinchikeysmolecular()

    def handlerxntype(self):
        if len(self.rxntypekeys) > 1:
            self.sample += [[self.info[each_key] for each_key in self.rxntypekeys]]
        elif len(self.rxntypekeys) == 1:
            self.sample += [[self.info[self.rxntypekeys[0]]]]
        else:
            if isinstance(self.params.smiles,list):
                self.sample += [['all' for i in range(len(self.params.smiles))]]
            else:
                self.sample += [['all']]
            
    def handletargets(self):
        if isinstance(self.params.target,list):
            self.sample += [[self.info[each_key] for each_key in self.params.target]]
        else:
            self.sample += [[self.info[self.params.target]]]

    def handleadditionals(self):
        if isinstance(self.params.additional,list):
            self.sample += [[self.info[each_key] for each_key in self.params.additional]]
        elif isinstance(self.params.additional,str):
            self.sample += [[self.info[self.params.additional]]]
    
    def handleaddonsmolecular(self):
        if len(self.infoRaddonkeys) > 1:
            self.sample += [[torch.Tensor(self.info[each_key]) for each_key in self.infoRaddonkeys]]
        else:
            self.sample += [[torch.Tensor(self.info[self.infoRaddonkeys[0]])]]
    
    def handleaddonsreaction(self):
        if len(self.infoRaddonkeys) > 1:
            self.sample += [[torch.Tensor(self.info[each_key]) for each_key in self.infoRaddonkeys],
                             [torch.Tensor(self.info[each_key]) for each_key in self.infoPaddonkeys]]
        else:
            self.sample += [[torch.Tensor(self.info[self.infoRaddonkeys[0]])],
                             [torch.Tensor(self.info[self.infoPaddonkeys[0]])]]
        
    def handleaddons(self):
        if self.params.graph == 'reaction':
            self.handleaddonsreaction()
        elif self.params.graph == 'molecular':
            self.handleaddonsmolecular()
    
    def singlecompgraphcase(self,key):
            return [self.info[key]]
    
    def multicompcasebasefcn(self,keycheck = None):
        iterdict = {key: self.info[key] for key in keycheck}
        values = list(iterdict.values())  # list of lists
        # Step 2: Generate Cartesian product of all value lists
        combinations = list(itertools.product(*values))
        return [combinations]


    def handlegraphsmolecular(self):
        if len(self.infoRgraphkeys) > 1:
            self.sample += [self.multicompcasebasefcn(self.infoRgraphkeys)]
        elif len(self.infoRgraphkeys) == 1:
            self.sample += [self.singlecompgraphcase(self.infoRgraphkeys[0])]
    
    def handlegraphreaction(self):
        if len(self.infoRgraphkeys) > 1:
            self.sample += [[self.multicompcasebasefcn(self.infoRgraphkeys),
                             self.multicompcasebasefcn(self.infoPgraphkeys)]]
        elif len(self.infoRgraphkeys) == 1:
            self.sample += [[self.singlecompgraphcase(self.infoRgraphkeys[0]),
                             self.singlecompgraphcase(self.infoPgraphkeys[0])]]
    
    def handlegraph(self):
        if self.params.graph == 'reaction':
            self.handlegraphreaction()
        elif self.params.graph == 'molecular':
            self.handlegraphsmolecular()

    def creategraphsample(self):
        self.sample = []
        self._grabkeysfrominfo()
        self._checkgraphkeys()
        self.handleindex()
        self.handlerxntype()
        if self.params.fingerprint == False: self.handlegraph()
        self.handleinchikeys()
        self.handletargets()
        if self.params.additional is not None: self.handleadditionals()
        if self.params.addons is not None: self.handleaddons()
        
    def run(self):
        self.creategraphsample()
        return self.sample
    