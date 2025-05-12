
import pandas as pd
import torch,dgl
from icecream import ic

class Expansions:
    def __init__(self):
        pass

    def createPandasDataframe(self,names,types,Rgraphs,smiles,targets,additionals=None,Raddons=None,Pgraphs=None,Paddons=None):
        df = dict()
        df['names'] = names
        df['types'] = types
        df['smiles'] = smiles
        df['targets'] = targets
        df['Rgraphs'] = Rgraphs
        if additionals != None: df['additionals'] = additionals
        if Raddons != None: df['Radd'] = Raddons
        if Pgraphs != None: df['Pgraphs'] = Pgraphs
        if Paddons != None: df['Padd'] = Paddons
        df = pd.DataFrame(df)
        return df 

    def processnames(self,df):
        names = df['names'].explode(ignore_index=False)
        return names.tolist()
    
    def processtypesandsmiles(self,df,col='smiles',multicomp=False):
        if col in ['targets','additionals']: df[col] = df[col].apply(lambda x: torch.tensor(x) if isinstance(x, (list, tuple)) else torch.tensor([x]))
        return df[col].tolist()
        
    def processgraphs(self,df,graph='Rgraphs',multicomp=False):
        if multicomp == False:
            graphs = df[graph].explode(ignore_index=False)
        else:
            graphs = df[graph].explode(ignore_index=False)
            graphs = graphs.explode()
        return graphs.tolist()
    
    def processaddons(self,df,col='Radd',multicomp=False):
        return df[col].tolist()
    
    def processResults(self,names,types,Rgraphs,smiles,targets,additionals=None,Raddons=None,Pgraphs=None,Paddons=None,multicomp=False):
        df = self.createPandasDataframe(names,types,Rgraphs,smiles,targets,additionals,Raddons,Pgraphs,Paddons)
        names = self.processnames(df)
        types = self.processtypesandsmiles(df,col='types',multicomp=multicomp)
        smiles = self.processtypesandsmiles(df,col='smiles',multicomp=multicomp)
        targets = self.processtypesandsmiles(df,col='targets')
        Rgraphs = self.processgraphs(df,graph='Rgraphs',multicomp=multicomp)
        if additionals is not None: additionals = self.processtypesandsmiles(df,col='additionals')
        if Raddons is not None: Raddons = self.processaddons(df,col='Radd')
        if Pgraphs is not None: Pgraphs = self.processgraphs(df,graph='Pgraphs',multicomp=True)
        if Paddons is not None: Paddons = self.processaddons(df,col='Padd',multicomp=True)
        return names,types,Rgraphs, smiles,targets,additionals,Raddons,Pgraphs,Paddons
    

def molecularexpandfcnfortargets(names,types,Rgraphs, smiles,targets,multicomp=False):
        processor = Expansions()
        names,types,Rgraphs, smiles,targets,_,_,_,_ = processor.processResults(names,types,Rgraphs, smiles,targets,multicomp=multicomp)
        Rgraphs = batchgraphs(Rgraphs)
        return names,types,Rgraphs, smiles,targets
    
def molecularexpandfcnforadditionals(names,types,Rgraphs, smiles,targets,additionals,multicomp=False):
    processor = Expansions()
    names,types,Rgraphs, smiles,targets,additionals,_,_,_ = processor.processResults(names,types,Rgraphs, smiles,targets,additionals,multicomp=multicomp)
    #ic(names,types,Rgraphs, smiles,targets,additionals)
    Rgraphs = batchgraphs(Rgraphs)
    return names,types,Rgraphs, smiles,targets,additionals

def molecularexpandfcnforaddons(names,types,Rgraphs, smiles,targets,Radd,multicomp=False):
    processor = Expansions()
    names,types,Rgraphs, smiles,targets,_,Radd,_,_ = processor.processResults(names,types,Rgraphs, smiles,targets,Raddons=Radd,multicomp=multicomp)
    Rgraphs = batchgraphs(Rgraphs)
    return names,types,Rgraphs, smiles,targets,Radd

def molecularexpandfcnforallprops(names,types,Rgraphs, smiles,targets,additionals,Radd):
    processor = Expansions()
    names,types,Rgraphs, smiles,targets,additionals,Radd,_,_ = processor.processResults(names,types,Rgraphs, smiles,targets,additionals,Raddons=Radd)
    Rgraphs = batchgraphs(Rgraphs)
    return names,types,Rgraphs, smiles,targets,additionals,Radd


def batchgraphs(graph):
    if isinstance(graph, (list, tuple)):
        graphdb = pd.DataFrame(graph)
        graphs = [] 
        for col in graphdb.columns:
            listofgraph = graphdb[col].tolist()
            graphs.append(dgl.batch(listofgraph))
        return graphs
    else: 
        return dgl.batch(graph)

def reactionexpandfcnfortargets(names,types,Rgraphs, Pgraphs,smiles,targets):
        processor = Expansions()
        names,types,Rgraphs, smiles,targets,_,_,_,_ = processor.processResults(names,types,Rgraphs, smiles,targets,Pgraphs=Pgraphs)
        Rgraphs = batchgraphs(Rgraphs)
        Pgraphs = batchgraphs(Pgraphs)
        return names,types,Rgraphs,Pgraphs, smiles,targets
    
def reactionexpandfcnforadditionals(names,types,Rgraphs,Pgraphs, smiles,targets,additionals):
    processor = Expansions()
    names,types,Rgraphs, smiles,targets,additionals,_,Pgraphs,_ = processor.processResults(names,types,Rgraphs, smiles,targets,additionals,Pgraphs=Pgraphs)
    Rgraphs = batchgraphs(Rgraphs)
    Pgraphs = batchgraphs(Pgraphs)
    return names,types,Rgraphs,Pgraphs, smiles,targets,additionals

def reactionexpandfcnforaddons(names,types,Rgraphs,Pgraphs, smiles,targets,Radd,Padd):
    processor = Expansions()
    names,types,Rgraphs, smiles,targets,_,Radd,Pgraphs,Padd = processor.processResults(names,types,Rgraphs, smiles,targets,Raddons=Radd,Pgraphs=Pgraphs,Paddons=Padd)
    Rgraphs = batchgraphs(Rgraphs)
    Pgraphs = batchgraphs(Pgraphs)
    return names,types,Rgraphs,Pgraphs, smiles,targets,Radd

def reactionexpandfcnforallprops(names,types,Rgraphs,Pgraphs, smiles,targets,additionals,Radd,Padd):
    processor = Expansions()
    names,types,Rgraphs, smiles,targets,additionals,Radd,Pgraphs,Padd = processor.processResults(names,types,Rgraphs, smiles,targets,additionals,Raddons=Radd,Pgraphs=Pgraphs,Paddons=Padd)
    Rgraphs = batchgraphs(Rgraphs)
    Pgraphs = batchgraphs(Pgraphs)
    return names,types,Rgraphs,Pgraphs, smiles,targets,additionals,Radd,Padd

