import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from ..dataset.Dataset import GraphDataset,FingerprintDataset
from ..dataset.external.dask import GraphDaskDataset,FingerprintDaskDataset
from ..dataset.external.json import GraphJSONDataset,FingerprintJSONDataset
from ..dataset.external.toml import GraphTOMLDataset,FingerprintTOMLDataset
from ..dataset.external.hdf import GraphHDF5Dataset,FingerprintHDF5Dataset
from ..dataset.external.mongo import GraphMongoDataset,FingerprintMongoDataset
from ..dataset.external.sqlite import GraphSQLDataset,FingerprintSQLDataset
from ..dataset.external.pandas import GraphPandasDataset,FingerprintPandasDataset
from ..dataset.external.spark import GraphSparkDataset,FingerprintSparkDataset
from ..dataset.external.polars import GraphParquetDataset,FingerprintParquetDataset
from ..dataset.external.postgres import GraphPostgresQLDataset,FingerprintPostgresQLDataset
from ..dataset.collate.molecule import MolecularCollator
from ..dataset.collate.reaction import ReactionCollator

@dataclass
class DataLoaderParams:
    data_path: str
    exclude: Optional[str] = None
    test_only: bool = False
    root: str = ''
    class_choice: Optional[str] = None
    randomize: bool = False
    fold: Optional[int] = None
    foldtype: Optional[str] = None
    size: Optional[int] = None
    target: Optional[str] = None
    additionals: Optional[List[str]] = None
    addons: bool = False
    graph: Optional[str] = None  # Updated to replace 'molecular' with 'graph' for clarity
    fingerprint: bool = False  # Added to specify if fingerprint datasets are used
    imblearn: bool = False  # Added to handle imbalanced datasets
    batch_size: int = 32


class DataLoaderCommands:
    def __init__(self,arguments):
        self.params = arguments

    def setuploader(self):
        self.egatdataset = dict()
        self.egatdataloader = dict()
        if self.params.test_only:
            splits = ['train','test']
        else:
            splits = ['train','val','test']
        self.splits = splits

    def excludedata(self):
        ### Get the files to exclude. If not, just set it blank. 
        if self.params.exclude is not None:
            if os.path.isfile(os.path.join(self.params.data_path,self.params.exclude)): # Check if path exists
                exclude = []
                with open(os.path.join(self.params.data_path,self.params.exclude),'r') as f: # open file 
                    for lc,lines in enumerate(f):
                        exclude.append(lines.split('/')[-1].split('.json')[0]) # Add exclude files 
            else:
                self.logger.info(f'{os.path.join(self.params.data_path,self.params.exclude)} does not exist.')
                exclude = []
        else:
            self.logger.info(f'{os.path.join(self.params.data_path)} has not exclude file.')
            exclude = []
        return exclude 
    
    def choosegraphdataset(self):
        if self.params.dataset == 'onthefly':
            self.datasetfunction = GraphDataset
        elif self.params.dataset == 'dask':
            self.datasetfunction = GraphDaskDataset
        elif self.params.dataset == 'json':
            self.datasetfunction = GraphJSONDataset
        elif self.params.dataset == 'toml':
            self.datasetfunction = GraphTOMLDataset
        elif self.params.dataset == 'hdf5':
            self.datasetfunction = GraphHDF5Dataset
        elif self.params.dataset == 'mongo':
            self.datasetfunction = GraphMongoDataset
        elif self.params.dataset == 'sql':
            self.datasetfunction = GraphSQLDataset
        elif self.params.dataset == 'pandas':
            self.datasetfunction = GraphPandasDataset
        elif self.params.dataset == 'spark':
            self.datasetfunction = GraphSparkDataset
        elif self.params.dataset == 'polars':
            self.datasetfunction = GraphParquetDataset
        elif self.params.dataset == 'postgresql':
            self.datasetfunction = GraphPostgresQLDataset
        else:
            raise ValueError(f'Unknown dataset type: {self.params.dataset}')
    
    def choosefingerprintdataset(self):
        if self.params.dataset == 'onthefly':
            self.datasetfunction = FingerprintDataset
        elif self.params.dataset == 'dask':
            self.datasetfunction = FingerprintDaskDataset
        elif self.params.dataset == 'json':
            self.datasetfunction = FingerprintJSONDataset
        elif self.params.dataset == 'toml':
            self.datasetfunction = FingerprintTOMLDataset
        elif self.params.dataset == 'hdf5':
            self.datasetfunction = FingerprintHDF5Dataset
        elif self.params.dataset == 'mongo':
            self.datasetfunction = FingerprintMongoDataset
        elif self.params.dataset == 'sql':
            self.datasetfunction = FingerprintSQLDataset
        elif self.params.dataset == 'pandas':
            self.datasetfunction = FingerprintPandasDataset
        elif self.params.dataset == 'spark':
            self.datasetfunction = FingerprintSparkDataset
        elif self.params.dataset == 'polars':
            self.datasetfunction = FingerprintParquetDataset
        elif self.params.dataset == 'postgresql':
            self.datasetfunction = FingerprintPostgresQLDataset
        else:
            raise ValueError(f'Unknown dataset type: {self.params.dataset}')

    def choosedataset(self):
        if self.params.fingerprint == False:
            self.choosegraphdataset()
        else:
            self.choosefingerprintdataset()

    def createdatsets(self):
        for split in self.splits:
            if self.split == 'test':
                self.egatdataset[split] = self.datasetfunction(root=self.params.root,split=split, class_choice=self.params.test_class_choice, exclude=self.excluded,randomize=self.params.randomize,fold=self.params.fold,foldtype=self.params.foldtype,size=self.params.size,target=self.params.target,additional=self.params.additionals,hasaddons=self.params.addons,molecular=self.params.graph,test=True)
            else:
                self.egatdataset[split] = self.datasetfunction(root=self.params.root,split=split, class_choice=self.params.class_choice, exclude=self.excluded,randomize=self.params.randomize,fold=self.params.fold,foldtype=self.params.foldtype,size=self.params.size,target=self.params.target,additional=self.params.additionals,hasaddons=self.params.addons,molecular=self.params.graph)

    def addimbalanceddataset(self):
        if self.params.imblearn is not None:
            self.egatdataset['augtrain'] = self.egatdataset['train']
            self.splits.append('augtrain')
        
    def grabcollatefunction(self):
        if self.params.addons is not None: #Check if we need RDKit Global Features. If we do, load them.
            if self.params.additional is not None: # Check if there are added features. If we do, load them.
                if self.params.graph == 'molecular': # Check if we only need molecular features. If we do, only load R features. 
                    self.collator = MolecularCollator.allprops
                elif self.params.graph == 'reaction':
                    self.collator = ReactionCollator.allprops
            else:
                if self.params.graph == 'molecular': # Check if we only need molecular features. If we do, only load R features. 
                    self.collator = MolecularCollator.addons
                elif self.params.graph == 'reaction':
                    self.collator = ReactionCollator.addons
        else:
            if self.params.additional is not None:
                if self.params.graph == 'molecular':
                    self.collator = MolecularCollator.additionals
                elif self.params.graph == 'reaction':
                    self.collator = ReactionCollator.additionals
            else:
                if self.params.graph == 'molecular':
                    self.collator = MolecularCollator.targets
                elif self.params.graph == 'reaction':
                    self.collator = ReactionCollator.targets

    def createdataloader(self):
        for split in self.splits:
            self.egatloader[split] = torch.utils.data.DataLoader(self.egatdataset[split], batch_size=self.params.batch_size, shuffle=True, collate_fn=self.collator)
    
class EGATDataLoader(DataLoaderCommands):
    def __init__(self,arguments):
        super().__init__(arguments)
        self.setuploader()
        self.excluded = self.excludedata()
        self.grabcollatefunction()

    def __call__(self):
        self.choosedataset()
        self.createdatsets()
        self.addimbalanceddataset()
        self.createdataloader()


if __name__ == "__main__":
    # Example usage of the EGATDataLoader
    params = DataLoaderParams(
        data_path="/path/to/data",
        exclude="exclude_file.txt",
        test_only=False,
        root="/path/to/root",
        class_choice="classA",
        randomize=True,
        fold=1,
        foldtype="typeA",
        size=1000,
        target="target_property",
        additionals=["feature1", "feature2"],
        addons=True,
        graph="molecular",
        fingerprint=False,
        imblearn=True,
        batch_size=64,
        dataset="json"
    )

    # Initialize the data loader
    data_loader = EGATDataLoader(params)

    # Call the data loader to prepare datasets and dataloaders
    data_loader()

    # Access the dataloaders
    for split, loader in data_loader.egatloader.items():
        print(f"DataLoader for {split}:")
        for batch in loader:
            print(batch)
            break  # Print only the first batch for demonstration