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
from ..dataset.collate.molecule import MolecularCollator,MolecularFingerprintCollator
from ..dataset.collate.reaction import ReactionCollator,ReactionFingerprintCollator

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

# - Add the not argument for class_choice like class_choice_not
# - Add the test_class_choice and test_class_choice_not
class DataLoaderCommands:
    def __init__(self,arguments):
        self.params = arguments


    def setupsplits(self,splits):
        if self.params.combine_loader is None:
            if self.params.test_only:
                splits = ['train','test']
            else:
                splits = ['train','val','test']
        elif self.params.combine_loader == 'trainval':
            if self.params.test_only:
                splits = ['trainval','test']
            else:
                splits = ['trainval']
        elif self.params.combine_loader == 'all':
            if self.params.test_only:
                splits = ['trainval']
            else:
                splits = ['all']
        elif self.params.combine_loader == 'valtest':
            splits = ['train','valtest']
        elif self.params.combine_loader == 'traintest':
            if self.params.test_only:
                splits = ['train']
            else:
                splits = ['traintest','val']
        return splits

    def setuploader(self):
        self.egatdataset = dict()
        self.egatdataloader = dict()    
        self.splits = self.setupsplits([])

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
            self.egatdataset[split] = self.datasetfunction(self.params,split)
            
    def addimbalanceddataset(self):
        if self.params.imblearn is not None:
            self.egatdataset[f'aug{self.splits[0]}'] = self.egatdataset[self.splits[0]]
            self.splits.append(f'aug{self.splits[0]}')
        
    def grabcollatefunction(self):
        if self.params.graph == 'molecular': # Check if we only need molecular features. If we do, only load R features. 
            collatorclass = 'Molecular'
        elif self.params.graph == 'reaction':
            collatorclass = 'Reaction'
        else:
            raise ValueError(f'Unknown graph type: {self.params.graph}')
        
        if self.params.fingerprint == True:
            collatorclass += 'Fingerprint'

        if self.params.addons is not None: #Check if we need RDKit Global Features. If we do, load them.
            if self.params.additional is not None:
                mode = 'allprops'
            else:
                mode = 'addons'
        else:
            if self.params.additional is not None:
                mode = 'additionals'
            else:
                mode = 'targets'

        self.collator = globals()[f'{collatorclass}Collator'](mode=mode,multicomp=True if isinstance(self.params.smiles,list) else False)

    def createdataloader(self):
        for split in self.splits:
            self.egatdataloader[split] = torch.utils.data.DataLoader(self.egatdataset[split], batch_size=self.params.batch_size, shuffle=self.params.shuffle_loader, collate_fn=self.collator,drop_last=self.params.drop_last)

    def createdataloaderdebug(self):
        for split in self.splits:
            self.egatdataloader[split] = torch.utils.data.DataLoader(self.egatdataset[split], batch_size=self.params.batch_size, shuffle=self.params.shuffle_loader,drop_last=self.params.drop_last)
    

class EGATDataLoader(DataLoaderCommands):
    def __init__(self,arguments):
        super().__init__(arguments)
        self.setuploader()
        #self.excluded = self.excludedata()
        self.grabcollatefunction()

    def __call__(self):
        self.choosedataset()
        self.createdatsets()
        self.addimbalanceddataset()
        self.createdataloader()
        #self.createdataloaderdebug()

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
            print('Batch: ',batch)
            break  # Print only the first batch for demonstration