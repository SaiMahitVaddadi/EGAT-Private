
from dataclasses import dataclass,field
from typing import Literal,Optional,List,Union,Type


from ..dataset.base.dataset_setups import BaseDatasetParams
from ..dataset.base.dicttosample import SampleParams
from ..dataset.base.helpers.dglgraphcreation import DGLGraphCreationParams
from ..processing.normalize import NormalizerParams
from ..loader.Loader import DataLoaderParams
from ..utils.database.csvfunctions import DenoteInputDataParams
from ..splitting.split import SplitParams
class TorchDatasetParams(BaseDatasetParams,SampleParams,DGLGraphCreationParams,NormalizerParams,DataLoaderParams,DenoteInputDataParams,SplitParams):
    pass



