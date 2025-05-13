from dataclasses import dataclass,field
from typing import Literal,Optional,List,Union,Type

from ..main.ml.egatsetup import EGATSetupParams
from ..main.ml.finetuning import FinetuningParams
from ..main.ml.gpusetups import GPUParams
from ..main.ml.losssetups import LossParams
from ..main.ml.optimizersetup import OptimizerParams
from ..main.ml.schedulersetup import SchedulerParams
from ..main.base.base import BaseEGATTrainParams
from ..main.base.batchiteration import BatchIterationParams
from ..main.base.learningrates import LearningRateParams
from ..main.base.metrics import MetricParams
from ..main.base.multitask import MultitaskParams
from ..main.base.obtainpreds import PredsParams
from ..main.base.torchsaver import TorchSaverParams
from ..main.train import TrainParams
from ..models.base.blocks.aggblock import AggregationBlockParams
from ..models.base.blocks.egatblock import EGATBlockParams
from ..models.base.blocks.predictionblock import PredictionBlockParams

class EGATMLParams(BaseEGATTrainParams,EGATSetupParams,FinetuningParams,GPUParams,LossParams,OptimizerParams,SchedulerParams,
                   BatchIterationParams,LearningRateParams,MetricParams,MultitaskParams,PredsParams,TorchSaverParams,TrainParams,
                   AggregationBlockParams,EGATBlockParams,PredictionBlockParams):
    pass


