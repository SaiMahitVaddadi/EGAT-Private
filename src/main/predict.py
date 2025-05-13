from .base.base import MLTrainandPredictBase
from dataclasses import dataclass
from typing import List, Union, Optional



@dataclass
class PredictParams:
    molecular: bool
    target: Union[str, List[str]]
    batch_size: int
    model_type: str
    additionals: Optional[Union[str, List[str]]] = None
    Norm: Optional[str] = None
    hasaddons: bool = False
    Embed: bool = False
    AttnMaps: bool = False
    tweights: Optional[List[float]] = None
    save_path: str



class Predict(MLTrainandPredictBase):
    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments
    
    def Predict(self):
        self.LoadMLNecessities()
        self.PredictLoop()
    
   
    