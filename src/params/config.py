from .graph import *
from .dataset import *
from .model import *
import argparse
import json
import toml
import os,yaml
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict
from ..dataset.external.graphgen import GraphGenParams
@dataclass
class EGATParams(GraphGenerationParams,TorchDatasetParams,EGATMLParams,GraphGenParams):
    pass

class Config:
    def __init__(self, config_path: str):
        self.params = EGATParams()
        self.config_path = config_path
        self.override_params()

    def load_config(self) -> Dict[str, Any]:
        ext = os.path.splitext(self.config_path)[1]
        with open(self.config_path, 'r') as f:
            if ext == '.json':
                return json.load(f)
            elif ext in ('.yaml', '.yml'):
                return yaml.safe_load(f)
            elif ext == '.toml':
                return toml.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {ext}")

    def override_params(self):
        config_dict = self.load_config()
        param_fields = {f.name for f in fields(self.params)}
        for key, value in config_dict.items():
            if key in param_fields:
                setattr(self.params, key, value)
            else:
                raise KeyError(f"'{key}' is not a valid parameter in Params")

    def get_params(self) -> EGATParams:
        return self.params

    def export(self, export_path: str):
        ext = os.path.splitext(export_path)[1]
        data = asdict(self.params)
        with open(export_path, 'w') as f:
            if ext == '.json':
                json.dump(data, f, indent=4)
            elif ext in ('.yaml', '.yml'):
                yaml.dump(data, f)
            elif ext == '.toml':
                toml.dump(data, f)
            else:
                raise ValueError(f"Unsupported export file format: {ext}")
