from .graph import *
from .dataset import *
from .model import *
import argparse
import json
import toml



class Params(EGATParams,EGATModelParams,GraphParams):
    pass



class Config:
    def __init__(self):
        self.parser = self.parse_args()

    def parse_args(self):
        parser = argparse.ArgumentParser(description="Parse parameters for the EGAT model")

        for field in Params.__annotations__:
            field_type = type(getattr(Params, field, str))
            if field_type is bool:
                parser.add_argument(f'--{field}', type=lambda x: (str(x).lower() == 'true'), default=getattr(Params, field, None))
            else:
                parser.add_argument(f'--{field}', type=field_type, default=getattr(Params, field, None))

        return parser.parse_args()
    
    def run_cli(self):
        args = self.parser
        # You can add the logic to use the parsed arguments here
        print("Parsed arguments:", args)
    
    def save_params_to_file(self,output_file):
        if output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(self.parser.__dict__, f, indent=4)
        elif output_file.endswith('.toml'):
            with open(output_file, 'w') as f:
                toml.dump(self.parser.__dict__, f)
        else:
            raise ValueError("Unsupported file format. Please use .json or .toml")

    def read_file_to_params(self,input_file):
        if input_file.endswith('.json'):
            with open(input_file, 'r') as f:
                params_dict = json.load(f)
        elif input_file.endswith('.toml'):
            with open(input_file, 'r') as f:
                params_dict = toml.load(f)
        else:
            raise ValueError("Unsupported file format. Please use .json or .toml")

        for key, value in params_dict.items():
            if hasattr(self.parser, key):
                setattr(self.parser, key, value)

        return self.parser

    def read_files_to_params(self, input_files):
        params_dict = {}
        for input_file in input_files: 
            if input_file.endswith('.json'):
                with open(input_file, 'r') as f:
                    params_dict.update(json.load(f))
            elif input_file.endswith('.toml'):
                with open(input_file, 'r') as f:
                    params_dict.update(toml.load(f))
            else:
                raise ValueError("Unsupported file format. Please use .json or .toml")

