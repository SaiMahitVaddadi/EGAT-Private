''' 
To use this file as the command `egat`, you need to do the following:

1. Save the file with a proper name, for example, `egat.py`.
2. Make the file executable.
3. Move the file to a directory that is in your system's PATH.

Here are the steps:

1. Save the file as `egat.py`.

2. Make the file executable by running the following command in your terminal:
    ```bash
    chmod +x egat.py
    ```

3. Move the file to a directory that is in your system's PATH, for example, `/usr/local/bin`:
    ```bash
    sudo mv egat.py /usr/local/bin/egat
    ```

Now you should be able to run the command `egat` from your terminal, and it will execute the script.
Here is the pseudocode for the steps:

1. Save the file with the name `egat.py`.
    - Open your text editor.
    - Write the Python script to print "egat is cool".
    - Save the file as `egat.py`.

2. Make the file executable.
    - Open your terminal.
    - Navigate to the directory where `egat.py` is saved.
    - Run the command to make the file executable: `chmod +x egat.py`.

3. Move the file to a directory in your system's PATH.
    - In the terminal, run the command to move the file: `sudo mv egat.py /usr/local/bin/egat`.

4. Verify the command.
    - Open a new terminal window.
    - Run the command `egat` to ensure it prints "egat is cool".
'''

from ..params.config import Config,EGATParams
from ..main.train import Train
from ..main.predict import Predict
from ..main.tuning.hyperparamtertuning import Tune
from ..dataset.external.graphgen import GraphGeneration
import argparse
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Type


class EGAT:
    def __init__(self,config=None,mode=None,exportconfig=None,params=None):
        self.config = config 
        if self.config != None:
            self.setups = Config(self.config)
            self.params = self.setups.get_params()
            if exportconfig != None: self.setups.export(exportconfig)
        else:
            if params == None: self.params = EGATParams()
            else: self.params = params
        if mode == 'fingerprint':
            self.params.Embed = True 
        self.train = Train(params=self.params)
        self.predict = Predict(params=self.params)
        self.tuner = Tune(params=self.params)
        self.generator = GraphGeneration(params=self.params)

    def generate(self):
        self.generator.save()

    def train(self):
        self.train.TrainingProtocol()

    def predict(self):
        self.predict.Predict()

    def fingerprint(self):
        self.predict.Predict()

    def tune(self):
        self.tuner.tune()

    def analyze(self):
        pass

def add_param_args(parser: argparse.ArgumentParser, param_cls: Type[EGATParams]):
    for field in fields(param_cls):
        arg_type = field.type
        # Handle typing like Optional[...] or Union[...] if needed here
        parser.add_argument(f"--{field.name}", type=arg_type, default=None, help=f"(default: {field.default})")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flexible config system with file and CLI override support.")
    parser.add_argument('command',type=str,help="Command to run (train, predict, tune, generate, fingerprint)")
    parser.add_argument("--config", type=str, help="Path to config file (.json, .yaml, .toml)",default=None)
    parser.add_argument("--export", type=str, help="Path to export the final parameters",default=None)

    # Dynamically add Params fields
    add_param_args(parser, EGATParams)

    args = parser.parse_args()
    command = args.command
    config = args.config
    export = args.export

    args_dict = vars(args)
    known_args = {"config", "export"}
    cli_overrides = {k: v for k, v in args_dict.items() if k not in known_args and v is not None}

    cfg = Config(config_path=args.config, overrides=cli_overrides)
    params = cfg.get_params()

    egatobj = EGAT(config=config,mode=command,exportconfig=export)
    if command == 'train':
        egatobj.train()
    elif command == 'predict':
        egatobj.predict()
    elif command == 'tune':
        egatobj.tune()
    elif command == 'generate':
        egatobj.generate()
    elif command == 'fingerprint':
        egatobj.fingerprint()
    else:
        raise ValueError(f"Unknown command: {command}. Please use 'train', 'predict', 'tune', 'generate', or 'fingerprint'.")



