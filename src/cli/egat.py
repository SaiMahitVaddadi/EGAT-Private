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

from ..params.config import Config,Params
from ..main.train import Train
from ..main.predict import Predict
from ..main.tuning.hyperparamtertuning import Tune



class EGAT:
    def __init__(self,config=None,mode=None):
        self.config = config 
        self.setups = Config()
        self.params = Params()
        if mode == 'fingerprint':
            self.params.Embed = True 
        self.train = Train(params=self.params)
        self.predict = Predict(params=self.params)
        self.tuner = Tune(params=self.params)

    def configurate(self):
        pass

    def generate(self):
        pass

    def train(self):
        self.train.TrainingProtocol()

    def predict(self):
        self.predict.Run()

    def fingerprint(self):
        self.predict.Run()

    def tune(self):
        self.tuner.tune()




