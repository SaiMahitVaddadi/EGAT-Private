
import yaml
import itertools
import os 
from ..slurm import write_slurm_script




class GridSearchSLURM:
    def __init__(self,iterfile,configfile,slurm=False):
        self.iterfile = iterfile
        self.configfile = configfile
        self.iterfile = self.OpenIterfile()
        self.config = self.OpenConfig()
        self.slurm = slurm

    
    def OpenConfig(self):
        with open(self.configfile, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def OpenIterfile(self):
        with open(self.iterfile, 'r') as file:
            iterfile = yaml.safe_load(file)
        return iterfile
    
    def ChangeInput(self,input):
        self.config['input'] = input
    
    def ChangeSavePath(self,save_path):
        self.config['save_path'] = save_path
    
    def ChangeTarget(self,target):
        self.config['target'] = target
        
    
    def ChangeSize(self,size):
        self.config['size'] = size

    def GridSearch(self,inputname='config',folder = '/scratch/gilbreth/svaddadi/Molecule-Benchmarks/EGAT_Runs/PCQM/yaml/'):
        keys, values = zip(*self.iterfile.items())
        for v in itertools.product(*values):
            config_copy = self.config.copy()
            for key, value in zip(keys, v):
                config_copy[key] = value

            
            output_filename = f"{inputname}-{'-'.join([f'{key}-{value}' for key, value in zip(keys, v)])}"
            with open(output_filename+'.yml', 'w') as file: yaml.safe_dump(config_copy, file)
            print(f"Modified configuration saved to {output_filename}")
            if self.slurm: 
                write_slurm_script(output_filename,os.path.join(folder,output_filename))
                print(f"Slurm script saved to {output_filename}")


