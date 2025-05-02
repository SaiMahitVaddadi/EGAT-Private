import torch,logging

class GPUSetup:
    def __init__(self,arguments):
        self.params = arguments
        self.logger = logging.getLogger(__name__)

    def cudaloader(self):
        if torch.cuda.device_count() > 1 and self.params.parallel:
            device = torch.device("cuda")
            self.logger.info(f'Using {torch.cuda.device_count()} GPUs for parallel training.')
        else:
            device = torch.device("cuda:0")
            self.logger.info('Using single GPU for training.')
        return device
    
    def metalloader(self):
        if torch.cuda.device_count() > 1 and self.params.parallel:
            device = torch.device("mps")
            self.logger.info(f'Using {torch.cuda.device_count()} MPS devices for parallel training.')
        else:
            device = torch.device("mps")
            self.logger.info('Using single MPS device for training.')
        return device
    
    def rocmloader(self):
        if torch.cuda.device_count() > 1 and self.params.parallel:
            device = torch.device("hip")
            self.logger.info(f'Using {torch.cuda.device_count()} ROCm devices for parallel training.')
        else:
            device = torch.device("hip")
            self.logger.info('Using single ROCm device for training.')
        return device
    
    def LoadTorchSetup(self):
        ### Get the gradient enabled
        torch.set_grad_enabled(True)
        ### Set the GPU if available. Otherwise Stop. 
        if torch.cuda.is_available() and self.params.setup == 'cuda':
            device = self.cudaloader()
        else:
            if torch.backends.mps.is_available() and self.params.setup == 'mps':
                device = self.metalloader()
            elif torch.backends.hip.is_available() and self.params.setup == 'rocm':
                device = self.rocmloader()
            else:
                device = torch.device("cpu")
        self.device = device