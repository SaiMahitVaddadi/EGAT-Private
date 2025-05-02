import torch,logging

class OptimizerSetup:
    def __init__(self,arguments):
        self.params = arguments
        self.logger = logging.getLogger(__name__)

    def LoadOptimizer(self,model,lr=None):
            ### Load the optimizer
            if lr is None: lr = self.params.lr
            if self.params.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=self.params.weight_decay)
            elif self.params.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=self.params.weight_decay)
            elif self.params.optimizer == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=self.params.weight_decay)
            elif self.params.optimizer == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=self.params.weight_decay, momentum=self.params.momentum)
            elif self.params.optimizer == 'adagrad':
                optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=self.params.weight_decay)
            elif self.params.optimizer == 'nadam':
                optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=self.params.weight_decay)
            elif self.params.optimizer == 'adamax':
                optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=self.params.weight_decay)
            else:
                raise ValueError(f"Optimizer '{self.params.optimizer}' not supported")
            self.optimizer = optimizer