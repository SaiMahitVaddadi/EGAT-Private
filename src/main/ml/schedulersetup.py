import torch,logging
from dataclasses import dataclass

@dataclass
class SchedulerParams:
    scheduler: str
    learning_rate_min: float = 0.0
    momentum_orig: float = 0.1
    lr_decay: float = 0.1
    step_size: int = 10
    epochs: int = 100
    gamma: float = 0.1
    base_lr: float = 0.001
    max_lr: float = 0.01
    step_size_up: int = 2000
    steps_per_epoch: int = 100
    lr_lambda: callable = None
    warmup_steps: int = 0

class ScheduleSetup:
    def __init__(self,arguments):
        self.params = arguments
        self.logger = logging.getLogger(__name__)

    def LoadScheduler(self,optimizer):
        self.LEARNING_RATE_CLIP = self.params.learning_rate_min
        self.MOMENTUM_ORIGINAL = self.params.momentum_orig #.1
        self.MOMENTUM_DECAY = self.params.lr_decay
        self.MOMENTUM_DECAY_STEP = self.params.step_size

        ### Load the scheduler
        if self.params.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params.epochs, eta_min=0)
        elif self.params.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.params.step_size, gamma=self.params.gamma)
        elif self.params.scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        elif self.params.scheduler == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params.gamma)
        elif self.params.scheduler == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.params.base_lr, max_lr=self.params.max_lr, step_size_up=self.params.step_size_up, mode='triangular')
        elif self.params.scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.params.max_lr, steps_per_epoch=self.params.steps_per_epoch, epochs=self.params.epochs)
        elif self.params.scheduler == 'multiplicative':
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=self.params.lr_lambda)
        elif self.params.scheduler == 'warmup_linear_decay':
            def lr_lambda(epoch):
                if epoch < self.params.warmup_steps:
                    return epoch / self.params.warmup_steps
                else:
                    return max(0.0, (self.params.epochs - epoch) / (self.params.epochs - self.params.warmup_steps))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler
    
    def get_learning_rate(self, scheduler):
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # Retrieve learning rate from the optimizer's parameter groups
            return scheduler.optimizer.param_groups[0]['lr']
        else:
            return scheduler.optimizer.param_groups[0]['lr']