import torch,logging

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
        return scheduler