from ..ml.setup import MLSetup


class LearningRateFunctions(MLSetup):

    def __init__(self, arguments):
        super().__init__(arguments)
        self.params = arguments 
    
    def LoadLearningRate(self,epoch):
        if epoch < self.params.epoch:
            lr = self.get_learning_rate(self.scheduler)
            return lr
        else:
            return self.params.learning_rate_min
    
    def UpdateLR(self,epoch):
        if self.params.scheduler == 'plateau':
            self.scheduler.step(self.best_loss)
        else:
            self.scheduler.step()
        self.params.learning_rate = self.LoadLearningRate(epoch)
        self.logger.info(f"Learning rate is set to {self.params.learning_rate}.")
    
    def UpdateMomentum(self,epoch):
        momentum = self.MOMENTUM_ORIGINAL * (self.MOMENTUM_DECAY ** (epoch //self.MOMENTUM_DECAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        return momentum