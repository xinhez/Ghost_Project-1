from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


class Optimizer(Module):
    """\
    Optimizer
    """
    def __init__(self, config, parameters):
        self.parameters = parameters
        self.clip_norm = config.clip_norm
        self.optimizer = Adam(parameters, lr=config.learning_rate)
        if config.scheduler is None:
            self.scheduler = None
        else:
            self.scheduler = StepLR(self.optimizer, step_size=config.scheduler.step_size, gamma=config.scheduler.gamma)


    def zero_grad(self):
        self.optimizer.zero_grad()


    def step(self):
        if self.clip_norm is not None:
            clip_grad_norm_(self.parameters, self.clip_norm)
        self.optimizer.step()

    
    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()