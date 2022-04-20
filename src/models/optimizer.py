import torch.nn as nn
import torch.optim as optim


class Optimizer(nn.Module):
    """\
    Optimizer
    """

    def __init__(self, learning_rate, config, parameters):
        self.parameters = parameters
        self.clip_norm = config.clip_norm
        self.optimizer = optim.Adam(parameters, lr=learning_rate)
        self.scheduler = (
            None
            if config.scheduler is None
            else optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.scheduler.step_size,
                gamma=config.scheduler.gamma,
            )
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        if self.clip_norm is not None:
            nn.utils.clip_grad_norm_(self.parameters, self.clip_norm)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
