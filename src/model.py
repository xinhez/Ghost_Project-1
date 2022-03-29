from torch.nn import Module 

class Model(Module):
    def __init__(self, cfg):
        super().__init__()


    """
    update_cfg
        Refresh the model with the updated configuration. All previous training data will be lost.
    """
    def update_cfg(self, cfg):
        pass