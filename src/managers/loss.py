from torch.nn import Module

from src.managers.base import NamedObject, ObjectManager


class LatentMMDLoss(Module, NamedObject):
    name = 'latent_mmd'
    def forward(self, model):
        raise Exception("Not Implemented!")


class ReconstructionMMDLoss(Module, NamedObject):
    name = 'reconstruction_mmd'
    def forward(self, model):
        raise Exception("Not Implemented!")


class CELoss(Module, NamedObject):
    name = 'ce'
    def forward(self, model):
        raise Exception("Not Implemented!")


class SCELoss(Module, NamedObject):
    name = 'sce'
    def forward(self, model):
        raise Exception("Not Implemented!")


class ContrastiveLoss(Module, NamedObject):
    name = 'contrastive'
    def forward(self, model):
        raise Exception("Not Implemented!")


class DiscriminatorLoss(Module, NamedObject):
    name = 'discriminator'
    def forward(self, model):
        raise Exception("Not Implemented!")


class GeneratorLoss(Module, NamedObject):
    name = 'generator'
    def forward(self, model):
        raise Exception("Not Implemented!")


class ReconstructionLoss(Module, NamedObject):
    name = 'reconstruction'
    def forward(self, model):
        raise Exception("Not Implemented!")


class TranslationLoss(Module, NamedObject):
    name = 'translation'
    def forward(self, model):
        raise Exception("Not Implemented!")


class LossManager(ObjectManager):
    name = 'losses'
    constructors = [
        # Batch Alignment Losses
        LatentMMDLoss, ReconstructionMMDLoss,
        # Classification Losses
        SCELoss, 
        # Clustering Losses
        CELoss,
        # Translation Losses
        ContrastiveLoss, DiscriminatorLoss, GeneratorLoss, ReconstructionLoss, TranslationLoss,
    ]