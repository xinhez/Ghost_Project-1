from src.managers.base import NamedObject, ObjectManager

class Loss(NamedObject):
    name = 'Loss'
    def __init__(self, config):
        self.weight = config.weight


class LatentMMDLoss(Loss):
    name = 'latent_mmd'
    def __call__(self, model):
        raise Exception("Not Implemented!")


class ReconstructionMMDLoss(Loss):
    name = 'reconstruction_mmd'
    def __call__(self, model):
        raise Exception("Not Implemented!")


class SELoss(Loss):
    name = 'se'
    def __call__(self, model):
        raise Exception("Not Implemented!")


class DDC1Loss(Loss):
    name = 'ddc1'
    def __call__(self, model):
        raise Exception("Not Implemented!")


class DDC3Loss(Loss):
    name = 'ddc3'
    def __call__(self, model):
        raise Exception("Not Implemented!")


class SCELoss(Loss):
    name = 'sce'
    def __call__(self, model):
        raise Exception("Not Implemented!")


class ContrastiveLoss(Loss):
    name = 'contrastive'
    def __call__(self, model):
        raise Exception("Not Implemented!")


class DiscriminatorLoss(Loss):
    name = 'discriminator'
    def __call__(self, model):
        raise Exception("Not Implemented!")


class GeneratorLoss(Loss):
    name = 'generator'
    def __call__(self, model):
        raise Exception("Not Implemented!")


class ReconstructionLoss(Loss):
    name = 'reconstruction'
    def __call__(self, model):
        raise Exception("Not Implemented!")


class TranslationLoss(Loss):
    name = 'translation'
    def __call__(self, model):
        raise Exception("Not Implemented!")


class LossManager(ObjectManager):
    name = 'losses'
    constructors = [
        # Batch Alignment Losses
        LatentMMDLoss, ReconstructionMMDLoss,
        # Classification Losses
        SCELoss, 
        # Clustering Losses
        SELoss, DDC1Loss, DDC3Loss,
        # Translation Losses
        ContrastiveLoss, DiscriminatorLoss, GeneratorLoss, ReconstructionLoss, TranslationLoss,
    ]