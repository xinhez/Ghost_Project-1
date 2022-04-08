import torch
import torch.nn.functional as F

from src.managers.base import NamedObject, ObjectManager


class Loss(NamedObject):
    name = 'Loss'
    based_on_head = False
    def __init__(self, config):
        self.weight = config.weight


    @staticmethod
    def compute_distance(is_binary_input, input, output):
        if is_binary_input:
            return F.binary_cross_entropy(output, input)
        else:
            return F.mse_loss(output, input)


class LatentMMDLoss(Loss):
    name = 'latent_mmd'
    def __call__(self, model):
        raise Exception("Not Implemented!")
        return loss, None


class ReconstructionMMDLoss(Loss):
    name = 'reconstruction_mmd'
    def __call__(self, model):
        raise Exception("Not Implemented!")
        return loss, None


class SelfEntropyLoss(Loss):
    name = 'self_entropy'
    based_on_head = True
    def __call__(self, model):
        raise Exception("Not Implemented!")
        return loss, head_losses


class DDC1Loss(Loss):
    name = 'ddc1'
    based_on_head = True
    def __call__(self, model):
        raise Exception("Not Implemented!")
        return loss, head_losses


class DDC3Loss(Loss):
    name = 'ddc3'
    based_on_head = True
    def __call__(self, model):
        raise Exception("Not Implemented!")
        return loss, head_losses


class CrossEntropyLoss(Loss):
    name = 'cross_entropy'
    based_on_head = True
    def __call__(self, model):
        loss = 0 
        head_losses = []

        for head, cluster_outputs in enumerate(model.cluster_outputs):
            head_losses.append(
                F.cross_entropy(
                    cluster_outputs, model.labels, 
                    weight=torch.Tensor(model.config.class_weights).to(device=model.device_in_use),
                )
            )
            loss += head_losses[head]
            
        return loss, head_losses


class ContrastiveLoss(Loss):
    name = 'contrastive'
    def __call__(self, model):
        raise Exception("Not Implemented!")
        return loss, None


class DiscriminatorLoss(Loss):
    name = 'discriminator'
    def __call__(self, model):
        raise Exception("Not Implemented!")
        return loss, None


class GeneratorLoss(Loss):
    name = 'generator'
    def __call__(self, model):
        raise Exception("Not Implemented!")
        return loss, None


class ReconstructionLoss(Loss):
    name = 'reconstruction'
    def __call__(self, model):
        loss = 0
        for modality_index, (translations, modality) in enumerate(zip(model.translations, model.modalities)):
            reconstruction = translations[modality_index]
            loss += Loss.compute_distance(
                model.config.encoders[modality_index].is_binary_input, modality, reconstruction
            )
        loss /= model.n_modality
        return loss, None


class TranslationLoss(Loss):
    name = 'translation'
    def __call__(self, model):
        loss = 0 
        for modality_to_index, (translations, modality) in enumerate(zip(model.translations, model.modalities)):
            for modality_from_index, translation in enumerate(translations):
                if modality_to_index != modality_from_index:
                    loss += Loss.compute_distance(
                        model.config.encoders[modality_to_index].is_binary_input, modality, translation
                    )
        loss /= (model.n_modality ** 2 - model.n_modality)
        return loss, None


class LossManager(ObjectManager):
    name = 'losses'
    constructors = [
        # Batch Alignment Losses
        LatentMMDLoss, ReconstructionMMDLoss,
        # Classification Losses
        CrossEntropyLoss, 
        # Clustering Losses
        SelfEntropyLoss, DDC1Loss, DDC3Loss,
        # Translation Losses
        ContrastiveLoss, DiscriminatorLoss, GeneratorLoss, ReconstructionLoss, TranslationLoss,
    ]