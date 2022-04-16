import torch
import torch.nn.functional as F

from src.managers.base import NamedObject, ObjectManager


class Loss(NamedObject):
    name = 'Loss'
    based_on_head = False
    def __init__(self, config, _):
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
        total_loss = 0 
        head_losses = []

        for cluster_outputs in model.cluster_outputs:
            loss = F.cross_entropy(
                cluster_outputs, model.labels, 
                weight=torch.Tensor(model.config.class_weights).to(device=model.device_in_use),
            )

            loss /= model.n_head
            loss *= self.weight

            total_loss += loss
            head_losses.append(loss)
            
        return total_loss, head_losses


class ContrastiveLoss(Loss):
    name = 'contrastive'
    based_on_head = True
    def __init__(self, config, model):
        super().__init__(config, model)
        self.sampling_ratio = config.sampling_ratio
        self.tau = config.tau
        self.eye = torch.eye(model.n_output, device=model.device_in_use)


    @staticmethod
    def _cosine_similarity(projections):
        h = F.normalize(projections, p=2, dim=1)
        return h @ h.t()


    def _draw_negative_samples(self, cluster_outputs, v, pos_indices):
        predictions = cluster_outputs.argmax(axis=1).detach()
        predictions = torch.cat(v * [predictions], dim=0)

        weights = (1 - self.eye[predictions])[:, predictions[[pos_indices]]].T
        n_negative_samples = int(self.sampling_ratio * predictions.size(0))
        negative_sample_indices = torch.multinomial(weights, n_negative_samples, replacement=True)
        return negative_sample_indices


    @staticmethod
    def _get_positive_samples(logits, v, n):
        diagonals = []
        inds = []
        for i in range(1, v):
            diagonal_offset = i * n
            diag_length = (v - i) * n
            _upper = torch.diagonal(logits, offset=diagonal_offset)
            _lower = torch.diagonal(logits, offset=-1 * diagonal_offset)
            _upper_inds = torch.arange(0, diag_length)
            _lower_inds = torch.arange(i * n, v * n)
            diagonals += [_upper, _lower]
            inds += [_upper_inds, _lower_inds]

        pos = torch.cat(diagonals, dim=0)
        pos_inds = torch.cat(inds, dim=0)
        return pos, pos_inds


    def __call__(self, model):
        total_loss = 0 
        head_losses = []

        logits = ContrastiveLoss._cosine_similarity(torch.cat(model.latents, dim=0)) / self.tau
        pos, pos_inds = ContrastiveLoss._get_positive_samples(logits, model.n_modality, model.n_sample)

        for cluster_outputs in model.cluster_outputs:
            neg_inds = self._draw_negative_samples(cluster_outputs, model.n_modality, pos_inds)
            neg = logits[pos_inds.view(-1, 1), neg_inds]
            inputs = torch.cat((pos.view(-1, 1), neg), dim=1)
            labels = torch.zeros(
                model.n_modality * (model.n_modality - 1) * model.n_sample, device=model.device_in_use, dtype=torch.long
            )
            loss = F.cross_entropy(inputs, labels)

            loss /= model.n_head
            loss *= self.weight

            total_loss += loss
            head_losses.append(loss)

        return total_loss, head_losses


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
        loss *= self.weight
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
        loss *= self.weight
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