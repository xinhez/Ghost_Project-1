import torch
import torch.nn.functional as F

from src.managers.base import NamedObject, ObjectManager


class Loss(NamedObject):
    name = 'Loss'
    based_on_head = False
    def __init__(self, config, _):
        self.weight = config.weight or 1


    @staticmethod
    def compute_distance(is_binary_input, input, output):
        if is_binary_input:
            return F.binary_cross_entropy(output, input)
        else:
            return F.mse_loss(output, input)


class LatentMMDLoss(Loss):
    name = 'latent_mmd'
    """\
    Adapted from https://github.com/KrishnaswamyLab/SAUCIE/blob/master/model.py
    """
    def __init__(self, config, _):
        self.weight = config.weight or 0.01
    @staticmethod
    def _pairwise_dists(x1, x2):
        """Helper function to calculate pairwise distances between tensors x1 and x2."""
        r1 = torch.sum(x1 * x1, dim=1, keepdim=True)
        r2 = torch.sum(x2 * x2, dim=1, keepdim=True)
        D = r1 - 2 * torch.matmul(x1, x2.t()) + r2.t()
        return D


    @staticmethod
    def _gaussian_kernel_matrix(dist, device):
        """Multi-scale RBF kernel."""
        sigmas = torch.tensor([10, 15, 20, 50], device=device)

        beta = 1. / (2. * (torch.unsqueeze(sigmas, 1)))

        s = torch.matmul(beta, torch.reshape(dist, (1, -1)))

        return torch.reshape(torch.sum(torch.exp(-s), dim=0), dist.shape) / len(sigmas)


    def __call__(self, model):
        loss = 0
        batches=model.batches
        
        for latent in model.latents:
            e = latent / torch.mean(latent)
            K = LatentMMDLoss._pairwise_dists(e, e)
            K = K / torch.max(K)
            K = LatentMMDLoss._gaussian_kernel_matrix(K, model.device_in_use)

            # reference batch
            ref_batch = 0
            ref_indices = torch.where(batches == ref_batch)[0]
            n_ref = ref_indices.shape[0]
            ref_var = torch.sum(K[ref_indices].t()[ref_indices]) / (n_ref ** 2)

            # nonreference
            for nonref_batch in batches.unique():
                if nonref_batch != ref_batch:
                    nonref_indices = torch.where(batches == nonref_batch)[0]
                    n_nonref = nonref_indices.shape[0]
            
                    nonref_var = torch.sum(K[nonref_indices].t()[nonref_indices]) / (n_nonref ** 2)
                    covar = torch.sum(K[ref_indices].t()[nonref_indices]) / n_ref / n_nonref

                    loss += torch.abs(ref_var + nonref_var - 2 * covar)

        loss /= model.n_modality
        loss *= self.weight
        return loss, None


class ReconstructionMMDLoss(Loss):
    name = 'reconstruction_mmd'
    """\
    Adapted from https://github.com/KrishnaswamyLab/SAUCIE/blob/master/model.py
    """
    def __call__(self, model):
        eps = 1e-5
        loss = 0
        for modality_index, (translations, modality) in enumerate(zip(model.translations, model.modalities)):
            reconstruction = translations[modality_index]
            batches=model.batches
            
            ref_batch = 0
            ref_indices = torch.where(batches == ref_batch)[0]
            ref_rc = reconstruction[ref_indices]
            ref_gt = modality[ref_indices]
            loss += F.mse_loss(ref_rc,ref_gt)

            for nonref_batch in torch.unique(batches):
                if nonref_batch != ref_batch:
                    nonref_indices = torch.where(batches == nonref_batch)[0]
                    nonref_rc = reconstruction[nonref_indices]
                    nonref_gt = modality[nonref_indices]

                    nonref_rc_mean = torch.mean(nonref_rc, dim=0, keepdim=True)
                    nonref_rc_std  = torch.std(nonref_rc, dim=0, keepdim=True)
                    nonref_rc_normalized = (nonref_rc - nonref_rc_mean) / (nonref_rc_std + eps)

                    nonref_gt_mean = torch.mean(nonref_gt, dim=0, keepdim=True)
                    nonref_gt_std  = torch.std(nonref_gt, dim=0, keepdim=True)
                    nonref_gt_normalized = (nonref_gt - nonref_gt_mean) / (nonref_gt_std + eps)

                    loss += F.mse_loss(nonref_rc_normalized, nonref_gt_normalized)

        loss /= model.n_modality
        loss *= self.weight
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
    def __init__(self, config, model):
        super().__init__(config, model)
        self.class_weights = torch.tensor(model.config.class_weights, device=model.device_in_use, dtype=torch.float)
    def __call__(self, model):
        total_loss = 0 
        head_losses = []

        for cluster_outputs in model.cluster_outputs:
            loss = F.cross_entropy(
                cluster_outputs, model.labels, 
                weight=self.class_weights,
            )

            loss /= model.n_head
            loss *= self.weight

            total_loss += loss
            head_losses.append(loss)
            
        return total_loss, head_losses


class ContrastiveLoss(Loss):
    """\
    Adapted From: https://github.com/DanielTrosten/mvc/blob/main/src/lib/loss.py
    """
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