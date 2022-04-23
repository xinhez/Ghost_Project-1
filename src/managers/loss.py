import torch
import torch.nn.functional as F

from src.managers.base import NamedObject, ObjectManager


class BaseLoss(NamedObject):
    name = "Loss"
    based_on_head = False
    based_on_discriminator = False
    eps = 1e-9

    def __init__(self, config, _):
        self.weight = config.weight or 1

    @staticmethod
    def compute_distance(is_binary_input, output, target):
        """\
        Compute the distance between target and output with BCE if binary data or MSE for all others.
        """
        if is_binary_input:
            return F.binary_cross_entropy(output, target)
        else:
            return F.mse_loss(output, target)


class ReconstructionMMDLoss(BaseLoss):
    name = "reconstruction_mmd"
    """\
    Adapted from https://github.com/KrishnaswamyLab/SAUCIE/blob/master/model.py
    """

    def __call__(self, model):
        loss = 0
        batches = model.batches

        for modality_index, (translations, modality) in enumerate(
            zip(model.translations, model.modalities)
        ):
            reconstruction = translations[modality_index]

            for batch in torch.unique(batches):
                indices = torch.where(batches == batch)[0]
                sample_count = indices.shape[0]

                if sample_count > 1:
                    rc = reconstruction[indices]
                    gt = modality[indices]

                    rc_mean = torch.mean(rc, dim=0, keepdim=True)
                    rc_std = torch.std(rc, dim=0, keepdim=True)
                    rc_normalized = (rc - rc_mean) / (rc_std + self.eps)

                    gt_mean = torch.mean(gt, dim=0, keepdim=True)
                    gt_std = torch.std(gt, dim=0, keepdim=True)
                    gt_normalized = (gt - gt_mean) / (gt_std + self.eps)

                    loss += F.mse_loss(rc_normalized, gt_normalized)

        loss /= model.n_modality
        loss *= self.weight
        return loss, None


class SelfEntropyLoss(BaseLoss):
    name = "self_entropy"
    """
    Entropy regularization to prevent trivial solution.
    """
    based_on_head = True

    def __init__(self, config, model):
        super().__init__(config, model)
        self.weight = config.weight or 0.1

    def __call__(self, model):
        eps = 1e-8
        total_loss = 0
        head_losses = []

        for cluster_outputs in model.cluster_outputs:
            prob_mean = cluster_outputs.mean(dim=0)
            prob_mean[(prob_mean < eps).data] = eps
            loss = (prob_mean * torch.log(prob_mean)).sum()

            loss /= model.n_modality
            loss *= self.weight

            total_loss += loss
            head_losses.append(loss)

        return total_loss, head_losses


class DDCLoss(BaseLoss):
    name = "ddc"
    """
    Adapted From: https://github.com/DanielTrosten/mvc/blob/main/src/lib/loss.py
    """

    def __init__(self, config, model):
        super().__init__(config, model)
        self.eye = torch.eye(model.n_output, device=model.device_in_use)

    @staticmethod
    def triu(X):
        """\ 
        Sum of strictly upper triangular part.
        """
        return torch.sum(torch.triu(X, diagonal=1))

    @staticmethod
    def _atleast_epsilon(X, eps=1e-9):
        """
        Ensure that all elements are >= `eps`.
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    @staticmethod
    def d_cs(A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(
            torch.diagonal(nom), 0
        )

        nom = DDCLoss._atleast_epsilon(nom)
        dnom_squared = DDCLoss._atleast_epsilon(dnom_squared, eps=BaseLoss.eps**2)

        d = (
            2
            / (n_clusters * (n_clusters - 1))
            * DDCLoss.triu(nom / torch.sqrt(dnom_squared))
        )
        return d

    @staticmethod
    def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=BaseLoss.eps):
        """\
        Compute a Gaussian kernel matrix from a distance matrix.
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(-dist / (2 * sigma2))
        return k

    @staticmethod
    def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=BaseLoss.eps):
        """\
        Compute a Gaussian kernel matrix from a distance matrix.
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(-dist / (2 * sigma2))
        return k

    @staticmethod
    def cdist(X, Y):
        """\
        Pairwise distance between rows of X and rows of Y.
        """
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X ** 2, dim=1, keepdim=True)
        y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    @staticmethod
    def vector_kernel(x, rel_sigma=0.15):
        """\
        Compute a kernel matrix from the rows of a matrix.
        """
        return DDCLoss.kernel_from_distance_matrix(DDCLoss.cdist(x, x), rel_sigma)

    def __call__(self, model):
        total_loss = 0
        head_losses = []

        for hidden, cluster_outputs in zip(model.hiddens, model.cluster_outputs):
            hidden_kernel = DDCLoss.vector_kernel(hidden)
            # L_1 loss
            loss = DDCLoss.d_cs(cluster_outputs, hidden_kernel, model.n_output)

            # L_3 loss
            m = torch.exp(-DDCLoss.cdist(cluster_outputs, self.eye))
            loss += DDCLoss.d_cs(m, hidden_kernel, model.n_output)
            loss *= self.weight

            total_loss += loss
            head_losses.append(loss)

        return total_loss, head_losses


class CrossEntropyLoss(BaseLoss):
    name = "cross_entropy"
    based_on_head = True

    def __init__(self, config, model):
        super().__init__(config, model)
        self.class_weights = torch.tensor(
            model.config.class_weights, device=model.device_in_use, dtype=torch.float
        )

    def __call__(self, model):
        total_loss = 0
        head_losses = []

        for cluster_outputs in model.cluster_outputs:
            loss = F.cross_entropy(
                cluster_outputs, model.labels, weight=self.class_weights,
            )

            loss /= model.n_head
            loss *= self.weight

            total_loss += loss
            head_losses.append(loss)

        return total_loss, head_losses


class ContrastiveLoss(BaseLoss):
    name = "contrastive"
    """\
    Adapted From: https://github.com/DanielTrosten/mvc/blob/main/src/lib/loss.py
    """
    based_on_head = True

    def __init__(self, config, model):
        super().__init__(config, model)
        self.sampling_ratio = config.sampling_ratio or 0.25
        self.tau = config.tau or 0.1
        self.eye = torch.eye(model.n_output, device=model.device_in_use)

    @staticmethod
    def _cosine_similarity(projections):
        h = F.normalize(projections, p=2, dim=1)
        return h @ h.t()

    def _draw_negative_samples(self, predictions, v, pos_indices):
        predictions = torch.cat(v * [predictions], dim=0)
        weights = (1 - self.eye[predictions])[:, predictions[[pos_indices]]].T
        n_negative_samples = int(self.sampling_ratio * predictions.size(0))
        negative_sample_indices = torch.multinomial(
            weights, n_negative_samples, replacement=True
        )
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
        if model.n_modality == 1:
            return 0, [0] * model.n_head
            
        total_loss = 0
        head_losses = []

        logits = (
            ContrastiveLoss._cosine_similarity(torch.cat(model.latents, dim=0))
            / self.tau
        )
        pos, pos_inds = ContrastiveLoss._get_positive_samples(
            logits, model.n_modality, model.n_sample
        )

        for predictions in model.predictions:
            if len(torch.unique(predictions)) > 1:
                neg_inds = self._draw_negative_samples(
                    predictions, model.n_modality, pos_inds
                )
                neg = logits[pos_inds.view(-1, 1), neg_inds]
                inputs = torch.cat((pos.view(-1, 1), neg), dim=1)
                labels = torch.zeros(
                    model.n_modality * (model.n_modality - 1) * model.n_sample,
                    device=model.device_in_use,
                    dtype=torch.long,
                )
                loss = F.cross_entropy(inputs, labels)

                loss /= model.n_head
                loss *= self.weight
            else:
                loss = 0

            total_loss += loss
            head_losses.append(loss)

        return total_loss, head_losses


class DiscriminatorLoss(BaseLoss):
    name = "discriminator"
    based_on_discriminator = True

    def __init__(self, config, model):
        super().__init__(config, model)
        self.weight = config.weight or 0.5

    def __call__(self, model):
        loss = 0

        for real_output in model.discriminator_real_outputs:
            loss += F.mse_loss(
                real_output, torch.ones_like(real_output, device=model.device_in_use)
            )

        for fake_output in model.discriminator_fake_outputs:
            loss += F.mse_loss(
                fake_output, torch.zeros_like(fake_output, device=model.device_in_use)
            )

        loss /= model.n_modality
        loss *= self.weight
        return loss, None


class GeneratorLoss(BaseLoss):
    name = "generator"
    based_on_discriminator = True

    def __init__(self, config, model):
        super().__init__(config, model)
        self.weight = config.weight or 0.5

    def __call__(self, model):
        loss = 0

        for generator_output in model.generator_outputs:
            loss += F.binary_cross_entropy(
                generator_output,
                torch.ones_like(generator_output, device=model.device_in_use),
            )

        loss /= model.n_modality
        loss *= self.weight
        return loss, None


class ReconstructionLoss(BaseLoss):
    name = "reconstruction"

    def __call__(self, model):
        loss = 0

        for modality_index, (translations, modality) in enumerate(
            zip(model.translations, model.modalities)
        ):
            reconstruction = translations[modality_index]
            loss += BaseLoss.compute_distance(
                model.config.encoders[modality_index].is_binary_input,
                reconstruction,
                modality,
            )

        loss /= model.n_modality
        loss *= self.weight
        return loss, None


class TranslationLoss(BaseLoss):
    name = "translation"

    def __call__(self, model):
        loss = 0
        for modality_to_index, (translations, modality) in enumerate(
            zip(model.translations, model.modalities)
        ):
            for modality_from_index, translation in enumerate(translations):
                if modality_to_index != modality_from_index:
                    loss += BaseLoss.compute_distance(
                        model.config.encoders[modality_to_index].is_binary_input,
                        translation,
                        modality,
                    )
        loss /= model.n_modality ** 2 - model.n_modality
        loss *= self.weight
        return loss, None


class LossManager(ObjectManager):
    name = "losses"
    constructors = [
        # Batch Alignment Losses
        ReconstructionMMDLoss,
        # Classification Losses
        CrossEntropyLoss,
        # Clustering Losses
        SelfEntropyLoss,
        DDCLoss,
        # Translation Losses
        ContrastiveLoss,
        DiscriminatorLoss,
        GeneratorLoss,
        ReconstructionLoss,
        TranslationLoss,
    ]
