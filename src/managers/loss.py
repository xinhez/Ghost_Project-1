from managers.base import NamedObject, ObjectManager


class BaseLoss(NamedObject):
    pass


class SCELoss(BaseLoss):
    name = 'sce'


class ContrastiveLoss(BaseLoss):
    name = 'contrastive'


class LossManager(ObjectManager):
    name = 'losses'