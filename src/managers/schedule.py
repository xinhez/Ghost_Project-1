from copyreg import constructor
from managers.base import AlternativelyNamedObject, ObjectManager


class BaseSchedule(AlternativelyNamedObject):
    pass


class ClassificationSchedule(BaseSchedule):
    name = 'classification'


class ClusteringSchedule(BaseSchedule):
    name = 'clustering'


class LatentBatchAlignmentSchedule(BaseSchedule):
    name = 'latent_batch_alignment'


class TranslationSchedule(BaseSchedule):
    name = 'translation'


class TranslationBatchAlignmentSchedule(BaseSchedule):
    name = 'translation_batch_alignment'


class ScheduleManager(ObjectManager):
    """\
    Schedule

    Each schedule determines which losses to compute and which optimizers should step.
    It should also save the best performing model based on its reference loss term.
    """
    name = 'schedules'
    constructors = [
        ClassificationSchedule, ClusteringSchedule, LatentBatchAlignmentSchedule, TranslationBatchAlignmentSchedule, 
        TranslationSchedule
    ]