from copyreg import constructor
from managers.base import AlternativelyNamedObject, ObjectManager


class BaseSchedule(AlternativelyNamedObject):
    def step(self, modalities, labels, translations, cluster_outputs):
        raise Exception("Not Implemented!")


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
    """
    name = 'schedules'
    constructors = [
        ClassificationSchedule, ClusteringSchedule, LatentBatchAlignmentSchedule, TranslationBatchAlignmentSchedule, 
        TranslationSchedule
    ]