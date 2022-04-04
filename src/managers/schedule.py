from src.managers.base import AlternativelyNamedObject, ObjectManager


class BaseSchedule(AlternativelyNamedObject):
    def step(self, model):
        raise Exception("Not Implemented!")


class ClassificationSchedule(BaseSchedule):
    name = 'classification'
    def __init__(self, losses):
        self.losses = []
        self.optimizers = []


class ClusteringSchedule(BaseSchedule):
    name = 'clustering'
    def __init__(self, losses):
        self.losses = []
        self.optimizers = []


class LatentBatchAlignmentSchedule(BaseSchedule):
    name = 'latent_batch_alignment'
    def __init__(self, losses):
        self.losses = []
        self.optimizers = []


class TranslationSchedule(BaseSchedule):
    name = 'translation'
    def __init__(self, losses):
        self.losses = []
        self.optimizers = []


class TranslationBatchAlignmentSchedule(BaseSchedule):
    name = 'translation_batch_alignment'
    def __init__(self, losses):
        self.losses = []
        self.optimizers = []


class ScheduleManager(ObjectManager):
    """\
    Schedule

    Each schedule determines which losses to compute and which optimizers should step.
    """
    name = 'schedules'
    constructors = [
        ClassificationSchedule, ClusteringSchedule, TranslationSchedule,
        LatentBatchAlignmentSchedule, TranslationBatchAlignmentSchedule, 
    ]