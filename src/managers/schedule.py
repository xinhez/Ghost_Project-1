from copyreg import constructor
from managers.base import AlternativelyNamedObject, ObjectManager


class BaseSchedule(AlternativelyNamedObject):
    pass


class ClassificationSchedule(BaseSchedule):
    name = 'classification'


class ClusteringSchedule(BaseSchedule):
    name = 'clustering'


class TranslationSchedule(BaseSchedule):
    name = 'translation'


class ScheduleManager(ObjectManager):
    """\
    Schedule
    """
    constructors = [ClassificationSchedule, ClusteringSchedule, TranslationSchedule]
