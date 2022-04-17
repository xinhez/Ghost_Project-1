from src.utils import convert_to_lowercase


class NamedObject:
    @classmethod
    def has_name(cls, name):
        return convert_to_lowercase(name) == cls.name


class AlternativelyNamedObject(NamedObject):
    @classmethod
    def get_short_name(cls):
        return "".join([word[0] for word in cls.name.split("_")])

    @classmethod
    def get_spaced_name(cls):
        return " ".join([word for word in cls.name.split("_")])

    @classmethod
    def has_name(cls, name):
        return convert_to_lowercase(name) in [
            cls.name,
            cls.get_short_name(),
            cls.get_spaced_name(),
        ]


class ObjectManager(NamedObject):
    @classmethod
    def get_constructor_names(cls):
        return [constructor.name for constructor in cls.constructors]

    @classmethod
    def get_constructor_by_name(cls, name):
        for constructor in cls.constructors:
            if constructor.has_name(name):
                return constructor
        raise Exception(f"Only {cls.get_constructor_names()} {cls.name} are supported.")
