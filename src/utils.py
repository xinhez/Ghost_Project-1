import numpy as np


def count_unique(items):
    """\
    Count of the number of unique items in the given list.
    """
    return len(np.unique(items))


def convert_to_lowercase(item):
    """\
    If the given item is a string, return its lowercase.
    """
    if isinstance(item, str):
        return item.lower()
    else:
        return item


class ObjectManager():
    @classmethod
    def get_constructor_names(cls):
        return [constructor.name for constructor in cls.constructors]


    @classmethod
    def get_constructor_by_name(cls, name):
        for constructor in cls.constructors:
            if constructor.has_name(name):
                return constructor
        raise Exception(f"Only {cls.get_constructor_names()} are supported.")


class NamedObject():
    @classmethod
    def has_name(cls, name):
        name = convert_to_lowercase(name)
        return name == cls.name


class AlternativelyNamedObject(NamedObject):
    @property
    def short_name(self):
        return ''.join([word[0] for word in self.name.split('_')])
    
    @property
    def spaced_name(self):
        return ' '.join([word for word in self.name.split('_')])

    @classmethod
    def has_name(cls, name):
        name = convert_to_lowercase(name)
        return name in [cls.name, cls.short_name, cls.spaced_name]
