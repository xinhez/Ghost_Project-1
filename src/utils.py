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