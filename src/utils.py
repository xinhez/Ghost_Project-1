import numbers
import numpy as np
import torch


def average_dictionary_values_by_count(dictionary, count):
    if count < 1:
        raise Exception("Please use positive count to average dictionary values.")
    for key in dictionary:
        dictionary[key] /= count
    return dictionary


def count_unique(l):
    """\
    Count of the number of unique items in the given list.
    """
    return len(np.unique(l))


def convert_to_lowercase(item):
    """\
    If the given item is a string, return its lowercase.
    """
    if isinstance(item, str):
        return item.lower()
    else:
        return item


def combine_tensor_lists(list0, list1):
    """\
    Combine lists of tensors.
    This method does not check types if at least one of the supplied lists is empty.
    """
    if len(list0) == 0:
        return list1 
    elif len(list1) == 0:
        return list0
    elif len(list0) != len(list1):
        raise Exception("Please only combine lists of the same length.")
        
    combined_lists = []
    for l0, l1 in zip(list0, list1):
        if isinstance(l0, list) or isinstance(l1, list):
            combined_lists.append(combine_tensor_lists(l0, l1))
        elif not isinstance(l0, torch.Tensor):
            raise Exception(f"Type {type(l0)} is not supported in combine_tensor_lists.")
        elif not isinstance(l1, torch.Tensor):
            raise Exception(f"Type {type(l1)} is not supported in combine_tensor_lists.")
        else:
            combined_lists.append(torch.cat([l0, l1], dim=0))
    return combined_lists


def sum_value_dictionaries(dictionary0, dictionary1):
    """\
    Sum values in dictionaries. 
    This method does not check types if at least one of the supplied dictionaries is empty.
    """
    if len(dictionary0) == 0:
        return dictionary1
    elif len(dictionary1) == 0:
        return dictionary0
        
    combined_dictionary = {}
    for key in set(dictionary0.keys()).union(set(dictionary1.keys())):
        if not isinstance(dictionary0.get(key, 0), numbers.Number):
            raise Exception(f"{dictionary0.get(key, 0)} is not supported as dictionary value.")
        elif not isinstance(dictionary0.get(key, 1), numbers.Number):
            raise Exception(f"{dictionary0.get(key, 1)} is not supported as dictionary value.")
        else:
            combined_dictionary[key] = dictionary0.get(key, 0) + dictionary1.get(key, 0)
    return combined_dictionary