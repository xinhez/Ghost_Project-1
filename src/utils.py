import numpy as np
import random
import torch


def amplify_value_dictionary_by_batch_size(dictionary, batch_size):
    amplified_dictionary = {}
    for key in dictionary:
        amplified_dictionary[key] = dictionary[key] * batch_size
    return amplified_dictionary


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


def set_random_seed(n, r, t, seed=3407):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def sum_value_dictionaries(dictionary0, dictionary1):
    """\
    Sum values in dictionaries. 
    This method does not check types if at least one of the supplied dictionaries is empty.
    """
    if not dictionary0:
        return dictionary1
    elif not dictionary1:
        return dictionary0
        
    combined_dictionary = {}
    for key in set(dictionary0.keys()).union(set(dictionary1.keys())):
        combined_dictionary[key] = dictionary0.get(key, 0) + dictionary1.get(key, 0)
    return combined_dictionary


def sum_value_lists(list0, list1):
    """\
    Sum values in lists. 
    This method does not check types if at least one of the supplied dictionaries is empty.
    """
    if len(list0) == 0:
        return list1
    if len(list1) == 0:
        return list0
    elif len(list0) != len(list1):
        raise Exception("Please sum value lists of the same length.")
    else:
        combined_list = []
        for value0, value1 in zip(list0, list1):
            combined_list.append(value0 + value1)
    return combined_list