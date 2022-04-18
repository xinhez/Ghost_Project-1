import numpy as np
import torch


def amplify_value_dictionary_by_sample_size(dictionary, sample_size):
    amplified_dictionary = {}
    for key in dictionary:
        amplified_dictionary[key] = dictionary[key] * sample_size
    return amplified_dictionary


def average_dictionary_values_by_sample_size(dictionary, sample_size):
    if sample_size < 1:
        raise Exception("Please use positive count to average dictionary values.")
    for key in dictionary:
        dictionary[key] /= sample_size
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
        elif not torch.is_tensor(l0):
            raise Exception(
                f"Type {type(l0)} is not supported in combine_tensor_lists."
            )
        elif not torch.is_tensor(l1):
            raise Exception(
                f"Type {type(l1)} is not supported in combine_tensor_lists."
            )
        else:
            combined_lists.append(torch.cat([l0, l1], dim=0))
    return combined_lists


def move_tensor_list_to_cpu(tensors):
    cpu_tensors = []
    for tensor in tensors:
        if isinstance(tensor, list):
            cpu_tensors.append(move_tensor_list_to_cpu(tensor))
        else:
            cpu_tensors.append(tensor.detach().cpu())
    return cpu_tensors


def set_random_seed(n, r, t, seed):
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.benchmark = False
    t.backends.cudnn.deterministic = True
    n.random.seed(seed)
    r.seed(seed)


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
