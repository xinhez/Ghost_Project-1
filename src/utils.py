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


def inplace_combine_tensor_lists(lists, new_list):
    """\
    In place add a new (nested) tensor list to current collections.
    This operation will move all concerned tensors to CPU.
    """
    if len(lists) == 0:
        for new_l in new_list:
            if isinstance(new_l, list):
                l = []
                inplace_combine_tensor_lists(l, new_l)
                lists.append(l)
            else:
                lists.append([new_l.detach().cpu()])
    else:
        for l, new_l in zip(lists, new_list):
            if isinstance(new_l, list):
                inplace_combine_tensor_lists(l, new_l)
            else:
                l.append(new_l.detach().cpu())


def concat_tensor_lists(lists):
    new_lists = []
    for l in lists:
        if len(l) == 0:
            raise Exception("Cannot concatenate empty tensor list.")
        if isinstance(l[0], list):
            new_lists.append(concat_tensor_lists(l))
        else:
            new_lists.append(torch.cat(l, dim=0))
    return new_lists


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
