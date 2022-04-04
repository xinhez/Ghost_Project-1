import numpy as np
import torch


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


def combine_tensor_lists(list0, list1):
    if len(list0) == 0:
        return list1 
    elif len(list1) == 0:
        return list0
    elif len(list0) != len(list1):
        raise Exception("Please only combine lists of the same length.")
        
    combined_lists = []
    for l0, l1 in zip(list0, list1):
        if isinstance(l0, torch.Tensor):
            combined_lists.append(torch.cat([l0, l1], dim=0))
        elif isinstance(l0, list):
            combined_lists.append(combine_tensor_lists(l0, l1))
        else:
            raise Exception(f"Type {type(l0)} is not supported in combine_tensor_lists.")
    return combined_lists