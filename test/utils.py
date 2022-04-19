from pandas import concat
import torch

from unittest import TestCase

from src.utils import (
    average_dictionary_values_by_sample_size,
    concat_tensor_lists,
    count_unique,
    convert_to_lowercase,
    inplace_combine_tensor_lists,
)
from src.utils import sum_value_dictionaries, sum_value_lists


class TestUtils(TestCase):
    def test_average_dictionary_values_by_count(self):
        self.assertRaises(Exception, average_dictionary_values_by_sample_size, {}, 0)
        self.assertEqual({}, average_dictionary_values_by_sample_size({}, 1))

        d = {
            "test_key0": 2,
            "test_key1": 20,
        }
        expected_d = {
            "test_key0": 1,
            "test_key1": 10,
        }
        self.assertEqual(expected_d, average_dictionary_values_by_sample_size(d, 2))

    def test_count_unique(self):
        self.assertEqual(0, count_unique([]))

        l = [0, 1, "a", "b", 0, 1]
        self.assertEqual(4, count_unique(l))

    def test_convert_to_lowercase(self):
        self.assertEqual(0, convert_to_lowercase(0))
        self.assertEqual("a", convert_to_lowercase("a"))
        self.assertEqual("a", convert_to_lowercase("A"))
        self.assertEqual("abca", convert_to_lowercase("AbCa"))

    def test_combine_tensor_lists(self):
        t00 = torch.tensor([1, 10, 100])
        t01 = torch.tensor([2, 20, 200])
        t10 = torch.tensor([3, 30, 300])
        t11 = torch.tensor([4, 40, 400])
        o = torch.tensor([9, 90, 900, 9000])
        l = torch.tensor([11, 12, 13, 21, 22, 23])
        list0 = [[[t00, t01], [t10, t11]], o, l]

        t002 = torch.tensor([5, 50, 500])
        t012 = torch.tensor([6, 60, 600])
        t102 = torch.tensor([7, 70, 700])
        t112 = torch.tensor([8, 80, 800])
        o2 = torch.tensor([902, 903, 904, 905])
        l2 = torch.tensor([14, 15, 16, 24, 25, 26])
        list1 = [[[t002, t012], [t102, t112]], o2, l2]

        list0_out = [[[[t00], [t01]], [[t10], [t11]]], [o], [l]]
        list1_out = [
            [[[t00, t002], [t01, t012]], [[t10, t102], [t11, t112]]],
            [o, o2],
            [l, l2],
        ]

        a = []
        inplace_combine_tensor_lists(a, list0)
        self.assertEqual(a, list0_out)

        inplace_combine_tensor_lists(a, list1)
        self.assertEqual(a, list1_out)

    def test_concat_tensor_lists(self):
        t00 = torch.tensor([1, 10, 100])
        t01 = torch.tensor([2, 20, 200])
        t10 = torch.tensor([3, 30, 300])
        t11 = torch.tensor([4, 40, 400])
        o = torch.tensor([9, 90, 900, 9000])
        l = torch.tensor([11, 12, 13, 21, 22, 23])
        list0 = [[[t00, t01], [t10, t11]], o, l]

        t002 = torch.tensor([5, 50, 500])
        t012 = torch.tensor([6, 60, 600])
        t102 = torch.tensor([7, 70, 700])
        t112 = torch.tensor([8, 80, 800])
        o2 = torch.tensor([902, 903, 904, 905])
        l2 = torch.tensor([14, 15, 16, 24, 25, 26])
        list1 = [[[t002, t012], [t102, t112]], o2, l2]

        lists = []
        inplace_combine_tensor_lists(lists, list0)
        inplace_combine_tensor_lists(lists, list1)

        t003 = torch.tensor([1, 10, 100, 5, 50, 500])
        t013 = torch.tensor([2, 20, 200, 6, 60, 600])
        t103 = torch.tensor([3, 30, 300, 7, 70, 700])
        t113 = torch.tensor([4, 40, 400, 8, 80, 800])
        o3 = torch.tensor([9, 90, 900, 9000, 902, 903, 904, 905])
        l3 = torch.tensor([11, 12, 13, 21, 22, 23, 14, 15, 16, 24, 25, 26])

        lists_out = [[[t003, t013], [t103, t113]], o3, l3]
        self.assertTrue(
            torch.equal(concat_tensor_lists(lists)[0][0][0], lists_out[0][0][0])
        )
        self.assertTrue(
            torch.equal(concat_tensor_lists(lists)[0][0][1], lists_out[0][0][1])
        )
        self.assertTrue(
            torch.equal(concat_tensor_lists(lists)[0][1][0], lists_out[0][1][0])
        )
        self.assertTrue(
            torch.equal(concat_tensor_lists(lists)[0][1][1], lists_out[0][1][1])
        )
        self.assertTrue(torch.equal(concat_tensor_lists(lists)[1], lists_out[1]))
        self.assertTrue(torch.equal(concat_tensor_lists(lists)[2], lists_out[2]))

    def test_combine_value_dictionaries(self):
        self.assertEqual({}, sum_value_dictionaries({}, {}))

        d = {
            "test_key": 0,
        }
        self.assertEqual(d, sum_value_dictionaries({}, d))
        self.assertEqual(d, sum_value_dictionaries(d, {}))

        d0 = {
            "test_key0": 0,
        }
        d1 = {
            "test_key1": 1,
        }
        expected_d = {
            "test_key0": 0,
            "test_key1": 1,
        }
        self.assertEqual(expected_d, sum_value_dictionaries(d0, d1))

    def test_sum_value_lists(self):
        self.assertEqual([], sum_value_lists([], []))
        self.assertEqual([1], sum_value_lists([], [1]))
        self.assertEqual([1], sum_value_lists([1], []))
        self.assertEqual([3], sum_value_lists([1], [2]))
        self.assertRaises(Exception, sum_value_lists, [1], [2, 3])
