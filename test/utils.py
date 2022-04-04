import torch

from unittest import TestCase

from src.utils import average_dictionary_values_by_count, count_unique, convert_to_lowercase, combine_tensor_lists, sum_value_dictionaries


class TestUtils(TestCase):
    def test_average_dictionary_values_by_count(self):
        self.assertRaises(Exception, average_dictionary_values_by_count, {}, 0)
        self.assertEqual({}, average_dictionary_values_by_count({}, 1))

        d = {
            'test_key0': 2,
            'test_key1': 20, 
        }
        expected_d = {
            'test_key0': 1,
            'test_key1': 10,
        }
        self.assertEqual(expected_d, average_dictionary_values_by_count(d, 2))


    def test_count_unique(self):
        self.assertEqual(0, count_unique([]))

        l = [0, 1, 'a', 'b', 0, 1]
        self.assertEqual(4, count_unique(l))


    def test_convert_to_lowercase(self):
        self.assertEqual(0, convert_to_lowercase(0))
        self.assertEqual('a', convert_to_lowercase('a'))
        self.assertEqual('a', convert_to_lowercase('A'))
        self.assertEqual('abca', convert_to_lowercase('AbCa'))


    def test_combine_tensor_lists(self):
        self.assertEqual([], combine_tensor_lists([], []))

        l = [torch.Tensor([1])]
        self.assertEqual(l, combine_tensor_lists([], l))
        self.assertEqual(l, combine_tensor_lists(l, []))

        l0 = [torch.Tensor([1])]
        l1 = [[]]
        self.assertEqual(l0, combine_tensor_lists(l0, l1))

        l0 = [torch.Tensor([1]), []]
        l1 = [[], torch.Tensor([2])]
        expected_l = [torch.Tensor([1]), torch.Tensor([2])]
        self.assertEqual(expected_l, combine_tensor_lists(l0, l1))

        l = [1]
        self.assertRaises(Exception, combine_tensor_lists, l, l)


    def test_combine_value_dictionaries(self):
        self.assertEqual({}, sum_value_dictionaries({}, {}))

        d = {
            'test_key': 0,
        }
        self.assertEqual(d, sum_value_dictionaries({}, d))
        self.assertEqual(d, sum_value_dictionaries(d, {}))

        d0 = {
            'test_key0': 0,
        }
        d1 = {
            'test_key1': 1,
        }
        expected_d = {
            'test_key0': 0,
            'test_key1': 1,
        }
        self.assertEqual(expected_d, sum_value_dictionaries(d0, d1))


        d = {
            'test_key0': 0,
            'test_key_list': 'test_exception',
        }

        self.assertRaises(Exception, sum_value_dictionaries, d, d)