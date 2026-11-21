import unittest

import numpy as np

from benchmark_detectors import craft_regions, project_regions, rectangle, score


class BenchmarkTests(unittest.TestCase):
    def test_duplicate_predictions_do_not_both_match(self):
        box = rectangle([0, 0, 10, 10])
        self.assertEqual(score([box], [box, box], [], .5),
                         {"tp": 1, "fp": 1, "fn": 0, "ignored": 0})

    def test_ignore_only_suppresses_unmatched_predictions(self):
        text = rectangle([0, 0, 10, 10])
        glare = rectangle([20, 20, 10, 10])
        self.assertEqual(score([text], [text, glare], [text, glare], .5),
                         {"tp": 1, "fp": 0, "fn": 0, "ignored": 1})

    def test_threshold_distinguishes_partial_overlap(self):
        label = rectangle([0, 0, 10, 10])
        prediction = rectangle([2, 0, 10, 10])
        self.assertEqual(score([label], [prediction], [], .5)["tp"], 1)
        self.assertEqual(score([label], [prediction], [], .75)["tp"], 0)

    def test_inverse_coordinate_mapping(self):
        transform = np.array([[.5, 0, 2], [0, .5, 4], [0, 0, 1.]])
        mapped, = project_regions([rectangle([2, 4, 5, 6])], transform)
        np.testing.assert_allclose(mapped, rectangle([0, 0, 10, 12]))

    def test_craft_box_formats(self):
        class Reader:
            def detect(self, *_args, **_kwargs):
                return ([[[1, 7, 2, 8]]], [[[[0, 0], [3, 1], [3, 4], [0, 3]]]])

        regions = craft_regions(Reader(), np.zeros((10, 10, 3), np.uint8))
        self.assertEqual(len(regions), 2)
        np.testing.assert_array_equal(regions[0], rectangle([1, 2, 6, 6]))
