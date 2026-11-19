"""Literal mathematical fixtures; no data, Torch, GPU, or model calls."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.interaction_statistics import compute_real_statistics


class StatisticsFixtures(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat(np.arange(4), 2)
        self.components = np.tile([10, 20], 4)
        self.x = np.array([[-1.], [1.], [1.], [-1.], [1.], [-1.], [-1.], [1.]])

    def test_cross_group_positive_covariance(self):
        # Component influences are -1/2,+1/2; 2/(2-1)*sum(v²)=1.
        out = compute_real_statistics(self.x, self.groups, self.components)
        np.testing.assert_allclose(out['target'], [0], atol=0)
        np.testing.assert_allclose(out['scale'], [1], atol=0)
        np.testing.assert_allclose(out['uncertainty'], [1], atol=0)

    def test_cross_group_cancellation_is_not_independent_groups(self):
        x = np.tile([[-1.], [1.]], (4, 1))
        out = compute_real_statistics(x, self.groups, self.components)
        np.testing.assert_allclose(out['uncertainty'], [0], atol=0)
        distinct = compute_real_statistics(x, self.groups, np.arange(8))
        np.testing.assert_allclose(distinct['uncertainty'], [1 / math.sqrt(7)], rtol=1e-14)

    def test_image_weighted_mean_not_equal_identity_mean(self):
        x = np.array([[0.], [0.], [6.], [0.], [0.], [0.], [0.], [0.], [0.]])
        out = compute_real_statistics(x, np.array([0,0,0,1,1,2,2,3,3]),
                                      np.array([1,1,2,1,2,1,2,1,2]))
        np.testing.assert_allclose(out['target'], [.5], rtol=1e-14)
        np.testing.assert_allclose(out['scale'], [math.sqrt(2)], rtol=1e-14)
        np.testing.assert_allclose(out['uncertainty'], [2/3], rtol=1e-14)
        self.assertEqual(out['cluster_weight_effective_counts'][0], 9/5)

    def test_repeat_presentations_add_no_independent_information(self):
        a = compute_real_statistics(self.x, self.groups, self.components)
        b = compute_real_statistics(np.repeat(self.x, 3, axis=0),
                                    np.repeat(self.groups, 3), np.repeat(self.components, 3))
        for key in ('target', 'scale', 'uncertainty', 'cluster_weight_effective_counts'):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-14)
        self.assertEqual(a['n_clusters'], b['n_clusters'])

    def test_shared_offset_invariance_and_positive_scale_equivariance(self):
        a = compute_real_statistics(self.x, self.groups, self.components)
        b = compute_real_statistics(3*self.x+123, self.groups, self.components)
        for key in ('target', 'scale', 'uncertainty'):
            np.testing.assert_allclose(b[key], 3*np.asarray(a[key]), rtol=1e-14, atol=1e-14)

    def test_group_relabel_one_bit_changes_target_sign_only(self):
        x = self.x + np.repeat([1,2,3,8],2)[:,None]
        a = compute_real_statistics(x, self.groups, self.components)
        b = compute_real_statistics(x, self.groups ^ 1, self.components)
        np.testing.assert_allclose(b['target'], -np.asarray(a['target']), rtol=1e-14)
        for key in ('scale','uncertainty'):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-14)

    def test_row_and_component_name_invariance(self):
        a = compute_real_statistics(self.x, self.groups, self.components)
        perm = np.array([7,2,5,0,6,1,3,4])
        b = compute_real_statistics(self.x[perm], self.groups[perm], 91-self.components[perm])
        for key in ('target', 'scale', 'uncertainty'):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-14)

    def test_constant_coordinates_use_explicit_floor(self):
        x = np.repeat([0.,0.,0.,4.], 2)[:,None]
        out = compute_real_statistics(x, self.groups, self.components)
        self.assertEqual(out['target'], [1.])
        self.assertEqual(out['scale'], [.001])
        self.assertEqual(out['uncertainty'], [0.])
        self.assertFalse(out['confidence_bound'])

    def test_invalid_inputs_rejected(self):
        invalid = [
            (self.x, self.groups.astype(float), self.components, {}),
            (self.x, self.groups, np.zeros(8,dtype=int), {}),
            (self.x, self.groups, self.components, {'scale_floor': float('nan')}),
            (self.x, self.groups, self.components, {'scale_floor': 0}),
            (np.full_like(self.x, np.nan), self.groups, self.components, {}),
            (self.x[:6], self.groups[:6], self.components[:6], {}),
            (self.x, self.groups, self.components.astype(float), {}),
        ]
        for args in invalid:
            with self.subTest(args=str(args[3])):
                with self.assertRaises(ValueError):
                    compute_real_statistics(*args[:3], **args[3])


if __name__ == '__main__':
    unittest.main()
