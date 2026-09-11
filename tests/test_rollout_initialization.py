import unittest
from unittest import mock

import numpy as np

from sousvide.synthesize.synthesize_helper import generate_perturbations


class RolloutInitializationTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.zeros((4,15))
        self.reference[:,0] = [0.0,0.05,0.10,0.15]
        self.reference[:,1:7] = np.arange(24).reshape(4,6)
        self.reference[:,10] = 1.0

    def test_jitter_is_preserved_and_early_starts_shift_to_second_sample(self):
        samples = generate_perturbations(
            np.array([0.0,0.025,0.075,0.125]),self.reference,np.zeros(10),
            include_previous=True)
        self.assertEqual([s['t0'] for s in samples],[0.05,0.05,0.075,0.125])
        for sample,index in zip(samples,[1,1,1,2]):
            np.testing.assert_array_equal(sample['x0'],self.reference[index,1:11])
            np.testing.assert_array_equal(sample['x_prev'],self.reference[index-1,1:11])

    def test_shared_noise_and_normalized_sign_aligned_quaternions(self):
        self.reference[0,7:11] = [0.0,0.0,0.6,0.8]
        self.reference[1,7:11] = [0.0,0.0,-0.8,-0.6]
        original = self.reference.copy()
        noise = np.linspace(0.01,0.1,10)
        with mock.patch('numpy.random.uniform',return_value=noise) as draw:
            sample = generate_perturbations(
                np.array([0.075]),self.reference,np.full(10,0.2),
                include_previous=True)[0]
        self.assertEqual(draw.call_count,1)
        for key,index in [('x_prev',0),('x0',1)]:
            expected = self.reference[index,1:11]+noise
            np.testing.assert_allclose(sample[key][:6],expected[:6])
            expected_q = expected[6:]/np.linalg.norm(expected[6:])
            self.assertAlmostEqual(abs(np.dot(sample[key][6:],expected_q)),1.0)
            self.assertAlmostEqual(np.linalg.norm(sample[key][6:]),1.0)
        self.assertGreaterEqual(np.dot(sample['x_prev'][6:],sample['x0'][6:]),0)
        np.testing.assert_array_equal(self.reference,original)

    def test_events_require_two_samples(self):
        for count in [0,1]:
            with self.assertRaisesRegex(ValueError,'at least two'):
                generate_perturbations(
                    np.array([0.0]),self.reference[:count],np.zeros(10),
                    include_previous=True)

    def test_non_event_start_remains_at_first_sample_without_predecessor(self):
        sample = generate_perturbations(
            np.array([0.0]),self.reference[:1],np.zeros(10))[0]
        self.assertEqual(sample['t0'],0.0)
        self.assertNotIn('x_prev',sample)
        np.testing.assert_array_equal(sample['x0'],self.reference[0,1:11])
