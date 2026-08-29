import unittest

from circadian_clock.model import generate_default_parameters


class ModelParameterTests(unittest.TestCase):
    def test_k5_migration_preserves_legacy_default_halfpoint(self):
        parameters = generate_default_parameters()
        legacy_effective_k5 = 0.8 ** 4
        self.assertAlmostEqual(parameters['K5'], legacy_effective_k5)

    def test_default_w_gain_and_half_life(self):
        parameters = generate_default_parameters()
        self.assertAlmostEqual(parameters['nu13'] / parameters['nu14'], 4.0)
        self.assertAlmostEqual(0.69314718056 / parameters['nu14'],
                               3.4657359028)


if __name__ == '__main__':
    unittest.main()
