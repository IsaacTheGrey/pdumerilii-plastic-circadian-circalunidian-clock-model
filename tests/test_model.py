import unittest

from circadian_clock.model import (DEFAULT_INITIAL_STATE,
                                   generate_default_parameters,
                                   goodwin_model_lunar)


class ModelParameterTests(unittest.TestCase):
    def test_default_coefficients_match_seven_state_model(self):
        expected = {
            'nu1': 0.7, 'nu2': 0.5,
            'nu3': 0.45, 'nu4': 0.3,
            'nu5': 0.7, 'nu6': 0.35,
            'nu7': 0.3, 'nu8': 0.2,
            'nu9': 0.1, 'nu10': 0.2,
            'nu11': 0.2, 'nu12': 0.05,
            'nu13': 0.8, 'nu14': 0.2,
            'K1': 1.0, 'K2': 1.0, 'K3': 1.0, 'K4': 1.0,
            'K5': 0.4096, 'K6': 1.0, 'K7': 1.0, 'K8': 1.0,
            'K9': 1.0, 'K_W': 2.0,
            'hill': 4, 'hill_S': 1.5, 'hill_W': 1.0,
            'b': 1.0, 'c': 0.5, 'd': 0.5,
        }
        self.assertEqual(generate_default_parameters(), expected)

    def test_k5_migration_preserves_legacy_default_halfpoint(self):
        parameters = generate_default_parameters()
        legacy_effective_k5 = 0.8 ** 4
        self.assertAlmostEqual(parameters['K5'], legacy_effective_k5)

    def test_default_w_gain_and_half_life(self):
        parameters = generate_default_parameters()
        self.assertAlmostEqual(parameters['nu13'] / parameters['nu14'], 4.0)
        self.assertAlmostEqual(0.69314718056 / parameters['nu14'],
                               3.4657359028)

    def test_default_model_has_seven_states(self):
        parameters = generate_default_parameters()
        derivatives = goodwin_model_lunar(
            DEFAULT_INITIAL_STATE, 0.0, parameters)
        self.assertEqual(len(DEFAULT_INITIAL_STATE), 7)
        self.assertEqual(len(derivatives), 7)


if __name__ == '__main__':
    unittest.main()
