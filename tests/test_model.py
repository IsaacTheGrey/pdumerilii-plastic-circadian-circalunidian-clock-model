import unittest

from circadian_clock.model import (CWO_LUNAR_DELAY_KEY,
                                   DEFAULT_INITIAL_STATE, _LT_OVERRIDE_KEY,
                                   _lunar_drive,
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
            CWO_LUNAR_DELAY_KEY: 0.0,
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

    def test_positive_lunar_delay_shifts_cwo_drive_later(self):
        delay_h = 24.0
        delayed = {CWO_LUNAR_DELAY_KEY: delay_h}
        self.assertAlmostEqual(_lunar_drive(delay_h, delayed),
                               _lunar_drive(0.0, {}))

    def test_negative_lunar_delay_advances_cwo_drive(self):
        advance_h = 24.0
        advanced = {CWO_LUNAR_DELAY_KEY: -advance_h}
        self.assertAlmostEqual(_lunar_drive(0.0, advanced),
                               _lunar_drive(advance_h, {}))

    def test_frozen_lunar_drive_takes_precedence_over_delay(self):
        parameters = {
            CWO_LUNAR_DELAY_KEY: 48.0,
            _LT_OVERRIDE_KEY: 0.63,
        }
        self.assertAlmostEqual(_lunar_drive(123.0, parameters), 0.63)


if __name__ == '__main__':
    unittest.main()
