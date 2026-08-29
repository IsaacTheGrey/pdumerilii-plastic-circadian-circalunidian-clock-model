import unittest

import numpy as np

from circadian_clock.analysis import (_resolve_sweep_lunar_drives,
                                      analyze_lunar_limit_cycles,
                                      dominant_period, rhythm_metrics)
from circadian_clock.model import generate_default_parameters


class RhythmMetricsTests(unittest.TestCase):
    def setUp(self):
        self.dt = 0.1
        self.t = np.arange(0.0, 300.0, self.dt)

    def test_recovers_clean_circadian_period(self):
        signal = 2.0 + 0.5 * np.sin(2.0 * np.pi * self.t / 24.0)
        metrics = rhythm_metrics(signal, self.dt)
        self.assertTrue(metrics['rhythmic'])
        self.assertAlmostEqual(metrics['period'], 24.0, places=2)

    def test_rejects_slow_nonoscillatory_drift(self):
        signal = 1.0 + 0.2 * self.t / self.t[-1]
        self.assertTrue(np.isnan(dominant_period(signal, self.dt)))

    def test_rejects_damped_oscillation(self):
        signal = 2.0 + np.exp(-self.t / 45.0) * np.sin(
            2.0 * np.pi * self.t / 24.0)
        metrics = rhythm_metrics(signal, self.dt)
        self.assertFalse(metrics['rhythmic'])
        self.assertIn(metrics['status'], {'low_amplitude', 'not_sustained'})

    def test_extracts_complete_limit_cycles_across_lunar_month(self):
        cycles = analyze_lunar_limit_cycles(
            generate_default_parameters(), dt=0.2, n_burn_cycles=2)
        self.assertGreaterEqual(len(cycles), 25)
        self.assertTrue(all(18.0 <= cycle['period_h'] <= 30.0
                            for cycle in cycles))
        self.assertLess(cycles[0]['lunar_phase_h'],
                        cycles[-1]['lunar_phase_h'])

    def test_parameter_sweeps_exclude_lunar_phases_by_default(self):
        drives = _resolve_sweep_lunar_drives(None, False)
        self.assertEqual(drives, {'Constant mean': 0.75})

    def test_parameter_sweeps_can_include_fixed_lunar_phases(self):
        drives = _resolve_sweep_lunar_drives(None, True)
        self.assertEqual(drives, {
            'Full moon': 0.50,
            'Mean drive': 0.75,
            'New moon': 1.00,
        })


if __name__ == '__main__':
    unittest.main()
