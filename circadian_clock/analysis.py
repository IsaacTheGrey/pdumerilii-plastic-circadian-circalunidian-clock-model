"""Period detection and sensitivity / plasticity sweeps.

All period-detection routines take a 1-D time series (typically the CLK/BMAL
trace ``sol[:, 0]``). The previous mixed 1-D/2-D convention caused
``IndexError`` in plasticity sweeps; everything is unified here.
"""

from __future__ import annotations

import csv
from multiprocessing import Pool

import numpy as np
from scipy.signal import periodogram

from .model import T_LUNAR
from .simulation import integrate_model


# ---------------------------------------------------------------------------
# Spectral utilities
# ---------------------------------------------------------------------------

def normalize_oscillations(solution: np.ndarray) -> np.ndarray:
    """Divide each column by its mean (mean-normalisation)."""
    return solution / solution.mean(axis=0)


def compute_periodogram(signal_1d: np.ndarray, dt: float,
                        pad_factor: int = 10):
    """Periodogram of a 1-D time series.

    Returns ``(frequencies, power)`` with the zero-frequency bin removed.
    Frequency resolution is increased ``pad_factor``-fold by zero-padding
    the FFT input.
    """
    signal_1d = np.asarray(signal_1d).ravel()
    freqs, power = periodogram(signal_1d, fs=1.0 / dt,
                               nfft=len(signal_1d) * pad_factor)
    nonzero = freqs > 0
    return freqs[nonzero], power[nonzero]


def dominant_period(signal_1d: np.ndarray, dt: float,
                    band=(18.0, 30.0), pad_factor: int = 10) -> float:
    """Dominant Fourier period of ``signal_1d`` within ``band`` (hours).

    Returns ``nan`` if no peak falls in band or if the signal is essentially
    flat (amplitude < 5 % of the mean across the segment).
    """
    sig = np.asarray(signal_1d).ravel()
    if np.ptp(sig) < 0.05 * max(abs(sig.mean()), 1e-9):
        return np.nan
    freqs, power = compute_periodogram(sig, dt, pad_factor=pad_factor)
    periods = 1.0 / freqs
    in_band = (periods >= band[0]) & (periods <= band[1])
    if not in_band.any():
        return np.nan
    idx_band = np.where(in_band)[0]
    best = idx_band[np.argmax(power[in_band])]
    return periods[best]


# ---------------------------------------------------------------------------
# Sensitivity sweep
# ---------------------------------------------------------------------------

def _sensitivity_worker(args):
    """Pool worker — must be module-level for pickling."""
    parameter_name, value, base_parameters, y0, t, last_osc, band = args
    p = base_parameters.copy()
    p[parameter_name] = value

    sol = integrate_model(y0, t, p)

    dt = t[1] - t[0]
    n_keep = int(last_osc / dt)
    bmal = sol[-n_keep:, 0]
    bmal = bmal / bmal.mean() if bmal.mean() > 0 else bmal

    period = dominant_period(bmal, dt, band=band)
    return value, period


def analyze_sensitivity(parameter_name, parameter_range, base_parameters,
                        y0, t, last_osc: float = 144.0,
                        band=(18.0, 30.0)):
    """Sweep one parameter, return dominant circadian period at each value."""
    work = [(parameter_name, v, base_parameters, y0, t, last_osc, band)
            for v in parameter_range]
    with Pool() as pool:
        results = pool.map(_sensitivity_worker, work)
    results.sort(key=lambda r: r[0])
    return [r[1] for r in results]


def parameter_sweep(base_parameters, y0, t, parameter_ranges,
                    last_osc: float = 144.0, band=(18.0, 30.0)):
    """Sweep several parameters. Returns ``{param: (range, periods)}``."""
    out = {}
    for name, values in parameter_ranges.items():
        periods = analyze_sensitivity(name, values, base_parameters, y0, t,
                                      last_osc=last_osc, band=band)
        out[name] = (values, periods)
    return out


def save_sensitivity_results(sensitivity_results: dict, filename: str) -> None:
    """Write sweep results to CSV in long format."""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Parameter', 'Value',
                                               'Dominant Period'])
        writer.writeheader()
        for name, (values, periods) in sensitivity_results.items():
            for v, per in zip(values, periods):
                writer.writerow({'Parameter': name, 'Value': v,
                                 'Dominant Period': per})


# ---------------------------------------------------------------------------
# CWO plasticity sweep (used for the paper's central claim)
# ---------------------------------------------------------------------------

def analyze_cwo_plasticity(base_parameters, y0, t,
                           cwo_params=('nu11', 'nu14'),
                           n_points: int = 50,
                           fold_range=(0.1, 5.0),
                           band=(18.0, 30.0)):
    """How CWO synthesis (nu11) and degradation (nu14) shift the period.

    Returns ``{param: (values, periods)}`` with NaN at any value that fails
    the circadian-band filter.
    """
    dt = t[1] - t[0]
    out = {}

    for name in cwo_params:
        base_val = base_parameters[name]
        values = np.linspace(fold_range[0] * base_val,
                             fold_range[1] * base_val, n_points)

        periods = []
        for v in values:
            p = base_parameters.copy()
            p[name] = v
            sol = integrate_model(y0, t, p)
            window = int(300.0 / dt)
            bmal = sol[-window:, 0]
            periods.append(dominant_period(bmal, dt, band=band))

        out[name] = (values, periods)
    return out


# ---------------------------------------------------------------------------
# Lunar plasticity scan — period as a function of position in the lunar month
# ---------------------------------------------------------------------------

def analyze_lunar_period_oscillation(base_parameters, y0,
                                     dt: float = 0.1,
                                     n_lunar_cycles: int = 2,
                                     window_size_h: float = 72.0,
                                     step_size_h: float = 12.0,
                                     band=(18.0, 30.0)):
    """Local circadian period across one lunar month.

    The system is integrated for ``n_lunar_cycles`` lunar months; only the
    last cycle is analysed (transients discarded). A sliding periodogram
    window of ``window_size_h`` hours is stepped every ``step_size_h`` hours.

    Returns
    -------
    lunar_times_h : ndarray
        Window-midpoint time within the lunar month (0 .. T_LUNAR).
    periods_h : ndarray
        Dominant period in band (NaN where none).
    """
    total_time = T_LUNAR * n_lunar_cycles
    t = np.arange(0.0, total_time, dt)
    print(f"Integrating model for {total_time:.0f} h ({n_lunar_cycles} lunar cycles)...")
    sol = integrate_model(y0, t, base_parameters)

    window_pts = int(window_size_h / dt)
    step_pts   = int(step_size_h   / dt)
    start_idx  = int(T_LUNAR / dt) * (n_lunar_cycles - 1)

    lunar_times, observed_periods = [], []
    print("Sliding-window period analysis on final lunar cycle...")
    for i in range(start_idx, len(t) - window_pts, step_pts):
        segment = sol[i: i + window_pts, 0]
        period = dominant_period(segment, dt, band=band)
        midpoint_time = t[i + window_pts // 2] % T_LUNAR
        lunar_times.append(midpoint_time)
        observed_periods.append(period)

    order = np.argsort(lunar_times)
    return np.array(lunar_times)[order], np.array(observed_periods)[order]