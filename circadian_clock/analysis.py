"""Period detection and sensitivity / plasticity sweeps.

All period-detection routines take a 1-D time series (typically the CLK/BMAL
trace ``sol[:, 0]``). The previous mixed 1-D/2-D convention caused
``IndexError`` in plasticity sweeps; everything is unified here.
"""

from __future__ import annotations

import csv
from multiprocessing import Pool

import numpy as np
from scipy.signal import find_peaks, periodogram

from .model import DEFAULT_INITIAL_STATE, T_LUNAR, _LT_OVERRIDE_KEY
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
    Zero-padding interpolates the displayed spectrum but does not add
    independent frequency information.
    """
    signal_1d = np.asarray(signal_1d).ravel()
    freqs, power = periodogram(signal_1d, fs=1.0 / dt,
                               nfft=len(signal_1d) * pad_factor)
    nonzero = freqs > 0
    return freqs[nonzero], power[nonzero]


def _refined_peak_times(signal_1d: np.ndarray, peaks: np.ndarray,
                        dt: float) -> np.ndarray:
    """Return peak times with three-point parabolic sub-step refinement."""
    sig = np.asarray(signal_1d, dtype=float)
    refined = []
    for peak in peaks:
        offset = 0.0
        if 0 < peak < len(sig) - 1:
            left, centre, right = sig[peak - 1: peak + 2]
            denom = left - 2.0 * centre + right
            if denom != 0.0:
                offset = float(np.clip(0.5 * (left - right) / denom,
                                       -0.5, 0.5))
        refined.append((peak + offset) * dt)
    return np.asarray(refined)


def rhythm_metrics(signal_1d: np.ndarray, dt: float,
                   band=(18.0, 30.0), min_intervals: int = 4,
                   min_relative_amplitude: float = 0.05,
                   max_interval_cv: float = 0.15,
                   min_sustained_ratio: float = 0.25) -> dict:
    """Measure period and oscillator quality from time-domain peaks.

    A period is reported only for a sustained, sufficiently regular rhythm.
    This prevents a slow lunar trend or a damped trajectory from being
    assigned the edge of the requested Fourier band.
    """
    sig = np.asarray(signal_1d, dtype=float).ravel()
    result = {
        'period': np.nan,
        'relative_amplitude': np.nan,
        'interval_cv': np.nan,
        'n_peaks': 0,
        'sustained_ratio': np.nan,
        'rhythmic': False,
        'status': 'invalid_signal',
    }
    if len(sig) < 3 or not np.isfinite(sig).all() or dt <= 0:
        return result

    amplitude = float(np.ptp(sig))
    scale = max(abs(float(np.mean(sig))), 1e-9)
    relative_amplitude = amplitude / scale
    result['relative_amplitude'] = relative_amplitude
    if relative_amplitude < min_relative_amplitude:
        result['status'] = 'low_amplitude'
        return result

    thirds = [chunk for chunk in np.array_split(sig, 3) if len(chunk)]
    segment_amplitudes = np.asarray([np.ptp(chunk) for chunk in thirds])
    max_segment_amplitude = float(segment_amplitudes.max())
    sustained_ratio = (float(segment_amplitudes.min()) /
                       max(max_segment_amplitude, 1e-12))
    result['sustained_ratio'] = sustained_ratio
    if sustained_ratio < min_sustained_ratio:
        result['status'] = 'not_sustained'
        return result

    min_distance = max(1, int(0.75 * band[0] / dt))
    prominence = max(0.10 * amplitude, 1e-12)
    peaks, _ = find_peaks(sig, distance=min_distance,
                          prominence=prominence)
    result['n_peaks'] = int(len(peaks))
    if len(peaks) < min_intervals + 1:
        result['status'] = 'insufficient_peaks'
        return result

    peak_times = _refined_peak_times(sig, peaks, dt)
    intervals = np.diff(peak_times)
    valid = intervals[(intervals >= band[0]) & (intervals <= band[1])]
    if len(valid) < min_intervals:
        result['status'] = 'period_outside_band'
        return result

    period = float(np.median(valid))
    interval_cv = float(np.std(valid, ddof=1) / np.mean(valid)) \
        if len(valid) > 1 else 0.0
    result['interval_cv'] = interval_cv
    if interval_cv > max_interval_cv:
        result['status'] = 'irregular_period'
        return result

    result.update(period=period, rhythmic=True, status='rhythmic')
    return result


def dominant_period(signal_1d: np.ndarray, dt: float,
                    band=(18.0, 30.0), pad_factor: int = 10) -> float:
    """Period of a sustained rhythm within ``band``, or ``nan``.

    ``pad_factor`` is retained for API compatibility. Period estimation is
    now based on refined time-domain peaks; the periodogram remains available
    separately for display and corroboration.
    """
    del pad_factor
    return rhythm_metrics(signal_1d, dt, band=band,
                          min_intervals=2)['period']


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
                                               'Dominant Period'],
                                lineterminator='\n')
        writer.writeheader()
        for name, (values, periods) in sensitivity_results.items():
            for v, per in zip(values, periods):
                writer.writerow({'Parameter': name, 'Value': v,
                                 'Dominant Period': per})


def _fixed_phase_sweep_worker(args):
    """Evaluate one parameter/mechanism at one frozen lunar drive."""
    (name, fold, lunar_label, lunar_drive, base_parameters, y0, t,
     analysis_points, band) = args
    p = base_parameters.copy()
    p[_LT_OVERRIDE_KEY] = lunar_drive

    if name == 'W gain (nu13/nu14)':
        # Change protein abundance gain while preserving W's time constant.
        p['nu13'] *= fold
        parameter_value = p['nu13'] / p['nu14']
    elif name == 'W turnover speed':
        # Scale production and degradation together: gain stays fixed while
        # the response time 1/nu14 changes.
        p['nu13'] *= fold
        p['nu14'] *= fold
        parameter_value = p['nu14']
    else:
        p[name] *= fold
        parameter_value = p[name]

    sol = integrate_model(y0, t, p)
    tail = sol[-analysis_points:, :]
    metrics = rhythm_metrics(tail[:, 0], t[1] - t[0], band=band)
    return {
        'Parameter': name,
        'Fold Change': float(fold),
        'Parameter Value': float(parameter_value),
        'Drive Condition': lunar_label,
        'Fixed L': float(lunar_drive),
        'Period (h)': metrics['period'],
        'Relative Amplitude': metrics['relative_amplitude'],
        'Interval CV': metrics['interval_cv'],
        'Peak Count': metrics['n_peaks'],
        'Sustained Ratio': metrics['sustained_ratio'],
        'Mean C': float(np.mean(tail[:, 5])),
        'Mean W': float(np.mean(tail[:, 6])),
        'W Gain (nu13/nu14)': float(p['nu13'] / p['nu14']),
        'W Turnover Rate (h^-1)': float(p['nu14']),
        'W Half-life (h)': float(np.log(2.0) / p['nu14']),
        'Rhythmic': metrics['rhythmic'],
        'Status': metrics['status'],
    }


def _run_fixed_phase_sweep(names, base_parameters, y0, folds,
                           lunar_drives, dt, settle_hours, analysis_hours,
                           band, processes):
    folds = np.asarray(folds, dtype=float)
    if not np.any(np.isclose(folds, 1.0)):
        folds = np.sort(np.append(folds, 1.0))
    t = np.arange(0.0, settle_hours + analysis_hours, dt)
    analysis_points = int(analysis_hours / dt)
    work = [
        (name, fold, lunar_label, lunar_drive, base_parameters, y0, t,
         analysis_points, band)
        for name in names
        for lunar_label, lunar_drive in lunar_drives.items()
        for fold in folds
    ]
    if processes == 1:
        rows = [_fixed_phase_sweep_worker(item) for item in work]
    else:
        with Pool(processes=processes) as pool:
            rows = pool.map(_fixed_phase_sweep_worker, work)
    return rows


def _resolve_sweep_lunar_drives(lunar_drives, include_lunar_phases: bool):
    """Return fixed drive conditions for a publication parameter sweep.

    An explicitly supplied mapping always takes precedence. Otherwise the
    default is a single constant-mean drive, which removes lunar phase from
    the sweep. Setting ``include_lunar_phases`` adds the two drive extremes.
    """
    if lunar_drives is not None:
        return lunar_drives
    if include_lunar_phases:
        return {
            'Full moon': 0.50,
            'Mean drive': 0.75,
            'New moon': 1.00,
        }
    return {'Constant mean': 0.75}


def analyze_cwo_phase_sensitivity(base_parameters, y0,
                                  cwo_params=('nu11', 'nu12',
                                              'nu13', 'nu14'),
                                  folds=None,
                                  lunar_drives=None,
                                  dt: float = 0.1,
                                  settle_hours: float = 2400.0,
                                  analysis_hours: float = 720.0,
                                  band=(18.0, 30.0),
                                  processes: int | None = 1,
                                  include_lunar_phases: bool = False):
    """Publication sweep of all C/W kinetics at fixed lunar drive(s).

    The returned long-format rows include period, oscillator-quality metrics,
    and mean C/W abundance. Non-rhythmic simulations retain a row with a NaN
    period and an explicit failure status. By default the lunar multiplier is
    held at its cycle mean (L=0.75), preserving average CWO synthesis while
    removing phase-dependent forcing. Set ``include_lunar_phases=True`` to
    compare fixed full-moon (0.5), mean (0.75), and new-moon (1.0) drives.
    A custom ``lunar_drives`` mapping takes precedence over this switch.
    """
    if folds is None:
        folds = np.geomspace(0.5, 2.0, 13)
    lunar_drives = _resolve_sweep_lunar_drives(
        lunar_drives, include_lunar_phases)
    return _run_fixed_phase_sweep(
        cwo_params, base_parameters, y0, folds, lunar_drives, dt,
        settle_hours, analysis_hours, band, processes)


def analyze_w_gain_turnover(base_parameters, y0, folds=None,
                            lunar_drives=None, dt: float = 0.1,
                            settle_hours: float = 2400.0,
                            analysis_hours: float = 720.0,
                            band=(18.0, 30.0),
                            processes: int | None = 1,
                            include_lunar_phases: bool = False):
    """Separate W abundance gain from turnover at fixed lunar drive(s).

    ``W gain`` varies nu13/nu14 at fixed nu14. ``W turnover speed`` scales
    nu13 and nu14 together, preserving their ratio while changing the W
    response time. The default lunar multiplier is held at its mean, L=0.75.
    Set ``include_lunar_phases=True`` to also evaluate fixed full-moon and
    new-moon drives. A custom ``lunar_drives`` mapping takes precedence.
    """
    if folds is None:
        folds = np.geomspace(0.5, 2.0, 13)
    lunar_drives = _resolve_sweep_lunar_drives(
        lunar_drives, include_lunar_phases)
    names = ('W gain (nu13/nu14)', 'W turnover speed')
    return _run_fixed_phase_sweep(
        names, base_parameters, y0, folds, lunar_drives, dt,
        settle_hours, analysis_hours, band, processes)


def save_publication_sweep_results(rows: list[dict], filename: str) -> None:
    """Write long-format validated sweep rows to CSV."""
    if not rows:
        raise ValueError('No sweep rows to save')
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]),
                                lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CWO plasticity sweep (used for the paper's central claim)
# ---------------------------------------------------------------------------

def analyze_cwo_plasticity(base_parameters, y0, t,
                           cwo_params=('nu11', 'nu14'),
                           n_points: int = 50,
                           fold_range=(0.1, 5.0),
                           band=(18.0, 30.0),
                           lunar_drive: float = 0.75):
    """Legacy two-parameter sweep at a frozen lunar drive.

    Returns ``{param: (values, periods)}`` with NaN at any value that fails
    the sustained-rhythmicity checks. New analyses should prefer
    :func:`analyze_cwo_phase_sensitivity`.
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
            p[_LT_OVERRIDE_KEY] = lunar_drive
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


def analyze_lunar_limit_cycles(base_parameters, y0=None,
                               x_index: int = 0, y_index: int = 2,
                               dt: float = 0.1,
                               n_burn_cycles: int = 2,
                               band=(18.0, 30.0)) -> list[dict]:
    """Extract individual circadian phase-plane loops over a lunar month.

    Each loop runs from one CLK/BMAL peak to the next. Its lunar phase is the
    midpoint of that peak-to-peak interval in the final simulated month.
    A short extension beyond the month boundary permits extraction of the
    complete loop centred near the final full moon.
    """
    if y0 is None:
        y0 = DEFAULT_INITIAL_STATE

    analysis_start = n_burn_cycles * T_LUNAR
    analysis_end = analysis_start + T_LUNAR
    t = np.arange(0.0, analysis_end + band[1], dt)
    sol = integrate_model(y0, t, base_parameters)

    x = sol[:, x_index]
    start_search = max(0, int((analysis_start - band[1]) / dt))
    x_search = x[start_search:]
    prominence = max(0.10 * np.ptp(x_search), 1e-12)
    peaks, _ = find_peaks(
        x_search,
        distance=max(1, int(0.75 * band[0] / dt)),
        prominence=prominence)
    peaks = peaks + start_search

    cycles = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        period_h = (right - left) * dt
        midpoint = 0.5 * (t[left] + t[right])
        if not (analysis_start <= midpoint < analysis_end):
            continue
        if not (band[0] <= period_h <= band[1]):
            continue
        cycles.append({
            'lunar_phase_h': float((midpoint - analysis_start) % T_LUNAR),
            'period_h': float(period_h),
            'x': np.asarray(sol[left:right + 1, x_index]),
            'y': np.asarray(sol[left:right + 1, y_index]),
        })

    cycles.sort(key=lambda cycle: cycle['lunar_phase_h'])
    return cycles
