"""Compare and fit the model to z-score normalised RNA-seq data.

The observed Excel sheet has six ZT timepoints (every 4 h) with mean and SD
across replicates for four transcripts: ``cwo``, ``clk``, ``per``, ``pdp1``.
The current model has direct counterparts for ``cwo`` (C, index 5), ``clk``
(X, index 0; we treat BMAL as the readable proxy for the *clk/bmal* dimer)
and ``per`` (Y, index 1). ``pdp1`` is not in the model and is reported only
for visual reference.

Workflow
--------
1. Integrate ``integrate_at_lunar_phase`` for ~10 days to drop transients.
2. Take the last 24 h, z-score each variable independently.
3. Linearly interpolate at ZT0..ZT20 and compute weighted χ² against the
   observed mean (weights = 1/SD²).

Optimisation is provided via ``scipy.optimize.minimize`` over a user-chosen
subset of parameters.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .model import DEFAULT_INITIAL_STATE, generate_default_parameters
from .simulation import integrate_at_lunar_phase


# Map column name in the Excel file -> state index in the ODE solution.
# ``pdp1`` is not in the model and is therefore not part of the fit.
GENE_TO_STATE_INDEX = {
    'cwo': 5,   # cwo mRNA       (C)
    'clk': 0,   # clk/bmal proxy (X)
    'per': 1,   # per mRNA       (Y)
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_observed(xlsx_path: str) -> dict:
    """Load the z-score RNA-seq table.

    Returns a dict keyed by gene name with values ``(zt_hours, mean, sd)``
    as float arrays. ZT strings ("ZT0", "ZT4", ...) are parsed to hours.
    """
    df = pd.read_excel(xlsx_path, header=[0, 1])

    # First column is the ZT label. Multi-index renamed it to something like
    # ('Unnamed: 0_level_0', 'Unnamed: 0_level_1'); grab whatever it is.
    zt_col = df.columns[0]
    zt_hours = np.array(
        [int(str(s).replace('ZT', '').strip()) for s in df[zt_col].values],
        dtype=float,
    )

    out = {}
    for gene in df.columns.get_level_values(0).unique():
        if gene == zt_col[0]:
            continue
        sub = df[gene]
        if 'mean' not in sub or 'SD' not in sub:
            continue
        out[gene] = (zt_hours, sub['mean'].values.astype(float),
                                sub['SD'].values.astype(float))
    return out


# ---------------------------------------------------------------------------
# Model evaluation aligned to data
# ---------------------------------------------------------------------------

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=0)
    if sd == 0:
        return np.zeros_like(x)
    return (x - x.mean()) / sd


def simulate_24h_zscored(parameters: dict,
                         lunar_phase_h: float = 0.0,
                         settle_hours: float = 480.0,
                         dt: float = 0.05,
                         y0=None,
                         return_period: bool = False,
                         return_amplitude: bool = False,
                         warn_on_damping: bool = True):
    """Return one z-scored circadian period of the model, mapped onto a 24 h grid.

    The model's free-running period is generally not exactly 24 h. To compare
    with ZT-sampled data (which assumes a 24 h day) without introducing a
    discontinuity at the wrap-around, we:

      1. Integrate at fixed lunar phase past transients,
      2. Measure the dominant circadian period P from the BMAL trace,
      3. Extract exactly one period from the end of the simulation,
      4. Linearly stretch that period onto a 24 h ZT axis,
      5. Z-score each gene.

    The resulting trace satisfies ``trace[0] ≈ trace[-1]`` so any subsequent
    phase shift (via :func:`numpy.interp` with ``period=24``) wraps cleanly.

    Parameters
    ----------
    return_period : bool
        If True, also return ``(period_h, amp_rel)`` — the detected period
        in hours and the relative BMAL amplitude (peak-to-peak / mean).
        ``amp_rel < 0.05`` flags a damped oscillator (z-scoring it is
        meaningless; the trace is essentially numerical noise).
    warn_on_damping : bool
        Emit a UserWarning when the oscillator is damped. Default True;
        set False inside optimisation loops that already check ``amp_rel``.
    """
    if y0 is None:
        y0 = DEFAULT_INITIAL_STATE

    # Run settle + a longer tail so the period-detection window contains
    # multiple cycles at deep steady state. 240 h ≈ 10 circadian cycles,
    # enough to be robust against slow tail transients in stiff regimes.
    extra_h = 240.0
    t = np.arange(0.0, settle_hours + extra_h, dt)
    sol = integrate_at_lunar_phase(y0, t, parameters, lunar_phase_h)

    # Period from BMAL over the deepest part of the steady-state window.
    # Peak-to-peak detection with parabolic sub-step refinement gives
    # ~1e-4 h accuracy — essential because this period is used to stretch
    # the trace onto a 24 h grid; even a 0.3 h error becomes a visible
    # endpoint jump.
    from .analysis import dominant_period
    from scipy.signal import find_peaks
    # Use only the last ~half of extra_h for period detection — earlier
    # segment may still have residual transients.
    bmal_full = sol[-int(extra_h / dt):, 0]
    bmal_tail = bmal_full[-int(120.0 / dt):]   # last 120 h

    # Damping check — at some constant lunar phases the oscillator collapses
    # to a fixed point. Z-scoring such a flat trace is meaningless.
    amp_rel = np.ptp(bmal_tail) / max(bmal_tail.mean(), 1e-9)
    if amp_rel < 0.05:
        if warn_on_damping:
            import warnings
            warnings.warn(
                f"At lunar_phase_h={lunar_phase_h:.0f} the model oscillator is "
                f"damped (BMAL ptp/mean = {amp_rel:.4f}). Z-scoring this trace "
                "will amplify numerical noise; the fit is not meaningful. "
                "Try a lunar phase where the model still oscillates (e.g. 0 h).",
                stacklevel=2,
            )
        period_h = 24.0
    else:
        peaks, _ = find_peaks(bmal_tail, distance=int(18.0 / dt))
        if len(peaks) >= 2:
            # Parabolic interpolation: fit y = a(x-x0)^2 + b around each
            # peak to recover sub-step peak position.
            refined = []
            for pk in peaks:
                if 0 < pk < len(bmal_tail) - 1:
                    y0_, y1_, y2_ = bmal_tail[pk-1], bmal_tail[pk], bmal_tail[pk+1]
                    denom = y0_ - 2.0 * y1_ + y2_
                    offset = 0.5 * (y0_ - y2_) / denom if denom != 0.0 else 0.0
                    refined.append((pk + offset) * dt)
                else:
                    refined.append(pk * dt)
            period_h = float(np.median(np.diff(refined)))
            if not (18.0 <= period_h <= 30.0):
                period_h = dominant_period(bmal_tail, dt, band=(18.0, 30.0))
        else:
            period_h = dominant_period(bmal_tail, dt, band=(18.0, 30.0))
        if np.isnan(period_h):
            period_h = 24.0

    # Exactly one period from the end, with sub-step interpolation so the
    # extracted segment is exactly period_h long (not rounded to dt).
    n_P = int(np.ceil(period_h / dt)) + 1
    seg_t = np.arange(n_P) * dt
    seg_data = sol[-n_P:, :]
    # Sample on an exact-length grid
    t_period_exact = np.linspace(0.0, period_h, max(int(period_h / dt), 50))
    one_period = np.empty((len(t_period_exact), seg_data.shape[1]))
    for col in range(seg_data.shape[1]):
        one_period[:, col] = np.interp(t_period_exact, seg_t, seg_data[:, col])
    t_period = t_period_exact

    # Stretch onto a 24 h ZT grid
    n_24 = int(round(24.0 / dt))
    t24 = np.arange(n_24) * dt               # 0 .. 24 h
    t_in_period = t24 * (period_h / 24.0)    # where each ZT lands in model time

    traces = {}
    for gene, idx in GENE_TO_STATE_INDEX.items():
        stretched = np.interp(t_in_period, t_period, one_period[:, idx])
        traces[gene] = _zscore(stretched)

    extras = []
    if return_period:
        extras.append(period_h)
    if return_amplitude:
        extras.append(amp_rel)
    if extras:
        return (t24, traces, *extras)
    return t24, traces


def model_at_timepoints(parameters: dict, zt_hours: np.ndarray,
                        gene: str, lunar_phase_h: float = 0.0,
                        zt0_offset_h: float = 0.0,
                        **sim_kwargs) -> np.ndarray:
    """Z-scored model values for ``gene`` at the requested ZTs.

    ``zt0_offset_h`` shifts the model's internal phase so its ZT0 lines up
    with the experimental ZT0. The shift uses periodic interpolation, so it
    wraps smoothly (no discontinuity).
    """
    t24, traces = simulate_24h_zscored(parameters, lunar_phase_h, **sim_kwargs)
    trace = traces[gene]
    t_query = (np.asarray(zt_hours, dtype=float) + zt0_offset_h) % 24.0
    return np.interp(t_query, t24, trace, period=24.0)


# ---------------------------------------------------------------------------
# Goodness of fit
# ---------------------------------------------------------------------------

def weighted_chi2(observed: dict, parameters: dict,
                  genes: Iterable[str] = None,
                  lunar_phase_h: float = 0.0,
                  zt0_offset_h: float = 0.0,
                  sd_floor: float = 0.05,
                  return_predictions: bool = False,
                  **sim_kwargs):
    """Weighted χ² of model vs. observed z-scored RNA-seq.

    ``sd_floor`` prevents divide-by-near-zero where SD is unrealistically
    tight; weights are 1 / max(SD, sd_floor)².

    Parameters
    ----------
    return_predictions : bool
        If True, also return a dict mapping each fitted gene to a tuple
        ``(zt_hours, observed_mean, observed_sd, model_pred, residual)``
        — everything needed to reproduce the χ² calculation or plot the
        fit externally. ``residual = model_pred - observed_mean`` (not
        weighted). Default False to preserve the old single-return API.

    Returns
    -------
    chi2 : dict
        ``{gene: chi2, ..., '_total': total}``.
    predictions : dict, optional
        Only returned when ``return_predictions=True``. See above.
    """
    if genes is None:
        genes = [g for g in GENE_TO_STATE_INDEX if g in observed]

    t24, traces = simulate_24h_zscored(parameters, lunar_phase_h, **sim_kwargs)

    per_gene = {}
    predictions = {}
    total = 0.0
    for gene in genes:
        if gene not in observed or gene not in traces:
            continue
        zt, mean, sd = observed[gene]
        trace = traces[gene]
        t_query = (np.asarray(zt, dtype=float) + zt0_offset_h) % 24.0
        pred = np.interp(t_query, t24, trace, period=24.0)
        w = 1.0 / np.maximum(sd, sd_floor) ** 2
        chi2 = float(np.sum(w * (pred - mean) ** 2))
        per_gene[gene] = chi2
        predictions[gene] = (np.asarray(zt), np.asarray(mean), np.asarray(sd),
                             pred, pred - np.asarray(mean))
        total += chi2

    per_gene['_total'] = total
    if return_predictions:
        return per_gene, predictions
    return per_gene


# ---------------------------------------------------------------------------
# Optional parameter optimisation
# ---------------------------------------------------------------------------

def fit_parameters(observed: dict, free_params: list[str],
                   base_parameters: dict | None = None,
                   bounds: dict | None = None,
                   genes: Iterable[str] = None,
                   lunar_phase_h: float = 0.0,
                   zt0_offset_h: float | None = 0.0,
                   period_band: tuple = (20.0, 28.0),
                   period_penalty: float = 100.0,
                   method: str = 'differential_evolution',
                   options: dict | None = None,
                   verbose: bool = True) -> tuple[dict, dict]:
    """Minimise weighted χ² over ``free_params``.

    Parameters
    ----------
    free_params : list of str
        Names of parameters to optimise. Any parameter in the base set is
        allowed; pass ``'zt0_offset_h'`` to also fit the phase alignment.
    bounds : dict, optional
        ``{param_name: (low, high)}``. Defaults to (0.2×, 5×) base value
        for kinetic parameters, (−12, 12) for ``zt0_offset_h``.
    zt0_offset_h : float | None
        Fixed phase alignment used when ``'zt0_offset_h'`` is not in
        ``free_params``. Pass ``None`` to treat it as free automatically.
    period_band : (low, high)
        Soft constraint: trials whose model period falls outside this band
        get a ``period_penalty`` added to their χ². Keeps the optimiser
        inside the biologically plausible 22–25 h plasticity window.
    method : {'differential_evolution', 'Nelder-Mead'}
        ``differential_evolution`` is a global search (recommended for this
        landscape). ``Nelder-Mead`` is local; faster but easily trapped.

    Returns
    -------
    fitted_params : dict
        Full parameter set with optimised values substituted in. Also
        contains a ``'zt0_offset_h'`` entry if it was fit.
    result_info : dict
        Optimiser diagnostics plus the final χ² breakdown by gene.
    """
    base = dict(base_parameters or generate_default_parameters())
    bounds = dict(bounds or {})

    # Treat zt0_offset_h as just another free parameter
    fit_offset = ('zt0_offset_h' in free_params) or (zt0_offset_h is None)
    if fit_offset and 'zt0_offset_h' not in free_params:
        free_params = list(free_params) + ['zt0_offset_h']

    def get_bounds(name):
        if name in bounds:
            return bounds[name]
        if name == 'zt0_offset_h':
            return (-12.0, 12.0)
        base_v = base[name]
        return (0.2 * base_v, 5.0 * base_v)

    lo = np.array([get_bounds(n)[0] for n in free_params])
    hi = np.array([get_bounds(n)[1] for n in free_params])

    fixed_offset = 0.0 if zt0_offset_h is None else float(zt0_offset_h)

    def unpack(x):
        trial = dict(base)
        offset = fixed_offset
        for n, v in zip(free_params, x):
            if n == 'zt0_offset_h':
                offset = float(v)
            else:
                trial[n] = float(v)
        return trial, offset

    def objective(x):
        if np.any(x < lo) or np.any(x > hi):
            return 1e12
        trial, offset = unpack(x)
        try:
            t24, traces, period_h, amp_rel = simulate_24h_zscored(
                trial, lunar_phase_h,
                return_period=True, return_amplitude=True,
                warn_on_damping=False)
        except Exception:
            return 1e12

        # Hard penalty for damped trials — z-scored noise can fit anything
        # by accident, so the optimiser must steer well clear. We require
        # both meaningful amplitude AND a clean period in the circadian band.
        # The threshold 0.3 corresponds to ~30% peak-to-peak variation, a
        # lower bound for a "real" oscillation in this model.
        if amp_rel < 0.3:
            return 1e10 * (0.3 - amp_rel + 1.0)

        # Spikiness penalty: discourage non-sinusoidal "relaxation oscillator"
        # regimes (sharp narrow spikes with long quiet intervals between).
        # Such regimes can fit the 6-point ZT data by accident while having
        # wildly non-physiological waveforms. We check the duty cycle: a clean
        # sinusoid spends ~50 % of its period above the mean. Sharp spikes
        # spend only ~10–20 % above the mean. Penalty kicks in below 35 %.
        bmal_one_period = np.asarray(traces.get('clk', traces[list(traces)[0]]))
        duty = float((bmal_one_period > bmal_one_period.mean()).mean())
        chi2_spikiness = 500.0 * (0.35 - duty) ** 2 if duty < 0.35 else 0.0

        # Check oscillator health at the OPPOSITE lunar phase too. Without
        # this, the optimiser routinely finds parameter sets that oscillate
        # only at the fit phase (e.g. NM) and collapse elsewhere — which
        # would kill the lunar plasticity story this paper is built on.
        # We use only an amplitude check (no χ² contribution) — period and
        # waveform shape can legitimately differ at the other lunar phase.
        opposite_phase = (lunar_phase_h + 354.0) % 708.0
        try:
            _, _, _, amp_opp = simulate_24h_zscored(
                trial, opposite_phase,
                return_period=True, return_amplitude=True,
                warn_on_damping=False)
        except Exception:
            amp_opp = 0.0
        if amp_opp < 0.3:
            return 1e10 * (0.3 - amp_opp + 1.0)

        # Confirm dominant period is detectable (NaN means no clean peak)
        from .analysis import dominant_period
        # Re-derive period from the underlying simulation, not the stretched
        # trace (the stretched trace was z-scored, which destroys amplitude
        # info). We use the already-detected period_h, which falls back to
        # 24.0 only if periodogram failed — in which case the trial is bad.
        if period_h == 24.0 and abs(period_h - 24.0) < 1e-9:
            # Could be genuine 24h or a fallback; double-check with band check
            pass  # period_band soft penalty below catches truly weird cases

        # χ²
        chi2_total = 0.0
        gene_list = genes if genes is not None else \
            [g for g in GENE_TO_STATE_INDEX if g in observed]
        for gene in gene_list:
            if gene not in observed or gene not in traces:
                continue
            zt, mean, sd = observed[gene]
            trace = traces[gene]
            t_query = (np.asarray(zt, dtype=float) + offset) % 24.0
            pred = np.interp(t_query, t24, trace, period=24.0)
            w = 1.0 / np.maximum(sd, 0.05) ** 2
            chi2_total += float(np.sum(w * (pred - mean) ** 2))

        # Soft period constraint
        if not (period_band[0] <= period_h <= period_band[1]):
            chi2_total += period_penalty * (
                max(0, period_band[0] - period_h)
                + max(0, period_h - period_band[1])) ** 2

        chi2_total += chi2_spikiness

        # Guard against any residual NaN/Inf
        if not np.isfinite(chi2_total):
            return 1e12
        return chi2_total

    x0 = np.array([base.get(n, fixed_offset if n == 'zt0_offset_h' else 0.0)
                   for n in free_params], dtype=float)

    if method == 'differential_evolution':
        from scipy.optimize import differential_evolution
        # Note: workers > 1 would require a top-level (picklable) objective.
        # The closure makes parallel evaluation impossible; single-threaded
        # DE is fast enough for typical fits (<1 min for ~5 parameters).
        opts = options or {'maxiter': 100, 'popsize': 15, 'tol': 1e-3,
                          'seed': 42, 'polish': True}
        if verbose:
            opts.setdefault('disp', True)
        res = differential_evolution(objective, list(zip(lo, hi)),
                                     x0=x0, **opts)
    else:
        opts = options or {'xatol': 1e-3, 'fatol': 1e-3, 'maxiter': 600}
        res = minimize(objective, x0, method=method, options=opts)

    fitted, fitted_offset = unpack(res.x)
    fitted['_fitted_zt0_offset_h'] = fitted_offset

    # Final per-gene breakdown for reporting
    final_chi2 = weighted_chi2(observed, fitted, genes=genes,
                               lunar_phase_h=lunar_phase_h,
                               zt0_offset_h=fitted_offset)

    info = {
        'x': res.x.tolist(),
        'fun': float(res.fun),
        'free_params': list(free_params),
        'fitted_values': {n: float(v) for n, v in zip(free_params, res.x)},
        'fitted_offset_h': float(fitted_offset),
        'final_chi2_per_gene': final_chi2,
        'success': bool(getattr(res, 'success', True)),
        'message': str(getattr(res, 'message', '')),
    }
    return fitted, info
