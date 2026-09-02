"""
ODE definition for a Goodwin-type oscillator with circalunar drive on
clockwork orange (CWO).

State variables
---------------
    X   CLK/BMAL protein         (transcription activator)
    Y   per mRNA
    Z   PER/tr-CRY protein       (negative feedback on X)
    R   rev-erb mRNA
    S   REV-ERB protein          (additional negative feedback)
    C   cwo mRNA
    W   CWO protein              (binds E-boxes; primary lunar target)

Lunar modulation
----------------
A single drive with period T_LUNAR = 708 h (~29.5 d):

    L_t(t) = 0.75 - 0.25 · cos(2π (t - Δ_CWO) / T_LUNAR)

At the default Δ_CWO = 0, L_t troughs (0.5) at t = 0 (FM) and peaks (1.0)
at t = T_LUNAR/2 = 354 h (NM). Positive Δ_CWO delays the CWO transcriptional
response relative to the lunar calendar; negative values advance it. Because
L_t multiplies CWO synthesis, the unshifted model makes CWO oscillate with
higher mean / larger amplitude at NM and lower / smaller at FM, matching the
biological expectation in *Platynereis dumerilii*.
"""

from __future__ import annotations

import numpy as np

# Lunar month length in hours.
T_LUNAR: float = 708.0
CWO_LUNAR_DELAY_KEY: str = "cwo_lunar_delay_h"

# State order shared by simulations, fitting, plotting, and tests.
DEFAULT_INITIAL_STATE: tuple[float, ...] = (
    1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
)

# Sentinel parameter key used by simulation.integrate_at_lunar_phase to
# freeze the lunar drive at a chosen phase. If present in the parameter
# dict, its value replaces the time-varying L_t.
_LT_OVERRIDE_KEY: str = "_L_t_override"


def _lunar_drive(t: float, parameters: dict) -> float:
    """Time-shifted lunar drive, or a frozen value if overridden.

    Positive ``cwo_lunar_delay_h`` delays the waveform; negative values
    advance it. Parameter dictionaries from older versions remain valid.
    """
    if _LT_OVERRIDE_KEY in parameters:
        return parameters[_LT_OVERRIDE_KEY]
    delay_h = float(parameters.get(CWO_LUNAR_DELAY_KEY, 0.0))
    shifted_time = t - delay_h
    return 0.75 - 0.25 * np.cos(2.0 * np.pi * shifted_time / T_LUNAR)


def goodwin_model_lunar(y, t, parameters):
    """Right-hand side for scipy.integrate.odeint.

    Parameters
    ----------
    y : sequence of 7 floats
        Current state [X, Y, Z, R, S, C, W].
    t : float
        Current time in hours.
    parameters : dict
        Parameter dictionary; see :func:`generate_default_parameters`.
    """
    X, Y, Z, R, S, C, W = y
    p = parameters

    L_t = _lunar_drive(t, p)

    # LSODA can briefly test a slightly negative concentration while taking
    # an otherwise valid step. Fractional Hill exponents are undefined there,
    # so rate-law concentrations are clipped at the physical boundary.
    Z_rate = max(float(Z), 0.0)
    S_rate = max(float(S), 0.0)
    W_rate = max(float(W), 0.0)

    # Hill repressions. K_W is a dedicated half-repression constant for the
    # CWO arm so the inhibition strength can be tuned independently of
    # K1 (which sets the PER and REV-ERB inhibitions).
    inhib_Z = p['K1']**p['hill'] / (p['K1']**p['hill'] + Z_rate**p['hill'])
    inhib_W = (p['K_W']**p['hill_W'] /
               (p['K_W']**p['hill_W'] + W_rate**p['hill_W']))
    inhib_S = (p['K1']**p['hill_S'] /
               (p['K1']**p['hill_S'] + S_rate**p['hill_S']))

    # Positive feedback driving X (CLK/BMAL).
    PFL = p['b'] + p['c'] * X + p['d'] * W

    # Core clock
    dXdt = (p['nu1'] * inhib_Z * inhib_W * inhib_S * PFL
            - p['nu2'] * X / (p['K2'] + X))
    dYdt = (p['nu3'] * X * p['K3']**p['hill_W'] /
            (p['K3']**p['hill_W'] + W_rate**p['hill_W'])
            - p['nu4'] * Y / (p['K4'] + Y))
    dZdt =  p['nu5'] * Y - p['nu6'] * Z / (p['K6'] + Z)
    dRdt =  p['nu7'] * X - p['nu8'] * R / (p['K7'] + R)
    dSdt =  p['nu9'] * R - p['nu10'] * S / (p['K8'] + S)

    # CWO arm — synthesis is lunar-modulated through L_t
    dCdt = (p['nu11'] * L_t * X * p['K5']**p['hill_W'] /
            (p['K5']**p['hill_W'] + W_rate**p['hill_W'])
            - p['nu12'] * C / (p['K9'] + C))
    # Linear protein degradation (see module docstring)
    dWdt = p['nu13'] * C - p['nu14'] * W

    return [dXdt, dYdt, dZdt, dRdt, dSdt, dCdt, dWdt]


def generate_default_parameters() -> dict:
    """Return the default parameter set.

    The CWO arm is identical to the original kinetics *except* that the
    protein degradation is now linear (``-nu14·W``). This prevents the
    runaway accumulation caused by saturating Michaelis-Menten while
    keeping the protein magnitude (~4·<C>) and the positive-feedback
    contribution ``d·W`` similar to the original model, so the oscillator
    behaves as it did before.
    """
    return {
        # Core clock kinetics
        'nu1': 0.7,  'nu2': 0.5,
        'nu3': 0.45, 'nu4': 0.3,
        'nu5': 0.7,  'nu6': 0.35,
        'nu7': 0.3,  'nu8': 0.2,
        'nu9': 0.1,  'nu10': 0.2,

        # CWO arm. Linear protein degradation (see goodwin_model_lunar).
        # nu13/nu14 = 4 -> W settles around 4·<C> ≈ 1.5–2 a.u., similar
        # role in PFL and inhibition as the original parameter set.
        'nu11': 0.2,   # cwo mRNA synthesis
        'nu12': 0.05,  # cwo mRNA degradation V_max
        'nu13': 0.8,   # CWO translation
        'nu14': 0.2,   # CWO linear degradation (h^-1); half-life ~3.5 h
        # Positive values delay the lunar modulation of cwo transcription;
        # negative values advance it. This is a phase parameter, not a rate.
        CWO_LUNAR_DELAY_KEY: 0.0,

        # Michaelis / Hill constants. K_W is a dedicated half-repression
        # constant for the CWO arm; hill_W = 1 (non-cooperative) keeps the
        # oscillator alive at the peak-CWO end of the lunar cycle. Higher
        # cooperativity (e.g. 2.5) causes the negative feedback to be too
        # switch-like and damps the clock at NM.
        'K1':  1.0, 'K2': 1.0, 'K3': 1.0, 'K4': 1.0,
        # The legacy equation used K5**hill with hill=4 and hill_W=1,
        # making its actual W half-repression point 0.8**4 = 0.4096.  Store
        # that effective half-point directly so the dimensionally consistent
        # hill_W expression above preserves the published baseline dynamics.
        'K5':  0.4096, 'K6': 1.0, 'K7': 1.0, 'K8': 1.0, 'K9': 1.0,
        'K_W': 2.0,
        'hill': 4, 'hill_S': 1.5, 'hill_W': 1.0,

        # Positive feedback loop (modest contribution from W; the
        # X→Y→Z negative feedback carries the oscillator on its own).
        'b': 1.0, 'c': 0.5, 'd': 0.5,
    }
