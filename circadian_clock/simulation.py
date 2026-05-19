"""Numerical integration of the lunar Goodwin model."""

from __future__ import annotations

import numpy as np
from scipy.integrate import odeint

from .model import (T_LUNAR, _LT_OVERRIDE_KEY, _lunar_drive,
                    goodwin_model_lunar)


def integrate_model(y0, t, parameters, mxstep: int = 5000):
    """Integrate the lunar Goodwin model with ``scipy.integrate.odeint``.

    ``mxstep`` is raised above the SciPy default (500) because the lunar
    drive is slow enough that LSODA can otherwise hit the step ceiling on
    multi-thousand-hour runs.
    """
    return odeint(goodwin_model_lunar, y0, t, args=(parameters,),
                  mxstep=mxstep)


def integrate_at_lunar_phase(y0, t, parameters, lunar_phase_h: float,
                             mxstep: int = 5000):
    """Integrate with the lunar drive frozen at a chosen phase.

    Useful for fitting 24 h RNA-seq waveforms: over one day the lunar
    drive moves by ~3 % of its range, so freezing it at the experiment's
    lunar phase is a clean assumption.

    Parameters
    ----------
    lunar_phase_h : float
        Time in hours relative to FM (full moon, t = 0, L_t trough).
        ``lunar_phase_h = 354`` corresponds to NM (L_t peak).
    """
    p = dict(parameters)  # don't mutate the caller's dict
    p[_LT_OVERRIDE_KEY] = _lunar_drive(lunar_phase_h, {})  # evaluate cleanly
    return odeint(goodwin_model_lunar, y0, t, args=(p,), mxstep=mxstep)