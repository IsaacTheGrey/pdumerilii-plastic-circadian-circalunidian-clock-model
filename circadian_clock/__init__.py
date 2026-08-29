"""
Goodwin-type model of the plastic circadian/circalunidian clock of
Platynereis dumerilii, with circalunar modulation of clockwork orange (CWO).

Modules
-------
model        : ODE definitions and default parameter set
simulation   : Thin wrapper around scipy.odeint
analysis     : Period detection, sensitivity sweeps, lunar plasticity scan
fitting      : RNA-seq loading and fixed-model comparison utilities
plotting     : Figure helpers
"""

from .model import (DEFAULT_INITIAL_STATE, T_LUNAR,
                    generate_default_parameters, goodwin_model_lunar)
from .simulation import integrate_model

__all__ = [
    "generate_default_parameters",
    "goodwin_model_lunar",
    "integrate_model",
    "T_LUNAR",
    "DEFAULT_INITIAL_STATE",
]
