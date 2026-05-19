"""
Goodwin-type model of the plastic circadian/circalunidian clock of
Platynereis dumerilii, with circalunar modulation of clockwork orange (CWO).

Modules
-------
model        : ODE definitions and default parameter set
simulation   : Thin wrapper around scipy.odeint
analysis     : Period detection, sensitivity sweeps, lunar plasticity scan
fitting      : Comparison of model output to z-score normalised RNA-seq data
plotting     : Figure helpers
"""

from .model import generate_default_parameters, goodwin_model_lunar, T_LUNAR
from .simulation import integrate_model

__all__ = [
    "generate_default_parameters",
    "goodwin_model_lunar",
    "integrate_model",
    "T_LUNAR",
]