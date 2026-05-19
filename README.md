# Plastic circadian/circalunidian clock model of *Platynereis dumerilii*

A Goodwin-type oscillator with circalunar modulation of *clockwork orange* (CWO),
implementing the plastic 22–25 h circadian period observed in the marine
annelid *Platynereis dumerilii*. The model accompanies the manuscript
**A circalunar oscillator and moonlight tune the circadian clock via cwo to adjust daily behaviour and physiology to moon phase**.

A single time-dependent drive on cwo synthesis is sufficient
to produce circalunar modulation of the circadian period within a biologically
plausible range, with CWO and PER as the only transcripts oscillating on a
lunar timescale and doing so in anti-phase.

## Model summary

Seven state variables: CLK/BMAL (X), *per* mRNA (Y), PER/tr-CRY (Z),
*rev-erb* mRNA (R), REV-ERB (S), *cwo* mRNA (C), CWO protein (W).
A single lunar drive `L(t) = 0.75 − 0.25·cos(2π t / T_lunar)` with
`T_lunar = 708 h` multiplies the CWO mRNA synthesis term, troughing at full
moon and peaking at new moon. See `circadian_clock/model.py` for the full ODE
system and the manuscript Methods for the symbolic equations.

## Repository layout

```
.
├── circadian_clock/        Python package
│   ├── model.py            ODE definitions and default parameters
│   ├── simulation.py       Wrappers around scipy.integrate.odeint
│   ├── analysis.py         Period detection, sensitivity & plasticity sweeps
│   ├── fitting.py          RNA-seq data fitting and goodness-of-fit
│   └── plotting.py         Figure helpers
├── run_simulation.py       End-to-end driver (produces all figures)
├── observed_gene_exp.xlsx  Observed z-scored RNA-seq expression (cwo, clk, per, pdp1)
├── figures/                Output directory (created on first run)
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Requires Python ≥ 3.10. Clone and install dependencies:

```bash
git clone https://github.com/[FILL IN: user/repo].git
cd [FILL IN: repo]
pip install -r requirements.txt
```

## Reproducing the figures

From the repository root:

```bash
python run_simulation.py
```

This runs, in order:

1. A long simulation showing steady-state trajectories, phase plot, and periodogram.
2. A parameter-sensitivity sweep over the CWO-relevant kinetics.
3. A sliding-window scan of the circadian period across one lunar month.
4. Comparison of z-scored model output against observed RNA-seq data.
5. A panel figure of gene oscillations across the lunar month.

Outputs (PNG figures and one CSV) land in `figures/`. Expensive sections can
be toggled off near the top of `main()` in `run_simulation.py`. The full run
takes 5–15 minutes on a modern laptop, depending on core count (the
sensitivity sweep parallelises automatically).

## Fitting to your own data

The fitting machinery in `circadian_clock/fitting.py` accepts any
`{gene: (zt_hours, mean, sd)}` mapping. To fit a subset of parameters against
the provided RNA-seq data:

```python
from circadian_clock.fitting import fit_parameters, load_observed
from circadian_clock.model import generate_default_parameters

obs = load_observed('observed_gene_exp.xlsx')
p0  = generate_default_parameters()

free = ['nu3', 'nu4', 'nu5', 'nu6', 'zt0_offset_h']
bounds = {'nu3': (0.1, 2.0), 'nu4': (0.05, 1.5),
          'nu5': (0.2, 2.0), 'nu6': (0.1, 1.5),
          'zt0_offset_h': (-12.0, 12.0)}

fitted, info = fit_parameters(
    obs, free, base_parameters=p0,
    bounds=bounds,
    lunar_phase_h=354.0,        # data sampled near NM (peak L_t, peak cwo)
    zt0_offset_h=None,           # fit the offset
    genes=['cwo', 'per'],        # skip noisy channels
    period_band=(22.0, 26.0),
    period_penalty=300.0,
    options={'maxiter': 40, 'popsize': 12, 'seed': 42, 'polish': False})
```

The optimiser uses differential evolution with biology-aware penalties: it
rejects damped trials, suppresses non-sinusoidal "relaxation-oscillator"
regimes via a duty-cycle term, and requires the oscillator to remain alive
at the opposite lunar phase so single-snapshot fitting does not collapse
lunar plasticity. See the docstring of `fit_parameters` for the full
list of knobs.

## Citation

If you use this code, please cite our manuscript nd the underlying Goodwin model framework as appropriate
(Goodwin 1965, *Adv. Enzyme Regul.*).


## Contact
Federico Scaramuzza
ORCID: 0000-0003-4360-3883

