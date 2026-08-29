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
│   ├── fitting.py          RNA-seq loading and model-overlay utilities
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
2. A fixed-phase sensitivity sweep of CWO production and degradation
   kinetics at full moon, mean lunar drive, and new moon.
3. A mechanistic sweep separating CWO protein abundance gain
   (`nu13/nu14`) from protein turnover speed.
4. A sliding-window scan of the circadian period across one lunar month.
5. A CLK/BMAL–PER phase-plane figure showing how the circadian limit cycle
   changes across the lunar month.
6. Comparison of z-scored model output against observed RNA-seq data.
7. A panel figure of gene oscillations across the lunar month.

Outputs (PNG figures and one CSV) land in `figures/`. Expensive sections can
be toggled off near the top of `main()` in `run_simulation.py`. The full run
takes 5–15 minutes on a modern laptop, depending on core count (the
sensitivity sweeps parallelise automatically).

## Period and rhythmicity validation

Sensitivity periods are estimated from refined CLK/BMAL peak times after a
long burn-in at a frozen lunar drive. A period is reported only when the
trajectory has sufficient amplitude, remains sustained across the analysis
window, contains enough peaks, and has regular inter-peak intervals. Failed
simulations remain in the CSV with a `Status` value and a missing period; they
are shown as crosses in the figures rather than being assigned the edge of a
Fourier search band.

All kinetic sensitivity axes are fold changes from the default parameter set.
The W mechanism analysis varies translation alone to change abundance gain,
then scales translation and degradation together to change turnover while
holding the gain constant.

## RNA-seq comparison

The end-to-end script can overlay the fixed default model on the supplied
z-scored RNA-seq observations. Automatic kinetic-parameter optimisation is
not part of the publication workflow; the sparse time-course data do not
constrain the full nonlinear model reliably enough to justify fitted kinetic
constants.

## Citation

If you use this code, please cite our manuscript nd the underlying Goodwin model framework as appropriate
(Goodwin 1965, *Adv. Enzyme Regul.*).


## Contact
Federico Scaramuzza
ORCID: 0000-0003-4360-3883
