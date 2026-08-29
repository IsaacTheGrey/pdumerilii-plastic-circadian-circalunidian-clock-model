"""End-to-end driver script.

Run this from the directory that contains the ``circadian_clock`` package:

    python run_simulation.py

The script performs, in order:
    1. A long simulation to display steady-state trajectories,
       phase plot and periodogram.
    2. A constant-mean-drive sweep of all four C/W kinetics.
    3. A sweep separating W abundance gain from W turnover speed.
    4. A sliding-window scan of circadian period across one lunar month.
    5. A phase-plane view of limit-cycle evolution over the lunar month.
    6. An overlay of z-scored model output on the observed RNA-seq data.
    7. A panel figure of gene oscillations across the lunar month.

CSV and PNG outputs are written into ./figures/ next to this script.
The expensive sweeps can be turned off with the toggles at the top of main().
"""

from __future__ import annotations

import os

import numpy as np

from circadian_clock.analysis import (
    analyze_cwo_phase_sensitivity,
    analyze_lunar_limit_cycles,
    analyze_lunar_period_oscillation,
    analyze_w_gain_turnover,
    compute_periodogram,
    dominant_period,
    normalize_oscillations,
    save_publication_sweep_results,
)
from circadian_clock.fitting import load_observed, weighted_chi2
from circadian_clock.model import (DEFAULT_INITIAL_STATE, T_LUNAR,
                                   generate_default_parameters)
from circadian_clock.plotting import (
    plot_cwo_amount_vs_period,
    plot_cwo_phase_sensitivity,
    plot_genes_over_lunar_month,
    plot_lunar_period_oscillation,
    plot_lunar_limit_cycles,
    plot_main_results,
    plot_model_vs_data,
    plot_w_gain_turnover,
)
from circadian_clock.simulation import integrate_model


# -------- I/O configuration ------------------------------------------------

OUTPUT_DIR = 'figures'
OBSERVED_XLSX = 'observed_gene_exp.xlsx'


OBSERVED_LUNAR_PHASE_H = 0

# Internal-model time at which ZT0 should be drawn. The model has no a
# priori ZT0 — pick the offset that lines up CLK/BMAL peak with biology.
ZT0_OFFSET_H = 20


def main():
    # ---- Toggles for expensive sections ---------------------------------
    RUN_CWO_SWEEPS  = True
    INCLUDE_LUNAR_PHASES_IN_SWEEPS = True
    RUN_LUNAR_SCAN  = True
    RUN_PHASE_PLANE = True
    RUN_RNASEQ_OVERLAY = False
    RUN_LUNAR_GENES = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 1. Steady-state simulation -------------------------------------
    dt = 0.05
    # Two burn-in lunar cycles plus one cycle for display/analysis are enough
    # to remove the slow initial transient without allocating a 100,000 h
    # trajectory in memory.
    t = np.arange(0.0, 3.0 * T_LUNAR, dt)
    # 7 state variables: [X, Y, Z, R, S, C, W]
    y0 = DEFAULT_INITIAL_STATE
    base_parameters = generate_default_parameters()

    print('1) Long simulation for steady-state trajectories...')
    sol = integrate_model(y0, t, base_parameters)

    last_osc = T_LUNAR  # one full lunar month
    n_keep = int(last_osc / dt)
    t_asymp = np.arange(0, last_osc, dt)
    t_abs_kept = t[-n_keep:]
    sol_asymp = normalize_oscillations(sol[-n_keep:, :])

    freqs, power = compute_periodogram(sol_asymp[:, 0], dt)
    periods = 1.0 / freqs
    dom_period = dominant_period(sol[-n_keep:, 0], dt, band=(18.0, 30.0))
    print(f'   Dominant circadian period under lunar forcing: '
          f'{dom_period:.2f} h')

    plot_main_results(t_asymp, sol_asymp, periods, power, dom_period,
                      save_path=os.path.join(OUTPUT_DIR, 'main_results.svg'),
                      t_absolute=t_abs_kept)

    # ---- 2–3. Validated CWO sensitivity and mechanism sweeps ------------
    if RUN_CWO_SWEEPS:
        phase_note = ('including fixed lunar phases'
                      if INCLUDE_LUNAR_PHASES_IN_SWEEPS
                      else 'without lunar phase comparisons')
        print(f'2) C/W kinetic sensitivity {phase_note}...')
        sweep_processes = min(4, os.cpu_count() or 1)
        sens = analyze_cwo_phase_sensitivity(
            base_parameters, y0, processes=sweep_processes,
            include_lunar_phases=INCLUDE_LUNAR_PHASES_IN_SWEEPS)
        save_publication_sweep_results(
            sens, os.path.join(OUTPUT_DIR, 'sensitivity_results.csv'))
        plot_cwo_phase_sensitivity(
            sens, save_path=os.path.join(OUTPUT_DIR, 'sensitivity.png'))

        print('3) W abundance-gain versus turnover sweep...')
        mechanisms = analyze_w_gain_turnover(
            base_parameters, y0, processes=sweep_processes,
            include_lunar_phases=INCLUDE_LUNAR_PHASES_IN_SWEEPS)
        save_publication_sweep_results(
            mechanisms,
            os.path.join(OUTPUT_DIR, 'w_gain_turnover_results.csv'))
        plot_w_gain_turnover(
            mechanisms,
            save_path=os.path.join(OUTPUT_DIR, 'cwo_plasticity.png'))
        plot_cwo_amount_vs_period(
            mechanisms,
            save_path=os.path.join(OUTPUT_DIR,
                                   'cwo_amount_vs_period.png'))

    # ---- 4. Lunar plasticity --------------------------------------------
    if RUN_LUNAR_SCAN:
        print('4) Sliding-window period across the lunar month...')
        lunar_t, lunar_per = analyze_lunar_period_oscillation(
            base_parameters, y0, dt=0.1, n_lunar_cycles=2)
        plot_lunar_period_oscillation(
            lunar_t, lunar_per,
            save_path=os.path.join(OUTPUT_DIR, 'lunar_period.png'))

    # ---- 5. Limit-cycle evolution over the lunar month -----------------
    if RUN_PHASE_PLANE:
        print('5) Phase-plane evolution across the lunar month...')
        cycles = analyze_lunar_limit_cycles(
            base_parameters, y0, x_index=0, y_index=2,
            dt=0.1, n_burn_cycles=2)
        plot_lunar_limit_cycles(
            cycles,
            save_path=os.path.join(OUTPUT_DIR,
                                   'lunar_limit_cycles.png'))

    # ---- 6. RNA-seq overlay ---------------------------------------------
    if RUN_RNASEQ_OVERLAY:
        print('6) Model vs. RNA-seq z-scores...')
        if not os.path.exists(OBSERVED_XLSX):
            print(f'   {OBSERVED_XLSX} not found; skipping.')
        else:
            observed = load_observed(OBSERVED_XLSX)
            chi2, preds = weighted_chi2(observed, base_parameters,
                                        lunar_phase_h=OBSERVED_LUNAR_PHASE_H,
                                        zt0_offset_h=ZT0_OFFSET_H,
                                        return_predictions=True)
            print('   Weighted chi-squared:')
            for g, v in chi2.items():
                print(f'     {g:8s} {v:.2f}')
            print('   Per-ZT predictions (z-score):')
            for gene, (zt, mean, sd, pred, resid) in preds.items():
                print(f'     {gene}:')
                print(f'       ZT       :' + ''.join(f'{int(z):>8d}' for z in zt))
                print(f'       observed :' + ''.join(f'{v:+8.3f}' for v in mean))
                print(f'       SD       :' + ''.join(f'{v:8.3f}' for v in sd))
                print(f'       model    :' + ''.join(f'{v:+8.3f}' for v in pred))
                print(f'       residual :' + ''.join(f'{v:+8.3f}' for v in resid))
            plot_model_vs_data(observed, base_parameters,
                               lunar_phase_h=OBSERVED_LUNAR_PHASE_H,
                               zt0_offset_h=ZT0_OFFSET_H,
                               save_path=os.path.join(OUTPUT_DIR,
                                                      'model_vs_data.png'))

    # ---- 7. Gene oscillations across the lunar month --------------------
    if RUN_LUNAR_GENES:
        print('7) Plotting gene oscillations across the lunar month...')
        plot_genes_over_lunar_month(
            base_parameters,
            genes={'cwo mRNA (C)': 5, 'CWO protein (W)': 6,
                   'per mRNA (Y)': 1, 'CLK/BMAL (X)': 0},
            n_lunar_cycles=1,
            zscore=True,
            save_path=os.path.join(OUTPUT_DIR, 'genes_over_lunar_month.png'),
        )

if __name__ == '__main__':
    main()
