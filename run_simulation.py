"""End-to-end driver script.

Run this from the directory that contains the ``circadian_clock`` package:

    python run_simulation.py

The script performs, in order:
    1. A long simulation to display steady-state trajectories,
       phase plot and periodogram.
    2. A parameter-sensitivity sweep over the CWO-relevant kinetics.
    3. A focused CWO plasticity sweep on nu11 and nu14.
    4. A sliding-window scan of circadian period across one lunar month.
    5. An overlay of z-scored model output on the observed RNA-seq data.
    6. A panel figure of gene oscillations across the lunar month.

CSV and PNG outputs are written into ./figures/ next to this script.
The expensive sweeps can be turned off with the toggles at the top of main().
"""

from __future__ import annotations

import os

import numpy as np

from circadian_clock.analysis import (
    analyze_cwo_plasticity,
    analyze_lunar_period_oscillation,
    compute_periodogram,
    normalize_oscillations,
    parameter_sweep,
    save_sensitivity_results,
)
from circadian_clock.fitting import load_observed, weighted_chi2
from circadian_clock.model import generate_default_parameters
from circadian_clock.plotting import (
    plot_genes_over_lunar_month,
    plot_lunar_period_oscillation,
    plot_main_results,
    plot_model_vs_data,
    plot_plasticity_range,
    plot_sensitivity_results,
)
from circadian_clock.simulation import integrate_model


# -------- I/O configuration ------------------------------------------------

OUTPUT_DIR = 'figures'
OBSERVED_XLSX = 'observed_gene_exp.xlsx'

# Where in the lunar cycle was the RNA-seq sampled?
# 0    = full-moon peak of L_t (highest CWO synthesis)
# 354  = new-moon trough of L_t
# Update this once you know the experimental phase.
OBSERVED_LUNAR_PHASE_H = 354.0

# Internal-model time at which ZT0 should be drawn. The model has no a
# priori ZT0 — pick the offset that lines up CLK/BMAL peak with biology.
ZT0_OFFSET_H = 6.0


def main():
    # ---- Toggles for expensive sections ---------------------------------
    RUN_SENSITIVITY = False
    RUN_PLASTICITY  = False
    RUN_LUNAR_SCAN  = True
    RUN_RNASEQ_FIT  = False
    RUN_LUNAR_GENES = True
    RUN_PARAM_FIT   = False   # slow (~2 min); set True to optimise kinetics

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 1. Steady-state simulation -------------------------------------
    dt = 0.01
    t = np.arange(0.0, 10000.0, dt)
    # 7 state variables: [X, Y, Z, R, S, C, W]
    y0 = [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    base_parameters = generate_default_parameters()

    print('1) Long simulation for steady-state trajectories...')
    sol = integrate_model(y0, t, base_parameters)

    last_osc = 1416  # one full lunar month
    n_keep = int(last_osc / dt)
    t_asymp = np.arange(0, last_osc, dt)
    sol_asymp = normalize_oscillations(sol[-n_keep:, :])

    freqs, power = compute_periodogram(sol_asymp[:, 0], dt)
    periods = 1.0 / freqs
    dom_idx = int(np.argmax(power))
    dom_period = periods[dom_idx]
    print(f'   Dominant period (free-running): {dom_period:.2f} h')

    plot_main_results(t_asymp, sol_asymp, periods, power, dom_period,
                      save_path=os.path.join(OUTPUT_DIR, 'main_results.png'))

    # ---- 2. Parameter sensitivity sweep ---------------------------------
    if RUN_SENSITIVITY:
        print('2) Parameter sensitivity sweep...')
        targeted = ['nu11', 'nu13', 'nu14', 'hill_W', 'K_W', 'd']
        ranges = {}
        for p in targeted:
            if 'hill' in p:
                ranges[p] = np.linspace(1.0, 5.0, 20)
            elif p == 'K_W':
                ranges[p] = np.linspace(0.05, 2.0, 20)
            else:
                ranges[p] = np.linspace(0.2 * base_parameters[p],
                                        4.0 * base_parameters[p], 30)

        t_sweep = np.arange(0.0, 1500.0, dt)
        sens = parameter_sweep(base_parameters, y0, t_sweep, ranges,
                               last_osc=144.0, band=(18.0, 30.0))
        save_sensitivity_results(sens,
                                 os.path.join(OUTPUT_DIR,
                                              'sensitivity_results.csv'))
        plot_sensitivity_results(sens,
                                 save_path=os.path.join(OUTPUT_DIR,
                                                        'sensitivity.png'))

    # ---- 3. CWO plasticity ---------------------------------------------
    if RUN_PLASTICITY:
        print('3) CWO plasticity sweep (nu11, nu14)...')
        t_plast = np.arange(0.0, 1500.0, dt)
        plast = analyze_cwo_plasticity(base_parameters, y0, t_plast)
        plot_plasticity_range(plast,
                              save_path=os.path.join(OUTPUT_DIR,
                                                     'cwo_plasticity.png'))

    # ---- 4. Lunar plasticity --------------------------------------------
    if RUN_LUNAR_SCAN:
        print('4) Sliding-window period across the lunar month...')
        lunar_t, lunar_per = analyze_lunar_period_oscillation(
            base_parameters, y0, dt=0.1, n_lunar_cycles=2)
        plot_lunar_period_oscillation(
            lunar_t, lunar_per,
            save_path=os.path.join(OUTPUT_DIR, 'lunar_period.png'))

    # ---- 5. RNA-seq overlay ---------------------------------------------
    if RUN_RNASEQ_FIT:
        print('5) Model vs. RNA-seq z-scores...')
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

    # ---- 6. Gene oscillations across the lunar month --------------------
    if RUN_LUNAR_GENES:
        print('6) Plotting gene oscillations across the lunar month...')
        plot_genes_over_lunar_month(
            base_parameters,
            genes={'cwo mRNA (C)': 5, 'CWO protein (W)': 6,
                   'per mRNA (Y)': 1, 'CLK/BMAL (X)': 0},
            n_lunar_cycles=1,
            zscore=True,
            save_path=os.path.join(OUTPUT_DIR, 'genes_over_lunar_month.png'),
        )

    # ---- 7. Optional: fit kinetic parameters to RNA-seq data -----------
    # Set RUN_PARAM_FIT = True (toggle at top of main) to run.
    if RUN_PARAM_FIT and os.path.exists(OBSERVED_XLSX):
        import json
        from circadian_clock.fitting import fit_parameters
        print('7) Fitting kinetic parameters against RNA-seq data...')

        # Phase-determining parameters. Different arms of the loop set
        # different relative phases between transcripts; these are the ones
        # that move per/clk peaks without breaking the oscillator.
        free = ['nu4', 'nu6', 'K6', 'nu3', 'nu9', 'nu10', 'zt0_offset_h']
        bounds = {
            'nu4':  (0.1, 1.0),  'nu6':  (0.1, 1.5), 'K6': (0.2, 3.0),
            'nu3':  (0.15, 1.5), 'nu9':  (0.02, 0.5),
            'nu10': (0.05, 0.6),
        }
        observed = load_observed(OBSERVED_XLSX)
        fitted, info = fit_parameters(
            observed, free,
            base_parameters=base_parameters,
            lunar_phase_h=OBSERVED_LUNAR_PHASE_H,
            period_band=(20.0, 27.0),
            bounds=bounds,
            method='differential_evolution',
            options={'maxiter': 60, 'popsize': 15, 'seed': 42,
                     'tol': 1e-4, 'polish': True, 'disp': False},
            verbose=False,
        )
        print(f"   Final chi^2: {info['fun']:.2f}  "
              f"(per-gene: { {k: round(v, 1) for k, v in info['final_chi2_per_gene'].items()} })")
        print(f"   Fitted ZT0 offset: {info['fitted_offset_h']:+.2f} h")
        print('   Fitted values:')
        for n in free:
            if n == 'zt0_offset_h':
                continue
            print(f"     {n}: {base_parameters[n]:.4f} -> "
                  f"{info['fitted_values'][n]:.4f}")

        # Save fitted parameters in JSON for downstream use
        with open(os.path.join(OUTPUT_DIR, 'fitted_parameters.json'), 'w') as f:
            json.dump({
                'base': {k: v for k, v in base_parameters.items()
                         if not isinstance(v, str)},
                'fitted': {k: v for k, v in fitted.items()
                           if not k.startswith('_')},
                'fitted_zt0_offset_h': info['fitted_offset_h'],
                'final_chi2': info['final_chi2_per_gene'],
            }, f, indent=2)

        # Plot the fitted overlay
        fitted_params = {k: v for k, v in fitted.items()
                         if not k.startswith('_')}
        plot_model_vs_data(observed, fitted_params,
                           lunar_phase_h=OBSERVED_LUNAR_PHASE_H,
                           zt0_offset_h=info['fitted_offset_h'],
                           save_path=os.path.join(OUTPUT_DIR,
                                                  'model_vs_data_fitted.png'))


if __name__ == '__main__':
    main()