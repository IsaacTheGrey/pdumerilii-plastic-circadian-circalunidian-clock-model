"""Figure helpers.

Every plotting function accepts an optional ``save_path``; if given, the
figure is written there (publication-ready) instead of being shown.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _finish(fig, save_path: str | None):
    fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


# ---------------------------------------------------------------------------
# Time series / phase plot / periodogram
# ---------------------------------------------------------------------------

def plot_main_results(t_asymp, solution, periods, power, dominant_period,
                      save_path: str | None = None):
    """Three-panel summary: trajectories, phase plot, periodogram."""
    fig, axs = plt.subplots(1, 3, figsize=(22, 7))

    axs[0].plot(t_asymp, solution[:, 0], label='BMAL (X)')
    axs[0].plot(t_asymp, solution[:, 1], label='per (Y)')
    axs[0].plot(t_asymp, solution[:, 5], label='cwo mRNA (C)', color='orange')
    axs[0].plot(t_asymp, solution[:, 6], label='CWO protein (W)', color='red')
    axs[0].legend(fontsize=11, loc='upper right')
    axs[0].set_xlabel('Time [h]', fontsize=13)
    axs[0].set_ylabel('Concentration [a.u.]', fontsize=13)
    axs[0].set_xticks(np.arange(0, t_asymp[-1] + 1, 24))
    axs[0].tick_params(axis='both', which='major', labelsize=11)
    axs[0].set_title('Trajectories', fontsize=13)

    axs[1].plot(solution[:, 0], solution[:, 2], color='grey', lw=0.8)
    axs[1].set_xlabel('CLK/BMAL [a.u.]', fontsize=13)
    axs[1].set_ylabel('PER/tr-CRY [a.u.]', fontsize=13)
    axs[1].tick_params(axis='both', which='major', labelsize=11)
    axs[1].set_title('Phase plot', fontsize=13)

    axs[2].plot(periods, power)
    axs[2].set_xlim(0, 36)
    axs[2].set_xlabel('Period [h]', fontsize=13)
    axs[2].set_ylabel('Power', fontsize=13)
    axs[2].set_xticks(np.arange(0, 37, 4))
    axs[2].tick_params(axis='both', which='major', labelsize=11)
    axs[2].set_title(f'Periodogram — dominant {dominant_period:.2f} h',
                     fontsize=13)

    _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def plot_sensitivity_results(sensitivity_results: dict,
                             save_path: str | None = None):
    """Two-up grid of period-vs-parameter curves."""
    params = list(sensitivity_results.keys())
    n = len(params)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows),
                            squeeze=False)
    for i, name in enumerate(params):
        ax = axs[i // ncols, i % ncols]
        values, periods = sensitivity_results[name]
        ax.plot(values, periods, marker='o', color='black',
                label='Model period')
        ax.axhline(24, color='red', linestyle='--', alpha=0.6, label='24 h')
        ax.axhspan(22, 25, color='green', alpha=0.10,
                   label='Target band (22–25 h)')
        ax.set_xlabel(f'Value of {name}', fontsize=12)
        ax.set_ylabel('Dominant period [h]', fontsize=12)
        ax.set_title(f'Sensitivity: {name}', fontsize=13)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=9)

    # Hide leftover empty subplots
    for j in range(n, nrows * ncols):
        axs[j // ncols, j % ncols].axis('off')

    _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Plasticity & lunar oscillation
# ---------------------------------------------------------------------------

def plot_plasticity_range(plasticity_results: dict,
                          save_path: str | None = None):
    """CWO-driven period plasticity overlay."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, (values, periods) in plasticity_results.items():
        ax.plot(values, periods, label=f'Effect of {name}',
                marker='s', markersize=4)
    ax.axhspan(22, 25, color='green', alpha=0.10,
               label='Target plasticity (22–25 h)')
    ax.axhline(24, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Parameter value [a.u.]', fontsize=12)
    ax.set_ylabel('Period [h]', fontsize=12)
    ax.set_title('CWO-driven period plasticity in P. dumerilii', fontsize=13)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    _finish(fig, save_path)


def plot_lunar_period_oscillation(lunar_times, periods,
                                  save_path: str | None = None):
    """Circadian period vs. lunar-month time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    valid = ~np.isnan(periods)
    ax.plot(lunar_times[valid], periods[valid],
            color='blue', lw=2.5, label='Measured period')
    ax.axhspan(22, 25, color='green', alpha=0.10,
               label='Target plasticity (22–25 h)')
    ax.axhline(24, color='red', ls='--', alpha=0.5)

    if valid.any():
        y_top = np.nanmax(periods[valid])
        ax.text(0,   y_top, 'Full moon (trough L)', ha='center', color='gray')
        ax.text(354, y_top, 'New moon (peak L)',    ha='center', color='gray')

    ax.set_xlabel('Time within lunar month [h]', fontsize=12)
    ax.set_ylabel('Circadian period [h]', fontsize=12)
    ax.set_title('Circadian-period plasticity across the lunar month',
                 fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Model vs. RNA-seq data
# ---------------------------------------------------------------------------

def plot_model_vs_data(observed: dict, parameters: dict,
                       lunar_phase_h: float = 0.0,
                       zt0_offset_h: float = 0.0,
                       genes=None,
                       save_path: str | None = None,
                       sim_kwargs: dict | None = None):
    """Overlay model and observed z-scored expression at the experimental ZTs.

    The model is evaluated *only* at the same ZT timepoints as the data, so
    each panel shows two sets of markers, one per condition. This matches
    what the weighted χ² actually measures and avoids visually hiding
    detail between data points behind a smooth curve.

    The model trace is computed by stretching exactly one circadian period
    of the simulation onto a 24 h ZT axis (see :func:`simulate_24h_zscored`),
    then sampled at the observed ZTs with periodic interpolation.
    """
    from .fitting import simulate_24h_zscored, GENE_TO_STATE_INDEX

    if genes is None:
        genes = [g for g in GENE_TO_STATE_INDEX if g in observed]

    sim_kwargs = sim_kwargs or {}
    t24, traces, period_h = simulate_24h_zscored(parameters, lunar_phase_h,
                                                 return_period=True,
                                                 **sim_kwargs)

    fig, axs = plt.subplots(1, len(genes), figsize=(5 * len(genes), 4.5),
                            squeeze=False)
    for ax, gene in zip(axs[0], genes):
        zt, mean, sd = observed[gene]
        trace = traces[gene]

        # Sample the model at the experimental ZTs (with optional phase shift)
        t_query = (np.asarray(zt, dtype=float) + zt0_offset_h) % 24.0
        model_at_zt = np.interp(t_query, t24, trace, period=24.0)

        ax.errorbar(zt, mean, yerr=sd, fmt='o', color='black',
                    capsize=3, markersize=7, label='Observed mean ± SD',
                    zorder=2)
        ax.plot(zt, model_at_zt, 's', color='C0', markersize=9,
                markerfacecolor='C0', markeredgecolor='white',
                markeredgewidth=1.5, label='Model (z)', zorder=3)

        ax.set_title(gene, fontsize=13)
        ax.set_xlabel('ZT [h]', fontsize=12)
        ax.set_ylabel('z-score', fontsize=12)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=9)

    fig.suptitle(f'Model vs. observed  —  lunar phase {lunar_phase_h:.0f} h, '
                 f'model period {period_h:.2f} h', fontsize=13)
    _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Circadian oscillations across one lunar month
# ---------------------------------------------------------------------------

def plot_genes_over_lunar_month(parameters: dict,
                                genes=None,
                                n_lunar_cycles: int = 1,
                                dt: float = 0.1,
                                y0=None,
                                zscore: bool = False,
                                show_lunar_drive: bool = True,
                                save_path: str | None = None):
    """Continuous time series of chosen variables across one or more lunar months.

    The lunar drive is active (not frozen). The first lunar cycle is integrated
    silently as a transient burn-in; the next ``n_lunar_cycles`` are plotted.
    Daily oscillations appear as the high-frequency component; lunar modulation
    appears as the slow envelope.

    Parameters
    ----------
    genes : dict | list | None
        Either a dict ``{label: state_index}`` for full control, or a list of
        names from :data:`fitting.GENE_TO_STATE_INDEX` (``'cwo'``, ``'clk'``,
        ``'per'``), or ``None`` for a sensible default panel of four traces.
    n_lunar_cycles : int
        How many lunar cycles to plot (after the burn-in cycle).
    zscore : bool
        Z-score each trace independently. Useful when overlaying traces with
        very different magnitudes (e.g. mRNA vs. high-amplitude CWO protein).
    show_lunar_drive : bool
        Overlay the L_t drive on a secondary axis of each panel.
    """
    from .fitting import GENE_TO_STATE_INDEX, _zscore
    from .model import T_LUNAR
    from .simulation import integrate_model

    if genes is None:
        genes = {
            'cwo mRNA (C)':    5,
            'CWO protein (W)': 6,
            'per mRNA (Y)':    1,
            'CLK/BMAL (X)':    0,
        }
    elif isinstance(genes, (list, tuple)):
        genes = {g: GENE_TO_STATE_INDEX[g] for g in genes}

    if y0 is None:
        y0 = [1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # Two burn-in lunar cycles, then n_lunar_cycles for display. The active
    # lunar drive prevents the system from ever reaching a fixed steady
    # state, so a single burn-in cycle can leave visible residual
    # transients in the slow variables.
    n_burn_cycles = 2
    t_total = T_LUNAR * (n_burn_cycles + n_lunar_cycles)
    t = np.arange(0.0, t_total, dt)
    print(f'Integrating for {t_total:.0f} h '
          f'({n_burn_cycles} burn-in + {n_lunar_cycles} display lunar cycle(s))...')
    sol = integrate_model(y0, t, parameters)

    n_burn = int(T_LUNAR * n_burn_cycles / dt)
    t_plot = t[n_burn:] - t[n_burn]      # restart at 0 for plot readability
    sol_plot = sol[n_burn:, :]
    from .model import _lunar_drive
    L_t = np.array([_lunar_drive(ti, {}) for ti in t_plot])

    n_panels = len(genes)
    fig, axs = plt.subplots(n_panels, 1, figsize=(13, 2.4 * n_panels),
                            sharex=True, squeeze=False)

    for ax, (label, idx) in zip(axs[:, 0], genes.items()):
        y = sol_plot[:, idx]
        if zscore:
            y = _zscore(y)
        ax.plot(t_plot, y, color='C0', lw=0.6)
        ax.set_ylabel(label + ('\n(z)' if zscore else ''), fontsize=10)
        ax.grid(True, alpha=0.3)

        if show_lunar_drive:
            ax2 = ax.twinx()
            ax2.plot(t_plot, L_t, color='crimson', alpha=0.45, lw=1.2, ls='--')
            ax2.set_ylim(0.4, 1.1)
            ax2.tick_params(axis='y', colors='crimson', labelsize=8)
            ax2.set_ylabel(r'$L_t$', color='crimson', fontsize=9, rotation=0,
                           labelpad=10, va='center')

    # Lunar-phase markers (FM at every k * T_LUNAR, NM at +T_LUNAR/2)
    for ax in axs[:, 0]:
        for k in range(n_lunar_cycles + 1):
            x_fm = k * T_LUNAR
            x_nm = (k + 0.5) * T_LUNAR
            if x_fm <= t_plot[-1]:
                ax.axvline(x_fm, color='gray', ls=':', lw=0.8, alpha=0.6)
            if x_nm <= t_plot[-1]:
                ax.axvline(x_nm, color='gray', ls=':', lw=0.8, alpha=0.6)

    # Label FM / NM at the top of the first panel
    top_ax = axs[0, 0]
    ylim = top_ax.get_ylim()
    yt = ylim[1] - 0.04 * (ylim[1] - ylim[0])
    for k in range(n_lunar_cycles + 1):
        x_fm = k * T_LUNAR
        x_nm = (k + 0.5) * T_LUNAR
        if x_fm <= t_plot[-1]:
            top_ax.text(x_fm, yt, 'FM', ha='center', va='top',
                        fontsize=8, color='gray')
        if x_nm <= t_plot[-1]:
            top_ax.text(x_nm, yt, 'NM', ha='center', va='top',
                        fontsize=8, color='gray')

    axs[-1, 0].set_xlabel('Time within displayed lunar window [h]', fontsize=11)
    axs[-1, 0].set_xticks(np.arange(0, t_plot[-1] + 1, T_LUNAR / 4))
    fig.suptitle('Circadian oscillations across the lunar month', fontsize=13)
    _finish(fig, save_path)