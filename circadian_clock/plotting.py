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

def _finish(fig, save_path: str | None, tight: bool = True):
    if tight:
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
                      save_path: str | None = None,
                      t_absolute=None, highlight_window_h: float = 24.0):
    """Three-panel summary: trajectories, phase plot, periodogram.

    Parameters
    ----------
    t_absolute : array-like, optional
        Absolute simulation time (hours) for each row of ``solution``. When
        supplied, the phase plot highlights the trajectory segments that fall
        near full moon (lunar phase ≈ 0) and new moon (lunar phase ≈
        T_LUNAR/2) — the two extremes of the lunar drive. Without it, the
        phase plot is drawn in a single colour as before.
    highlight_window_h : float
        Half-width (hours) of the window around FM and NM to highlight.
        Points whose lunar phase is within this distance of 0 or T_LUNAR/2
        are coloured. Default 24 h (~one circadian cycle either side).
    """
    fig, axs = plt.subplots(1, 3, figsize=(22, 7))

    axs[0].plot(t_asymp, solution[:, 0], label='CLK:BMAL (X)')
    axs[0].plot(t_asymp, solution[:, 1], label='per/tr-cry (Y)')
    #axs[0].plot(t_asymp, solution[:, 5], label='cwo mRNA (C)', color='orange')
    axs[0].plot(t_asymp, solution[:, 6], label='CWO protein (W)', color='red')
    axs[0].legend(fontsize=11, loc='upper right')
    axs[0].set_xlabel('Time [h]', fontsize=13)
    axs[0].set_ylabel('Concentration [a.u.]', fontsize=13)
    axs[0].set_xticks(np.arange(0, t_asymp[-1] + 1, 24))
    axs[0].tick_params(axis='both', which='major', labelsize=11)
    axs[0].set_title('Trajectories', fontsize=13)

    # Phase plot — full trajectory in grey as a backdrop
    axs[1].plot(solution[:, 0], solution[:, 2], color='lightgrey', lw=0.8,
                zorder=1)

    if t_absolute is not None:
        from .model import T_LUNAR
        t_absolute = np.asarray(t_absolute, dtype=float)
        lunar_phase = t_absolute % T_LUNAR

        # Distance to FM (phase 0, equivalently T_LUNAR) and to NM (T_LUNAR/2),
        # measured circularly so the wrap-around at 0/T_LUNAR is handled.
        d_fm = np.minimum(lunar_phase, T_LUNAR - lunar_phase)
        d_nm = np.abs(lunar_phase - T_LUNAR / 2.0)

        fm_mask = d_fm <= highlight_window_h
        nm_mask = d_nm <= highlight_window_h

        # Plot highlighted segments as scatter so disjoint runs don't get
        # joined by spurious connecting lines.
        axs[1].scatter(solution[fm_mask, 0], solution[fm_mask, 2],
                       s=6, color='#1f77b4', zorder=3,
                       label='Full moon')
        axs[1].scatter(solution[nm_mask, 0], solution[nm_mask, 2],
                       s=6, color='#d62728', zorder=3,
                       label='New moon')
        axs[1].legend(fontsize=10, loc='best')

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


def _plot_validated_sweep(rows, parameters, title,
                          save_path: str | None = None):
    """Plot fixed-phase sweep rows, marking loss of rhythmicity explicitly."""
    ncols = 2
    nrows = (len(parameters) + ncols - 1) // ncols
    fig, axs = plt.subplots(nrows, ncols, figsize=(13, 4.8 * nrows),
                            squeeze=False, sharex=True, sharey=True)
    phase_styles = {
        'Constant mean': ('#111827', 'o'),
        'Full moon': ('#2563eb', 'o'),
        'Mean drive': ('#374151', 's'),
        'New moon': ('#dc2626', '^'),
    }

    for index, parameter in enumerate(parameters):
        ax = axs[index // ncols, index % ncols]
        subset = [row for row in rows if row['Parameter'] == parameter]
        for phase, (colour, marker) in phase_styles.items():
            phase_rows = sorted(
                (row for row in subset if row['Drive Condition'] == phase),
                key=lambda row: row['Fold Change'])
            if not phase_rows:
                continue
            folds = np.asarray([row['Fold Change'] for row in phase_rows])
            periods = np.asarray([row['Period (h)'] for row in phase_rows],
                                 dtype=float)
            ax.plot(folds, periods, color=colour, marker=marker, ms=5,
                    lw=1.8, label=phase)
            failed = np.isnan(periods)
            if failed.any():
                ax.scatter(folds[failed], np.full(failed.sum(), 18.25),
                           marker='x', s=45, color=colour, linewidth=1.5)

        ax.axvline(1.0, color='0.55', lw=1.0, ls=':')
        ax.axhline(24.0, color='0.35', lw=1.0, ls='--')
        ax.axhspan(22.0, 25.0, color='#16a34a', alpha=0.09)
        ax.set_xscale('log', base=2)
        ax.set_xticks([0.5, 0.7071, 1.0, 1.4142, 2.0])
        ax.set_xticklabels(['0.5', '0.71', '1', '1.41', '2'])
        ax.set_xlim(0.48, 2.08)
        ax.set_ylim(18.0, 30.0)
        ax.set_title(parameter)
        ax.set_xlabel('Fold change from baseline')
        ax.set_ylabel('Period [h]')
        ax.grid(True, which='both', linestyle=':', alpha=0.35)

    for index in range(len(parameters), nrows * ncols):
        axs[index // ncols, index % ncols].axis('off')

    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3,
                   bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(title, y=1.02, fontsize=14)
    fig.text(0.5, 0.005,
             '× at 18.25 h denotes loss of a sustained circadian rhythm',
             ha='center', fontsize=9, color='0.35')
    _finish(fig, save_path)


def plot_cwo_phase_sensitivity(rows, save_path: str | None = None):
    """Four-parameter C/W sensitivity with optional fixed lunar phases."""
    n_conditions = len({row['Drive Condition'] for row in rows})
    drive_text = ('at fixed lunar phases' if n_conditions > 1
                  else 'without lunar phase comparisons')
    _plot_validated_sweep(
        rows, ['nu11', 'nu12', 'nu13', 'nu14'],
        f'CWO kinetic sensitivity {drive_text}', save_path)


def plot_w_gain_turnover(rows, save_path: str | None = None):
    """Separate CWO abundance gain and turnover at optional fixed phases."""
    n_conditions = len({row['Drive Condition'] for row in rows})
    drive_text = ('at fixed lunar phases' if n_conditions > 1
                  else 'without lunar phase comparisons')
    _plot_validated_sweep(
        rows, ['W gain (nu13/nu14)', 'W turnover speed'],
        f'CWO protein abundance versus turnover {drive_text}',
        save_path)


def plot_cwo_amount_vs_period(rows, save_path: str | None = None):
    """Plot circadian period against mean simulated CWO protein abundance.

    The two panels keep the W-gain and W-turnover perturbations separate.
    Point colour shows parameter fold change and marker shape shows the fixed
    lunar drive. The star marks the baseline parameter value (1x). Rows
    without a valid sustained period are omitted.
    """
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    parameters = ['W gain (nu13/nu14)', 'W turnover speed']
    phase_markers = {
        'Constant mean': 'o',
        'Full moon': 'o',
        'Mean drive': 's',
        'New moon': '^',
    }
    valid_folds = [row['Fold Change'] for row in rows
                   if np.isfinite(row['Period (h)'])]
    max_log_fold = max(abs(np.log2(valid_folds)).max(), 1.0)
    norm = Normalize(-max_log_fold, max_log_fold)
    cmap = plt.get_cmap('coolwarm')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)

    for ax, parameter in zip(axs, parameters):
        subset = [row for row in rows if row['Parameter'] == parameter]
        for phase, marker in phase_markers.items():
            phase_rows = sorted(
                (row for row in subset if row['Drive Condition'] == phase),
                key=lambda row: row['Fold Change'])
            if not phase_rows:
                continue

            mean_w = np.asarray([row['Mean W'] for row in phase_rows],
                                dtype=float)
            periods = np.asarray([row['Period (h)'] for row in phase_rows],
                                 dtype=float)
            log_folds = np.log2([row['Fold Change'] for row in phase_rows])
            valid = np.isfinite(mean_w) & np.isfinite(periods)
            ax.scatter(mean_w[valid], periods[valid], c=log_folds[valid],
                       cmap=cmap, norm=norm, marker=marker, s=50,
                       edgecolor='0.2', linewidth=0.45, label=phase,
                       zorder=3)

            baseline = np.asarray([
                np.isclose(row['Fold Change'], 1.0) for row in phase_rows
            ]) & valid
            if baseline.any():
                ax.scatter(mean_w[baseline], periods[baseline], marker='*',
                           s=150, color='white', edgecolor='black',
                           linewidth=0.9, zorder=4)

        ax.axhline(24.0, color='0.35', lw=1.0, ls='--')
        ax.axhspan(22.0, 25.0, color='#16a34a', alpha=0.09)
        ax.set_title(parameter)
        ax.set_xlabel('Mean CWO protein, W [a.u.]')
        ax.set_ylabel('Period [h]')
        ax.grid(True, linestyle=':', alpha=0.35)

    handles, labels = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=3,
                   bbox_to_anchor=(0.46, 0.89))
    colourbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axs,
                             pad=0.025, fraction=0.028)
    colourbar.set_label('Parameter fold change (log2 colour scale)')
    fig.suptitle('Average CWO abundance in relation to circadian period',
                 y=0.97, fontsize=14)
    fig.text(0.46, 0.035, 'Star denotes the baseline parameter set',
             ha='center', fontsize=9, color='0.35')
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.14, top=0.79,
                        wspace=0.10)
    _finish(fig, save_path, tight=False)


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
            color='blue', lw=2.5, label='Model-estimated period')
    ax.axhspan(22, 25, color='green', alpha=0.10,
               label='Target plasticity (22–25 h)')
    ax.axhline(24, color='red', ls='--', alpha=0.5)

    if valid.any():
        ax.text(0.01, 0.97, 'Full moon (trough L)', transform=ax.transAxes,
                ha='left', va='top', color='gray')
        ax.text(0.50, 0.97, 'New moon (peak L)', transform=ax.transAxes,
                ha='center', va='top', color='gray')

    ax.set_xlabel('Time within lunar month [h]', fontsize=12)
    ax.set_ylabel('Circadian period [h]', fontsize=12)
    ax.set_title('Circadian-period plasticity across the lunar month',
                 fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    _finish(fig, save_path)


def plot_lunar_limit_cycles(cycles, save_path: str | None = None):
    """Phase-plane loops across the lunar month, using CLK/BMAL versus PER."""
    if not cycles:
        raise ValueError('No complete circadian cycles were supplied')

    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from .model import T_LUNAR

    cmap = plt.get_cmap('twilight_shifted')
    norm = Normalize(0.0, T_LUNAR)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8), sharex=True, sharey=True)

    paths = [np.column_stack([cycle['x'], cycle['y']]) for cycle in cycles]
    phases = np.asarray([cycle['lunar_phase_h'] for cycle in cycles])
    collection = LineCollection(paths, cmap=cmap, norm=norm,
                                linewidths=1.4, alpha=0.82)
    collection.set_array(phases)
    axs[0].add_collection(collection)
    axs[0].autoscale_view()
    axs[0].set_title('All circadian loops over one lunar month')

    representative_phases = [
        ('FM', 0.0),
        ('FM+1', T_LUNAR / 4.0),
        ('NM', T_LUNAR / 2.0),
        ('NM+1', 3.0 * T_LUNAR / 4.0),
    ]
    for label, target_phase in representative_phases:
        cycle = min(
            cycles,
            key=lambda item: min(
                abs(item['lunar_phase_h'] - target_phase),
                T_LUNAR - abs(item['lunar_phase_h'] - target_phase)))
        colour = cmap(norm(target_phase))
        axs[1].plot(cycle['x'], cycle['y'], color=colour, lw=2.2,
                    label=f"{label}: P={cycle['period_h']:.2f} h")
        axs[1].scatter(cycle['x'][0], cycle['y'][0], color=colour,
                       s=28, edgecolor='white', linewidth=0.7, zorder=3)

    axs[1].set_title('Representative lunar phases')
    axs[1].legend(fontsize=9, frameon=True)

    for ax in axs:
        ax.set_xlabel('CLK/BMAL (X) [a.u.]')
        ax.set_ylabel('PER/tr-CRY (Z) [a.u.]')
        ax.grid(True, linestyle=':', alpha=0.35)
        ax.set_box_aspect(1)

    colourbar = fig.colorbar(collection, ax=axs, pad=0.025, fraction=0.04)
    colourbar.set_label('Lunar phase')
    colourbar.set_ticks([0.0, T_LUNAR / 4.0, T_LUNAR / 2.0,
                         3.0 * T_LUNAR / 4.0, T_LUNAR])
    colourbar.set_ticklabels(['FM', 'FM+1', 'NM', 'NM+1', 'FM'])
    fig.suptitle('Evolution of the circadian limit cycle over the lunar month',
                 fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.84,
                        wspace=0.22)
    _finish(fig, save_path, tight=False)


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
        ``'per'``), or ``None`` for a sensible default panel.
    n_lunar_cycles : int
        How many lunar cycles to plot (after the burn-in cycle).
    zscore : bool
        Z-score each trace independently. Useful when overlaying traces with
        very different magnitudes (e.g. mRNA vs. high-amplitude CWO protein).
    show_lunar_drive : bool
        Overlay the L_t drive on a secondary axis of each panel.
    """
    from .fitting import GENE_TO_STATE_INDEX, _zscore
    from .model import DEFAULT_INITIAL_STATE, T_LUNAR
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
        y0 = DEFAULT_INITIAL_STATE

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
