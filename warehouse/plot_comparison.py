"""Render the manuscript's four-panel warehouse trajectory comparison."""
from pathlib import Path

import numpy as np

from .protocol import check_sha256, read_json, reference, track_length

PANELS = (
    ('Path2_Hand_Swing01', 'Carrying Mode: Handheld arm swing'),
    # Acquisition ID Path3_Trouser_Pocket05 is numbered 02 in this package.
    ('Path3_Trouser_Pocket02', 'Carrying Mode: Trouser pocket'),
    ('Path3_Jacket_Pocket02', 'Carrying Mode: Upper-body pocket'),
    ('Path1_Freedom03', 'Carrying Mode: Unrestricted'),
)
METHODS = ('reference_xy', 'pdr_display_xy', 'hn_display_xy')
LABELS = ('Ground_truth', 'Our previous PDR study', 'HN-Transformer')
COLORS = ('#888888', '#C58C63', '#0072B2')
FIGURE_NAME = 'warehouse_2x2_Path2_Hand_Swing01.png'


def check_2x2_alignment(figure, axes, tolerance_pt=1.5):
    figure.canvas.draw()
    scale = np.tile(figure.get_size_inches(), 2) * 72
    boxes = np.array([ax.get_position().bounds for ax in axes.flat]) * scale
    deviations = [np.ptp(boxes[:, 2:], axis=0).max()]
    for a, b, fields in ((0, 1, [1, 3]), (2, 3, [1, 3]),
                         (0, 2, [0, 2]), (1, 3, [0, 2])):
        deviations.append(np.abs(boxes[a, fields] - boxes[b, fields]).max())
    if not np.isfinite(boxes).all() or max(deviations) > tolerance_pt:
        raise ValueError('2x2 panels are not aligned.')


def load_comparison_tracks(folder):
    manifest = read_json(folder / 'manifest.json')
    records = {record['sequence']: record for record in manifest['records']}
    panel_tracks = []
    for sequence, _ in PANELS:
        record = records[sequence]
        path = folder / f'{sequence}.npz'
        check_sha256(path, record['cache_sha256'])
        reference_xy, _ = reference(record['route'])
        with np.load(path, allow_pickle=False) as arrays:
            trajectories = [arrays[key] for key in METHODS]
            np.testing.assert_array_equal(trajectories[0], reference_xy)
            for method, xy in zip(('pdr', 'hn'), trajectories[1:], strict=True):
                np.testing.assert_allclose(track_length(xy), track_length(arrays[f'{method}_xy']),
                                           rtol=0, atol=1e-9)
        panel_tracks.append(trajectories)
    return panel_tracks


def plot_comparison(tracks: Path, output: Path):
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.font_manager import FontProperties, findfont

    panel_tracks = load_comparison_tracks(tracks)
    font = 'Times New Roman'
    try:
        findfont(FontProperties(family=font), fallback_to_default=False)
    except ValueError:
        font = 'DejaVu Serif'

    with mpl.rc_context({'font.family': font, 'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': True,
                         'axes.spines.right': True, 'axes.linewidth': .7, 'legend.frameon': False}):
        figure = Figure(figsize=(7.2, 5.0))
        FigureCanvasAgg(figure)
        axes = figure.subplots(2, 2)
        figure.subplots_adjust(left=.09, right=.98, bottom=.115, top=.78, wspace=.26, hspace=.40)
        for ax, (sequence, title), trajectories in zip(axes.flat, PANELS, panel_tracks, strict=True):
            for j, xy in enumerate(trajectories):
                ax.plot(xy[:, 1], xy[:, 0], color=COLORS[j], ls='--' if j == 0 else '-',
                        lw=(.9, 1., 1.7)[j], label=LABELS[j], zorder=(1, 2, 3)[j])
            ax.set(xlabel='Y (m)', ylabel='X (m)')
            ax.margins(x=.06, y=.10)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_axisbelow(True)
            ax.grid(True, linestyle='--', linewidth=.5, color='#999999', alpha=.4)
            ax.tick_params(direction='out', top=True, right=True, bottom=True, left=True,
                           labeltop=False, labelright=False, length=3, width=.65)
            for spine in ax.spines.values():
                spine.set_visible(True)
            ax.set_title(title, fontsize=10, pad=8)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        legend = figure.legend(handles, labels, ncol=3, loc='upper right', bbox_to_anchor=(.7, .9),
                               fontsize=8.5, handlelength=2.5, columnspacing=2., frameon=True,
                               fancybox=False, edgecolor='black', facecolor='white', framealpha=1.)
        legend.get_frame().set_linewidth(.8)
        check_2x2_alignment(figure, axes)
        for ax, (sequence, _), trajectories in zip(axes.flat, PANELS, panel_tracks, strict=True):
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            for xy in trajectories:
                if not (np.all((xmin <= xy[:, 1]) & (xy[:, 1] <= xmax))
                        and np.all((ymin <= xy[:, 0]) & (xy[:, 0] <= ymax))):
                    raise ValueError(f'{sequence}: Trajectory extends beyond the plot limits.')
        figure.savefig(output, dpi=600)
