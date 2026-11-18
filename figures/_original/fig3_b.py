"""Saved Figure 3b plotting statements; no model execution."""
import numpy as np
import matplotlib.pyplot as plt

CLS = ["k562", "rpe1", "hepg2", "jurkat"]
CL_LABELS = {"k562": "K562", "rpe1": "RPE1", "hepg2": "HepG2", "jurkat": "Jurkat"}
MODEL_STYLE_3to1 = {
    "STATE": {"color": "#8A94A6", "marker": "o", "label": "STATE"},
    "Ridge": {"color": "#149E9A", "marker": "s", "label": "Ridge (DepMap)"},
    "MLP": {"color": "#0D7377", "marker": "^", "label": "MLP (DepMap)"},
    "BioLORD": {"color": "#D98736", "marker": "P", "label": "BioLORD"},
    "scGen": {"color": "#AA5C97", "marker": "D", "label": "scGen"},
}

def plot_panel_b(results, out_dir=OUT, five_seed=None):
    plt.rcParams.update({'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 10, 'axes.labelsize': 11, 'axes.linewidth': 0.6, 'axes.edgecolor': '#1F2937', 'xtick.color': '#1F2937', 'ytick.color': '#1F2937', 'text.color': '#1F2937', 'svg.fonttype': 'none'})
    (fig, axis) = plt.subplots(figsize=(6.2, 3.5))
    axis.set_title('Three cell lines source', fontsize=11, pad=8)
    positions = np.arange(len(CLS))
    offsets = np.linspace(-0.27, 0.27, len(MODEL_STYLE_3to1))
    summary = {}
    for (model_index, (model, style)) in enumerate(MODEL_STYLE_3to1.items()):
        summary[model] = {}
        for (target_index, target) in enumerate(CLS):
            values = np.asarray(results[model][target], dtype=float)
            if not len(values) or not np.all(np.isfinite(values)):
                raise ValueError(f'Missing or invalid values for {model}/{target}')
            mean = float(values.mean())
            deviation = float(values.std()) if len(values) > 1 else None
            position = positions[target_index] + offsets[model_index]
            if deviation is not None:
                axis.errorbar(position, mean, yerr=deviation, fmt='none', ecolor=style['color'], elinewidth=1.0, capsize=2, capthick=0.8, zorder=2)
            axis.scatter(position, mean, s=45, marker=style['marker'], color=style['color'], edgecolor='white', linewidth=0.5, zorder=3, label=style['label'] if target_index == 0 else None)
            summary[model][target] = {'values': values.tolist(), 'mean': mean, 'std': deviation, 'n_runs': len(values)}
    axis.axhline(0, color='#1F2937', linewidth=0.5, alpha=0.4)
    axis.set_xticks(positions)
    axis.set_xticklabels([CL_LABELS[target] for target in CLS], rotation=45, ha='right')
    axis.set_xlabel('Target cell line')
    axis.set_ylabel('Perturbed-reference Pearson')
    axis.yaxis.grid(True, color='#E5E7EB', linewidth=0.5)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False, fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(out_dir / f'fig3b_3to1_transfer.{extension}', dpi=300, bbox_inches='tight')
    plt.close(fig)

plot_panel_b(results, OUT)
