
from matplotlib import rcParams

_STYLES = {
    "thesis_v1": {
        "figure.figsize": (7.5, 4.5),
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "savefig.dpi": 120,
        "font.family": "DejaVu Sans",
    }
}

def apply(style: str = "thesis_v1"):
    for k, v in _STYLES.get(style, {}).items():
        rcParams[k] = v
    return style
