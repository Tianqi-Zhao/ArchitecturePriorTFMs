"""Shared vector-panel rendering; no PDF cropping or manuscript checkout needed."""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARCH_NAMES = {"1d": "Row-token", "2d": "Alternating-axis"}
ARCH_COLORS = {"1d": "#526784", "2d": "#348d88"}
MODEL_NAMES = {"nano": "Alternating-axis", "v2": "TabPFN v2"}
SCOPE_COLORS = {"context": "#0072b2", "query": "#d55e00"}
DIM_COLORS = {10: "#0072b2", 20: "#d55e00"}
ROLE_COLORS = {"active": "#D55E00", "inactive": "#F2BD86"}
COMPONENT_COLORS = {"E": "#2D333B", "C": "#D55E00", "Z": "#C5C9CF", "N": "#80868E"}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7, "axes.linewidth": .6,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    })


def decorate(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E1E5E9", linewidth=.45)
    ax.tick_params(length=2.5, width=.6)


def seed_points(ax, x, values, color, marker="o", filled=True):
    values = np.asarray(values)
    ax.scatter(x + np.linspace(-.045, .045, len(values)), values, s=8,
               color=color, alpha=.45, zorder=4)
    ax.errorbar(x, values.mean(), yerr=values.std(ddof=0), fmt=marker,
                ms=4, color=color, mfc=color if filled else "white", mew=.8,
                capsize=3, elinewidth=1, zorder=5)


class Renderer:
    def __init__(self, output_dir, formats=("pdf", "png")):
        self.root = Path(output_dir)
        self.formats = formats
        self.families = {}
        self.extras = []
        style()

    def save(self, fig, relative):
        destination = self.root / "figures" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        for extension in self.formats:
            fig.savefig(destination.with_suffix("." + extension), dpi=180)
        plt.close(fig)

    def grid(self, name, panels, *, columns=3, panel_stem=None):
        """Each panel callable is rendered both alone and in a complete figure."""
        panel_stem = panel_stem or name
        columns = min(columns, len(panels))
        rows = math.ceil(len(panels) / columns)
        fig, axes = plt.subplots(rows, columns, squeeze=False,
                                 figsize=(3.55 * columns, 2.85 * rows), layout="constrained")
        filenames = []
        primary_extension = "pdf" if "pdf" in self.formats else self.formats[0]
        for index, draw in enumerate(panels):
            letter = chr(97 + index)
            draw(axes.flat[index])
            title = axes.flat[index].get_title()
            axes.flat[index].set_title("")
            axes.flat[index].set_title(f"({letter}) " + title, loc="left")
            panel, ax = plt.subplots(figsize=(3.55, 2.85), layout="constrained")
            draw(ax)
            relative = Path("panels") / name / f"{panel_stem}__{letter}"
            self.save(panel, relative)
            filenames.append(str(Path("figures") / relative.with_suffix("." + primary_extension)))
        for ax in axes.flat[len(panels):]:
            ax.remove()
        self.save(fig, Path(name))
        self.families[name] = {"panels": filenames,
                               "combined": f"figures/{name}.{primary_extension}"}

    def legend(self, name, handles):
        fig = plt.figure(figsize=(7.1, .38))
        fig.legend(handles=handles, loc="center", ncol=len(handles), frameon=False,
                   fontsize=8, handlelength=1.1, columnspacing=1.1)
        relative = Path("panels") / name / f"{name}__legend"
        self.save(fig, relative)
        self.extras.extend(str(Path("figures") / relative.with_suffix("." + ext)) for ext in self.formats)
