"""Render the formal Experiment 0 figures from evaluated CSV tables.

This module intentionally contains no exploratory plotting. Its two public
entry points build the paper's Part I bundle and the TabDPT release appendix.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from experiment_core.openml import REAL_REGRESSION_DATASETS

from experiments_arch.experiment_0_pretrained.evaluate import (
    FRIEDMAN_SWEEP_MODEL_SUMMARY_COLUMNS,
    MODEL_CONTRAST_COLUMNS,
    PAIRED_EFFECT_COLUMNS,
    PART_I2_CONTRAST_COLUMNS,
    PART_I2_DATASET_DIMENSION_COLUMNS,
    PART_I2_FRIEDMAN_SWEEP_SUMMARY_COLUMNS,
    PART_I2_PDP_CURVE_COLUMNS,
    PART_I2_SUMMARY_COLUMNS,
    PART_I2_TASK_EFFECT_COLUMNS,
    SUMMARY_COLUMNS,
    require_benchmark_sizes,
)


# --- Figure vocabulary and rendering contract ------------------------------

MODEL_LABELS = {
    "tabdpt_retrieval_off": "TabDPT v1.2 (retrieval off)",
    "tabpfn_v2": "TabPFN v2",
    "tabpfn_v2.5": "TabPFN v2.5",
    "tabicl_v2": "TabICL v2",
    "tabpfn_v3": "TabPFN v3",
    "tabdpt_v1_0_retrieval_off": "TabDPT v1.0",
    "tabdpt_v1_1_retrieval_off": "TabDPT v1.1",
    "tabdpt_v1_3_retrieval_off": "TabDPT v1.3",
}
# The release figures order the four TabDPT releases; the primary model is the
# v1.2 checkpoint used in the main-text comparison.
TABDPT_VERSION_MODELS = (
    "tabdpt_v1_0_retrieval_off",
    "tabdpt_v1_1_retrieval_off",
    "tabdpt_retrieval_off",
    "tabdpt_v1_3_retrieval_off",
)
TABDPT_VERSION_BASELINE = "tabpfn_v2"
TABDPT_VERSION_LABELS = {
    "tabdpt_v1_0_retrieval_off": "TabDPT v1.0",
    "tabdpt_v1_1_retrieval_off": "TabDPT v1.1",
    "tabdpt_retrieval_off": "TabDPT v1.2",
    "tabdpt_v1_3_retrieval_off": "TabDPT v1.3",
    "tabpfn_v2": "TabPFN v2 (reference)",
}
MODEL_ORDER = tuple(MODEL_LABELS)
MODEL_MARKERS = {
    "tabdpt_retrieval_off": "o",
    "tabpfn_v2": "s",
    "tabpfn_v2.5": "^",
    "tabicl_v2": "D",
    "tabpfn_v3": "v",
    "tabdpt_v1_0_retrieval_off": "v",
    "tabdpt_v1_1_retrieval_off": "^",
    "tabdpt_v1_3_retrieval_off": "P",
}
PART_I2_SOURCES = ("synthetic", "real")
OPENML_DIMENSION_TIERS = (
    ("low", r"Low-dimensional ($d_{\mathrm{base}} \leq 8$)", 3, 8),
    ("medium", r"Medium-dimensional ($9 \leq d_{\mathrm{base}} \leq 18$)", 9, 18),
    ("high", r"High-dimensional ($d_{\mathrm{base}} \geq 21$)", 21, None),
)
# The long tier labels overrun a 1.7 in panel at final size; the tier bounds
# stay exact, only the leading word is abbreviated.
# Split so the plain tier name carries line one and the width bound carries
# line two: the title then fits a 1.6 in panel at the size the mathtext
# subscript needs.
OPENML_DIMENSION_TIER_SHORT = {
    "low": ("Low-dimensional", r"$d_{\mathrm{base}} \leq 8$"),
    "medium": ("Mid-dimensional", r"$9 \leq d_{\mathrm{base}} \leq 18$"),
    "high": ("High-dimensional", r"$d_{\mathrm{base}} \geq 21$"),
}
PDP_REPRESENTATIVE_DATASETS = (
    ("synthetic", "friedman1-task-000"),
    ("real", "openml-44973"),  # Grid Stability
)
PDP_REPRESENTATIVE_RATIO = 7.0
PDP_REPRESENTATIVE_REPEAT = 0
PDP_QUANTILE_LABELS = ("5", "27.5", "50", "72.5", "95")

# The publication figures are authored at their final printed size, so
# ``\includegraphics[width=\linewidth]`` applies a 1.0x scale and the rendered
# point sizes below are the point sizes on paper. Previously the bundle was
# drawn at 9.2-16.0 in and shrunk 2-3x by LaTeX, which pushed almost every
# glyph under the 5 pt legibility floor.
PUBLICATION_WIDTH_IN = 5.5  # ICLR single-column \linewidth
PUBLICATION_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    # Matplotlib renders mathtext sub/superscripts at 0.7x the surrounding
    # size, so any string carrying $R^2$ or $d_{\mathrm{base}}$ must sit at
    # >= 7.15 pt to keep those glyphs above the 5 pt floor.
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.titlesize": 8.0,
    "axes.linewidth": 0.6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.labelpad": 2.0,
    "axes.titlepad": 4.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.minor.size": 1.3,
    "ytick.minor.size": 1.3,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.1,
    "legend.labelspacing": 0.35,
    "legend.borderaxespad": 0.2,
    "lines.markeredgewidth": 0.3,
    "grid.linewidth": 0.4,
    "savefig.dpi": 600,
}
# One saturated hero colour for the system that degrades, and a single cool
# family for the four comparators, so the figure reads as "one outlier against
# a unified baseline". This also removes the red/green pair of the previous
# tab10 mapping, which is the standard colour-vision-deficiency failure case.
PUBLICATION_MODEL_COLORS = {
    "tabdpt_retrieval_off": "#B64342",
    "tabpfn_v2": "#0F4D92",
    "tabpfn_v2.5": "#3775BA",
    "tabicl_v2": "#9A4D8E",
    "tabpfn_v3": "#42949E",
    # Release order is ordinal, so the four TabDPT releases take one ramp of
    # the hero red with v1.2 keeping its main-text hue; the TabPFN v2
    # reference keeps its own navy.
    "tabdpt_v1_0_retrieval_off": "#F0A6A4",
    "tabdpt_v1_1_retrieval_off": "#D96B68",
    "tabdpt_v1_3_retrieval_off": "#7B2220",
}
# rho = 0.2 and 0.4 land 2.8% and 5.6% along a linear axis that ends at 7, so
# labelling them runs into the "0" label and renders as "0.2.4". Keep the dose
# spacing linear and demote the two small doses to unlabelled minor ticks.
PUBLICATION_RATIO_MAJOR_TICKS = (0.0, 1.0, 2.0, 4.0, 7.0)
# Panel (c) compares two metrics rather than two models, so it needs hues that
# are not in the model palette and stay distinguishable in greyscale.
PART_I_METRIC_COLORS = {"r2_drop": "#272727", "permutation_shift": "#E69F00"}
PART_I_BENCHMARKS = {"synthetic": "Friedman", "real": "OpenML"}
# Where one panel carries both benchmarks, line style is the benchmark and
# colour stays the model, so neither encoding has to do double duty.
PART_I_BENCHMARK_STYLE = {"synthetic": (0, (2.4, 1.2)), "real": "solid"}
PART_I2_MODELS = ("tabdpt_retrieval_off", "tabpfn_v2")
# Figure 1 is stored as four single-panel files that LaTeX tiles back into a
# 2 x 2 grid at 0.5\linewidth each, so every tile is exactly half the column
# wide and shares one plot rectangle. The values reproduce the plot areas of
# the earlier single-file layout (2.22 x 1.11 in, 0.53 in column gutter).
# Section 2 places two panels side by side inside a half-column wrapfigure,
# each at 0.49\linewidth of that 0.5\textwidth box. Authoring the tiles at
# that final width is what keeps the type legible: the same panels drawn on a
# 2.75 in canvas were being shrunk to 0.49x, which put their labels near 3 pt.
PART_I_TILE_WIDTH_IN = 0.49 * PUBLICATION_WIDTH_IN / 2.0
PART_I_TILE_LEFT_IN = 0.42
PART_I_TILE_RIGHT_IN = 0.04
PART_I_TILE_PLOT_HEIGHT_IN = 0.78
# The tiles are physically small, so they run a notch below the 5.5 in
# figures' type scale while staying clear of the 5 pt floor.
PART_I_TILE_LABEL_PT = 7.0
PART_I_TILE_TICK_PT = 6.0
PART_I_TILE_LEGEND_PT = 6.0
# The four-model key in panel (a) competes with the curves for room, so it
# runs one step smaller while staying above the 5 pt floor.
PART_I_TILE_LEGEND_SMALL_PT = 5.5
# Figure 1 compares the four systems the running text names; the complete
# five-system comparison stays in the appendix.
PART_I_OVERVIEW_EXCLUDED_MODELS = ("tabpfn_v2.5",)
# "TabDPT v1.2 (retrieval off)" does not fit a 0.9 in plot; the retrieval
# setting is stated in the running text instead.
PART_I_TILE_SHORT_LABELS = {"tabdpt_retrieval_off": "TabDPT v1.2"}
# (top, bottom) margins: the top row needs a title above and tick labels
# below; the bottom row needs only the panel letter above but tick labels and
# the x label below. Their sum across the row seam keeps the old row gap.
# Both rows now carry an x label, so both reserve space below; only the
# titled row reserves extra space above.
PART_I_TOP_TILE_MARGINS_IN = (0.19, 0.31)
PART_I_BOTTOM_TILE_MARGINS_IN = (0.16, 0.31)
# Feature identity in the PDP panels is nominal and must not borrow the model
# hues: a red curve there would read as TabDPT everywhere else. This is an
# Okabe-Ito-derived set with the hero red and baseline navy left out.
PUBLICATION_FEATURE_COLORS = (
    "#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#7570B3",
    "#8C6D31", "#66A61E", "#00A0B0", "#666666", "#B07AA1",
)
# Minimum size for any string that contains a mathtext sub/superscript.
PUBLICATION_MATH_FONTSIZE = 7.5
PUBLICATION_LINEWIDTH = 1.1
PUBLICATION_MARKERSIZE = 2.9
PUBLICATION_BAND_ALPHA = 0.07
PUBLICATION_BAND_EDGE_ALPHA = 0.55
PUBLICATION_BAND_EDGE_WIDTH = 0.35


# --- Shared file and rendering utilities -----------------------------------

def _read(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"plot input columns do not match schema: {path}")
        return list(reader)


def _atomic_savefig(figure, path: Path, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}.partial{path.suffix}")
    try:
        figure.savefig(temporary, **kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _model_colors() -> dict[str, tuple[float, ...]]:
    """Return the fixed, colour-vision-safe publication palette."""
    from matplotlib.colors import to_rgb

    return {
        model_id: to_rgb(color)
        for model_id, color in PUBLICATION_MODEL_COLORS.items()
    }


def _publication_band(
    axis,
    x: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    color,
    zorder: float = 1.0,
    edges: bool = True,
) -> None:
    """Draw an IQR ribbon as a very light wash plus two thin boundary lines.

    A solid alpha-0.14 wash of several overlapping models turns into opaque
    grey mud at final size. Thin edges keep each interval traceable while
    letting the median lines stay the most salient mark. Callers that stack
    several bands per model drop the edges.
    """
    axis.fill_between(
        x, lower, upper, color=color, alpha=PUBLICATION_BAND_ALPHA,
        linewidth=0, zorder=zorder,
    )
    if edges:
        for bound in (lower, upper):
            axis.plot(
                x, bound, color=color, alpha=PUBLICATION_BAND_EDGE_ALPHA,
                linewidth=PUBLICATION_BAND_EDGE_WIDTH, zorder=zorder,
            )


def _openml_catalog_dimensions() -> dict[str, tuple[str, int]]:
    return {
        f"openml-{spec.dataset_id}": (spec.name, int(spec.n_features))
        for spec in REAL_REGRESSION_DATASETS
    }


def _openml_tier(d_base: int) -> str | None:
    for key, _label, lower, upper in OPENML_DIMENSION_TIERS:
        if d_base >= lower and (upper is None or d_base <= upper):
            return key
    return None


def _median_iqr(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
    )


def _add_corner_panel_labels(axes, *, first: str = "a") -> None:
    """Anchor a bold lowercase panel label at each panel's top-left corner.

    Fixed point offsets keep the labels at the same optical position whatever
    the panel width, and leave the title free to carry only the condition.
    ``first`` lets a figure split across files continue its lettering.
    """
    panel_index = 0
    for axis in np.asarray(axes, dtype=object).ravel():
        if not axis.get_visible():
            continue
        axis.annotate(
            chr(ord(first) + panel_index),
            xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(-13.0, 3.0), textcoords="offset points",
            fontsize=8.0, fontweight="bold", ha="left", va="bottom",
            annotation_clip=False,
        )
        panel_index += 1


def _publication_alignment_gate(
    figure, stem: str, exemptions: list[dict[str, object]] | None = None
) -> None:
    """Run the render-time panel-alignment audit when QA is requested.

    Enabled by pointing ``NATURE_FIGURE_ALIGNMENT_DIR`` at a writable
    directory; a normal run is unaffected.  The auditor is vendored in
    ``vendor/``, so no external ``PYTHONPATH`` is needed.
    """
    directory = os.environ.get("NATURE_FIGURE_ALIGNMENT_DIR")
    if not directory:
        return
    from experiments_arch.experiment_0_pretrained.vendor.audit_panel_alignment import (
        require_matplotlib_panel_alignment,
    )

    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    require_matplotlib_panel_alignment(
        figure,
        json_out=str(target / f"{stem}.alignment.json"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        exemptions=exemptions or [],
        strict=True,
    )


# In the four-panel metric rows, (a, b) and (c, d) are two y-sharing pairs on
# different scales, so only (a) and (c) carry tick labels. The wider gutter in
# front of (c) is the scale change and is deliberate.
PUBLICATION_PAIRED_ROW_EXEMPTIONS = [
    {
        "panels": ["a", "b", "c", "d"],
        "checks": ["horizontal-gutter"],
        "reason": (
            "two y-sharing panel pairs on different scales; the wider middle "
            "gutter carries the tick labels that mark the scale change"
        ),
    }
]


def _save_publication_figure(
    figure,
    directory: Path,
    stem: str,
    exemptions: list[dict[str, object]] | None = None,
) -> Path:
    """Export the vector master plus a raster copy at the authored size.

    ``bbox_inches`` is deliberately not used: cropping would change the final
    width and reintroduce the LaTeX rescaling that shrank the type.
    """
    _publication_alignment_gate(figure, stem, exemptions)
    saved = _atomic_savefig(figure, directory / f"{stem}.pdf")
    _atomic_savefig(figure, directory / f"{stem}.png", dpi=600)
    return saved


# --- Part I.1 data preparation and appendix panels --------------------------

def _raw_r2_summary(
    effects: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Aggregate absolute R2 using the same repeat-then-dataset hierarchy."""
    by_dataset: dict[tuple[str, str, float, str], list[float]] = {}
    for row in effects:
        value = float(row["r2_at_dose"])
        if not np.isfinite(value):
            continue
        key = (
            row["source"],
            row["model_id"],
            float(row["requested_irrelevant_ratio"]),
            row["dataset_id"],
        )
        by_dataset.setdefault(key, []).append(value)

    by_condition: dict[tuple[str, str, float], list[float]] = {}
    for (source, model_id, ratio, _dataset_id), values in by_dataset.items():
        by_condition.setdefault((source, model_id, ratio), []).append(
            float(np.mean(values))
        )

    rows: list[dict[str, str]] = []
    for (source, model_id, ratio), values in sorted(by_condition.items()):
        array = np.asarray(values, dtype=np.float64)
        rows.append({
            "source": source,
            "model_id": model_id,
            "requested_irrelevant_ratio": format(ratio, ".17g"),
            "r2_at_dose_mean": format(float(np.mean(array)), ".17g"),
            "r2_at_dose_std": format(
                float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
                ".17g",
            ),
            "r2_at_dose_median": format(float(np.median(array)), ".17g"),
            "r2_at_dose_q25": format(float(np.quantile(array, 0.25)), ".17g"),
            "r2_at_dose_q75": format(float(np.quantile(array, 0.75)), ".17g"),
        })
    return rows


def _plot_openml_dimension_r2_drop(
    effects: list[dict[str, str]], path: Path
) -> Path | None:
    """Plot dataset-first OpenML dose responses in three base-width tiers."""
    import matplotlib.pyplot as plt

    catalog = _openml_catalog_dimensions()
    real = [
        row for row in effects
        if row["source"] == "real" and row["dataset_id"] in catalog
    ]
    if not real:
        return None
    model_colors = _model_colors()
    models = [
        model_id for model_id in MODEL_ORDER
        if any(row["model_id"] == model_id for row in real)
    ]
    ratios = sorted({float(row["requested_irrelevant_ratio"]) for row in real})
    positions = np.asarray(ratios, dtype=np.float64)
    by_dataset: dict[tuple[str, str, float, str], list[float]] = {}
    for row in real:
        tier = _openml_tier(catalog[row["dataset_id"]][1])
        if tier is None:
            continue
        key = (
            tier,
            row["model_id"],
            float(row["requested_irrelevant_ratio"]),
            row["dataset_id"],
        )
        by_dataset.setdefault(key, []).append(float(row["r2_drop"]))

    figure, axes = plt.subplots(
        1, len(OPENML_DIMENSION_TIERS),
        figsize=(PUBLICATION_WIDTH_IN, 2.35),
        sharex=True, sharey=True, squeeze=False,
        layout="constrained",
    )
    flat_axes = axes[0]
    for axis, (tier, _label, _lower, _upper) in zip(
        flat_axes, OPENML_DIMENSION_TIERS
    ):
        tier_datasets = {
            dataset_id for (key_tier, _model, _ratio, dataset_id) in by_dataset
            if key_tier == tier
        }
        for model_id in models:
            medians, q25s, q75s = [], [], []
            for ratio in ratios:
                dataset_values = [
                    float(np.mean(values))
                    for (key_tier, model, key_ratio, _dataset), values
                    in by_dataset.items()
                    if key_tier == tier
                    and model == model_id
                    and key_ratio == ratio
                ]
                median, q25, q75 = _median_iqr(dataset_values)
                medians.append(median)
                q25s.append(q25)
                q75s.append(q75)
            median_array = np.asarray(medians)
            q25_array = np.asarray(q25s)
            q75_array = np.asarray(q75s)
            finite = (
                np.isfinite(median_array)
                & np.isfinite(q25_array)
                & np.isfinite(q75_array)
            )
            _publication_band(
                axis, positions[finite], q25_array[finite],
                q75_array[finite], color=model_colors[model_id],
            )
            axis.plot(
                positions[finite], median_array[finite],
                color=model_colors[model_id],
                linewidth=PUBLICATION_LINEWIDTH,
                marker=MODEL_MARKERS.get(model_id, "o"),
                markersize=PUBLICATION_MARKERSIZE,
                markeredgecolor="white", markeredgewidth=0.3, zorder=3,
                label=MODEL_LABELS[model_id],
            )
        name, bound = OPENML_DIMENSION_TIER_SHORT[tier]
        axis.set_title(
            f"{name}\n{bound} ($n={len(tier_datasets)}$)",
            fontsize=PUBLICATION_MATH_FONTSIZE,
        )
        _publication_dose_axis(axis, ratios)
        axis.set_xlabel(r"Null-feature ratio $\rho$")
    flat_axes[0].set_ylabel(r"$R^2$ drop from clean baseline")
    figure.legend(
        handles=_publication_model_handles(plt, models),
        loc="outside lower center", ncol=3,
    )
    _add_corner_panel_labels(axes)
    saved = _save_publication_figure(figure, path.parent, path.stem)
    plt.close(figure)
    return saved


# --- Part I.2 data preparation and appendix panels --------------------------

def _read_part_i2_rows(
    path: Path, columns: tuple[str, ...]
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError("Part I.2 plot input columns do not match schema")
        return list(reader)


def _read_representative_pdp_curves(path: Path) -> list[dict[str, str]]:
    """Stream only the appendix PDP examples from the full curve export."""
    selected: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PART_I2_PDP_CURVE_COLUMNS:
            raise ValueError("Part I.2 PDP curve columns do not match schema")
        for row in reader:
            if (
                (row["source"], row["dataset_id"]) in PDP_REPRESENTATIVE_DATASETS
                and row["model_id"] in ("tabdpt_retrieval_off", "tabpfn_v2")
                and math.isclose(
                    float(row["requested_irrelevant_ratio"]),
                    PDP_REPRESENTATIVE_RATIO,
                )
                and int(row["transform_repeat"]) == PDP_REPRESENTATIVE_REPEAT
            ):
                selected.append(row)
    return selected


def _representative_pdp_feature_label(
    source: str, role: str, source_feature_index: int, feature_bank_id: int
) -> str:
    if source == "synthetic":
        return (
            rf"Relevant $x_{{{source_feature_index + 1}}}$"
            if role == "active"
            else rf"Added $z_{{{feature_bank_id + 1}}}$"
        )
    source_label = rf"$x_{{{source_feature_index + 1}}}$"
    if role == "original":
        return rf"Original {source_label}"
    return rf"Added $z_{{{feature_bank_id + 1}}}$ from {source_label}"


def _plot_representative_pdp_curves(
    rows: list[dict[str, str]], output_dir: Path
) -> tuple[Path, ...]:
    """Plot column-resolved PDPs for one Friedman and one OpenML task."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _openml_catalog_dimensions()
    paths: list[Path] = []
    models = ("tabdpt_retrieval_off", "tabpfn_v2")
    feature_palette = PUBLICATION_FEATURE_COLORS

    for source, dataset_id in PDP_REPRESENTATIVE_DATASETS:
        roles = ("active" if source == "synthetic" else "original", "irrelevant")
        dataset_rows = [
            row for row in rows
            if row["source"] == source and row["dataset_id"] == dataset_id
        ]
        if not dataset_rows:
            raise ValueError(f"PDP example is missing: {source}/{dataset_id}")
        dataset_name = dataset_rows[0]["dataset_name"]
        source_ids = sorted({
            int(row["source_feature_index"])
            for row in dataset_rows
            if int(row["source_feature_index"]) >= 0
        })
        colors = {
            source_id: feature_palette[index % len(feature_palette)]
            for index, source_id in enumerate(source_ids)
        }

        figure, axes = plt.subplots(
            2, 2, figsize=(PUBLICATION_WIDTH_IN, 4.4), sharex=True,
            sharey="col", squeeze=False, layout="constrained",
        )
        for model_index, model_id in enumerate(models):
            for role_index, role in enumerate(roles):
                axis = axes[model_index, role_index]
                panel_rows = [
                    row for row in dataset_rows
                    if row["model_id"] == model_id
                    and row["feature_role"] == role
                ]
                if not panel_rows:
                    raise ValueError(
                        f"PDP panel is missing: {dataset_id}/{model_id}/{role}"
                    )
                by_feature: dict[
                    tuple[int, int, int], dict[int, float]
                ] = {}
                for row in panel_rows:
                    identity = (
                        int(row["source_feature_index"]),
                        int(row["copy_block_id"]),
                        int(row["feature_bank_id"]),
                    )
                    by_feature.setdefault(identity, {})[
                        int(row["grid_index"])
                    ] = float(row["pdp_value"])

                feature_order = sorted(
                    by_feature,
                    key=(
                        (lambda item: (item[2], item[0], item[1]))
                        if role == "irrelevant"
                        else (lambda item: (item[0], item[2], item[1]))
                    ),
                )
                for source_id, _copy_block, bank_id in feature_order:
                    curve = by_feature[(source_id, _copy_block, bank_id)]
                    if set(curve) != set(range(len(PDP_QUANTILE_LABELS))):
                        continue
                    values = np.asarray(
                        [curve[index] for index in range(len(PDP_QUANTILE_LABELS))],
                        dtype=np.float64,
                    )
                    values = values - values[len(values) // 2]
                    axis.plot(
                        np.arange(len(values)),
                        values,
                        color=(
                            colors[source_id] if source == "real" or role == "active"
                            else feature_palette[bank_id % len(feature_palette)]
                        ),
                        linewidth=PUBLICATION_LINEWIDTH,
                        marker="o",
                        markersize=PUBLICATION_MARKERSIZE,
                        markeredgecolor="white",
                        markeredgewidth=0.25,
                        label=_representative_pdp_feature_label(
                            source, role, source_id, bank_id
                        ),
                    )
                axis.axhline(0.0, color="0.55", linewidth=0.55, linestyle="--")
                axis.grid(False)
                axis.set_axisbelow(True)
                role_title = {
                    "active": "relevant columns",
                    "original": "original columns",
                    "irrelevant": "added null columns",
                }[role]
                axis.set_title(f"{MODEL_LABELS[model_id]}: {role_title}")
                if model_index == len(models) - 1:
                    axis.set_xlabel("Feature quantile (%)")
                if role_index == 0:
                    axis.set_ylabel(r"Centered PDP response")
                axis.set_xticks(
                    np.arange(len(PDP_QUANTILE_LABELS)), PDP_QUANTILE_LABELS
                )
                # The five entries are column identities, not models, so they
                # stay per panel; a fixed corner and two columns keep them out
                # of the curves and halve the block height.
                # No in-panel legend: the PDP fans cross at the centre
                # quantile and cover every corner, so an inset legend always
                # lands on a curve. Both rows of a column share the same
                # column identities, so one legend per column carries them.

        # One shared legend per column, outside the panels: the left column
        # lists the relevant/original columns and the right column the added
        # null ones, identically for both models.
        for column_index, location in enumerate(
            ("outside lower left", "outside lower right")
        ):
            handles, labels = axes[0, column_index].get_legend_handles_labels()
            if handles:
                # Three columns only fit short "Relevant x1" style labels; the
                # longer "Added z1 from x8" labels overrun into the next
                # column's handle, so they drop to two columns.
                longest = max(len(label) for label in labels)
                figure.legend(
                    handles, labels, loc=location,
                    ncol=3 if longest <= 10 else 2,
                    fontsize=PUBLICATION_MATH_FONTSIZE, frameon=False,
                    handlelength=1.2, handletextpad=0.4, columnspacing=1.4,
                    labelspacing=0.3,
                )
        if source == "synthetic":
            figure.suptitle(
                f"Friedman task {dataset_id.rsplit('-', 1)[-1]} "
                r"($N=1000$, $\sigma=1$, $\rho=7$)"
            )
        else:
            d_base = metadata.get(dataset_id, (dataset_name, len(source_ids)))[1]
            figure.suptitle(
                f"{dataset_name} (OpenML {dataset_id.rsplit('-', 1)[-1]}, "
                rf"$d_{{\mathrm{{base}}}}={d_base}$, $\rho=7$)"
            )
        _add_corner_panel_labels(axes)
        paths.append(_save_publication_figure(
            figure, output_dir, f"appendix_part_i2_pdp_curves_{dataset_id}"
        ))
        plt.close(figure)
    return tuple(paths)


def _part_i2_panel_role(source: str, role: str) -> str:
    if role == "irrelevant":
        return "Null features"
    return "Relevant features" if source == "synthetic" else "Original features"


def _plot_part_i2_openml_dimension_metric(
    rows: list[dict[str, str]],
    dimensions: list[dict[str, str]],
    metric: str,
    path: Path,
) -> Path | None:
    """Plot OpenML functional sensitivity by pre-augmentation width tier."""
    import matplotlib.pyplot as plt

    dimension_map = {
        row["dataset_id"]: int(row["d_base"])
        for row in dimensions if row["source"] == "real"
    }
    selected = [
        row for row in rows
        if row["source"] == "real"
        and row["metric"] == metric
        and row["dataset_id"] in dimension_map
    ]
    if not selected:
        return None
    models = [
        model_id for model_id in MODEL_ORDER
        if any(row["model_id"] == model_id for row in selected)
    ]
    ratios = sorted({
        float(row["requested_irrelevant_ratio"]) for row in selected
    })
    positions = np.asarray(ratios, dtype=np.float64)
    model_colors = _model_colors()
    roles = ("original", "irrelevant")
    figure, axes = plt.subplots(
        len(roles), len(OPENML_DIMENSION_TIERS),
        figsize=(PUBLICATION_WIDTH_IN, 3.9),
        sharex=True, sharey="row", squeeze=False,
        layout="constrained",
    )
    for row_index, role in enumerate(roles):
        for column_index, (tier, _label, _lower, _upper) in enumerate(
            OPENML_DIMENSION_TIERS
        ):
            axis = axes[row_index, column_index]
            tier_datasets = {
                row["dataset_id"] for row in selected
                if row["feature_role"] == role
                and _openml_tier(dimension_map[row["dataset_id"]]) == tier
            }
            for model_id in models:
                medians, q25s, q75s = [], [], []
                for ratio in ratios:
                    values = [
                        float(row["value"]) for row in selected
                        if row["feature_role"] == role
                        and row["model_id"] == model_id
                        and float(row["requested_irrelevant_ratio"]) == ratio
                        and _openml_tier(dimension_map[row["dataset_id"]]) == tier
                    ]
                    median, q25, q75 = _median_iqr(values)
                    medians.append(median)
                    q25s.append(q25)
                    q75s.append(q75)
                median_array = np.asarray(medians)
                q25_array = np.asarray(q25s)
                q75_array = np.asarray(q75s)
                finite = (
                    np.isfinite(median_array)
                    & np.isfinite(q25_array)
                    & np.isfinite(q75_array)
                )
                _publication_band(
                    axis, positions[finite], q25_array[finite],
                    q75_array[finite], color=model_colors[model_id],
                )
                axis.plot(
                    positions[finite], median_array[finite],
                    color=model_colors[model_id],
                    linewidth=PUBLICATION_LINEWIDTH,
                    marker=MODEL_MARKERS.get(model_id, "o"),
                    markersize=PUBLICATION_MARKERSIZE,
                    markeredgecolor="white", markeredgewidth=0.3,
                    zorder=3, label=MODEL_LABELS[model_id],
                )
            if row_index == 0:
                name, bound = OPENML_DIMENSION_TIER_SHORT[tier]
                axis.set_title(
                    f"{name}\n{bound} ($n={len(tier_datasets)}$)",
                    fontsize=PUBLICATION_MATH_FONTSIZE,
                )
            _publication_dose_axis(axis, ratios)
            if row_index == len(roles) - 1:
                axis.set_xlabel(r"Null-feature ratio $\rho$")
            if column_index == 0:
                metric_label = (
                    "Normalized prediction shift"
                    if metric == "permutation_shift"
                    else "Normalized PDP variation"
                )
                axis.set_ylabel(
                    f"{'Original' if role == 'original' else 'Null'} features\n"
                    f"{metric_label}"
                )
    figure.legend(
        handles=_publication_model_handles(plt, models),
        loc="outside lower center", ncol=min(len(models), 3),
    )
    _add_corner_panel_labels(axes)
    saved = _save_publication_figure(figure, path.parent, path.stem)
    plt.close(figure)
    return saved


# --- Shared formal-figure curve primitives ---------------------------------

def _publication_models(
    rows: list[dict[str, str]], allowed: tuple[str, ...] | None = None
) -> list[str]:
    present = {row["model_id"] for row in rows}
    order = allowed or MODEL_ORDER
    models = [model_id for model_id in order if model_id in present]
    return models + sorted(present - set(models)) if allowed is None else models


def _publication_model_handles(
    plt, models: list[str], labels: dict[str, str] | None = None
):
    from matplotlib.lines import Line2D

    colors = _model_colors()
    overrides = labels or {}
    return [
        Line2D(
            [0], [0], color=colors[model_id], linewidth=1.4,
            marker=MODEL_MARKERS.get(model_id, "o"), markersize=3.2,
            label=overrides.get(
                model_id, MODEL_LABELS.get(model_id, model_id)
            ),
        )
        for model_id in models
    ]


def _ratio_tick_label(value: float) -> str:
    label = format(value, "g")
    return label[1:] if 0.0 < value < 1.0 and label.startswith("0") else label


def _publication_ratio_ticks(axis, ratios: list[float]) -> None:
    """Label the dose grid without letting the small doses collide.

    The smallest dose is always labelled so the left end of every panel is
    identifiable; any remaining sub-1 dose stays an unlabelled minor tick.
    """
    labelled = {min(ratios)} | {
        value for value in ratios if value in PUBLICATION_RATIO_MAJOR_TICKS
    }
    major = [value for value in ratios if value in labelled]
    minor = [value for value in ratios if value not in labelled]
    axis.set_xticks(major, [_ratio_tick_label(value) for value in major])
    axis.set_xticks(minor, minor=True)


def _publication_dose_axis(axis, ratios: list[float]) -> None:
    axis.axhline(
        0.0, color="0.45", linewidth=0.55, linestyle="--", zorder=0,
    )
    _publication_ratio_ticks(axis, ratios)
    # No y-grid: these panels carry inset condition legends, and a full-width
    # gridline runs straight through their labels. Tick marks carry value
    # reading without that collision.
    axis.grid(False)
    axis.set_xlim(min(ratios) - 0.12, max(ratios) + 0.12)
    axis.set_axisbelow(True)
    axis.margins(y=0.08)


def _plot_publication_condition_curves(
    axis,
    rows: list[dict[str, str]],
    *,
    level_column: str,
    reverse_levels: bool,
    condition_title: str,
    median_column: str,
    q25_column: str,
    q75_column: str,
    models: list[str],
    show_condition_legend: bool = True,
    condition_legend_loc: str = "upper left",
    condition_legend_bbox: tuple[float, float] | None = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    levels = sorted(
        {float(row[level_column]) for row in rows}, reverse=reverse_levels
    )
    ratios = sorted({float(row["requested_irrelevant_ratio"]) for row in rows})
    x = np.asarray(ratios, dtype=np.float64)
    strengths = np.linspace(0.42, 1.0, len(levels))
    linewidths = np.linspace(0.55, 1.35, len(levels))
    marker_sizes = np.linspace(1.9, 3.2, len(levels))
    colors = _model_colors()
    for model_index, model_id in enumerate(models):
        base_color = np.asarray(colors[model_id][:3])
        lookup = {
            (float(row[level_column]), float(row["requested_irrelevant_ratio"])): row
            for row in rows if row["model_id"] == model_id
        }
        for level_index, (level, strength, linewidth, marker_size) in enumerate(
            zip(levels, strengths, linewidths, marker_sizes)
        ):
            selected = [lookup.get((level, ratio)) for ratio in ratios]
            median = np.asarray([
                float(row[median_column]) if row is not None else np.nan
                for row in selected
            ])
            q25 = np.asarray([
                float(row[q25_column]) if row is not None else np.nan
                for row in selected
            ])
            q75 = np.asarray([
                float(row[q75_column]) if row is not None else np.nan
                for row in selected
            ])
            finite = np.isfinite(median) & np.isfinite(q25) & np.isfinite(q75)
            color = tuple(1.0 - (1.0 - base_color) * strength)
            # Four condition levels per model means eight boundary lines per
            # model, so these bands stay wash-only.
            _publication_band(
                axis, x[finite], q25[finite], q75[finite],
                color=color, zorder=1 + level_index, edges=False,
            )
            axis.plot(
                x[finite], median[finite], color=color,
                linewidth=float(linewidth),
                marker=MODEL_MARKERS.get(model_id, "o"),
                markersize=float(marker_size), markeredgecolor="white",
                markeredgewidth=0.3,
                zorder=2 + level_index + model_index * 0.01,
            )
    if show_condition_legend:
        handles = []
        for level, strength, linewidth, marker_size in zip(
            levels, strengths, linewidths, marker_sizes
        ):
            gray = str(1.0 - 0.68 * strength)
            handles.append(Line2D(
                [0], [0], color=gray, linewidth=float(linewidth), marker="o",
                markersize=float(marker_size), markeredgecolor="white",
                markeredgewidth=0.3, label=format(level, "g"),
            ))
        # No add_artist() here: axis.legend() already attaches the legend, and
        # re-adding it draws every entry twice. That duplicate is invisible in
        # a 220 dpi raster but is a real overlapping text run in vector output.
        # This legend sits inside the panel, where the y-grid would otherwise
        # run straight through its labels, so it carries an opaque white patch
        # with no visible edge and is drawn above the grid.
        legend = axis.legend(
            handles=handles, title=condition_title, loc=condition_legend_loc,
            bbox_to_anchor=condition_legend_bbox,
            frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
            handlelength=1.8, fontsize=6.5, title_fontsize=6.5,
            labelspacing=0.28, handletextpad=0.5, borderaxespad=0.3,
            borderpad=0.3,
        )
        legend.get_title().set_multialignment("left")
        legend.set_zorder(20)
    _publication_dose_axis(axis, ratios)


def _dataset_median_iqr(
    rows: list[dict[str, str]], value_column: str
) -> dict[tuple[str, float], tuple[float, float, float]]:
    by_dataset: dict[tuple[str, float, str], list[float]] = {}
    for row in rows:
        value = float(row[value_column])
        if np.isfinite(value):
            key = (
                row["model_id"],
                float(row["requested_irrelevant_ratio"]),
                row["dataset_id"],
            )
            by_dataset.setdefault(key, []).append(value)
    grouped: dict[tuple[str, float], list[float]] = {}
    for (model_id, ratio, _dataset_id), values in by_dataset.items():
        grouped.setdefault((model_id, ratio), []).append(float(np.mean(values)))
    return {key: _median_iqr(values) for key, values in grouped.items()}


def _plot_publication_dataset_curves(
    axis,
    rows: list[dict[str, str]],
    *,
    value_column: str,
    models: list[str],
) -> None:
    import matplotlib.pyplot as plt

    colors = _model_colors()
    ratios = sorted({float(row["requested_irrelevant_ratio"]) for row in rows})
    summary = _dataset_median_iqr(rows, value_column)
    x = np.asarray(ratios, dtype=np.float64)
    for model_index, model_id in enumerate(models):
        values = [summary.get((model_id, ratio), (np.nan,) * 3) for ratio in ratios]
        median = np.asarray([value[0] for value in values])
        q25 = np.asarray([value[1] for value in values])
        q75 = np.asarray([value[2] for value in values])
        finite = np.isfinite(median) & np.isfinite(q25) & np.isfinite(q75)
        _publication_band(
            axis, x[finite], q25[finite], q75[finite],
            color=colors[model_id], zorder=1 + model_index * 0.01,
        )
        axis.plot(
            x[finite], median[finite], color=colors[model_id],
            linewidth=PUBLICATION_LINEWIDTH,
            marker=MODEL_MARKERS.get(model_id, "o"),
            markersize=PUBLICATION_MARKERSIZE,
            markeredgecolor="white", markeredgewidth=0.3,
            zorder=3 + model_index * 0.01,
        )
    _publication_dose_axis(axis, ratios)


def _plot_publication_summary_curves(
    axis,
    rows: list[dict[str, str]],
    *,
    prefix: str,
    models: list[str],
) -> None:
    import matplotlib.pyplot as plt

    colors = _model_colors()
    ratios = sorted({float(row["requested_irrelevant_ratio"]) for row in rows})
    x = np.asarray(ratios, dtype=np.float64)
    for model_index, model_id in enumerate(models):
        lookup = {
            float(row["requested_irrelevant_ratio"]): row
            for row in rows if row["model_id"] == model_id
        }
        selected = [lookup.get(ratio) for ratio in ratios]
        median = np.asarray([
            float(row[f"{prefix}_median"]) if row is not None else np.nan
            for row in selected
        ])
        finite = np.isfinite(median)
        # Medians only. The across-dataset interquartile range describes the
        # heterogeneity of the benchmark collection, not the comparison this
        # panel makes: on OpenML it spans roughly a full unit of R2 while the
        # five systems sit within about 0.03 of one another, so plotting it
        # buries the very difference the panel exists to show. Paired
        # dispersion for the effect itself is reported in the dose-response
        # figure, which subtracts each system's own clean baseline.
        axis.plot(
            x[finite], median[finite], color=colors[model_id],
            linewidth=PUBLICATION_LINEWIDTH,
            marker=MODEL_MARKERS.get(model_id, "o"),
            markersize=PUBLICATION_MARKERSIZE,
            markeredgecolor="white", markeredgewidth=0.3,
            zorder=3 + model_index * 0.01,
        )
    _publication_ratio_ticks(axis, ratios)
    axis.set_xlim(min(ratios) - 0.12, max(ratios) + 0.12)
    axis.grid(False)
    axis.set_axisbelow(True)
    axis.margins(y=0.08)


def _write_publication_csv(
    rows: list[dict[str, str]], path: Path, columns: tuple[str, ...]
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# --- Main-text Part I overview ---------------------------------------------

def _dataset_medians(
    rows: list[dict[str, str]],
    *,
    value_column: str,
    source: str,
    model_id: str,
    ratios: list[float],
) -> list[float]:
    """Median across datasets of the per-dataset mean over repeats."""
    by_dataset: dict[tuple[float, str], list[float]] = {}
    for row in rows:
        if row["source"] != source or row["model_id"] != model_id:
            continue
        value = float(row[value_column])
        if np.isfinite(value):
            key = (float(row["requested_irrelevant_ratio"]), row["dataset_id"])
            by_dataset.setdefault(key, []).append(value)
    grouped: dict[float, list[float]] = {}
    for (ratio, _dataset), values in by_dataset.items():
        grouped.setdefault(ratio, []).append(float(np.mean(values)))
    return [
        float(np.median(grouped[ratio])) if grouped.get(ratio) else float("nan")
        for ratio in ratios
    ]


def _win_fraction(
    contrasts: list[dict[str, str]],
    *,
    delta_column: str,
    source: str | None,
    ratios: list[float],
    predicate=None,
) -> list[float]:
    """Percentage of datasets where model_a exceeds model_b.

    Averaging repeats first keeps one vote per dataset, so a task with noisy
    repeats cannot outvote a task with clean ones. ``predicate`` selects the
    metric and feature role for the Part I.2 contrast files, which stack
    several quantities in one table.
    """
    by_dataset: dict[tuple[float, str], list[float]] = {}
    for row in contrasts:
        if source is not None and row["source"] != source:
            continue
        if predicate is not None and not predicate(row):
            continue
        delta = float(row[delta_column])
        if np.isfinite(delta):
            key = (float(row["requested_irrelevant_ratio"]), row["dataset_id"])
            by_dataset.setdefault(key, []).append(delta)
    grouped: dict[float, list[float]] = {}
    for (ratio, _dataset), values in by_dataset.items():
        grouped.setdefault(ratio, []).append(float(np.mean(values)))
    return [
        100.0 * float(np.mean([value > 0.0 for value in grouped[ratio]]))
        if grouped.get(ratio) else float("nan")
        for ratio in ratios
    ]


def _is_irrelevant_permutation(row: dict[str, str]) -> bool:
    return (
        row["metric"] == "permutation_shift"
        and row["feature_role"] == "irrelevant"
    )


def _part_i_tile(plt, *, top_row: bool):
    """Create one Figure 1 tile with the shared fixed plot rectangle.

    No layout engine is used: every tile places its axes at the same inch
    offsets, so the four separately stored files line up exactly when LaTeX
    tiles them back into a 2 x 2 grid.
    """
    top, bottom = (
        PART_I_TOP_TILE_MARGINS_IN if top_row else PART_I_BOTTOM_TILE_MARGINS_IN
    )
    # Every y label is a single rotated line, so one left margin serves both
    # rows; the wider margin the lower row used to need is no longer required.
    left = PART_I_TILE_LEFT_IN
    width = PART_I_TILE_WIDTH_IN
    height = top + PART_I_TILE_PLOT_HEIGHT_IN + bottom
    figure = plt.figure(figsize=(width, height))
    axis = figure.add_axes([
        left / width,
        bottom / height,
        (width - left - PART_I_TILE_RIGHT_IN) / width,
        PART_I_TILE_PLOT_HEIGHT_IN / height,
    ])
    axis.tick_params(labelsize=PART_I_TILE_TICK_PT)
    return figure, axis


def _plot_part_i_overview(
    effects: list[dict[str, str]],
    contrasts: list[dict[str, str]],
    part_i2_contrasts: list[dict[str, str]],
    part_i2_synthetic: list[dict[str, str]],
    part_i2_real: list[dict[str, str]],
    models: list[str],
    path: Path,
    functional_path: Path,
) -> tuple[Path, ...]:
    """Draw the two main-text Part I figures as separately stored panels.

    The panels carry four different roles rather than four metrics. The
    overview figure establishes the dose response on both benchmarks; the
    functional figure shows the effect holds on almost every individual task
    rather than being carried by a few, and links it to functional dependence
    on the added coordinates. Each panel is saved as ``<stem>_<letter>`` so
    the paper can place them independently, and each figure letters its own
    panels from (a) because the two are numbered separately in the text.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = _model_colors()
    saved: list[Path] = []
    # Panels (a, b) show the systems the running text names; the complete
    # comparison including TabPFN v2.5 stays in the appendix.
    overview_models = [
        model_id for model_id in models
        if model_id not in PART_I_OVERVIEW_EXCLUDED_MODELS
    ]

    for letter, source in zip("ab", PART_I2_SOURCES):
        figure, axis = _part_i_tile(plt, top_row=True)
        source_rows = [row for row in effects if row["source"] == source]
        ratios = sorted({
            float(row["requested_irrelevant_ratio"]) for row in source_rows
        })
        x = np.asarray(ratios, dtype=np.float64)
        for model_index, model_id in enumerate(overview_models):
            median = np.asarray(_dataset_medians(
                source_rows, value_column="r2_drop", source=source,
                model_id=model_id, ratios=ratios,
            ))
            finite = np.isfinite(median)
            axis.plot(
                x[finite], median[finite], color=colors[model_id],
                linewidth=PUBLICATION_LINEWIDTH,
                marker=MODEL_MARKERS.get(model_id, "o"),
                markersize=PUBLICATION_MARKERSIZE,
                markeredgecolor="white", markeredgewidth=0.3,
                zorder=3 + model_index * 0.01,
            )
        _publication_dose_axis(axis, ratios)
        # The unit counts move to the caption: the full title overruns a
        # 1.35 in tile.
        axis.set_title(
            PART_I_BENCHMARKS[source], fontsize=PART_I_TILE_LABEL_PT
        )
        axis.set_xlabel(
            r"Null-feature ratio $\rho$", fontsize=PART_I_TILE_LABEL_PT
        )
        if letter == "a":
            # Held at the mathtext size so the $R^2$ superscript, which
            # renders at 0.7x, clears the 5 pt floor.
            axis.set_ylabel(
                r"$R^2$ drop", fontsize=PUBLICATION_MATH_FONTSIZE
            )
            # The model key lives in (a): its upper left is empty because
            # only TabDPT leaves the floor, and only at the larger doses.
            axis.legend(
                handles=_publication_model_handles(
                    plt, overview_models, labels=PART_I_TILE_SHORT_LABELS
                ),
                # Tightened after the plot was flattened: at the previous
                # size the four rows reached far enough down that the TabDPT
                # curve crossed the bottom entry.
                loc="upper left", frameon=False, borderaxespad=0.25,
                labelspacing=0.15, handlelength=1.0, handletextpad=0.35,
                fontsize=PART_I_TILE_LEGEND_SMALL_PT,
            )
        _add_corner_panel_labels([axis], first=letter)
        saved.append(_save_publication_figure(
            figure, path.parent, f"{path.stem}_{letter}"
        ))
        plt.close(figure)

    # Both metrics on one axis: colour is the metric, line style stays the
    # benchmark. This is where Experiment I.1 and I.2 meet, so the paired
    # statement in the text has a panel behind it for both quantities.
    figure, axis = _part_i_tile(plt, top_row=False)
    for source in ("real", "synthetic"):
        ratios = sorted({
            float(row["requested_irrelevant_ratio"]) for row in contrasts
            if row["source"] == source
            and float(row["requested_irrelevant_ratio"]) > 0.0
        })
        series = (
            (PART_I_METRIC_COLORS["r2_drop"],
             _win_fraction(contrasts, delta_column="delta_r2_drop",
                           source=source, ratios=ratios)),
            (PART_I_METRIC_COLORS["permutation_shift"],
             _win_fraction(part_i2_contrasts, delta_column="delta_a_minus_b",
                           source=source, ratios=ratios,
                           predicate=_is_irrelevant_permutation)),
        )
        for color, fraction in series:
            axis.plot(
                ratios, fraction, color=color,
                linewidth=PUBLICATION_LINEWIDTH,
                linestyle=PART_I_BENCHMARK_STYLE[source], marker="o",
                markersize=PUBLICATION_MARKERSIZE, markeredgecolor="white",
                markeredgewidth=0.3,
            )
    axis.axhline(50.0, color="0.45", linewidth=0.55, linestyle=":", zorder=1)
    _publication_dose_axis(axis, ratios)
    axis.set_ylim(45.0, 103.0)
    # "worse", not "exceeds": a larger value means TabDPT degrades more for
    # both metrics plotted here, which the comparative reading inverts.
    # A single rotated line has to fit the 0.78 in plot height, about 15
    # characters at this size, so the comparator itself is left to the caption
    # and to (b)'s model key rather than named on the axis.
    axis.set_ylabel(
        "TabDPT worse (%)",
        fontsize=PART_I_TILE_LABEL_PT,
    )
    axis.set_xlabel(
        r"Null-feature ratio $\rho$", fontsize=PART_I_TILE_LABEL_PT
    )
    # Only the metric needs a key here; the caption carries the solid/dashed
    # benchmark convention, which (b) reuses.
    axis.legend(
        handles=[
            Line2D([0], [0], color=PART_I_METRIC_COLORS["r2_drop"],
                   linewidth=PUBLICATION_LINEWIDTH, label=r"$R^2$ drop"),
            Line2D([0], [0],
                   color=PART_I_METRIC_COLORS["permutation_shift"],
                   linewidth=PUBLICATION_LINEWIDTH, label="Null shift"),
        ],
        # Held at the mathtext size so the $R^2$ superscript clears the 5 pt
        # floor, which is also why the second entry is abbreviated: the full
        # "null-feature shift" is wider than this 0.89 in plot at 7.5 pt.
        # Lifted clear of the chance line that "lower right" would sit on.
        loc="lower right", bbox_to_anchor=(1.0, 0.10),
        fontsize=PUBLICATION_MATH_FONTSIZE, frameon=False,
        borderaxespad=0.3, labelspacing=0.25,
        handlelength=1.1, handletextpad=0.4,
    )
    _add_corner_panel_labels([axis], first="a")
    saved.append(_save_publication_figure(
        figure, functional_path.parent, f"{functional_path.stem}_a"
    ))
    plt.close(figure)

    figure, axis = _part_i_tile(plt, top_row=False)
    for source, source_rows in (
        ("real", part_i2_real), ("synthetic", part_i2_synthetic)
    ):
        selected = [
            row for row in source_rows
            if row["metric"] == "permutation_shift"
            and row["feature_role"] == "irrelevant"
        ]
        ratios = sorted({
            float(row["requested_irrelevant_ratio"]) for row in selected
        })
        for model_id in PART_I2_MODELS:
            median = np.asarray(_dataset_medians(
                selected, value_column="value", source=source,
                model_id=model_id, ratios=ratios,
            ))
            finite = np.isfinite(median)
            axis.plot(
                np.asarray(ratios)[finite], median[finite],
                color=colors[model_id], linewidth=PUBLICATION_LINEWIDTH,
                linestyle=PART_I_BENCHMARK_STYLE[source],
                marker=MODEL_MARKERS.get(model_id, "o"),
                markersize=PUBLICATION_MARKERSIZE, markeredgecolor="white",
                markeredgewidth=0.3, zorder=3,
            )
    _publication_dose_axis(axis, ratios)
    axis.set_ylabel(
        "Null-feature shift", fontsize=PART_I_TILE_LABEL_PT
    )
    axis.set_xlabel(
        r"Null-feature ratio $\rho$", fontsize=PART_I_TILE_LABEL_PT
    )
    # Unlike (a), colour here is the model, not the metric, so this panel
    # needs its own key: nothing else in the figure names the two systems.
    axis.legend(
        handles=_publication_model_handles(
            plt, list(PART_I2_MODELS), labels=PART_I_TILE_SHORT_LABELS
        ),
        # Upper right: TabDPT decays across the sweep and TabPFN v2 sits near
        # the floor, so the top right is the one corner both curves vacate.
        loc="upper right", frameon=False, borderaxespad=0.25,
        labelspacing=0.15, handlelength=1.0, handletextpad=0.35,
        fontsize=PART_I_TILE_LEGEND_SMALL_PT,
    )
    _add_corner_panel_labels([axis], first="b")
    saved.append(_save_publication_figure(
        figure, functional_path.parent, f"{functional_path.stem}_b"
    ))
    plt.close(figure)
    return tuple(saved)


# --- TabDPT release appendix ------------------------------------------------

def _version_dataset_medians(
    rows: list[dict[str, str]], value_column: str
) -> dict[tuple[str, float], list[tuple[str, float]]]:
    """Average repeats per dataset, keeping dataset identity for pairing."""

    by_dataset: dict[tuple[str, float, str], list[float]] = {}
    for row in rows:
        value = float(row[value_column])
        if not np.isfinite(value):
            continue
        key = (
            row["model_id"], float(row["requested_irrelevant_ratio"]),
            row["dataset_id"],
        )
        by_dataset.setdefault(key, []).append(value)
    grouped: dict[tuple[str, float], list[tuple[str, float]]] = {}
    for (model_id, ratio, dataset_id), values in by_dataset.items():
        grouped.setdefault((model_id, ratio), []).append(
            (dataset_id, float(np.mean(values)))
        )
    return grouped


def _version_exceedance_fraction(
    rows: list[dict[str, str]],
    *,
    model_id: str,
    baseline_id: str,
    ratio: float,
    value_column: str,
) -> tuple[float, int]:
    """Percentage of shared datasets where the model has a larger R2 drop."""

    grouped = _version_dataset_medians(rows, value_column)
    model = dict(grouped.get((model_id, ratio), []))
    baseline = dict(grouped.get((baseline_id, ratio), []))
    shared = sorted(set(model) & set(baseline))
    if not shared:
        return float("nan"), 0
    exceedances = [model[key] > baseline[key] for key in shared]
    return 100.0 * float(np.mean(exceedances)), len(shared)


def _version_median_iqr(
    rows: list[dict[str, str]],
    *,
    model_id: str,
    ratio: float,
    value_column: str,
) -> tuple[float, float, float, int]:
    grouped = _version_dataset_medians(rows, value_column)
    values = [value for _dataset, value in grouped.get((model_id, ratio), [])]
    if not values:
        return float("nan"), float("nan"), float("nan"), 0
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.median(array)),
        float(np.quantile(array, 0.25)),
        float(np.quantile(array, 0.75)),
        len(values),
    )


def _version_handles(plt, models: tuple[str, ...]):
    from matplotlib.lines import Line2D

    colors = _model_colors()
    return [
        Line2D(
            [0], [0], color=colors[model_id], linewidth=1.4,
            marker=MODEL_MARKERS.get(model_id, "o"), markersize=3.2,
            label=TABDPT_VERSION_LABELS.get(model_id, model_id),
        )
        for model_id in models
    ]


class DefaultTabDPTVersionPlotter:
    """Appendix figures and table for the TabDPT release replication.

    The four releases share one red ramp in release order and TabPFN v2 is
    drawn as a reference in its main-text navy, so each panel answers whether
    every release degrades and how far it sits from the robust comparator.
    """

    def plot(
        self,
        *,
        part_i1_effects: list[dict[str, str]],
        part_i2_friedman: list[dict[str, str]],
        part_i2_openml: list[dict[str, str]],
        output_root: Path,
    ) -> dict[str, object]:
        import matplotlib as mpl

        with mpl.rc_context(PUBLICATION_RC):
            return self._build(
                part_i1_effects=part_i1_effects,
                part_i2_friedman=part_i2_friedman,
                part_i2_openml=part_i2_openml,
                output_root=Path(output_root).resolve(),
            )

    def _build(
        self,
        *,
        part_i1_effects: list[dict[str, str]],
        part_i2_friedman: list[dict[str, str]],
        part_i2_openml: list[dict[str, str]],
        output_root: Path,
    ) -> dict[str, object]:
        import matplotlib.pyplot as plt

        figures_dir = output_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        models = TABDPT_VERSION_MODELS + (TABDPT_VERSION_BASELINE,)
        paths: list[Path] = []

        # Figure 1: Part I.1 dose response over the clean-task baseline.
        figure, axes = plt.subplots(
            2, 2, figsize=(PUBLICATION_WIDTH_IN, 4.3), squeeze=False,
            layout="constrained",
        )
        raw_summary = _raw_r2_summary(part_i1_effects)
        for column, (source, title) in enumerate(
            (("synthetic", "Friedman (350 task-settings)"),
             ("real", "OpenML (42 datasets)"))
        ):
            drop_axis = axes[0][column]
            _plot_publication_dataset_curves(
                drop_axis,
                [row for row in part_i1_effects if row["source"] == source],
                value_column="r2_drop", models=list(models),
            )
            drop_axis.set_title(title)
            raw_axis = axes[1][column]
            _plot_publication_summary_curves(
                raw_axis,
                [row for row in raw_summary if row["source"] == source],
                prefix="r2_at_dose", models=list(models),
            )
            raw_axis.set_xlabel(r"Null-feature ratio $\rho$")
        axes[0][0].set_ylabel(r"Median $R^2$ drop")
        axes[1][0].set_ylabel(r"Median $R^2$")
        axes[0][1].sharey(axes[0][0])
        figure.legend(
            handles=_version_handles(plt, models),
            loc="outside lower center", ncol=3,
        )
        _add_corner_panel_labels(axes)
        paths.append(_save_publication_figure(
            figure, figures_dir, "appendix_part_i1_tabdpt_versions_r2_drop"
        ))
        plt.close(figure)

        # Figure 2: Part I.2 permutation shift with its positive controls.
        figure, axes = plt.subplots(
            1, 4, figsize=(PUBLICATION_WIDTH_IN, 2.3), sharex=True,
            squeeze=False, layout="constrained",
        )
        flat_axes = axes[0]
        flat_axes[1].sharey(flat_axes[0])
        flat_axes[3].sharey(flat_axes[2])
        panels = (
            (part_i2_friedman, "Friedman", "synthetic", "active"),
            (part_i2_openml, "OpenML", "real", "original"),
            (part_i2_friedman, "Friedman", "synthetic", "irrelevant"),
            (part_i2_openml, "OpenML", "real", "irrelevant"),
        )
        for index, (rows, source_label, source, role) in enumerate(panels):
            axis = flat_axes[index]
            selected = [
                row for row in rows
                if row["metric"] == "permutation_shift"
                and row["feature_role"] == role
            ]
            _plot_publication_dataset_curves(
                axis, selected, value_column="value", models=list(models)
            )
            axis.set_title(
                source_label + "\n" + _part_i2_panel_role(source, role)
            )
            axis.set_xlabel(r"Null-feature ratio $\rho$")
            if index in (0, 2):
                axis.set_ylabel("Normalized prediction shift")
            else:
                axis.tick_params(axis="y", labelleft=False)
        figure.legend(
            handles=_version_handles(plt, models),
            loc="outside lower center", ncol=3,
        )
        _add_corner_panel_labels(axes)
        paths.append(_save_publication_figure(
            figure, figures_dir, "appendix_part_i2_tabdpt_versions_permutation",
            # Same layout as the main Part I.2 appendix row: the wider middle
            # gutter carries the tick labels of the second y scale.
            PUBLICATION_PAIRED_ROW_EXEMPTIONS,
        ))
        plt.close(figure)

        table_path = self._write_table(
            part_i1_effects, part_i2_friedman, part_i2_openml, output_root
        )
        return {"figures": paths, "table": table_path}

    def _write_table(
        self,
        part_i1_effects: list[dict[str, str]],
        part_i2_friedman: list[dict[str, str]],
        part_i2_openml: list[dict[str, str]],
        output_root: Path,
    ) -> Path:
        """Write the maximum-dose summary as CSV plus LaTeX tabular rows."""

        ratio = max(
            float(row["requested_irrelevant_ratio"]) for row in part_i1_effects
        )
        irrelevant = {
            source: [
                row for row in rows
                if row["metric"] == "permutation_shift"
                and row["feature_role"] == "irrelevant"
            ]
            for source, rows in (
                ("synthetic", part_i2_friedman), ("real", part_i2_openml)
            )
        }
        rows: list[dict[str, object]] = []
        for model_id in TABDPT_VERSION_MODELS + (TABDPT_VERSION_BASELINE,):
            entry: dict[str, object] = {
                "model_id": model_id,
                "label": TABDPT_VERSION_LABELS.get(model_id, model_id),
                "requested_irrelevant_ratio": ratio,
            }
            for source, prefix in (("synthetic", "friedman"), ("real", "openml")):
                effects = [
                    row for row in part_i1_effects if row["source"] == source
                ]
                median, q25, q75, count = _version_median_iqr(
                    effects, model_id=model_id, ratio=ratio,
                    value_column="r2_drop",
                )
                entry[prefix + "_r2_drop_median"] = median
                entry[prefix + "_r2_drop_q25"] = q25
                entry[prefix + "_r2_drop_q75"] = q75
                entry[prefix + "_n"] = count
                shift = _version_median_iqr(
                    irrelevant[source], model_id=model_id, ratio=ratio,
                    value_column="value",
                )
                entry[prefix + "_null_shift_median"] = shift[0]
                entry[prefix + "_null_shift_q25"] = shift[1]
                entry[prefix + "_null_shift_q75"] = shift[2]
                exceedance, _shared = _version_exceedance_fraction(
                    effects, model_id=model_id,
                    baseline_id=TABDPT_VERSION_BASELINE, ratio=ratio,
                    value_column="r2_drop",
                )
                entry[prefix + "_exceeds_percent_vs_tabpfn_v2"] = exceedance
            rows.append(entry)

        csv_path = output_root / "tabdpt_versions_max_dose.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        def number(value: float) -> str:
            # Four decimals round the robust comparator's near-zero entries to
            # "0.0000", which reads as an exact zero, so only those keep a
            # fifth digit; everything else matches the other Part I tables.
            digits = 5 if abs(value) < 5e-5 else 4
            return format(value, f".{digits}f").replace("-", "$-$")

        def cell(value: float, low: float, high: float) -> str:
            return number(value) + " [" + number(low) + ", " + number(high) + "]"

        lines = []
        for entry in rows:
            if entry["model_id"] == TABDPT_VERSION_BASELINE:
                win = "---"
            else:
                win = "{:.1f} / {:.1f}".format(
                    entry["friedman_exceeds_percent_vs_tabpfn_v2"],
                    entry["openml_exceeds_percent_vs_tabpfn_v2"],
                )
            lines.append(" & ".join([
                str(entry["label"]),
                cell(
                    entry["friedman_r2_drop_median"],
                    entry["friedman_r2_drop_q25"],
                    entry["friedman_r2_drop_q75"],
                ),
                cell(
                    entry["openml_r2_drop_median"],
                    entry["openml_r2_drop_q25"],
                    entry["openml_r2_drop_q75"],
                ),
                cell(
                    entry["friedman_null_shift_median"],
                    entry["friedman_null_shift_q25"],
                    entry["friedman_null_shift_q75"],
                ),
                cell(
                    entry["openml_null_shift_median"],
                    entry["openml_null_shift_q25"],
                    entry["openml_null_shift_q75"],
                ),
                win,
            ]) + r" \\")
        tex_path = output_root / "tabdpt_versions_max_dose_rows.tex"
        tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return csv_path

# --- Complete Part I publication bundle ------------------------------------

class DefaultPublicationPlotter:
    """Build the compact median/IQR Part I publication evaluation bundle."""

    def plot(
        self,
        *,
        artifact_root: Path,
    ) -> dict[str, tuple[Path, ...]]:
        import matplotlib as mpl

        artifact_root = Path(artifact_root).resolve()
        with mpl.rc_context(PUBLICATION_RC):
            return self._build(
                part_i1_dir=artifact_root / "evaluations_part_i1",
                part_i2_dir=artifact_root / "evaluations_part_i2",
                output_root=artifact_root / "publication",
            )

    def _build(
        self,
        *,
        part_i1_dir: Path,
        part_i2_dir: Path,
        output_root: Path,
    ) -> dict[str, tuple[Path, ...]]:
        import matplotlib.pyplot as plt

        part_i1_dir = Path(part_i1_dir).resolve()
        part_i2_dir = Path(part_i2_dir).resolve()
        output_root = Path(output_root).resolve()
        part_i1_output = output_root / "evaluations_part_i1"
        part_i2_output = output_root / "evaluations_part_i2"
        part_i1_figures = part_i1_output / "figures"
        part_i2_figures = part_i2_output / "figures"

        effects = _read(
            part_i1_dir / "paired_effects.csv", PAIRED_EFFECT_COLUMNS
        )
        part_i1_summary = _read(
            part_i1_dir / "summary.csv", SUMMARY_COLUMNS
        )
        part_i1_contrasts = _read(
            part_i1_dir / "model_contrasts.csv", MODEL_CONTRAST_COLUMNS
        )
        sweep_rows = _read(
            part_i1_dir / "friedman_condition_summary.csv",
            FRIEDMAN_SWEEP_MODEL_SUMMARY_COLUMNS,
        )
        part_i2_rows = _read_part_i2_rows(
            part_i2_dir / "task_effects.csv", PART_I2_TASK_EFFECT_COLUMNS
        )
        part_i2_summary = _read_part_i2_rows(
            part_i2_dir / "summary_openml.csv", PART_I2_SUMMARY_COLUMNS
        )
        dimensions = _read_part_i2_rows(
            part_i2_dir / "dataset_dimensions.csv",
            PART_I2_DATASET_DIMENSION_COLUMNS,
        )
        representative_pdp_rows = _read_representative_pdp_curves(
            part_i2_dir / "pdp_curves.csv"
        )
        part_i2_sweep_rows = _read_part_i2_rows(
            part_i2_dir / "task_effects.csv", PART_I2_TASK_EFFECT_COLUMNS
        )
        part_i2_sweep_rows = [
            row for row in part_i2_sweep_rows if row["source"] == "synthetic"
        ]
        part_i2_sweep_summary = _read_part_i2_rows(
            part_i2_dir / "summary_friedman_conditions.csv",
            PART_I2_FRIEDMAN_SWEEP_SUMMARY_COLUMNS,
        )
        # Friedman is always the 350 sweep task-settings and OpenML the 42
        # datasets.
        require_benchmark_sizes(effects, "Part I.1 paired effects")
        require_benchmark_sizes(part_i2_sweep_rows, "Part I.2 Friedman sweep")
        require_benchmark_sizes(
            [row for row in part_i2_rows if row["source"] == "real"],
            "Part I.2 OpenML",
        )

        part_i1_models = _publication_models(sweep_rows)
        figure, axes = plt.subplots(
            1, 3, figsize=(PUBLICATION_WIDTH_IN, 2.3), sharey=True,
            squeeze=False, layout="constrained",
        )
        flat_axes = axes[0]
        noise_rows = [row for row in sweep_rows if row["sweep"] == "noise_std"]
        _plot_publication_condition_curves(
            flat_axes[0], noise_rows, level_column="noise_std",
            reverse_levels=False,
            condition_title=r"Noise std. $\sigma$",
            median_column="r2_drop_median", q25_column="r2_drop_q25",
            q75_column="r2_drop_q75", models=part_i1_models,
        )
        flat_axes[0].set_title("Friedman: noise sweep\n" r"($N=1000$)")
        sample_rows = [row for row in sweep_rows if row["sweep"] == "sample_size"]
        _plot_publication_condition_curves(
            flat_axes[1], sample_rows, level_column="n_total",
            reverse_levels=True, condition_title=r"Total samples $N$",
            median_column="r2_drop_median", q25_column="r2_drop_q25",
            q75_column="r2_drop_q75", models=part_i1_models,
        )
        flat_axes[1].set_title(
            "Friedman: sample-size sweep\n" r"($\sigma=1$)"
        )
        real_effects = [row for row in effects if row["source"] == "real"]
        _plot_publication_dataset_curves(
            flat_axes[2], real_effects, value_column="r2_drop",
            models=part_i1_models,
        )
        flat_axes[2].set_title("OpenML\n" r"(42 datasets)")
        for axis in flat_axes:
            axis.set_xlabel(r"Null-feature ratio $\rho$")
        flat_axes[0].set_ylabel(r"Median $R^2$ drop")
        figure.legend(
            handles=_publication_model_handles(plt, part_i1_models),
            loc="outside lower center", ncol=3,
        )
        _add_corner_panel_labels(axes)
        # Demoted to the appendix: the two sweeps stratify one result by a
        # nuisance factor, which is a boundary check rather than a step in the
        # main argument.
        sweeps_i1 = _save_publication_figure(
            figure, part_i1_figures, "appendix_part_i1_r2_drop_median_iqr"
        )
        plt.close(figure)

        raw_summary = _raw_r2_summary(effects)
        figure, axes = plt.subplots(
            1, 2, figsize=(PUBLICATION_WIDTH_IN, 2.5), squeeze=False,
            layout="constrained",
        )
        flat_axes = axes[0]
        for axis, source, title in zip(
            flat_axes, ("synthetic", "real"),
            ("Friedman (350 task-settings)", "OpenML (42 datasets)"),
        ):
            source_rows = [row for row in raw_summary if row["source"] == source]
            _plot_publication_summary_curves(
                axis, source_rows, prefix="r2_at_dose", models=part_i1_models
            )
            axis.set_title(title)
            axis.set_xlabel(r"Null-feature ratio $\rho$")
        flat_axes[0].set_ylabel(r"Median $R^2$")
        figure.legend(
            handles=_publication_model_handles(plt, part_i1_models),
            loc="outside lower center", ncol=3,
        )
        _add_corner_panel_labels(axes)
        # No "_median_iqr" suffix: this panel now reports medians only.
        raw_i1 = _save_publication_figure(
            figure, part_i1_figures, "appendix_part_i1_raw_r2_median"
        )
        plt.close(figure)
        dimension_i1 = _plot_openml_dimension_r2_drop(
            effects,
            part_i1_figures / "appendix_part_i1_openml_by_dimension_median_iqr.pdf",
        )
        part_i2_models = list(PART_I2_MODELS)
        real_part_i2 = [row for row in part_i2_rows if row["source"] == "real"]

        # The single main-text figure. Friedman takes the seven-configuration
        # sweep rows so that both halves of the figure pool the configurations
        # the same way the surrounding text describes.
        # The Friedman contrast comes from the sweep evaluation and the OpenML
        # contrast from the Part I.2 evaluation, matching the two sources that
        # panel (d) plots.
        part_i2_contrasts = [
            row for row in _read_part_i2_rows(
                part_i2_dir / "model_contrasts.csv",
                PART_I2_CONTRAST_COLUMNS,
            ) if row["source"] == "synthetic"
        ] + [
            row for row in _read_part_i2_rows(
                part_i2_dir / "model_contrasts.csv", PART_I2_CONTRAST_COLUMNS
            ) if row["source"] == "real"
        ]
        overview = _plot_part_i_overview(
            effects, part_i1_contrasts, part_i2_contrasts,
            part_i2_sweep_rows, real_part_i2,
            part_i1_models, part_i1_figures / "main_part_i_overview.pdf",
            part_i1_figures / "main_part_i_functional.pdf",
        )
        part_i1_paths = tuple(
            path for path in (*overview, sweeps_i1, raw_i1, dimension_i1)
            if path is not None
        )
        part_i2_paths: list[Path] = []
        for metric, stem in (
            # Demoted to the appendix: its active/original panels are positive
            # controls, and the irrelevant-feature result it carries is now in
            # the main overview figure.
            ("permutation_shift", "appendix_part_i2_permutation_median_iqr"),
            ("pdp_variation", "appendix_part_i2_pdp_variation_median_iqr"),
        ):
            # One row of four panels: (a,b) active/original features share a
            # y-axis, (c,d) added irrelevant features share a second y-axis.
            figure, axes = plt.subplots(
                1, 4, figsize=(PUBLICATION_WIDTH_IN, 2.1), sharex=True,
                squeeze=False, layout="constrained",
            )
            flat_axes = axes[0]
            flat_axes[1].sharey(flat_axes[0])
            flat_axes[3].sharey(flat_axes[2])
            panels = (
                (part_i2_sweep_rows, "Friedman", "synthetic", "active"),
                (real_part_i2, "OpenML", "real", "original"),
                (part_i2_sweep_rows, "Friedman", "synthetic", "irrelevant"),
                (real_part_i2, "OpenML", "real", "irrelevant"),
            )
            metric_label = (
                "Normalized prediction shift"
                if metric == "permutation_shift"
                else "Normalized PDP variation"
            )
            for panel_index, (source_rows, source_label, source, role) in enumerate(panels):
                axis = flat_axes[panel_index]
                selected = [
                    row for row in source_rows
                    if row["metric"] == metric and row["feature_role"] == role
                ]
                _plot_publication_dataset_curves(
                    axis, selected, value_column="value", models=part_i2_models
                )
                # Two lines, because a single-line title overruns a 1.1 in
                # panel at the final printed width.
                axis.set_title(
                    f"{source_label}\n{_part_i2_panel_role(source, role)}"
                )
                axis.set_xlabel(r"Null-feature ratio $\rho$")
                if panel_index in (0, 2):
                    axis.set_ylabel(metric_label)
                else:
                    axis.tick_params(axis="y", labelleft=False)
            figure.legend(
                handles=_publication_model_handles(plt, part_i2_models),
                loc="outside lower center", ncol=2,
            )
            _add_corner_panel_labels(axes)
            path = _save_publication_figure(
                figure, part_i2_figures, stem,
                PUBLICATION_PAIRED_ROW_EXEMPTIONS,
            )
            plt.close(figure)
            part_i2_paths.append(path)

        part_i2_paths.extend(_plot_representative_pdp_curves(
            representative_pdp_rows, part_i2_figures
        ))

        figure, axes = plt.subplots(
            2, 2, figsize=(PUBLICATION_WIDTH_IN, 4.2), sharex=True,
            sharey="row", squeeze=False, layout="constrained",
        )
        for column_index, (sweep, level_column, reverse, title, legend_title) in enumerate((
            (
                "noise_std", "noise_std", False,
                r"Noise sweep ($N=1000$)", r"Noise std. $\sigma$",
            ),
            (
                "sample_size", "n_total", True,
                r"Sample-size sweep ($\sigma=1$)", r"Total samples $N$",
            ),
        )):
            for row_index, role in enumerate(("active", "irrelevant")):
                axis = axes[row_index, column_index]
                selected = [
                    row for row in part_i2_sweep_summary
                    if row["sweep"] == sweep
                    and row["metric"] == "permutation_shift"
                    and row["feature_role"] == role
                ]
                _plot_publication_condition_curves(
                    axis, selected, level_column=level_column,
                    reverse_levels=reverse, condition_title=legend_title,
                    median_column="median", q25_column="q25", q75_column="q75",
                    models=part_i2_models,
                    show_condition_legend=row_index == 0,
                    # The top row's y-axis is stretched down to the zero
                    # reference line, so the empty band is between the curves
                    # (~0.47 and up) and zero. "lower left" lands on the zero
                    # line and "center left" lands on the flat 0.47 curve, so
                    # anchor it explicitly inside that band.
                    condition_legend_loc="lower left",
                    condition_legend_bbox=(0.02, 0.14),
                )
                if row_index == 0:
                    axis.set_title(title)
                else:
                    axis.set_xlabel(r"Null-feature ratio $\rho$")
                if column_index == 0:
                    axis.set_ylabel(
                        f"{'Relevant' if role == 'active' else 'Null'} features\n"
                        "Normalized prediction shift"
                    )
        figure.legend(
            handles=_publication_model_handles(plt, part_i2_models),
            loc="outside lower center", ncol=2,
        )
        _add_corner_panel_labels(axes)
        part_i2_paths.append(_save_publication_figure(
            figure, part_i2_figures,
            "appendix_part_i2_friedman_sweeps_permutation_median_iqr",
        ))
        plt.close(figure)

        for metric, stem in (
            ("permutation_shift", "permutation"),
            ("pdp_variation", "pdp_variation"),
        ):
            path = _plot_part_i2_openml_dimension_metric(
                part_i2_rows, dimensions, metric,
                part_i2_figures / (
                    f"appendix_part_i2_openml_by_dimension_{stem}_median_iqr.pdf"
                ),
            )
            if path is not None:
                part_i2_paths.append(path)

        _write_publication_csv(
            part_i1_summary, part_i1_output / "summary.csv", SUMMARY_COLUMNS
        )
        _write_publication_csv(
            sweep_rows, part_i1_output / "friedman_sweep_summary.csv",
            FRIEDMAN_SWEEP_MODEL_SUMMARY_COLUMNS,
        )
        _write_publication_csv(
            [row for row in part_i2_summary if row["source"] == "real"],
            part_i2_output / "openml_summary.csv", PART_I2_SUMMARY_COLUMNS,
        )
        _write_publication_csv(
            part_i2_sweep_summary,
            part_i2_output / "friedman_sweep_summary.csv",
            PART_I2_FRIEDMAN_SWEEP_SUMMARY_COLUMNS,
        )
        _write_publication_csv(
            [row for row in dimensions if row["source"] == "real"],
            part_i2_output / "openml_dataset_dimensions.csv",
            PART_I2_DATASET_DIMENSION_COLUMNS,
        )
        manifests = {
            part_i1_output: {
                "complete": True,
                "statistic": "median_iqr",
                "friedman_tasks_per_setting": 50,
                "excludes_original_200_task_friedman_suite": True,
                "figures": [str(path.relative_to(part_i1_output)) for path in part_i1_paths],
            },
            part_i2_output: {
                "complete": True,
                "statistic": "median_iqr",
                "friedman_tasks_per_setting": 50,
                "excludes_original_200_task_friedman_suite": True,
                "figures": [str(path.relative_to(part_i2_output)) for path in part_i2_paths],
            },
        }
        for directory, manifest in manifests.items():
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / f".manifest.{uuid4().hex}.partial"
            try:
                temporary.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, directory / "manifest.json")
            finally:
                temporary.unlink(missing_ok=True)
        return {
            "part_i1": part_i1_paths,
            "part_i2": tuple(part_i2_paths),
        }


__all__ = [
    "DefaultPublicationPlotter",
    "DefaultTabDPTVersionPlotter",
]
