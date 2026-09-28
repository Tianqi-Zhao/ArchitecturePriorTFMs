"""Plot the shared 128-task source-message experiments and their saved intervals."""

from functools import partial

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from .render import MODEL_NAMES, SCOPE_COLORS, DIM_COLORS, decorate

LABELS = {"P": r"Mean specificity $P_R$", "w": r"Median sensitivity $w$",
          "delta_E": r"Median $\Delta E$", "fidelity": r"Median fidelity $\widehat R^2_\Delta$"}


def curve(ax, condition, metric, role, dose, color, label):
    records = sorted((r for r in condition["records"] if np.isclose(r["dose"], dose)),
                     key=lambda r: r["layer"])
    values = np.array([[r["stats"][f"{metric}_{role}"][s] for s in ("center", "lo", "hi")]
                       for r in records], dtype=float)
    if not np.isfinite(values).any():
        return
    x = [r["layer"] for r in records]
    center, lo, hi = values.T
    ax.fill_between(x, lo, hi, color=color, alpha=.12, linewidth=0)
    ax.plot(x, center, color=color, ls="-" if role == "active" else "--", lw=1.1,
            marker="o", ms=2.8, mfc=color if role == "active" else "white", label=label)


def axes_style(ax, metric, model):
    ax.set(xlabel=r"Layer $\ell$", ylabel=LABELS[metric])
    ax.set_xticks([1, 2, 3] if model == "nano" else [1, 4, 8, 12])
    ax.yaxis.set_major_locator(MaxNLocator(4))
    if metric == "P":
        ax.set_ylim(0, 1.04)
    elif metric == "delta_E":
        ax.axhline(0, color="#777777", lw=.6)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    decorate(ax)


def build(bundle, renderer):
    data = bundle["interventions.json"]
    conditions = {(c["model"], c["d"], c["k"], c["scope"]): c for c in data["conditions"]}

    def scopes(ax, model, d, metric):
        for scope, color in SCOPE_COLORS.items():
            for role in ("active", "inactive"):
                curve(ax, conditions[model, d, 1, scope], metric, role, .1, color,
                      f"{scope.capitalize()}, {role}")
        ax.set_title(f"{MODEL_NAMES[model]}, d = {d}")
        axes_style(ax, metric, model)

    for d, name in ((10, "section4_scope_effects"), (20, "appendix_d_scope_effects")):
        renderer.grid(name, [partial(scopes, model=m, d=d, metric=metric)
                            for m in ("nano", "v2") for metric in ("P", "w", "delta_E")])

    def fidelity(ax, model):
        for d, color in DIM_COLORS.items():
            for role in ("active", "inactive"):
                curve(ax, conditions[model, d, 1, "both"], "fidelity", role, .1, color, f"d = {d}, {role}")
        ax.axhline(1, color="#777777", lw=.6, ls=":")
        ax.set_title(MODEL_NAMES[model])
        axes_style(ax, "fidelity", model)

    renderer.grid("appendix_d_fidelity", [partial(fidelity, model=m) for m in ("nano", "v2")], columns=2)

    dose_colors = {.05: "#3276A4", .1: "#C87826", .2: "#369781"}

    def doses(ax, model, d, metric, role):
        for dose, color in dose_colors.items():
            curve(ax, conditions[model, d, 1, "both"], metric, role, dose, color, rf"$\delta={dose:g}$")
        ax.set_title(f"{MODEL_NAMES[model]}, d = {d}, {role}")
        axes_style(ax, metric, model)
        if metric == "w":
            # Preserve the paper's common limits across dimensions for each role/model.
            high = max(r["stats"][f"w_{role}"]["hi"] for dimension in (10, 20)
                       for r in conditions[model, dimension, 1, "both"]["records"])
            ax.set_ylim(0, high * 1.15)

    for model in ("nano", "v2"):
        for d in (10, 20):
            renderer.grid(f"appendix_d_{model}_sensitivity_d{d}",
                          [partial(doses, model=model, d=d, metric=metric, role=role)
                           for metric in ("w", "delta_E") for role in ("active", "inactive")], columns=2)

    def support(ax, model, dense, metric):
        for d, color in DIM_COLORS.items():
            k = d if dense else d // 2
            for role in (("active",) if dense else ("active", "inactive")):
                curve(ax, conditions[model, d, k, "both"], metric, role, .1, color, f"d = {d}, {role}")
        ax.set_title(MODEL_NAMES[model] + (", k = d" if dense else ", k = d/2"))
        axes_style(ax, metric, model)

    for model in ("nano", "v2"):
        renderer.grid(f"appendix_e_support_{model}",
                      [partial(support, model=model, dense=dense, metric=metric)
                       for dense in (False, True) for metric in ("P", "w", "delta_E")])

    signed = {(r["model"], r["d"], r["scope"]): r for r in bundle["signed.json"]["records"]}

    def distribution(ax, model, d):
        for scope, color in SCOPE_COLORS.items():
            r = signed[model, d, scope]
            x = np.sort(np.asarray(r["signed_sensitivity"]))
            # Every task contributes; no histogram window or tail truncation.
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color,
                    label=f"{scope.capitalize()} ({r['negative_count']}/{len(x)} negative)")
            ax.scatter(x, np.arange(1, len(x) + 1) / len(x), s=3, color=color)
        ax.axvline(0, color="#777777", lw=.7, ls="--")
        ax.set_xscale("symlog", linthresh=.05)
        lo, hi = ax.get_xlim()
        ticks = [t for t in (-10, -1, -.1, 0, .1, 1, 10) if lo <= t <= hi]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.set(title=f"{MODEL_NAMES[model]}, d = {d}", xlabel=r"Signed sensitivity $s_g$",
               ylabel="Empirical cumulative probability", ylim=(0, 1.02))
        ax.legend(frameon=False, loc="upper left")
        decorate(ax)

    renderer.grid("appendix_signed_row_scope",
                  [partial(distribution, model=m, d=d) for m in ("nano", "v2") for d in (10, 20)], columns=2)

    records = {(r["d"], r["scope"]): r for r in bundle["response_matrices.json"]["records"]}

    def matrix(ax, d, scope, layer, scale):
        r = records[d, scope]
        active = r["active_source"] - 1
        im = ax.imshow(r["J"][layer], cmap="RdBu_r", vmin=-scale, vmax=scale, interpolation="nearest")
        ticks = [0, d // 2 - 1, d - 1]
        ax.set_xticks(ticks, [t + 1 for t in ticks])
        ax.set_yticks(ticks, [t + 1 for t in ticks])
        ax.set(xlabel=r"Source $g$", ylabel=r"Coefficient $r$",
               title=f"{scope.capitalize()} | Layer {layer + 1} | d = {d}")
        ax.add_patch(Rectangle((active - .5, -.5), 1, d, fill=False, edgecolor="#c58a16", lw=.8))
        ax.add_patch(Rectangle((active - .5, active - .5), 1, 1, fill=False, edgecolor="#c58a16", lw=1.6))
        ax.figure.colorbar(im, ax=ax, shrink=.82, pad=.03, ticks=[-scale, 0, scale])

    for d, name in ((10, "section4_response_matrices"), (20, "appendix_d_response_matrices")):
        matrices = np.array([records[d, s]["J"] for s in ("context", "query")])
        maxima = np.max(np.abs(matrices), axis=(0, 2, 3))
        powers = 10. ** np.floor(np.log10(np.maximum(maxima, np.finfo(float).tiny)))
        scales = np.where(maxima > 0, np.ceil(maxima / powers) * powers, 1.)
        renderer.grid(name, [partial(matrix, d=d, scope=scope, layer=layer, scale=scales[layer])
                             for scope in ("context", "query") for layer in range(3)])
