"""Figures for matched architectures, error components and task distributions."""

from functools import partial

import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter

from .render import (ARCH_NAMES, ARCH_COLORS, COMPONENT_COLORS, DIM_COLORS,
                     ROLE_COLORS, decorate, seed_points)


def build(bundle, renderer):
    conditions = {(c["d"], c["k"]): c for c in bundle["baselines.json"]["conditions"]}

    def values(d, k, metric, arch=None):
        a = conditions[d, k]["architectures"]
        if arch is not None:
            return np.asarray(a[arch]["values"][metric])
        return np.asarray(a["1d"]["values"][metric]) - a["2d"]["values"][metric]

    def performance(ax):
        for d, marker, linestyle, offset in ((10, "o", "--", -.045), (20, "s", "-", .025)):
            for arch, shift in (("1d", -.045), ("2d", .045)):
                x = np.arange(3) + offset + shift
                ys = [values(d, k, "E", arch) for k in (1, d // 2, d)]
                ax.plot(x, [v.mean() for v in ys], color=ARCH_COLORS[arch], ls=linestyle, lw=1.2)
                for xi, v in zip(x, ys):
                    seed_points(ax, xi, v, ARCH_COLORS[arch], marker, d == 20)
        ax.set(yscale="log", ylim=(.012, 1.4), xlim=(-.23, 2.23),
               xlabel="Support size k", ylabel=r"$\mathcal{E}_{a,k}$")
        ax.set_xticks([0, 1, 2], ["1", r"$d/2$", r"$d$"])
        ax.set_yticks([.02, .1, .5], ["0.02", "0.1", "0.5"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.legend(handles=[Line2D([], [], color=ARCH_COLORS[a], label=ARCH_NAMES[a]) for a in ARCH_NAMES] +
                  [Line2D([], [], color="#555555", ls=ls, marker=m, mfc="white" if d == 10 else "#555555",
                          label=f"d = {d}") for d, m, ls in ((10, "o", "--"), (20, "s", "-"))],
                  frameon=False, loc="upper right")
        decorate(ax)

    renderer.grid("section4_bayes_approximation_error", [performance])

    def absolute(ax, d, k):
        for x, arch in enumerate(ARCH_NAMES):
            bottom = 0.
            cumulative = np.zeros(3)
            for metric in ("C", "Z", "N"):
                v = values(d, k, metric, arch)
                ax.bar(x, v.mean(), .6, bottom=bottom, color=COMPONENT_COLORS[metric],
                       edgecolor="white", linewidth=.4)
                bottom += v.mean()
                cumulative += v
                ax.scatter(x + np.linspace(-.06, .06, 3), cumulative, s=8,
                           color=COMPONENT_COLORS[metric], edgecolor="white", linewidth=.3, zorder=4)
        ax.set_xticks([0, 1], ["Row-token", "Alternating-axis"])
        ax.set(title=f"d = {d}, k = {k}", ylabel="Task-normalized error")
        ax.margins(y=.30)
        ax.legend(handles=[Patch(color=COMPONENT_COLORS[m], label=t) for m, t in
                           (("C", "Coefficient"), ("Z", "Squared intercept"), ("N", "Residual"))],
                  frameon=False, loc="upper right")
        decorate(ax)

    for d in (10, 20):
        renderer.grid(f"part_iv2_absolute_errors_d{d}",
                      [partial(absolute, d=d, k=k) for k in (1, d // 2, d)],
                      panel_stem="part_iv2_absolute_errors_d10_inpanel_v5" if d == 10 else None)

    def gaps(ax, d, k):
        for x, metric in enumerate(("E", "C", "Z", "N")):
            v = values(d, k, metric)
            ax.bar(x, v.mean(), .6, color=COMPONENT_COLORS[metric])
            seed_points(ax, x, v, COMPONENT_COLORS[metric])
        component_sum = sum(values(d, k, metric).mean() for metric in ("C", "Z", "N"))
        ax.scatter(.25, component_sum, marker="D", s=22, facecolor="white", edgecolor="#333333", zorder=6,
                   label="Component sum")
        ax.axhline(0, color="#777777", lw=.6)
        ax.set_xticks(range(4), [r"$G$", r"$G^C$", r"$G^\zeta$", r"$G^N$"])
        ax.set(title=f"d = {d}, k = {k}", ylabel="Architecture gap")
        column = 0 if k == 1 else 2 if k == d else 1
        ax.set_ylim(*[(-.04, .84), (-.004, .074), (-.0027, .0034)][column])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        decorate(ax)
        if k < d:
            inset = ax.inset_axes([.57, .51, .39, .37])
            small = [values(d, k, metric) for metric in ("Z", "N")]
            for x, (metric, v) in enumerate(zip(("Z", "N"), small)):
                inset.bar(x, v.mean(), .6, color=COMPONENT_COLORS[metric])
                seed_points(inset, x, v, COMPONENT_COLORS[metric])
            low = min(0, *(min(v.min(), v.mean() - v.std(ddof=0)) for v in small))
            high = max(max(v.max(), v.mean() + v.std(ddof=0)) for v in small)
            pad = max(high - low, 1e-12) * .2
            inset.set_ylim(low - pad, high + pad)
            inset.set_xticks([0, 1], [r"$G^\zeta$", r"$G^N$"])
            inset.tick_params(labelsize=6, pad=1, length=2)
            inset.yaxis.set_major_locator(MaxNLocator(3))
            inset.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
            inset.yaxis.get_offset_text().set_size(6)
            inset.axhline(0, color="#777777", lw=.5)

    renderer.grid("part_iv2_component_gaps", [partial(gaps, d=d, k=k) for d, k in sorted(conditions)])
    renderer.legend("part_iv2_component_gaps", [
        Patch(color=COMPONENT_COLORS[m], label=label) for m, label in
        (("E", "Prediction"), ("C", "Coefficient"), ("Z", "Squared intercept"), ("N", "Residual"))
    ] + [Line2D([], [], ls="none", marker="D", mfc="white", mec="#333333", label="Component sum")])

    def reference(ax, learned=False):
        for d, marker in ((10, "o"), (20, "s")):
            x = np.arange(3) + (-.03 if d == 10 else .03)
            if learned:
                ys = [values(d, k, "E") for k in (1, d // 2, d)]
                y = [v.mean() for v in ys]
                for xi, v in zip(x, ys):
                    seed_points(ax, xi, v, DIM_COLORS[d], marker)
            else:
                y = [conditions[d, k]["dense_reference_gap"] if k < d else 0. for k in (1, d // 2, d)]
            ax.plot(x, y, marker=marker, color=DIM_COLORS[d], lw=1,
                    ls="--" if d == 10 else "-", label=f"d = {d}")
        ax.set_xticks([0, 1, 2], ["1", r"$d/2$", r"$d$"])
        ax.set(xlabel="Support size k", ylabel=r"$\widehat G_k$" if learned else r"$\Delta_k^{\mathrm{dense}}$",
               title="Architecture gap" if learned else "Dense-prior reference")
        if learned:
            ax.set_yscale("log")
        ax.legend(frameon=False)
        decorate(ax)

    renderer.grid("part_iv_reference_and_architecture_gaps", [reference, partial(reference, learned=True)], columns=2)

    def support(ax, d, k, per_coordinate=False):
        for x, arch in enumerate(ARCH_NAMES):
            bottom = 0.
            cumulative = np.zeros(3)
            for j, role in enumerate(("active", "inactive")):
                v = values(d, k, f"C_{role}", arch)
                if per_coordinate:
                    v = v / (k if role == "active" else d - k)
                    seed_points(ax, x + (-.15 if j == 0 else .15), v, ROLE_COLORS[role], "D", False)
                else:
                    ax.bar(x, v.mean(), .6, bottom=bottom, color=ROLE_COLORS[role], edgecolor="white", linewidth=.4)
                    cumulative += v
                    bottom += v.mean()
                    if k < d or role == "active":
                        ax.scatter(x + np.linspace(-.06, .06, 3), cumulative, s=10,
                                   color=ROLE_COLORS[role], edgecolor="white", linewidth=.3, zorder=5)
        ax.set_xticks([0, 1], list(ARCH_NAMES.values()))
        ax.set(title=f"d = {d}, k = {k}", ylabel="Error per coordinate" if per_coordinate else "Coefficient error")
        if per_coordinate:
            ax.set_yscale("log")
            ax.set_ylim(.0003, .25)
        else:
            column = 0 if k == 1 else 2 if k == d else 1
            ax.set_ylim(0, [.85, .085, .016][column])
        ax.legend(handles=[Patch(color=c, label=r.capitalize()) for r, c in ROLE_COLORS.items()], frameon=False)
        decorate(ax)

    renderer.grid("part_iv_support_levels", [partial(support, d=d, k=k) for d, k in sorted(conditions)])
    renderer.grid("part_iv_support_per_coordinate",
                  [partial(support, d=d, k=k, per_coordinate=True) for d, k in sorted(conditions) if k < d], columns=2)

    def histogram(ax, c):
        edges = np.asarray(c["edges"])
        density = np.asarray(c["histogram_counts"]) / (c["tasks"] * np.diff(edges))
        ax.stairs(density[0], edges, fill=True, color="#CFD5D9", label="Task mean over seeds")
        centers = (edges[:-1] + edges[1:]) / 2
        for seed, color in enumerate(("#3276A4", "#C87826", "#369781")):
            ax.plot(centers, density[seed + 1], color=color, lw=.8, label=f"Seed {seed}")
        ax.axvline(0, color="#777777", ls="--", lw=.7)
        ax.set(title=f"d = {c['d']}, k = {c['k']}", xlim=c["xlim"], ylim=(0, density.max() * 1.6),
               xlabel=r"Task gap $\widehat G_k^{(t)}$" + f"\n{c['outside_view_fraction']:.1%} of tasks outside view",
               ylabel="Density")
        ax.text(.98, .97, f"Positive: {c['positive_fraction']:.2%}\nMean: {c['mean']:.3g}\n"
                f"Median: {c['median']:.3g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.legend(frameon=False, loc="upper left", fontsize=6)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        decorate(ax)

    renderer.grid("part_iv_task_gap_distributions",
                  [partial(histogram, c=c) for c in sorted(bundle["task_gaps.json"]["conditions"],
                                                          key=lambda c: (c["d"], c["k"]))])
