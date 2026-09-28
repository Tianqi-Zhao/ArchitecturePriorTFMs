"""OLS/Bayes coefficient comparisons used in the current appendix."""

from functools import partial

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .render import ROLE_COLORS, decorate


def build(bundle, renderer):
    conditions = sorted(bundle["ols.json"]["conditions"], key=lambda c: (c["d"], c["k"]))

    def comparison(ax, c):
        names = ("OLS", "Row-token", "Alternating-axis")
        for x, name in enumerate(names):
            v = c["models"][name]
            ax.bar(x, v["active"], .6, color=ROLE_COLORS["active"], edgecolor="white", linewidth=.4)
            ax.bar(x, v["inactive"], .6, bottom=v["active"], color=ROLE_COLORS["inactive"],
                   edgecolor="white", linewidth=.4)
            ax.annotate(f"{v['total']:.3g}", (x, v["total"]), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(3), ["OLS", "Row-token", "Alternating-\naxis"])
        ax.set(title=f"d = {c['d']}, k = {c['k']}", ylabel="Coefficient error")
        column = 0 if c["k"] == 1 else 2 if c["k"] == c["d"] else 1
        high = max(r["models"]["OLS"]["total"] for r in conditions
                   if (0 if r["k"] == 1 else 2 if r["k"] == r["d"] else 1) == column)
        ax.set_ylim(0, 1.25 * high)
        ax.legend(handles=[Patch(color=color, label=role.capitalize()) for role, color in ROLE_COLORS.items()],
                  frameon=False)
        decorate(ax)

    renderer.grid("part_iv_ols_comparison", [partial(comparison, c=c) for c in conditions])

    def coefficients(ax, c):
        sample = c["scatter"]
        x, y = np.asarray(sample["c"]).ravel(), np.asarray(sample["bayes"]).ravel()
        support = np.asarray(sample["support"], dtype=bool).ravel()
        limit = max(abs(x).max(), abs(y).max()) * 1.05
        ax.plot([-limit, limit], [-limit, limit], "--", color="#888888", lw=.7, zorder=0)
        for role, mask in (("inactive", ~support), ("active", support)):
            ax.scatter(x[mask], y[mask], s=3, color=ROLE_COLORS[role], alpha=.5, linewidths=0,
                       label=role.capitalize())
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
               title=f"d = {c['d']}, k = {c['k']}", xlabel=r"OLS coefficient $c_j$",
               ylabel=r"Bayes coefficient $\eta_j^*$")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.legend(frameon=False)

    renderer.grid("part_iv_ols_coefficients", [partial(coefficients, c=c) for c in conditions])
