"""Figures for a set of scored runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .evaluate import binarize

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
CLASS_COLORS = {"non_binder": "#8e8d87", "binder": "#2a78d6"}

METRIC_PANELS = ["ipsae_min_mean", "sc_dockq_mean", "iptm_pae_mean", "n_interface_contacts_mean"]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.title.set_color(INK)


def _figure(width: float = 7.0, height: float = 6.0):
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    _style(ax)
    return fig, ax


def roc_curves(runs: dict[str, pd.DataFrame], path: Path) -> Path:
    """ROC per run, with the operating point of the fixed 0.5 cut marked on each curve."""
    fig, ax = _figure()
    for (name, df), color in zip(runs.items(), SERIES):
        usable = df.dropna(subset=["p_binder"])
        y_true = binarize(usable["y_true"])
        scores = usable["p_binder"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC {roc_auc_score(y_true, scores):.3f})")

        predicted = scores >= 0.5
        ax.scatter(
            [predicted[y_true == 0].mean()],
            [predicted[y_true == 1].mean()],
            color=color, s=70, zorder=3, edgecolor=SURFACE, linewidth=2,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color="#b4b3ad", label="chance")
    ax.set(xlabel="false positive rate", ylabel="true positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Ranking quality, with the 0.5 operating point marked", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK_MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def score_distributions(runs: dict[str, pd.DataFrame], path: Path) -> Path:
    """p(binder) by true class, one panel per run: shows where the 0.5 cut actually falls.

    Plotted as a share of each class so the two are comparable despite the 3:1 imbalance,
    and as outlines so the overlap is not mistaken for stacking.
    """
    fig, axes = plt.subplots(
        len(runs), 1, figsize=(7, 2.4 * len(runs)), facecolor=SURFACE, squeeze=False, sharex=True
    )
    bins = np.linspace(0, 1, 41)
    for ax, (name, df) in zip(axes.ravel(), runs.items()):
        _style(ax)
        usable = df.dropna(subset=["p_binder"])
        for label, color in CLASS_COLORS.items():
            values = usable.loc[usable["y_true"] == label, "p_binder"]
            weights = np.full(len(values), 1 / max(len(values), 1))
            ax.hist(
                values, bins=bins, weights=weights, color=color, alpha=0.3,
                histtype="stepfilled", edgecolor=color, linewidth=2, label=f"{label} (n={len(values)})",
            )
        ax.axvline(0.5, color=INK_MUTED, linestyle="--", linewidth=1.5)
        ax.set_title(name, fontsize=11, loc="left")
        ax.set_ylabel("share of class")
        ax.legend(fontsize=9, frameon=False, labelcolor=INK_MUTED)
    axes.ravel()[-1].set_xlabel("p(binder), dashed line = 0.5 decision threshold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def score_vs_metrics(name: str, df: pd.DataFrame, path: Path) -> Path:
    """p(binder) against the cofolding metrics it was given, colored by the wet-lab label."""
    panels = [c for c in METRIC_PANELS if c in df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5), facecolor=SURFACE, squeeze=False)
    usable = df.dropna(subset=["p_binder"])
    for ax, column in zip(axes.ravel(), panels):
        _style(ax)
        for label, color in CLASS_COLORS.items():
            group = usable[usable["y_true"] == label]
            ax.scatter(
                group[column], group["p_binder"], s=14, color=color,
                alpha=0.35 if label == "non_binder" else 0.8, label=label, linewidth=0,
            )
        ax.set(xlabel=column, ylabel="p(binder)")
    axes.ravel()[0].legend(fontsize=9, frameon=False, labelcolor=INK_MUTED, markerscale=1.6)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(f"{name}: predicted score against its own input metrics", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def render_all(runs: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    written = [
        roc_curves(runs, out_dir / "roc_curves.png"),
        score_distributions(runs, out_dir / "score_distributions.png"),
    ]
    for name, df in runs.items():
        written.append(score_vs_metrics(name, df, out_dir / f"score_vs_metrics_{name}.png"))
    return written
