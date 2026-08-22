from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

WIDTH_MM = 183
HEIGHT_MM = 78
VARIANTS = (
    "A0_independent_max",
    "A1_cross_source_consensus",
    "A2_consensus_hierarchy",
    "A3_full_fusion",
)
DISPLAY = (
    "Max\nevidence",
    "+ Cross-source\nconsensus",
    "+ Hierarchy\nconstraints",
    "+ Specificity\nownership",
)
DOMAIN_DISPLAY = {
    "container": "Containers",
    "daily_object": "Daily objects",
    "device": "Devices",
    "furniture": "Furniture",
    "tool_prop": "Tools / props",
    "vehicle": "Vehicles",
}


def _configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--by-domain", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stem",
        default="Fig_HPID_Split_cross_domain_fusion_ablation",
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.summary).set_index("variant").loc[list(VARIANTS)]
    by_domain = pd.read_csv(args.by_domain)
    counts = (
        by_domain.groupby("expected_domain", sort=False)["case_count"]
        .first()
        .to_dict()
    )
    matrix = (
        by_domain.pivot(
            index="expected_domain",
            columns="variant",
            values="mean_part_f1_at_025",
        )
        .loc[list(DOMAIN_DISPLAY), list(VARIANTS)]
        .to_numpy(dtype=float)
    )

    _configure()
    fig = plt.figure(
        figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
        constrained_layout=False,
    )
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(0.88, 1.35),
        left=0.075,
        right=0.985,
        top=0.88,
        bottom=0.25,
        wspace=0.38,
    )

    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(VARIANTS))
    means = summary["mean_part_f1_at_025"].to_numpy(dtype=float)
    lower = means - summary["ci95_low_part_f1_at_025"].to_numpy(dtype=float)
    upper = summary["ci95_high_part_f1_at_025"].to_numpy(dtype=float) - means
    color = "#176B75"
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([lower, upper]),
        color=color,
        marker="o",
        markersize=4.5,
        linewidth=1.5,
        capsize=2.5,
        capthick=0.8,
        zorder=3,
    )
    ax.fill_between(x, means, color=color, alpha=0.07, zorder=1)
    for index, value in enumerate(means):
        ax.text(
            index,
            value + upper[index] + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#17343A",
        )
    ax.set_xticks(x, DISPLAY)
    ax.set_ylabel("Part F1 at IoU 0.25")
    ax.set_ylim(0.0, 0.46)
    ax.set_yticks(np.arange(0.0, 0.46, 0.1))
    ax.grid(axis="y", color="#D9DEE2", linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_title("Overall progression", loc="left", fontweight="bold")
    ax.text(
        0.0,
        -0.34,
        "n = 65 cases; points are macro means; error bars are\n95% case-bootstrap confidence intervals.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#4B545A",
    )

    heat = fig.add_subplot(grid[0, 1])
    cmap = LinearSegmentedColormap.from_list(
        "hpid_signal", ("#F2F4F5", "#9CCFD1", "#176B75")
    )
    image = heat.imshow(matrix, vmin=0.0, vmax=0.45, cmap=cmap, aspect="auto")
    heat.set_xticks(np.arange(len(DISPLAY)), DISPLAY)
    row_labels = [
        f"{DOMAIN_DISPLAY[domain]} (n={int(counts[domain])})"
        for domain in DOMAIN_DISPLAY
    ]
    heat.set_yticks(np.arange(len(row_labels)), row_labels)
    heat.set_title("Cross-domain Part F1", loc="left", fontweight="bold")
    heat.spines[:].set_visible(False)
    heat.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    heat.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    heat.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    heat.tick_params(which="minor", bottom=False, left=False)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            heat.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if value >= 0.29 else "#172124",
            )
    bar = fig.colorbar(image, ax=heat, fraction=0.035, pad=0.025)
    bar.set_label("F1", labelpad=2)
    bar.set_ticks((0.0, 0.2, 0.4))
    heat.text(
        0.0,
        -0.34,
        "Domain rows are descriptive means. Furniture (n=4) and\nvehicle (n=2) strata are not used for standalone inference.",
        transform=heat.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#4B545A",
    )

    fig.text(0.015, 0.94, "a", fontsize=9, fontweight="bold")
    fig.text(0.425, 0.94, "b", fontsize=9, fontweight="bold")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = args.output_dir / args.stem
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
