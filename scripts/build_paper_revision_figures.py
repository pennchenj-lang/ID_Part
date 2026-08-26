from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

WIDTH_IN = 7.16
COLORS = {
    "blue": "#245A7A",
    "blue_mid": "#6F96AE",
    "blue_light": "#B8CEDA",
    "orange": "#B9673F",
    "teal": "#4F827B",
    "purple": "#756E87",
    "gray_dark": "#4D535A",
    "gray": "#858B92",
    "gray_mid": "#A9AEB4",
    "gray_light": "#D9DDE1",
    "ink": "#1C2024",
    "grid": "#E6E8EB",
}
METHOD_LABELS = {
    "sam2_raw": "SAM2 raw",
    "sam2_nms": "SAM2 + NMS",
    "sam2_max_ownership": "SAM2 + max ownership",
    "grounded_sam2_same_inventory": "Grounded-SAM2",
    "clipseg_ovparts_style": "CLIPSeg object-part",
    "hpid_split_a3": "HPID-Split A3",
}
VARIANT_LABELS = {
    "A0_independent_max": "A0\nindependent max",
    "A1_cross_source_consensus": "A1\nconsensus",
    "A2_consensus_hierarchy": "A2\n+ hierarchy",
    "A3_full_fusion": "A3\n+ ownership",
}
DOMAIN_LABELS = {
    "container": "Container",
    "daily_object": "Daily object",
    "device": "Device",
    "furniture": "Furniture",
    "tool_prop": "Tool / prop",
    "vehicle": "Vehicle",
}
ID_COLORS = [mpl.colors.to_hex(color) for color in mpl.colormaps["tab20"].colors]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.25,
        "lines.markersize": 3.8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
    }
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    svg_path = output / f"{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    # Matplotlib emits trailing spaces in multi-line SVG path data. Normalizing
    # them keeps the publication evidence clean under `git diff --check`.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        output / f"{name}.tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def _style_axis(ax: plt.Axes, *, y_grid: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if y_grid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.55)
        ax.set_axisbelow(True)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
    )


def _method_colors(count: int) -> list[str]:
    colors = [COLORS["gray_mid"]] * count
    if count >= 2:
        colors[-2] = COLORS["orange"]
        colors[-1] = COLORS["blue"]
    return colors


def _fit(image: Image.Image, size: tuple[int, int], fill: str = "white") -> Image.Image:
    image = image.convert("RGB")
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, fill)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def _colorize_id_map(id_map: np.ndarray) -> Image.Image:
    canvas = np.zeros((*id_map.shape, 3), dtype=np.uint8)
    for value in sorted(int(item) for item in np.unique(id_map) if int(item) > 0):
        color = np.asarray(mpl.colors.to_rgb(ID_COLORS[(value - 1) % len(ID_COLORS)]))
        canvas[id_map == value] = np.round(color * 255).astype(np.uint8)
    return Image.fromarray(canvas)


def _truth_map(case_path: Path) -> Image.Image:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    case_dir = case_path.parent
    with Image.open(case_dir / "source_crop.png") as source:
        shape = (source.height, source.width)
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    for index, part in enumerate(case["parts"], start=1):
        mask = np.asarray(Image.open(case_dir / str(part["mask_crop"])).convert("L")) >= 128
        color = np.asarray(mpl.colors.to_rgb(ID_COLORS[(index - 1) % len(ID_COLORS)]))
        canvas[mask] = np.round(color * 255).astype(np.uint8)
    return Image.fromarray(canvas)


def build_external_baselines(facts: dict[str, object], output: Path) -> None:
    external = dict(facts["external_baselines"])
    external["clipseg_ovparts_style"] = facts["clipseg_ovparts_style"]
    order = [
        "sam2_raw",
        "sam2_nms",
        "sam2_max_ownership",
        "grounded_sam2_same_inventory",
        "clipseg_ovparts_style",
        "hpid_split_a3",
    ]
    short_labels = ["Raw", "NMS", "Max-own.", "Grounded", "CLIPSeg", "HPID"]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 4.6), constrained_layout=True)
    x = np.arange(len(order))
    colors = _method_colors(len(order))

    for metric, label, color, marker in (
        ("part_f1_at_025", "F1@.25", COLORS["blue"], "o"),
        ("part_f1_at_050", "F1@.50", COLORS["blue_mid"], "s"),
        ("part_f1_at_075", "F1@.75", COLORS["gray_dark"], "^"),
    ):
        values = [float(external[key]["metrics"].get(metric, 0.0)) for key in order]
        axes[0, 0].plot(x, values, marker=marker, label=label, color=color)
    axes[0, 0].set_title("Strict Part F1", loc="left", fontweight="bold")
    axes[0, 0].set_xticks(x, short_labels, rotation=18, ha="right", rotation_mode="anchor")
    axes[0, 0].legend(ncol=3, loc="upper left", handlelength=1.2, columnspacing=0.9)
    _style_axis(axes[0, 0])
    _panel_label(axes[0, 0], "a")

    metrics = [
        ("part_f1_mean_025_075", "Mean Part F1"),
        ("semantic_f1_at_025", "Semantic F1@.25"),
        ("object_iou", "Object IoU"),
    ]
    for ax, (metric, title) in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]), metrics, strict=True
    ):
        values = [float(external[key]["metrics"].get(metric, 0.0)) for key in order]
        y = np.arange(len(order))
        ax.scatter(values, y, c=colors, s=28, edgecolors="white", linewidths=0.45, zorder=3)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlim(0, max(0.1, min(1.0, max(values) * 1.23)))
        ax.set_yticks(y, short_labels)
        ax.invert_yaxis()
        for index, value in enumerate(values):
            ax.text(value + ax.get_xlim()[1] * 0.025, index, f"{value:.3f}", ha="left", va="center", fontsize=6.2)
        _style_axis(ax, y_grid=False)
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.55)
    for ax, label in zip((axes[0, 1], axes[1, 0], axes[1, 1]), ("b", "c", "d"), strict=True):
        _panel_label(ax, label)
    _save(fig, output, "Fig4_external_baselines")


def build_ablation(facts: dict[str, object], output: Path) -> None:
    variants = dict(facts["fusion_ablation"])["variants"]
    order = list(VARIANT_LABELS)
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.75), constrained_layout=True)
    for metric, label, color, marker in (
        ("part_f1_at_025", "F1@.25", COLORS["blue"], "o"),
        ("part_f1_at_050", "F1@.50", COLORS["blue_mid"], "s"),
        ("part_f1_at_075", "F1@.75", COLORS["gray_dark"], "^"),
        ("part_f1_mean_025_075", "Mean F1", COLORS["orange"], "D"),
    ):
        values = [float(variants[key]["metrics"][metric]) for key in order]
        axes[0].plot(x, values, marker=marker, label=label, color=color)
    axes[0].set_xticks(x, [VARIANT_LABELS[key] for key in order])
    axes[0].set_ylabel("Case-macro score")
    axes[0].set_title("Strict Part-ID matching", loc="left", fontweight="bold")
    axes[0].legend(ncol=2, loc="upper left", handlelength=1.2, columnspacing=0.9)
    _style_axis(axes[0])
    _panel_label(axes[0], "a")

    for metric, label, color in (
        ("object_iou", "Object IoU", COLORS["gray_dark"]),
        ("mean_matched_iou_at_050", "Matched IoU@.50", COLORS["purple"]),
        ("mean_matched_boundary_f1_at_050", "Boundary F1@.50", COLORS["teal"]),
        ("semantic_f1_at_025", "Semantic F1@.25", COLORS["orange"]),
    ):
        values = [float(variants[key]["metrics"][metric]) for key in order]
        axes[1].plot(x, values, marker="o", label=label, color=color)
    axes[1].set_xticks(x, [VARIANT_LABELS[key] for key in order])
    axes[1].set_title("Boundary, semantics, and root support", loc="left", fontweight="bold")
    axes[1].legend(ncol=2, loc="upper left", handlelength=1.2, columnspacing=0.9)
    _style_axis(axes[1])
    _panel_label(axes[1], "b")
    _save(fig, output, "Fig5_fusion_ablation")


def _select_domain_medians(
    cases_csv: Path,
) -> dict[str, dict[str, str]]:
    rows = [
        row
        for row in _read_csv(cases_csv)
        if row["method"] == "hpid_split_a3"
    ]
    by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_domain[row["expected_domain"]].append(row)
    selected: dict[str, dict[str, str]] = {}
    for domain, domain_rows in by_domain.items():
        ordered = sorted(
            domain_rows,
            key=lambda row: (float(row["part_f1_at_025"]), row["case_id"]),
        )
        selected[domain] = ordered[(len(ordered) - 1) // 2]
    return selected


def build_qualitative(
    *,
    cases_csv: Path,
    manifest: Path,
    benchmark_root: Path,
    output: Path,
) -> dict[str, str]:
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    case_paths = {
        str(row["case_id"]): Path(str(row["case_path"]))
        for row in manifest_payload["cases"]
        if row.get("case_path")
    }
    selected = _select_domain_medians(cases_csv)
    domains = [key for key in DOMAIN_LABELS if key in selected]
    fig, axes = plt.subplots(
        len(domains),
        4,
        figsize=(WIDTH_IN, 6.2),
        constrained_layout=True,
    )
    titles = ["Source crop", "PACO part masks", "Fine Part IDs", "Editable Group IDs"]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontweight="bold")
    selected_ids: dict[str, str] = {}
    for row_index, domain in enumerate(domains):
        row = selected[domain]
        case_id = row["case_id"]
        selected_ids[domain] = case_id
        case_path = case_paths[case_id]
        package = benchmark_root / case_id
        panels = [
            Image.open(case_path.parent / "source_crop.png").convert("RGB"),
            _truth_map(case_path),
            _colorize_id_map(np.asarray(Image.open(package / "part_id_map.tiff"))),
            _colorize_id_map(np.asarray(Image.open(package / "group_id_map.tiff"))),
        ]
        for column, panel in enumerate(panels):
            axes[row_index, column].imshow(_fit(panel, (360, 270), "black"))
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])
            for spine in axes[row_index, column].spines.values():
                spine.set_visible(False)
        axes[row_index, 0].set_ylabel(
            DOMAIN_LABELS[domain],
            rotation=0,
            rotation_mode="anchor",
            ha="right",
            va="center",
            labelpad=7,
            fontsize=7.2,
            fontweight="bold",
        )
    _save(fig, output, "Fig6_domain_median_examples")
    return selected_ids


def build_quality(facts: dict[str, object], output: Path) -> None:
    quality = dict(facts["quality_exit"])
    statuses = list(quality["by_status"])
    coverage = list(quality["coverage"])
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.85), constrained_layout=True)

    status_order = ["ready", "review_recommended", "target_selection_required"]
    status_rows = {row["quality_status"]: row for row in statuses}
    labels = ["No review trigger", "Review recommended", "Target selection required"]
    x = np.arange(3)
    for metric, label, color in (
        ("mean_part_f1_at_050", "Part F1@.50", COLORS["blue"]),
        ("mean_semantic_f1_at_025", "Semantic F1@.25", COLORS["orange"]),
        ("mean_mean_matched_boundary_f1_at_050", "Boundary F1@.50", COLORS["gray_dark"]),
    ):
        values = [float(status_rows[key][metric]) for key in status_order]
        axes[0].plot(x, values, marker="o", label=label, color=color)
    axes[0].set_xticks(x, labels, rotation=18, ha="right", rotation_mode="anchor")
    axes[0].set_title("Quality-exit strata", loc="left", fontweight="bold")
    axes[0].legend(loc="upper right")
    _style_axis(axes[0])
    _panel_label(axes[0], "a")

    policy_order = [
        "original_ready_only",
        "original_ready_plus_review",
        "all_outputs",
    ]
    policy_rows = {row["acceptance_policy"]: row for row in coverage}
    xs = [float(policy_rows[key]["coverage"]) for key in policy_order]
    ys = [1.0 - float(policy_rows[key]["failure_rate"]) for key in policy_order]
    axes[1].plot(xs, ys, color=COLORS["blue"], marker="o")
    for key, x_value, y_value in zip(policy_order, xs, ys, strict=True):
        label = {
            "original_ready_only": "No-trigger only",
            "original_ready_plus_review": "No-trigger + review",
            "all_outputs": "All outputs",
        }[key]
        axes[1].annotate(label, (x_value, y_value), xytext=(4, 4), textcoords="offset points", fontsize=6.8)
    axes[1].set_xlabel("Coverage")
    axes[1].set_ylabel("1 - operational failure rate")
    axes[1].set_xlim(0, 1.04)
    axes[1].set_ylim(0, 1.04)
    axes[1].set_title("Coverage-accuracy trade-off", loc="left", fontweight="bold")
    _style_axis(axes[1])
    _panel_label(axes[1], "b")
    _save(fig, output, "Fig7_quality_exit")


def build_sensitivity(facts: dict[str, object], output: Path) -> None:
    rows = list(dict(facts["sensitivity"])["rows"])
    parameters = list(dict(facts["sensitivity"])["report"]["parameters"])
    short = {
        "full_agreement_overlap": "agreement overlap",
        "specificity_minimum_containment": "minimum containment",
        "specificity_root_minimum_candidate_score": "root candidate score",
        "specificity_host_suppression": "host suppression",
        "remainder_merge_distance_ratio": "remainder distance",
    }
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.95), constrained_layout=True)
    invariant_styles = ["-", "--", ":", "-."]
    invariant_index = 0
    for parameter in parameters:
        is_sensitive = parameter == "specificity_root_minimum_candidate_score"
        color = COLORS["blue"] if is_sensitive else COLORS["gray_mid"]
        linestyle = "-" if is_sensitive else invariant_styles[invariant_index % len(invariant_styles)]
        if not is_sensitive:
            invariant_index += 1
        selected = sorted(
            [row for row in rows if row["parameter"] == parameter],
            key=lambda row: float(row["factor"]),
        )
        factors = [(float(row["factor"]) - 1.0) * 100 for row in selected]
        baseline = next(row for row in selected if abs(float(row["factor"]) - 1.0) < 1e-9)
        for ax, metric in zip(
            axes,
            ("mean_part_f1_at_025", "mean_semantic_f1_at_025"),
            strict=True,
        ):
            base_value = float(baseline[metric])
            delta = [float(row[metric]) - base_value for row in selected]
            ax.plot(factors, delta, marker="o", color=color, linestyle=linestyle, alpha=1.0 if is_sensitive else 0.8, label=short[parameter])
    axes[0].set_title("Part F1@.25 change", loc="left", fontweight="bold")
    axes[1].set_title("Semantic F1@.25 change", loc="left", fontweight="bold")
    for ax in axes:
        ax.axhline(0, color=COLORS["gray_dark"], linewidth=0.7)
        ax.set_xlabel("One-factor perturbation (%)")
        ax.set_ylabel("Delta from frozen setting")
        ax.set_xticks([-20, -10, 0, 10, 20])
        _style_axis(ax)
    axes[1].legend(loc="best", handlelength=1.6)
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    _save(fig, output, "Fig8_parameter_sensitivity")


def build_runtime(facts: dict[str, object], output: Path) -> None:
    runtime = dict(facts["runtime"])
    stages = dict(runtime["stage_runtime_seconds"])
    labels = list(stages)
    medians = [float(stages[label]["median"]) for label in labels]
    p95s = [float(stages[label]["p95"]) for label in labels]
    order = np.argsort(medians)[::-1]
    labels = [labels[index].replace("_", " ") for index in order]
    medians = [medians[index] for index in order]
    p95s = [p95s[index] for index in order]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(WIDTH_IN, max(2.5, 0.33 * len(labels) + 0.9)), constrained_layout=True)
    ax.hlines(y, medians, p95s, color=COLORS["gray_light"], linewidth=5.5, zorder=1)
    ax.scatter(p95s, y, marker="|", s=80, color=COLORS["gray_dark"], linewidths=1.1, label="P95", zorder=3)
    ax.scatter(medians, y, marker="o", s=24, color=COLORS["blue"], edgecolors="white", linewidths=0.45, label="Median", zorder=4)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Recorded seconds per isolated case")
    ax.set_title("Recorded stage time", loc="left", fontweight="bold")
    ax.legend(loc="lower right", ncol=2)
    _style_axis(ax, y_grid=False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.7)
    _save(fig, output, "Fig9_runtime_breakdown")


def build_gate_link(dev_gate_dir: Path, output: Path) -> None:
    rows = _read_csv(dev_gate_dir / "gate_summary.csv")
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.75), constrained_layout=True)
    retained = [float(row["mean_candidate_count"]) for row in rows]
    axes[0].vlines(x, 0, retained, color=COLORS["gray_light"], linewidth=6)
    axes[0].scatter(x, retained, s=28, color=COLORS["blue"], edgecolors="white", linewidths=0.45, zorder=3)
    axes[0].set_xticks(x, ["C1 semantics", "C2 structure", "C3 appearance"])
    axes[0].set_ylabel("Mean retained candidates")
    axes[0].set_title("Serial gate retention", loc="left", fontweight="bold")
    _style_axis(axes[0])
    _panel_label(axes[0], "a")
    for metric, label, color in (
        ("mean_candidate_precision_at_050", "Precision@.50", COLORS["blue"]),
        ("mean_candidate_recall_at_050", "Recall@.50", COLORS["orange"]),
        ("mean_candidate_f1_at_050", "F1@.50", COLORS["gray_dark"]),
    ):
        axes[1].plot(x, [float(row[metric]) for row in rows], marker="o", label=label, color=color)
    axes[1].set_xticks(x, ["C1", "C2", "C3"])
    axes[1].set_title("Candidate gate trade-off", loc="left", fontweight="bold")
    axes[1].legend(loc="best")
    _style_axis(axes[1])
    _panel_label(axes[1], "b")
    _save(fig, output, "FigS1_gate_to_final_link")


def build_error_impact(facts: dict[str, object], output: Path) -> None:
    rows = list(dict(facts["quality_exit"])["error_impact"])
    labels = [row["error_type"].replace("_", " ") for row in rows]
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.9), constrained_layout=True)
    metrics = [
        ("affected_minus_unaffected_part_f1_at_050", "Part F1@.50", COLORS["blue"]),
        ("affected_minus_unaffected_mean_matched_boundary_f1_at_050", "Boundary F1@.50", COLORS["gray_dark"]),
        ("affected_minus_unaffected_semantic_f1_at_025", "Semantic F1@.25", COLORS["orange"]),
    ]
    for offset, (metric, label, color) in enumerate(metrics):
        ax.bar(x + (offset - 1) * width, [float(row[metric]) for row in rows], width=width, label=label, color=color)
    ax.axhline(0, color=COLORS["gray_dark"], linewidth=0.7)
    ax.set_xticks(x, labels, rotation=20, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Affected minus unaffected mean")
    ax.set_title("Audited failure-type impact", loc="left", fontweight="bold")
    ax.legend(ncol=3, loc="lower left")
    _style_axis(ax)
    _save(fig, output, "FigS2_error_type_impact")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build data-linked manuscript figures.")
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--dev-gate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    facts = json.loads(args.facts.read_text(encoding="utf-8"))

    build_external_baselines(facts, args.output)
    build_ablation(facts, args.output)
    build_quality(facts, args.output)
    build_sensitivity(facts, args.output)
    build_runtime(facts, args.output)
    build_gate_link(args.dev_gate_dir, args.output)
    build_error_impact(facts, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
