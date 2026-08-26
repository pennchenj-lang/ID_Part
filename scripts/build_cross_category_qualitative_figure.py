from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

DOMAIN_PREFIXES = (
    "container_",
    "daily_object_",
    "device_",
    "furniture_",
    "tool_prop_",
    "vehicle_",
)

EXPECTED_GROUPS = {
    "Container": {
        "container_body",
        "container_label",
        "container_lid",
    },
    "Daily object": {
        "daily_object_body",
        "daily_object_cap",
        "daily_object_label",
        "daily_object_neck",
    },
    "Device": {"device_body", "device_base", "device_handle"},
    "Furniture": {
        "furniture_frame",
        "furniture_backrest",
        "furniture_seat",
    },
    "Tool / prop": {
        "tool_prop_handle",
        "tool_prop_blade",
        "tool_prop_pivot",
    },
    "Vehicle": {
        "vehicle_body",
        "vehicle_bumper",
        "vehicle_grille",
        "vehicle_headlight",
        "vehicle_hood",
        "vehicle_mirror",
        "vehicle_roof",
        "vehicle_wheel",
        "vehicle_windshield",
    },
}

EXPECTED_INSTANCE_COUNTS = {
    "Vehicle": {"vehicle_headlight": 2, "vehicle_wheel": 2},
}


def parse_case(value: str) -> tuple[str, str, Path]:
    fields = value.split("|", maxsplit=2)
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(
            "case must be DOMAIN|CASE_ID|PACKAGE_DIRECTORY"
        )
    domain, case_id, directory = fields
    return domain.strip(), case_id.strip(), Path(directory).expanduser().resolve()


def display_part(name: str) -> str:
    value = name
    for prefix in DOMAIN_PREFIXES:
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return value.replace("_", " ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case(spec: tuple[str, str, Path]) -> dict[str, object]:
    domain, case_id, directory = spec
    source = directory / "source.png"
    group_map = directory / "group_id_preview.png"
    group_id_map = directory / "group_id_map.tiff"
    part_id_map = directory / "part_id_map.tiff"
    groups_path = directory / "groups.json"
    for required in (source, group_map, group_id_map, part_id_map, groups_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    with Image.open(group_map) as image:
        preview = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(group_id_map) as image:
        group_indices = np.asarray(image, dtype=np.uint16)
    group_colors: dict[int, tuple[float, float, float]] = {}
    for group in groups:
        group_index = int(group["group_index"])
        pixels = preview[group_indices == group_index]
        if pixels.size == 0:
            raise ValueError(f"group {group_index} has no preview pixels in {directory}")
        color = np.median(pixels, axis=0) / 255.0
        group_colors[group_index] = tuple(float(channel) for channel in color)
    return {
        "domain": domain,
        "case_id": case_id,
        "directory": directory,
        "source": source,
        "group_map": group_map,
        "group_id_map": group_id_map,
        "part_id_map": part_id_map,
        "groups_path": groups_path,
        "groups": groups,
        "group_colors": group_colors,
    }


def legend_entries(case: dict[str, object]) -> list[tuple[str, tuple[float, float, float]]]:
    groups = list(case["groups"])
    colors = dict(case["group_colors"])
    totals = Counter(str(group["semantic_name"]) for group in groups)
    seen: defaultdict[str, int] = defaultdict(int)
    entries: list[tuple[str, tuple[float, float, float]]] = []
    for group in groups:
        semantic = str(group["semantic_name"])
        seen[semantic] += 1
        label = display_part(semantic)
        if totals[semantic] > 1:
            label = f"{label} {seen[semantic]}"
        entries.append((label, colors[int(group["group_index"])]))
    return entries


def build_figure(cases: list[dict[str, object]], output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )
    fig = plt.figure(figsize=(7.12, 4.55), facecolor="white")
    outer = fig.add_gridspec(
        2,
        3,
        left=0.025,
        right=0.99,
        top=0.965,
        bottom=0.035,
        wspace=0.20,
        hspace=0.28,
    )

    for index, case in enumerate(cases):
        row, column = divmod(index, 3)
        sub = outer[row, column].subgridspec(
            3,
            2,
            height_ratios=(0.09, 0.66, 0.25),
            wspace=0.055,
            hspace=0.045,
        )
        title_ax = fig.add_subplot(sub[0, :])
        source_ax = fig.add_subplot(sub[1, 0])
        group_ax = fig.add_subplot(sub[1, 1])
        text_ax = fig.add_subplot(sub[2, :])

        with Image.open(case["source"]) as image:
            source_ax.imshow(image.convert("RGB"))
        with Image.open(case["group_map"]) as image:
            group_ax.imshow(image.convert("RGB"))

        for axis in (source_ax, group_ax):
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("black")
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.55)
                spine.set_edgecolor("#B7BDC5")

        source_ax.set_title("Source", fontsize=6.2, pad=2.0, color="#4C5560")
        group_ax.set_title(
            "Editable Group IDs", fontsize=6.2, pad=2.0, color="#4C5560"
        )
        title_ax.set_axis_off()
        title_ax.text(
            0.0,
            0.98,
            chr(ord("a") + index),
            transform=title_ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color="#1F252B",
            va="top",
        )
        title_ax.text(
            0.08,
            0.98,
            str(case["domain"]),
            transform=title_ax.transAxes,
            fontsize=7.4,
            fontweight="bold",
            color="#1F252B",
            va="top",
        )

        entries = legend_entries(case)
        text_ax.set_axis_off()
        text_ax.text(
            0.0,
            0.98,
            f"{len(entries)} groups  |  color key",
            transform=text_ax.transAxes,
            fontsize=6.2,
            fontweight="bold",
            color="#25313B",
            va="top",
        )
        columns = 3 if len(entries) > 6 else 2
        rows = int(np.ceil(len(entries) / columns))
        x_step = 1.0 / columns
        y_top = 0.67
        y_step = 0.57 if rows <= 2 else 0.205
        swatch_width = 0.034
        swatch_height = 0.105 if rows <= 2 else 0.085
        for entry_index, (label, color) in enumerate(entries):
            legend_row = entry_index // columns
            legend_column = entry_index % columns
            x = legend_column * x_step
            y = y_top - legend_row * y_step
            text_ax.add_patch(
                Rectangle(
                    (x, y - swatch_height * 0.55),
                    swatch_width,
                    swatch_height,
                    transform=text_ax.transAxes,
                    facecolor=color,
                    edgecolor="#343B42",
                    linewidth=0.35,
                    clip_on=False,
                )
            )
            text_ax.text(
                x + swatch_width + 0.012,
                y,
                label,
                transform=text_ax.transAxes,
                fontsize=4.8 if rows > 2 else 5.3,
                color="#4A555F",
                va="center",
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    fig.savefig(
        output.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def write_audit(cases: list[dict[str, object]], output: Path) -> None:
    forbidden_terms = (
        "highlight",
        "shadow",
        "shading",
        "specular",
        "illumination",
        "gradient",
        "visual_panel",
        "color_patch",
        "texture_patch",
    )
    rows = []
    forbidden_public_ids: list[str] = []
    semantic_failures: list[dict[str, object]] = []
    ownership_failures: list[str] = []
    for case in cases:
        semantic_names = [group["semantic_name"] for group in case["groups"]]
        for semantic_name in semantic_names:
            normalized = f"_{semantic_name.casefold().replace('-', '_')}_"
            if any(f"_{term}_" in normalized for term in forbidden_terms):
                forbidden_public_ids.append(semantic_name)
        expected = EXPECTED_GROUPS.get(str(case["domain"]))
        actual = set(semantic_names)
        if expected is not None and actual != expected:
            semantic_failures.append(
                {
                    "domain": case["domain"],
                    "missing": sorted(expected - actual),
                    "unexpected": sorted(actual - expected),
                }
            )
        expected_counts = EXPECTED_INSTANCE_COUNTS.get(str(case["domain"]), {})
        for semantic, count in expected_counts.items():
            actual_count = semantic_names.count(semantic)
            if actual_count != count:
                semantic_failures.append(
                    {
                        "domain": case["domain"],
                        "semantic_name": semantic,
                        "expected_instance_count": count,
                        "actual_instance_count": actual_count,
                    }
                )
        with Image.open(case["part_id_map"]) as image:
            fine_foreground = np.asarray(image, dtype=np.uint16) > 0
        with Image.open(case["group_id_map"]) as image:
            public_foreground = np.asarray(image, dtype=np.uint16) > 0
        ownership_complete = bool(np.array_equal(fine_foreground, public_foreground))
        if not ownership_complete:
            ownership_failures.append(str(case["case_id"]))
        rows.append(
            {
                "domain": case["domain"],
                "case_id": case["case_id"],
                "group_count": len(case["groups"]),
                "groups": semantic_names,
                "expected_groups": sorted(expected) if expected is not None else None,
                "expected_group_check_pass": expected is None or actual == expected,
                "complete_root_ownership": ownership_complete,
                "files": {
                    "source_sha256": sha256(case["source"]),
                    "group_map_sha256": sha256(case["group_map"]),
                    "group_id_map_sha256": sha256(case["group_id_map"]),
                    "part_id_map_sha256": sha256(case["part_id_map"]),
                    "groups_sha256": sha256(case["groups_path"]),
                },
            }
        )
    audit = {
        "figure": output.with_suffix(".png").name,
        "selection_role": "qualitative_legibility_audit",
        "selection_is_statistical_evidence": False,
        "quantitative_metrics_reused_from_frozen_outputs": False,
        "manual_mask_retouching": False,
        "colors_are_case_local_ids": True,
        "illumination_regions_are_evidence_only": True,
        "forbidden_photometric_public_id_terms": list(forbidden_terms),
        "forbidden_photometric_public_ids_detected": forbidden_public_ids,
        "photometric_public_id_check_pass": not forbidden_public_ids,
        "expected_physical_inventory_failures": semantic_failures,
        "expected_physical_inventory_check_pass": not semantic_failures,
        "complete_root_ownership_failures": ownership_failures,
        "complete_root_ownership_check_pass": not ownership_failures,
        "cases": rows,
    }
    if forbidden_public_ids:
        raise ValueError(
            "photometric regions leaked into public editable IDs: "
            + ", ".join(sorted(set(forbidden_public_ids)))
        )
    if semantic_failures:
        raise ValueError(
            "publication cases failed their expected physical inventory: "
            + json.dumps(semantic_failures, ensure_ascii=False)
        )
    if ownership_failures:
        raise ValueError(
            "public groups do not cover exactly the fine-part foreground: "
            + ", ".join(ownership_failures)
        )
    output.with_name(f"{output.name}_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the audited cross-category HPID-Split qualitative plate."
    )
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.case) != 6:
        raise ValueError("the publication plate requires exactly six cases")
    cases = [load_case(case) for case in args.case]
    output = args.output.expanduser().resolve()
    build_figure(cases, output)
    write_audit(cases, output)
    print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
