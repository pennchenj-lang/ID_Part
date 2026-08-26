from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

DOMAIN_PREFIXES = (
    "container_",
    "daily_object_",
    "device_",
    "furniture_",
    "tool_prop_",
    "vehicle_",
)


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
    groups_path = directory / "groups.json"
    evaluation_path = directory / "paco_evaluation.json"
    for required in (source, group_map, groups_path, evaluation_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    return {
        "domain": domain,
        "case_id": case_id,
        "directory": directory,
        "source": source,
        "group_map": group_map,
        "groups_path": groups_path,
        "evaluation_path": evaluation_path,
        "groups": groups,
        "evaluation": evaluation,
    }


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
        hspace=0.31,
    )

    for index, case in enumerate(cases):
        row, column = divmod(index, 3)
        sub = outer[row, column].subgridspec(
            3,
            2,
            height_ratios=(0.10, 0.72, 0.18),
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

        group_names = [
            display_part(str(group["semantic_name"])) for group in case["groups"]
        ]
        group_text = " · ".join(group_names)
        wrapped = "\n".join(textwrap.wrap(group_text, width=43, break_long_words=False))
        text_ax.set_axis_off()
        text_ax.text(
            0.0,
            0.95,
            f"{len(group_names)} groups",
            transform=text_ax.transAxes,
            fontsize=6.2,
            fontweight="bold",
            color="#25313B",
            va="top",
        )
        text_ax.text(
            0.0,
            0.64,
            wrapped,
            transform=text_ax.transAxes,
            fontsize=5.5,
            color="#58636E",
            va="top",
            linespacing=1.16,
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
    for case in cases:
        evaluation = case["evaluation"]
        semantic_names = [group["semantic_name"] for group in case["groups"]]
        for semantic_name in semantic_names:
            normalized = f"_{semantic_name.casefold().replace('-', '_')}_"
            if any(f"_{term}_" in normalized for term in forbidden_terms):
                forbidden_public_ids.append(semantic_name)
        rows.append(
            {
                "domain": case["domain"],
                "case_id": case["case_id"],
                "group_count": len(case["groups"]),
                "groups": semantic_names,
                "object_iou": evaluation.get("object_iou"),
                "group_f1_at_025": evaluation.get("editable_group_metrics", {}).get(
                    "part_discovery_f1_at_025"
                ),
                "semantic_part_recall": evaluation.get("semantic_part_recall"),
                "files": {
                    "source_sha256": sha256(case["source"]),
                    "group_map_sha256": sha256(case["group_map"]),
                    "groups_sha256": sha256(case["groups_path"]),
                    "evaluation_sha256": sha256(case["evaluation_path"]),
                },
            }
        )
    audit = {
        "figure": output.with_suffix(".png").name,
        "selection_role": "qualitative_legibility_audit",
        "selection_is_statistical_evidence": False,
        "manual_mask_retouching": False,
        "colors_are_case_local_ids": True,
        "illumination_regions_are_evidence_only": True,
        "forbidden_photometric_public_id_terms": list(forbidden_terms),
        "forbidden_photometric_public_ids_detected": forbidden_public_ids,
        "photometric_public_id_check_pass": not forbidden_public_ids,
        "cases": rows,
    }
    if forbidden_public_ids:
        raise ValueError(
            "photometric regions leaked into public editable IDs: "
            + ", ".join(sorted(set(forbidden_public_ids)))
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
