import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from time import perf_counter

from PIL import Image

from .resource_paths import DEFAULT_PROMPT_BANK, user_completion_config, user_hpid_home
from .scope_routing import route_extraction_scope
from .validation import validate_package

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_COMPLETION_CONFIG = REPOSITORY_ROOT / "configs" / "lama_sam2_evidence.local.json"


def _default_completion_config() -> Path:
    configured = os.environ.get("HPID_COMPLETION_CONFIG")
    if configured:
        return Path(configured).expanduser()
    repository_example = REPOSITORY_ROOT / "configs" / "lama_sam2_evidence.example.json"
    for candidate in (
        _LOCAL_COMPLETION_CONFIG,
        user_completion_config(),
        repository_example,
    ):
        if candidate.is_file():
            return candidate
    return user_completion_config()


DEFAULT_COMPLETION_CONFIG = _default_completion_config()


def _runtime_model_roots() -> tuple[Path, ...]:
    roots = [
        user_hpid_home() / "models",
        Path(sys.prefix).resolve().parent / "models",
        REPOSITORY_ROOT / "models",
    ]
    runtime_root = os.environ.get("HPID_RUNTIME_ROOT", "").strip()
    if runtime_root:
        roots.insert(0, Path(runtime_root).expanduser() / "models")
    return tuple(dict.fromkeys(root.resolve() for root in roots))


def _default_retrieval_index() -> Path:
    configured = os.environ.get("HPID_RETRIEVAL_INDEX")
    if configured:
        return Path(configured).expanduser()
    for model_root in _runtime_model_roots():
        for relative in (
            Path("hpid_paco_prototypes_v2") / "index.json",
            Path("retrieval") / "index.json",
        ):
            candidate = model_root / relative
            if candidate.is_file():
                return candidate
    return user_hpid_home() / "retrieval" / "index.json"


DEFAULT_RETRIEVAL_INDEX = _default_retrieval_index()


def _optional_path_from_environment(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def _default_asset_router_index() -> Path | None:
    configured = _optional_path_from_environment("HPID_ASSET_ROUTER_INDEX")
    if configured is not None:
        return configured
    for model_root in _runtime_model_roots():
        candidate = model_root / "hpid_siglip2_asset_router_v1"
        if candidate.is_dir() and (candidate / "index.json").is_file():
            return candidate
    return None


DEFAULT_ASSET_ROUTER_INDEX = _default_asset_router_index()


def _default_vlm_model() -> Path | None:
    configured = _optional_path_from_environment("HPID_VLM_MODEL")
    if configured is not None:
        return configured
    for model_root in _runtime_model_roots():
        for model_name in (
            "qwen3-vl-4b-instruct",
            "qwen3-vl-2b-instruct",
        ):
            candidate = model_root / model_name
            if candidate.is_dir() and (candidate / "config.json").is_file():
                return candidate
    return None


DEFAULT_VLM_MODEL = _default_vlm_model()


def _default_vlm_4bit(model: Path | None) -> bool:
    configured = os.environ.get("HPID_VLM_4BIT")
    if configured is not None:
        return configured.strip().lower() in {"1", "true", "yes"}
    return bool(model is not None and "4b" in model.name.casefold())


DEFAULT_VLM_4BIT = _default_vlm_4bit(DEFAULT_VLM_MODEL)
DEFAULT_DENSE_SEMANTIC_MODEL = os.environ.get(
    "HPID_DENSE_SEMANTIC_MODEL", "CIDAS/clipseg-rd64-refined"
)
DOMAIN_CHOICES = (
    "auto",
    "character",
    "vehicle",
    "furniture",
    "tool_prop",
    "container",
    "device",
    "daily_object",
    "structure",
    "natural_object",
    "terrain",
)
SCOPE_CHOICES = ("Primary asset", "Entire scene")
DECOMPOSITION_CHOICES = ("Automatic", "Prompt-guided")


def _runtime_timeout_seconds(
    *, quality: str, scope: str, complete_hidden_regions: bool
) -> int:
    configured = os.environ.get("HPID_WEB_TIMEOUT_SECONDS", "").strip()
    if configured:
        timeout = int(configured)
        if timeout < 1:
            raise ValueError("HPID_WEB_TIMEOUT_SECONDS must be positive")
        return timeout
    timeout = 120 if quality == "Fast" else 150
    if scope == "Entire scene":
        timeout += 30
    if complete_hidden_regions:
        timeout += 120
    return timeout


def _output_root() -> Path:
    root = Path(
        os.environ.get("HPID_OUTPUT_ROOT", str(user_hpid_home() / "runs"))
    ).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _launch_allowed_paths() -> list[str]:
    """Restrict Gradio file serving to the controlled HPID run directory."""
    return [str(_output_root())]


def _build_auto_command(
    image_path: Path,
    output_dir: Path,
    *,
    domain: str,
    quality: str,
    complete_hidden_regions: bool,
    scope: str = "Primary asset",
    decomposition_mode: str = "Automatic",
    part_prompts: str = "",
    asset_prompt: str = "",
    target_point: str = "",
    prompt_bank: Path = DEFAULT_PROMPT_BANK,
    completion_config: Path = DEFAULT_COMPLETION_CONFIG,
    retrieval_index: Path = DEFAULT_RETRIEVAL_INDEX,
    asset_router_index: Path | None = DEFAULT_ASSET_ROUTER_INDEX,
    dense_semantic_model: str = DEFAULT_DENSE_SEMANTIC_MODEL,
    vlm_model: Path | None = DEFAULT_VLM_MODEL,
    vlm_load_in_4bit: bool = DEFAULT_VLM_4BIT,
) -> list[str]:
    if quality not in {"Fast", "Ensemble"}:
        raise ValueError(f"unknown quality mode: {quality}")
    heavy_ensemble = os.environ.get("HPID_ENABLE_HEAVY_ENSEMBLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    grounding_model = "IDEA-Research/grounding-dino-tiny"
    command = [
        sys.executable,
        "-m",
        "hpid_split.cli",
        "auto",
        "--image",
        str(image_path),
        "--output",
        str(output_dir),
        "--prompt-bank",
        str(prompt_bank),
        "--grounding-model",
        grounding_model,
        "--device",
        "auto",
    ]
    if scope == "Primary asset":
        command.extend(["--root-mode", "primary"])
    elif scope == "Entire scene":
        command.extend(["--root-mode", "scene"])
        if quality == "Fast":
            scene_root_budget = (12, 48)
        elif heavy_ensemble:
            scene_root_budget = (24, 96)
        else:
            scene_root_budget = (16, 64)
        command.extend(
            [
                "--maximum-roots-per-domain",
                str(scene_root_budget[0]),
                "--maximum-total-roots",
                str(scene_root_budget[1]),
            ]
        )
    elif scope == "All detected assets":
        # Backward-compatible diagnostic option used by older callers.
        command.extend(["--root-mode", "all"])
    else:
        raise ValueError(f"unknown extraction scope: {scope}")
    fast_scene = quality == "Fast" and scope == "Entire scene"
    if not fast_scene:
        command.extend(
            [
                "--dense-semantic-fallback",
                "--dense-semantic-model",
                dense_semantic_model,
            ]
        )
    if asset_prompt.strip():
        command.extend(["--asset-prompt", asset_prompt.strip()])
    if target_point.strip():
        values = [value.strip() for value in target_point.split(",")]
        if len(values) != 2:
            raise ValueError("target point must contain x,y coordinates")
        try:
            x, y = (float(value) for value in values)
        except ValueError as error:
            raise ValueError(
                "target point must contain numeric x,y coordinates"
            ) from error
        command.extend(["--target-point", str(x), str(y)])
    if decomposition_mode == "Automatic":
        if part_prompts.strip():
            raise ValueError(
                "Part prompts require the Prompt-guided decomposition mode."
            )
        command.extend(["--decomposition-mode", "automatic"])
        if asset_router_index is not None and asset_router_index.is_dir():
            command.extend(["--asset-router-index", str(asset_router_index)])
        if (
            quality == "Ensemble"
            and heavy_ensemble
            and scope == "Primary asset"
            and retrieval_index.is_file()
        ):
            command.extend(["--retrieval-index", str(retrieval_index)])
    elif decomposition_mode == "Prompt-guided":
        if not part_prompts.strip():
            raise ValueError("Enter at least one part prompt.")
        command.extend(
            [
                "--decomposition-mode",
                "prompt-guided",
                "--part-prompts",
                part_prompts.strip(),
                "--guided-backend",
                "auto",
            ]
        )
    else:
        raise ValueError(f"unknown decomposition mode: {decomposition_mode}")
    if domain != "auto":
        if domain not in DOMAIN_CHOICES:
            raise ValueError(f"unknown domain: {domain}")
        command.extend(["--domains", domain])
    if quality == "Ensemble" and heavy_ensemble:
        additional_model = "IDEA-Research/grounding-dino-base"
        command.extend(
            [
                "--additional-grounding-model",
                additional_model,
            ]
        )
        command.extend(["--visual-points-per-crop", "20"])
        command.append("--semantic-part-multimask")
        if (
            decomposition_mode == "Automatic"
            and vlm_model is not None
            and vlm_model.exists()
        ):
            command.extend(
                [
                    "--vlm-parts",
                    "--vlm-model",
                    str(vlm_model),
                    "--vlm-query-mode",
                    "per-semantic",
                    "--vlm-maximum-queries",
                    "6" if scope == "Entire scene" else "10",
                    "--vlm-maximum-total-queries",
                    "12",
                    "--vlm-maximum-roots",
                    "6" if scope == "Entire scene" else "2",
                    "--vlm-maximum-root-audits",
                    "4",
                    "--vlm-maximum-semantic-audits",
                    "3",
                    "--vlm-maximum-physicality-audits",
                    "2" if scope == "Entire scene" else "0",
                    "--vlm-dynamic-inventory",
                ]
            )
            if vlm_load_in_4bit:
                command.append("--vlm-load-in-4bit")
    else:
        ensemble = quality == "Ensemble"
        command.extend(
            [
                "--visual-points-per-crop",
                (
                    "14"
                    if ensemble and scope == "Entire scene"
                    else "20"
                    if ensemble
                    else "12"
                    if scope == "Entire scene"
                    else "18"
                ),
            ]
        )
        command.extend(
            [
                "--grabcut-iterations",
                "1",
                "--maximum-grabcut-candidates",
                (
                    "28"
                    if ensemble and scope == "Entire scene"
                    else "40"
                    if ensemble
                    else "8"
                    if scope == "Entire scene"
                    else "32"
                ),
            ]
        )
        if ensemble:
            command.append("--semantic-part-multimask")
        use_proposal_first = bool(
            decomposition_mode == "Automatic"
            and (quality == "Fast" or scope == "Entire scene" or domain == "auto")
        )
        if use_proposal_first:
            command.extend(
                [
                    "--proposal-first-fast",
                    "--no-isolated-profile-resolution",
                    "--no-profile-refinement",
                ]
            )
            if not fast_scene:
                command.append("--adaptive-profile-refinement")
        if scope == "Entire scene":
            command.append("--no-scene-profile-root-queries")
            if quality == "Fast":
                command.extend(
                    [
                        "--no-relational-appearance",
                        "--no-ontology-scene-consensus",
                    ]
                )
    if complete_hidden_regions:
        if not completion_config.is_file():
            raise FileNotFoundError(
                f"completion configuration is missing: {completion_config}. "
                "Run: hpid-split setup-completion"
            )
        command.extend(["--completion-config", str(completion_config)])
    return command


def _part_rows(package_dir: Path) -> list[list[object]]:
    groups = json.loads((package_dir / "groups.json").read_text(encoding="utf-8"))
    return [
        [
            item["asset_id"],
            item["group_id"],
            item["semantic_name"],
            len(item["member_part_ids"]),
            item["area_px"],
            "review" if item.get("review_required") else "ready",
        ]
        for item in groups
    ]


def run_uploaded_image(
    image_path: str | None,
    domain: str,
    scope: str,
    quality: str,
    complete_hidden_regions: bool,
    decomposition_mode: str,
    asset_prompt: str,
    target_point: str,
    part_prompts: str,
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    list[list[object]],
    str | None,
    str,
]:
    if not image_path:
        return None, None, None, None, [], None, "No image selected."
    source = Path(image_path)
    if not source.is_file():
        return None, None, None, None, [], None, f"Image is missing: {source}"
    output_root = _output_root()
    run_id = uuid.uuid4().hex[:12]
    package_dir = output_root / f"run_{run_id}"
    try:
        scope_routing = route_extraction_scope(Image.open(source), scope)
        resolved_scope = scope_routing.resolved_scope
        command = _build_auto_command(
            source,
            package_dir,
            domain=domain,
            scope=resolved_scope,
            quality=quality,
            complete_hidden_regions=complete_hidden_regions,
            decomposition_mode=decomposition_mode,
            asset_prompt=asset_prompt,
            target_point=target_point,
            part_prompts=part_prompts,
        )
        started_at = perf_counter()
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=_runtime_timeout_seconds(
                quality=quality,
                scope=resolved_scope,
                complete_hidden_regions=complete_hidden_regions,
            ),
        )
        elapsed_seconds = perf_counter() - started_at
        validation = validate_package(package_dir)
        if not validation["valid"]:
            raise RuntimeError(
                "Exported package failed validation: "
                + "; ".join(str(item) for item in validation["errors"])
            )
        archive_path = Path(
            shutil.make_archive(str(package_dir), "zip", root_dir=package_dir)
        )
        rows = _part_rows(package_dir)
        quality_report = json.loads(
            (package_dir / "quality_report.json").read_text(encoding="utf-8")
        )
        status_line = completed.stdout.strip().splitlines()[-1]
        semantic_vlm = "on" if "--vlm-parts" in command else "off"
        scene_labels = (
            "provisional"
            if "--no-ontology-scene-consensus" in command
            else "verified-route"
        )
        hidden_completion = (
            "on" if "--completion-config" in command else "off"
        )
        quality_status = str(quality_report["status"])
        evidence_grade = str(quality_report["evidence_grade"])
        reasons = ", ".join(str(value) for value in quality_report["review_reasons"])
        if quality_status == "target_selection_required":
            message = (
                "Target selection required. Click the intended asset in the input "
                "image, then run again."
            )
        elif quality_status == "review_recommended":
            message = f"Review recommended: {reasons}."
        elif quality_status == "no_parts_found":
            message = "No reliable Part ID was found. Change the asset domain or prompt."
        elif quality_status == "invalid_hierarchy":
            message = f"Hierarchy validation failed: {reasons}."
        else:
            message = (
                "Package consistency checks passed; visual accuracy still requires "
                "review."
            )
        run_status = (
            f"{message} Grade={evidence_grade}; {status_line}; valid=True; "
            f"scope={resolved_scope} (requested {scope}); quality={quality}; "
            f"scope routing={scope_routing.diagnostics['status']}; "
            f"scene labels={scene_labels}; "
            f"semantic VLM={semantic_vlm}; "
            f"hidden completion={hidden_completion}."
            f" elapsed={elapsed_seconds:.1f}s."
        )
        target_candidates = package_dir / "target_candidates.png"
        return (
            str(package_dir / "group_id_preview.png"),
            str(package_dir / "part_id_preview.png"),
            str(package_dir / "group_overlay.png"),
            str(target_candidates) if target_candidates.is_file() else None,
            rows,
            str(archive_path),
            run_status,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or str(error)).strip()
        return None, None, None, None, [], None, details[-4000:]
    except subprocess.TimeoutExpired:
        shutil.rmtree(package_dir, ignore_errors=True)
        timeout = _runtime_timeout_seconds(
            quality=quality,
            scope=locals().get("resolved_scope", scope),
            complete_hidden_regions=complete_hidden_regions,
        )
        return (
            None,
            None,
            None,
            None,
            [],
            None,
            (
                f"Stopped after {timeout} seconds. Use Primary asset, select a "
                "target point, or add a target prompt before running again."
            ),
        )
    except (OSError, RuntimeError, ValueError) as error:
        return None, None, None, None, [], None, str(error)


def build_app():
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError(
            "Install the UI extra: pip install -e '.[ui,foundation]'"
        ) from error

    with gr.Blocks(title="HPID Split") as app:
        with gr.Row():
            image_input = gr.Image(type="filepath", label="Input image")
            with gr.Tabs():
                with gr.Tab("Editable groups"):
                    group_preview = gr.Image(
                        type="filepath", label="Editable-group preview"
                    )
                with gr.Tab("Fine parts"):
                    preview = gr.Image(type="filepath", label="Part-ID preview")
                with gr.Tab("Edge overlay"):
                    source_overlay = gr.Image(
                        type="filepath", label="Source edge overlay"
                    )
                with gr.Tab("Target review"):
                    target_candidates = gr.Image(
                        type="filepath", label="Target candidates"
                    )
        with gr.Row():
            domain = gr.Dropdown(
                choices=list(DOMAIN_CHOICES), value="auto", label="Asset domain"
            )
            scope = gr.Radio(
                choices=list(SCOPE_CHOICES),
                value="Primary asset",
                label="Extraction scope",
            )
            quality = gr.Radio(
                choices=["Fast", "Ensemble"], value="Fast", label="Quality"
            )
            complete_hidden = gr.Checkbox(value=False, label="Complete hidden regions")
        with gr.Row():
            decomposition_mode = gr.Radio(
                choices=list(DECOMPOSITION_CHOICES),
                value="Automatic",
                label="Part-ID mode",
            )
            asset_prompt = gr.Textbox(
                label="Target asset (optional)",
                placeholder="serving tray, power drill, computer mouse",
                lines=2,
            )
            target_point = gr.Textbox(
                label="Selected target point",
                value="",
                interactive=False,
            )
            part_prompts = gr.Textbox(
                label="Part prompts",
                placeholder="stock, magazine, receiver, trigger, sight",
                lines=2,
            )
        with gr.Row():
            run_button = gr.Button("Split Part IDs", variant="primary")
            clear_target_button = gr.Button("Clear target point")
        parts = gr.Dataframe(
            headers=[
                "Asset ID",
                "Group ID",
                "Semantic",
                "Fine parts",
                "Area",
                "Status",
            ],
            datatype=["str", "str", "str", "number", "number", "str"],
            interactive=False,
            label="Editable groups",
        )
        package = gr.File(label="HPID package")
        status = gr.Textbox(label="Status", interactive=False)
        def processing_message(selected_scope: str, selected_quality: str) -> str:
            if selected_scope == "Entire scene":
                return (
                    f"Processing the entire scene in {selected_quality} mode. "
                    "Stage timings will be reported when complete."
                )
            return f"Processing the primary asset in {selected_quality} mode."

        start_event = run_button.click(
            fn=processing_message,
            inputs=[scope, quality],
            outputs=status,
            queue=False,
            show_progress="hidden",
        )
        start_event.then(
            fn=run_uploaded_image,
            inputs=[
                image_input,
                domain,
                scope,
                quality,
                complete_hidden,
                decomposition_mode,
                asset_prompt,
                target_point,
                part_prompts,
            ],
            outputs=[
                group_preview,
                preview,
                source_overlay,
                target_candidates,
                parts,
                package,
                status,
            ],
            show_progress="full",
        )

        def remember_target_point(evt: gr.SelectData) -> str:
            index = evt.index
            if not isinstance(index, (list, tuple)) or len(index) < 2:
                return ""
            return f"{float(index[0]):.3f},{float(index[1]):.3f}"

        image_input.select(
            fn=remember_target_point,
            inputs=None,
            outputs=target_point,
            show_progress="hidden",
        )
        clear_target_button.click(
            fn=lambda: "",
            inputs=None,
            outputs=target_point,
            show_progress="hidden",
        )
    return app


def main() -> None:
    build_app().launch(allowed_paths=_launch_allowed_paths())


if __name__ == "__main__":
    main()
