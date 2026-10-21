"""Image-only PartCATSeg smoke inference using the frozen official implementation.

No data loader, annotation reader, evaluator, or class-conditioned output is used.
The input masks and object class are constants required by the upstream interface;
only the independent, unfiltered ``sem_seg_all`` output is exported. All assets
must already exist locally. This program never installs or downloads anything.

Default: official VOC-116 vocabulary, seed, prompt ensemble and image sizes.
An optional metadata-only vocabulary JSON supplies ``stuff_classes``,
``obj_classes`` and ``ignore_label`` for a declared
cross-dataset run. PACO runs must supply all 456 classes and ignore_label=65535.
The default unchunked decoder is the upstream decoder. Decoder chunking changes
only its independent class batch and requires an equivalence receipt, or a
same-model verification run against the unchunked VOC forward.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback
import types


TASK = Path(__file__).resolve().parent
CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
DINO_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
PATCH_VERSION = "partcatseg-device-local-assets-pillow-v2"
DECODER_VERSION = "official-conv-decoder-class-batch-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class FrozenOverlay(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Exact, hash-checked in-memory portability patches; no source file edits."""

    def __init__(self, source, manifest, device, local_dino_loader):
        self.entries = {}
        self.ledger = []
        self.local_dino_loader = local_dino_loader
        replacements = {
            "baselines.data.transforms.transform": [
                ("Image.LINEAR", "Image.BILINEAR", 1),
            ],
            "baselines.partcatseg": [
                ('self.device = "cuda"', f"self.device = {device!r}", 1),
            ],
            "baselines.modeling.transformer.part_cat_seg_predictor": [
                ('device = "cuda" if torch.cuda.is_available() else "cpu"', f"device = {device!r}", 1),
                (".cuda()", ".to(self.device)", 4),
            ],
            "baselines.modeling.backbone.dinov2_backbone": [
                ("torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')", "__partcatseg_local_dino__()", 1),
                ('torch.load(f, map_location="cpu")', 'torch.load(f, map_location="cpu", weights_only=False)', 1),
            ],
        }
        for module, edits in replacements.items():
            relative = module.replace(".", "/") + ".py"
            path = source / relative
            actual = sha256(path)
            if actual != manifest["files"].get(relative):
                raise RuntimeError(f"Frozen source hash mismatch: {relative}")
            text = path.read_text(encoding="utf-8")
            for old, new, count in edits:
                if text.count(old) != count:
                    raise RuntimeError(f"Unexpected patch target count in {relative}: {old}")
                text = text.replace(old, new)
                self.ledger.append({"file": relative, "original_sha256": actual,
                                    "old": old, "new": new, "count": count})
            compile(text, str(path), "exec")
            self.entries[module] = (path, text)

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.entries:
            return importlib.util.spec_from_file_location(fullname, self.entries[fullname][0], loader=self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, text = self.entries[module.__name__]
        module.__dict__["__partcatseg_local_dino__"] = self.local_dino_loader
        exec(compile(text, str(path), "exec"), module.__dict__)


@contextlib.contextmanager
def no_network():
    def blocked(*args, **kwargs):
        raise RuntimeError("Network access is disabled for inference; provision assets separately")
    old_connect, old_create = socket.socket.connect, socket.create_connection
    socket.socket.connect = blocked
    socket.create_connection = blocked
    try:
        yield
    finally:
        socket.socket.connect, socket.create_connection = old_connect, old_create


def install_decoder_chunking(aggregator, chunk_size):
    """Slice T only AFTER all joint-class attention, using the upstream decoder."""
    original = aggregator.conv_decoder

    def chunked(self, x, guidance):
        if x.shape[0] != 1:
            raise ValueError("This adapter and decoder chunking require image batch size 1")
        # Up.forward repeats the single-image guidance for each class. GroupNorm
        # and convolutions do not mix this class batch. No attention is chunked.
        import torch
        return torch.cat([original(x[:, :, start:start + chunk_size], guidance)
                          for start in range(0, x.shape[2], chunk_size)], dim=1)

    aggregator.conv_decoder = types.MethodType(chunked, aggregator)
    return original


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=TASK / "source/part-catseg")
    parser.add_argument("--source-manifest", type=Path, default=TASK / "source_manifest.json")
    parser.add_argument("--checkpoint", type=Path, default=TASK / "weights/partcatseg_voc.pth")
    parser.add_argument("--checkpoint-sha256", required=True, help="Frozen, independently recorded asset SHA-256")
    parser.add_argument("--clip-weights", type=Path, default=TASK / "weights/ViT-B-16.pt")
    parser.add_argument("--dino-repo", type=Path, default=TASK / "environment/dinov2")
    parser.add_argument("--dino-weights", type=Path, default=TASK / "weights/dinov2_vits14_pretrain.pth")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, help="One image; never an annotation manifest")
    inputs.add_argument("--cohort-manifest", type=Path, help="Frozen anonymous image-only cohort manifest")
    parser.add_argument("--output", type=Path, required=True, help="Directory for this one-image result")
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default="cpu")
    parser.add_argument("--vocabulary-json", type=Path)
    parser.add_argument("--label-registry", type=Path,
                        help="PACO metadata registry: ordered labels with label_index, paco_category_id, paco_name, rendered_name")
    parser.add_argument("--paco", action="store_true", help="Enforce complete 456-class metadata and ignore=65535")
    parser.add_argument("--verify-dummy-invariance", action="store_true",
                        help="Repeat forward with different constant masks/class and verify sem_seg_all invariance")
    parser.add_argument("--save-scores", action="store_true", help="Also save full sigmoid scores; can be large")
    parser.add_argument("--decoder-chunk-size", type=int, default=0)
    parser.add_argument("--verify-decoder", action="store_true", help="Compare full and chunked VOC outputs in this run")
    parser.add_argument("--decoder-equivalence-receipt", type=Path)
    parser.add_argument("--protocol-frozen", type=Path)
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if args.decoder_chunk_size < 0:
        parser.error("--decoder-chunk-size must be nonnegative")
    if args.verify_decoder and (args.decoder_chunk_size == 0 or args.vocabulary_json or args.label_registry or args.paco):
        parser.error("--verify-decoder requires positive chunk size and original VOC metadata")
    if args.decoder_chunk_size and not (args.verify_decoder or args.decoder_equivalence_receipt):
        parser.error("Chunking requires --verify-decoder or a verified VOC equivalence receipt")
    if args.vocabulary_json and args.label_registry:
        parser.error("Use one metadata interface: --vocabulary-json or --label-registry")
    if args.paco and not (args.vocabulary_json or args.label_registry):
        parser.error("--paco requires metadata-only --vocabulary-json or --label-registry")
    if args.cohort_manifest:
        if not (args.protocol_frozen and args.protocol_sha256 and args.label_registry):
            parser.error("Cohort inference requires --protocol-frozen, --protocol-sha256 and --label-registry")
        if args.verify_decoder or args.verify_dummy_invariance:
            parser.error("Cohort inference consumes prior validation receipts; perform validation in VOC smoke mode")
        if not args.decoder_equivalence_receipt or args.decoder_chunk_size <= 0:
            parser.error("This cohort protocol requires validated decoder chunking and --decoder-equivalence-receipt")
    return args


def validate_cohort_protocol(args, report):
    """Bind inference inputs and executable before model construction; no GT read."""
    if sha256(args.protocol_frozen) != args.protocol_sha256.lower():
        raise ValueError("Frozen protocol SHA mismatch")
    protocol = json.loads(args.protocol_frozen.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_COHORT_INFERENCE":
        raise ValueError("Protocol was not frozen before cohort inference")
    binding = protocol["hash_bindings"]
    expected = {"inference_manifest_sha256": sha256(args.cohort_manifest),
                "label_registry_sha256": sha256(args.label_registry),
                "checkpoint_sha256": args.checkpoint_sha256.lower(),
                "adapter_sha256": sha256(__file__),
                "source_manifest_sha256": sha256(args.source_manifest),
                "decoder_equivalence_sha256": sha256(args.decoder_equivalence_receipt)}
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen protocol hash_bindings do not match inference resources")
    if not re.fullmatch(r"[a-fA-F0-9]{64}", binding.get("scorer_sha256", "")):
        raise ValueError("Protocol must also freeze a scorer SHA; inference never imports that scorer")
    manifest = json.loads(args.cohort_manifest.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if len(cases) != 42:
        raise ValueError("The frozen reviewer-5 cohort must contain exactly 42 anonymous cases")
    seen = set()
    for case in cases:
        case_id = case["anonymous_id"]
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", case_id) or case_id in seen:
            raise ValueError("Anonymous IDs must be unique safe filenames")
        seen.add(case_id)
        if not re.fullmatch(r"[a-fA-F0-9]{64}", case["input_sha256"]):
            raise ValueError(f"Invalid frozen input hash: {case_id}")
    report["protocol"] = {"path": str(args.protocol_frozen.resolve()),
                           "sha256": args.protocol_sha256.lower(), "hash_bindings": binding}
    report["cohort"] = {"manifest_path": str(args.cohort_manifest.resolve()),
                         "manifest_sha256": expected["inference_manifest_sha256"], "case_count": len(cases)}
    return cases


def prepare_image_sample(image_path, cfg, ignore):
    """Official RGB read/resize and constant required interface tensors only."""
    import numpy as np
    import torch
    from detectron2.data import detection_utils
    from detectron2.structures import Instances
    from baselines.data import transforms as official_transforms
    original_image = detection_utils.read_image(str(image_path.resolve()), format=cfg.INPUT.FORMAT)
    height, width = original_image.shape[:2]
    augmentation = official_transforms.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, "choice")
    resized = augmentation.get_transform(original_image).apply_image(original_image)
    image_tensor = torch.as_tensor(np.ascontiguousarray(resized.transpose(2, 0, 1)))
    resized_shape = tuple(image_tensor.shape[-2:])
    placeholders = torch.full(resized_shape, ignore, dtype=torch.int64)
    instances = Instances(resized_shape)
    instances.gt_classes = torch.zeros(1, dtype=torch.int64)
    sample = {"image": image_tensor, "height": height, "width": width,
              "sem_seg": placeholders, "obj_part_sem_seg": placeholders.clone(), "instances": instances}
    return sample, {"path": str(image_path.resolve()), "sha256": sha256(image_path),
                    "original_shape": [height, width], "resized_shape": list(resized_shape)}


def predict_unfiltered(model, sample, class_count, device):
    """One genuine full-vocabulary forward; return CPU sem_seg_all only."""
    import torch
    if device != "cpu":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model([sample])[0]
        scores = output["sem_seg_all"]
        del output
    metrics = {"inference_seconds": time.perf_counter() - started}
    if device != "cpu":
        torch.cuda.synchronize()
        metrics["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    expected_shape = (class_count + 1, sample["height"], sample["width"])
    if scores.shape != expected_shape or not torch.isfinite(scores).all():
        raise RuntimeError(f"Invalid unfiltered score output: {scores.shape}")
    return scores, metrics


def decoder_binding(args, report):
    return {"decoder_version": DECODER_VERSION, "checkpoint_sha256": args.checkpoint_sha256.lower(),
            "source_commit": report["source_commit"], "config_sha256": report["config_sha256"],
            "source_manifest_sha256": report["source_manifest_sha256"],
            "adapter_sha256": report["adapter_sha256"],
            "torch": report["runtime"]["torch"], "device": args.device,
            "runtime_fingerprint": report["runtime"]["numerical_settings"],
            "chunk_size": args.decoder_chunk_size, "autocast": False}


def validate_decoder_receipt(args, report):
    receipt = json.loads(args.decoder_equivalence_receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS" or not receipt.get("validation_run_complete") or any(
            receipt.get(key) != value for key, value in decoder_binding(args, report).items()):
        raise ValueError("Decoder equivalence receipt does not match this runtime, model and chunk size")
    if args.cohort_manifest and receipt.get("dummy_target_invariance", {}).get("status") != "PASS":
        raise ValueError("Cohort requires a completed VOC dummy-target invariance check")
    report["decoder_equivalence_receipt"] = {"path": str(args.decoder_equivalence_receipt.resolve()),
                                             "sha256": sha256(args.decoder_equivalence_receipt)}


def infer_cohort(args, report, cfg, model, cases):
    """Reuse one loaded official model; seal every success and failure, without GT."""
    import numpy as np
    import torch
    from PIL import Image
    validate_decoder_receipt(args, report)
    install_decoder_chunking(model.sem_seg_head.predictor.transformer, args.decoder_chunk_size)
    report["decoder_chunk_size"] = args.decoder_chunk_size
    report["input_provenance"] = {"annotations_read": False, "dummy_mask_value": 65535,
                                  "dummy_gt_class": 0, "exported_output": "sem_seg_all",
                                  "category_conditioned_sem_seg_discarded": True}
    output_records = []
    label_dir = args.output / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        case_id = case["anonymous_id"]
        path = Path(case["input_path"])
        if not path.is_absolute():
            path = args.cohort_manifest.resolve().parent / path
        record = {"anonymous_id": case_id, "input_path": str(path.resolve()),
                  "input_sha256": case["input_sha256"], "status": "FAILED"}
        scores = sample = labels = None
        try:
            if sha256(path) != case["input_sha256"].lower():
                raise ValueError("Input image SHA mismatch")
            sample, image_info = prepare_image_sample(path, cfg, 65535)
            if image_info["sha256"] != case["input_sha256"].lower():
                raise ValueError("Input image changed while preparing model input")
            scores, metrics = predict_unfiltered(model, sample, 456, args.device)
            labels = scores.argmax(0).numpy().astype(np.uint16)
            label_path = label_dir / (case_id + ".npy")
            png_path = label_dir / (case_id + ".png")
            np.save(label_path, labels, allow_pickle=False)
            Image.fromarray(labels).save(png_path)
            if args.save_scores:
                np.save(label_dir / (case_id + "_scores.npy"), scores.numpy(), allow_pickle=False)
            record.update({"status": "COMPLETE", "label_path": str(label_path.resolve()),
                           "label_sha256": sha256(label_path), "png_path": str(png_path.resolve()),
                           "png_sha256": sha256(png_path), "shape": list(labels.shape),
                           "resized_shape": image_info["resized_shape"], **metrics})
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            del scores, sample, labels
            gc.collect()
            if args.device != "cpu":
                torch.cuda.empty_cache()
        output_records.append(record)
        save_json(args.output / "progress.json", {"status": "RUNNING", "cases": output_records,
                                                  "planned_cases": len(cases)})
        print(json.dumps({"anonymous_id": case_id, "status": record["status"],
                          "attempted": len(output_records), "planned": len(cases)}), flush=True)
    success_count = sum(record["status"] == "COMPLETE" for record in output_records)
    prediction_manifest = {"status": "SEALED", "protocol_sha256": args.protocol_sha256.lower(),
                           "input_manifest_sha256": report["cohort"]["manifest_sha256"],
                           "label_registry_sha256": report["label_registry"]["sha256"],
                           "adapter_sha256": report["adapter_sha256"], "annotations_read": False,
                           "background_index": 456, "dtype": "uint16", "cases": output_records,
                           "complete_cases": success_count, "failed_cases": len(cases) - success_count}
    manifest_path = args.output / "predictions_manifest.json"
    save_json(manifest_path, prediction_manifest)
    receipt = {"status": "completed", "meaning": "All frozen cases attempted; failures remain explicit",
               "predictions_manifest_path": str(manifest_path.resolve()),
               "predictions_manifest_sha256": sha256(manifest_path),
               "label_registry_sha256": report["label_registry"]["sha256"],
               "inference_manifest_sha256": report["cohort"]["manifest_sha256"],
               "protocol_sha256": args.protocol_sha256.lower(), "adapter_sha256": report["adapter_sha256"],
               "complete_cases": success_count, "failed_cases": len(cases) - success_count,
               "cases": [{key: value for key, value in record.items()
                          if key in {"anonymous_id", "status", "label_path", "label_sha256", "input_sha256", "error"}}
                         for record in output_records]}
    save_json(args.output / "receipt.json", receipt)
    save_json(args.output / "progress.json", {"status": "COMPLETED", "cases": output_records,
                                              "planned_cases": len(cases)})
    report["status"] = "COHORT_INFERENCE_COMPLETE"
    report["cohort_result"] = receipt
    report["scientific_scope"] = "Image-only sealed predictions; scoring is a separate process and all failures are retained"
    return report


def run(args, report):
    source = args.source.resolve()
    required_paths = [args.checkpoint, args.clip_weights, args.dino_weights,
                      args.source_manifest, args.dino_repo / "hubconf.py"]
    if args.image:
        required_paths.append(args.image)
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    cohort_cases = validate_cohort_protocol(args, report) if args.cohort_manifest else None
    assets = {"checkpoint": (args.checkpoint, args.checkpoint_sha256.lower()),
              "clip": (args.clip_weights, CLIP_SHA256), "dino": (args.dino_weights, DINO_SHA256)}
    report["assets"] = {}
    for name, (path, expected) in assets.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"{name} asset hash mismatch: {actual}")
        report["assets"][name] = {"path": str(path.resolve()), "sha256": actual}
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    verified_files = 0
    for relative, expected in manifest["files"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source):
            raise ValueError(f"Frozen source manifest path escapes source directory: {relative}")
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
        verified_files += 1
    report["source_integrity"] = {"status": "PASS", "verified_files": verified_files,
                                  "scope": "Every file in source_manifest.json"}
    report["source_commit"] = manifest["source_commit"]
    report["source_manifest_sha256"] = sha256(args.source_manifest)
    report["dino_source"] = {"path": str(args.dino_repo.resolve()),
                             "hubconf_sha256": sha256(args.dino_repo / "hubconf.py")}
    # Preserve the frozen tree even when Python imports it.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    sys.path.insert(1, str(source / "open_clip/src"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["XFORMERS_DISABLED"] = "1"

    import numpy as np
    import torch
    from PIL import Image

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable")
    report["runtime"] = {"python": sys.version, "torch": torch.__version__,
                         "cuda_build": torch.version.cuda, "device": args.device,
                         "xformers_disabled": True, "autocast": False,
                         "outer_torch_inference_mode": True}

    def load_local_dino():
        backbone = torch.hub.load(str(args.dino_repo.resolve()), "dinov2_vits14", source="local", pretrained=False)
        state = torch.load(args.dino_weights.resolve(), map_location="cpu", weights_only=True)
        backbone.load_state_dict(state, strict=True)
        return backbone

    overlay = FrozenOverlay(source, manifest, args.device, load_local_dino)
    sys.meta_path.insert(0, overlay)
    report["compatibility_patches"] = overlay.ledger
    # Genuine Detectron2 and genuine upstream modules; no stand-in operators.
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.data import MetadataCatalog, detection_utils
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.structures import Instances
    from baselines import add_mask_former_config
    from baselines.data import transforms as official_transforms
    from baselines.third_party import clip
    from timm.utils import random_seed

    def local_clip_download(url, root=None):
        if url != clip._MODELS["ViT-B/16"]:
            raise RuntimeError(f"Unexpected CLIP asset requested: {url}")
        return str(args.clip_weights.resolve())
    clip._download = local_clip_download

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.set_new_allowed(True)
    config_path = source / "configs/zero_shot/partcatseg_voc.yaml"
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(args.checkpoint.resolve())
    cfg.MODEL.DEVICE = args.device
    cfg.OUTPUT_DIR = str(args.output.resolve())
    cfg.EVAL_ONLY = True
    metadata_path = args.label_registry or args.vocabulary_json
    if metadata_path:
        vocab = json.loads(metadata_path.read_text(encoding="utf-8"))
        if args.label_registry:
            labels = vocab["labels"]
            if len(labels) != 456 or vocab.get("background_index") != 456:
                raise ValueError("PACO registry must contain 456 labels and background_index=456")
            if vocab.get("ignore_index") != 65535:
                raise ValueError("PACO registry must use ignore_index=65535")
            if [label["label_index"] for label in labels] != list(range(456)):
                raise ValueError("Registry labels must appear in exact contiguous label_index order 0..455")
            if len({label["paco_category_id"] for label in labels}) != 456:
                raise ValueError("PACO category IDs must be unique")
            for label in labels:
                if not isinstance(label["paco_name"], str) or not isinstance(label["rendered_name"], str):
                    raise ValueError("Registry class names must be strings")
            rendered = [label["rendered_name"] for label in labels]
            registry_objects = vocab["object_classes"]
            vocab = {"stuff_classes": rendered, "obj_classes": registry_objects,
                     "class_ids": list(range(456)), "ignore_label": 65535}
            report["label_registry"] = {"path": str(args.label_registry.resolve()),
                                        "sha256": sha256(args.label_registry), "background_index": 456}
        classes, objects, ignore = vocab["stuff_classes"], vocab["obj_classes"], vocab["ignore_label"]
        if not classes or len(set(classes)) != len(classes) or not objects or len(set(objects)) != len(objects):
            raise ValueError("Class and object vocabularies must be nonempty ordered unique strings")
        if any(not isinstance(name, str) or name.count("'s") != 1 or
               any(not part.strip() for part in name.split("'s")) for name in classes):
            raise ValueError("Each class must use the official object's part syntax")
        if any(name.split("'s")[0].strip() not in objects for name in classes):
            raise ValueError("All objects in part names must be listed in obj_classes")
        if not isinstance(ignore, int) or ignore < max(len(classes), len(objects)) or ignore > 65535:
            raise ValueError("ignore_label must fit uint16 and exceed every contiguous metadata index")
        if (args.paco or args.label_registry) and (len(classes) != 456 or ignore != 65535):
            raise ValueError("Formal PACO vocabulary must contain all 456 classes and ignore_label=65535")
        metadata_name = "partcatseg_image_only_" + sha256(metadata_path)[:16]
        MetadataCatalog.get(metadata_name).set(stuff_classes=classes, obj_classes=objects, ignore_label=ignore)
        cfg.DATASETS.TEST = (metadata_name,)
        report["vocabulary_adapter"] = {"path": str(metadata_path.resolve()),
                                         "sha256": sha256(metadata_path), "paco": bool(args.paco or args.label_registry)}
        ids = vocab.get("class_ids", list(range(len(classes))))
    else:
        meta = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        classes, objects, ignore = list(meta.stuff_classes), list(meta.obj_classes), int(meta.ignore_label)
        ids = list(range(len(classes)))
    if len(ids) != len(classes) or len(set(ids)) != len(ids) or any(
            not isinstance(value, int) or value < 0 or value > 65535 or value == ignore for value in ids):
        raise ValueError("class_ids must be unique uint16 labels, one per class, excluding ignore_label")
    if ids != list(range(len(classes))):
        raise ValueError("Prediction contract requires contiguous class indices; category IDs belong in the registry")
    if not metadata_path and len(classes) != 116:
        raise RuntimeError(f"Expected the official 116-class VOC test vocabulary, got {len(classes)}")
    cfg.freeze()
    if (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.FORMAT) != (640, 768, "RGB"):
        raise RuntimeError("Unexpected official image preprocessing config")
    report["config_sha256"] = sha256(config_path)
    report["seed"] = int(cfg.SEED)
    report["preprocessing"] = {"input_format": "RGB", "resize_shortest_edge": 640,
                               "resize_max_size": 768, "mapper_padding": False,
                               "model_ImageList_size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
                               "clip_internal_size": [384, 384], "dino_internal_size": [448, 448]}
    report["vocabulary"] = {"dataset": cfg.DATASETS.TEST[0], "class_count": len(classes),
                            "stuff_classes": classes, "obj_classes": objects,
                            "class_ids": ids, "background_index": len(classes), "ignore_label": ignore}
    report["prediction_contract"] = {"dtype": "uint16", "part_indices": [0, len(classes) - 1],
                                      "background_index": len(classes),
                                      "ignore_label_used_only_for_input_placeholders": ignore,
                                      "background_predictions_must_count_as_errors_on_valid_GT": True}
    save_json(args.output / "run.json", report)

    class AuditedCheckpointer(DetectionCheckpointer):
        # Torch >=2.6 changed torch.load's default. Assets above are explicitly
        # hash-bound local official checkpoints; preserve the old loader behavior.
        def _torch_load(self, path):
            return torch.load(path, map_location="cpu", weights_only=False)

        def _load_model(self, checkpoint):
            if checkpoint.get("matching_heuristics", False):
                raise RuntimeError("Heuristic checkpoint key matching is forbidden")
            raw = checkpoint["model"]
            if not isinstance(raw, dict) or not raw:
                raise ValueError("Checkpoint must contain a nonempty model state dictionary")
            strip_module = all(key.startswith("module.") for key in raw)
            normalized = {key[7:] if strip_module else key: value for key, value in raw.items()}
            current = self.model.state_dict()
            missing = sorted(set(current) - set(normalized))
            unexpected = sorted(set(normalized) - set(current))
            bad_shapes = [{"key": key, "checkpoint": list(normalized[key].shape),
                           "model": list(current[key].shape)} for key in sorted(set(current) & set(normalized))
                          if tuple(normalized[key].shape) != tuple(current[key].shape)]
            report["checkpoint_raw_audit"] = {"status": "FAIL" if missing or unexpected or bad_shapes else "PASS",
                                               "checkpoint_key_count": len(raw), "model_key_count": len(current),
                                               "module_prefix_removed_for_comparison": strip_module,
                                               "missing_keys": missing, "unexpected_keys": unexpected,
                                               "incorrect_shapes": bad_shapes, "heuristics": False}
            if missing or unexpected or bad_shapes:
                raise RuntimeError("Raw checkpoint/model keys or shapes differ; no inference result accepted")
            result = super()._load_model(checkpoint)
            report["checkpoint_compatibility"] = {
                "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
                "incorrect_shapes": [list(item) for item in result.incorrect_shapes]}
            if result.missing_keys or result.unexpected_keys or result.incorrect_shapes:
                raise RuntimeError("Checkpoint/model mismatch; inspect run.json before accepting any result")
            return result

    random_seed(cfg.SEED)
    # Fix arithmetic for class-batch decoder validation. The upstream setting
    # enables cuDNN benchmarking/TF32; different batch sizes may choose kernels
    # with visibly different rounding. No learned weights or operations change.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    report["numeric_execution_policy"] = {
        "float32_convolution": True, "tf32": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "reason": "Validate decoder batching under fixed full-float32 arithmetic",
        "upstream_cudnn_benchmark": bool(cfg.CUDNN_BENCHMARK),
    }
    report["runtime"]["numerical_settings"] = {
        "cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "xformers_disabled": True, "autocast": False,
        "gpu_name": torch.cuda.get_device_name(0) if args.device != "cpu" else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if args.device != "cpu" else None}
    build_started = time.perf_counter()
    model = build_model(cfg)
    AuditedCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    report["build_seconds"] = time.perf_counter() - build_started
    # Preserve upstream caches constructed before final checkpoint load. Do not
    # recompute the random-background text cache from the loaded parameter.
    report["text_cache_policy"] = "Preserved official seeded initialization order and pre-checkpoint caches"
    report["text_cache_sha256"] = {}
    for name in ("text_part_obj_features", "text_specific_part_features",
                 "text_part_obj_features_test", "text_specific_part_features_test"):
        cached = getattr(model.sem_seg_head.predictor, name)
        report["text_cache_sha256"][name] = hashlib.sha256(cached.detach().cpu().numpy().tobytes()).hexdigest()

    if cohort_cases is not None:
        return infer_cohort(args, report, cfg, model, cohort_cases)
    sample, report["image"] = prepare_image_sample(args.image, cfg, ignore)
    resized_shape = tuple(sample["image"].shape[-2:])
    placeholders = sample["sem_seg"]
    report["input_provenance"] = {"annotations_read": False, "dummy_mask_value": ignore,
                                  "dummy_gt_class": 0, "exported_output": "sem_seg_all",
                                  "category_conditioned_sem_seg_discarded": True}

    def predict(model_sample=sample):
        scores, metrics = predict_unfiltered(model, model_sample, len(classes), args.device)
        report.update(metrics)
        report.setdefault("forward_metrics", []).append(metrics)
        return scores

    aggregator = model.sem_seg_head.predictor.transformer
    binding = decoder_binding(args, report)
    if args.decoder_chunk_size and args.verify_decoder:
        reference = predict()
        install_decoder_chunking(aggregator, args.decoder_chunk_size)
        scores = predict()
        max_error = float((reference - scores).abs().max())
        same_labels = bool(torch.equal(reference.argmax(0), scores.argmax(0)))
        close = bool(torch.allclose(reference, scores, rtol=1e-5, atol=1e-6))
        receipt = {**binding, "status": "PASS" if close and same_labels else "FAIL",
                   "validation_run_complete": False,
                   "verification_vocabulary": "official VOC-116", "class_count": len(classes),
                   "image_sha256": report["image"]["sha256"], "max_abs_score_error": max_error,
                   "scores_allclose_rtol": 1e-5, "scores_allclose_atol": 1e-6,
                   "argmax_identical": same_labels,
                   "scope": "One VOC smoke image on the recorded device; not a universal proof"}
        save_json(args.output / "decoder_equivalence.json", receipt)
        report["decoder_equivalence"] = receipt
        if receipt["status"] != "PASS":
            raise RuntimeError("Decoder chunking equivalence check failed")
        del reference
    else:
        if args.decoder_chunk_size:
            validate_decoder_receipt(args, report)
            install_decoder_chunking(aggregator, args.decoder_chunk_size)
        scores = predict()
    report["decoder_chunk_size"] = args.decoder_chunk_size
    if args.verify_dummy_invariance:
        if len(objects) < 2:
            raise ValueError("At least two object names are needed for the dummy-class invariance check")
        changed_instances = Instances(resized_shape)
        changed_instances.gt_classes = torch.ones(1, dtype=torch.int64)
        changed_sample = {**sample, "instances": changed_instances,
                          "sem_seg": torch.zeros_like(placeholders),
                          "obj_part_sem_seg": torch.ones_like(placeholders)}
        changed_scores = predict(changed_sample)
        close = bool(torch.allclose(scores, changed_scores, rtol=1e-5, atol=1e-6))
        same_labels = bool(torch.equal(scores.argmax(0), changed_scores.argmax(0)))
        report["dummy_target_invariance"] = {
            "status": "PASS" if close and same_labels else "FAIL",
            "original": {"sem_seg": ignore, "obj_part_sem_seg": ignore, "gt_classes": 0},
            "changed": {"sem_seg": 0, "obj_part_sem_seg": 1, "gt_classes": 1},
            "max_abs_score_error": float((scores - changed_scores).abs().max()),
            "scores_allclose_rtol": 1e-5, "scores_allclose_atol": 1e-6,
            "argmax_identical": same_labels,
            "annotations_read": False}
        if not close or not same_labels:
            raise RuntimeError("sem_seg_all depends on dummy targets or is not numerically reproducible")
        del changed_scores
    else:
        report["dummy_target_invariance"] = {"status": "NOT_RUN"}
    if args.verify_decoder:
        report["decoder_equivalence"]["validation_run_complete"] = True
        report["decoder_equivalence"]["dummy_target_invariance"] = report["dummy_target_invariance"]
        save_json(args.output / "decoder_equivalence.json", report["decoder_equivalence"])
    labels = scores.argmax(0).numpy()
    lookup = np.asarray(ids + [len(classes)], dtype=np.uint16)
    mapped = lookup[labels]
    np.save(args.output / "prediction.npy", mapped, allow_pickle=False)
    Image.fromarray(mapped).save(args.output / "prediction.png")
    if args.save_scores:
        np.save(args.output / "sem_seg_all.npy", scores.numpy(), allow_pickle=False)
    report["outputs"] = {name: sha256(args.output / name) for name in
                         ["prediction.npy", "prediction.png"] + (["sem_seg_all.npy"] if args.save_scores else [])}
    report["predicted_pixels"] = {str(int(key)): int(count) for key, count in zip(*np.unique(mapped, return_counts=True))}
    report["status"] = "SMOKE_INFERENCE_COMPLETE"
    report["scientific_scope"] = "Image-only raw model output; no accuracy claim or annotation-based scoring"
    return report


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "run.json").exists():
        raise FileExistsError("Use a fresh output directory to preserve previous run evidence")
    report = {"status": "STARTED", "adapter_sha256": sha256(__file__),
              "patch_version": PATCH_VERSION, "argv": sys.argv[1:], "network_disabled": True}
    save_json(args.output / "run.json", report)
    try:
        with no_network():
            run(args, report)
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        save_json(args.output / "run.json", report)
        raise
    save_json(args.output / "run.json", report)
    print(json.dumps({"status": report["status"], "report": str((args.output / "run.json").resolve())}))


if __name__ == "__main__":
    main()
