# HPID-Split

HPID-Split decomposes one asset image into hierarchy-constrained, auditable
Part IDs. Its core contribution is the fusion and identity-assignment layer:
open-vocabulary models propose regions, while HPID-Split decides which
proposals can coexist, which describe the same physical part, who owns each
pixel, and how stable IDs and parent relations are assigned.

The current prompt bank covers characters, vehicles, furniture, tools/props,
containers, devices, structures, terrain, and natural objects. Automatic Fast
regression has been run on two character layouts, a knife, a firearm, a globe,
a repeated-rock game scene, and a two-building scene. These are unlabelled
software and structural checks, not a cross-domain accuracy benchmark.

## Research releases

Version `0.3.0` is the frozen research-preview baseline. See
[FROZEN_RELEASE.md](FROZEN_RELEASE.md) for its evidence boundary and
[the locked paper protocol](docs/PAPER_EXPERIMENT_PROTOCOL.md) for leakage
rules. The repository also includes portable results from 65 materialized
PACO-LVIS oracle-crop holdout cases in
[experiments/paper_v030_holdout](experiments/paper_v030_holdout). The automatic
condition reaches object IoU 0.7368 but Part-F1@0.25 only 0.3319 and semantic
part recall 0.1782; these negative results are intentionally retained. This
release is not advertised as error-free universal part segmentation.

Version `0.3.1` keeps package format `0.3.0` and revises the physical-group
layer. It adds semantic-shape rejection, repeated-part shape recovery,
open-versus-narrow-container interior rules, conservative repeated-instance
splitting, and quantized boundary refinement for deterministic export. The
fine Part-ID maps and proposal candidates used for the 226-case regroup audit
were frozen; only editable groups were regenerated. Because these cases were
inspected while the grouping rules were debugged, they are now a regression
set rather than an untouched independent test set.

Version `0.3.2` keeps the frozen `0.3.1` quantitative evidence unchanged and
repairs qualitative cross-domain grouping. A globe's meridian support is
recovered as one continuous structure even when its lower arc is rejected as
shading-like appearance evidence. Layered character clothing keeps the inner
top, outer garment, and lower garment separate while anatomical regions share
the body group. Scene instances now use a multicluster support-surface model,
bounded appearance-marker supplementation, and structure-aware projected-face
fusion so repeated trees and rocks can separate without turning every lit face
into a public ID. These post-freeze examples are engineering regressions, not
additional benchmark estimates.

Version `0.3.3` adds an all-candidate photometric boundary audit before final
physical grouping. Closed highlights, cast shadows, smooth illumination
gradients, colour patches, and texture patches remain evidence-only regions;
they cannot create a public Part ID without independent semantic and physical
structure support. Bounded optical components such as screens, lenses, and
windshields remain eligible when their own closed geometry is supported. The
release includes regression tests for same-material highlights and shadows and
an audited six-domain qualitative plate whose public ID names contain no
photometric-region labels. Frozen `0.3.1` benchmark numbers are unchanged.

## What is implemented

- One-image CLI and local upload UI.
- Two decomposition modes: label-free automatic structure discovery and
  prompt-guided Part IDs.
- A proposal-first Fast path that runs one global SAM2 mask pool, reuses it for
  roots and parts, and derives a primary-asset foreground from the image border.
  Grounding DINO remains a fallback when no coherent image-only root is found.
- A strictly serial three-level evidence policy: the routed semantic part
  inventory first defines admissible public parts; geometry and topology are
  evaluated only after that semantic gate passes; colour, luminance, texture,
  and edge closure are evaluated last and may only confirm or reject the
  structural match. A later stage cannot rescue an earlier failure. Appearance
  cues are image-space proxies, not physical material recognition.
- Cross-candidate semantic/structure matching for enclosed surfaces. For
  example, a phone-screen hypothesis supplies the semantic slot, while an
  independently inset, closed rectangular region supplies the actual display
  boundary. Both are rejected when the last-stage evidence is better explained
  by a highlight or shadow.
- Adaptive category-guided mask discovery. Profiles such as humanoids, globes,
  and microwaves request one bounded Grounded-SAM2 refinement pass; ordinary
  profiles keep the lower-latency proposal-first path.
- Scale-balanced appearance proposals reserve capacity for small details,
  medium parts, and large structural surfaces instead of allowing one scale to
  exhaust the candidate budget. Closed contours and enclosed interiors recover
  lenses, displays, doors, and framed panels; nested texture-only marks are
  suppressed when they do not form an independently editable structure.
- Repeated semantic support can merge fragmented observations of one physical
  region, while disconnected repeated scene instances retain separate asset and
  Part IDs.
- Category-independent silhouette-bottleneck proposals for attached lobes and
  appendages that SAM2 did not isolate cleanly.
- Scene-aware proposal routing: a selected object root cannot be reintroduced
  as a scene-layer child, each region is assigned to the most specific
  compatible root, and a broad terrain layer only fills unclaimed pixels.
- Scene-panel recovery closes small gaps in a background-ring proposal, derives
  the enclosed object envelope, and rejects duplicate rectangular panel fills.
  This operates on topology and proposal agreement rather than fixed image
  coordinates or a named test file.
- Optional reviewed-prototype learning: retrieve similar objects, recover their
  recurrent part inventory, and rerank masks with learned visual and geometry
  priors without adding a per-image code rule.
- Hierarchical Grounding DINO + SAM2 candidate generation.
- Masked CLIPSeg root-domain arbitration so broad detector labels cannot choose
  an unrelated ontology by confidence alone.
- Label-free, multiscale SAM2 automatic-mask proposals, spatially stable visual
  IDs, and containment-derived visual parent relations.
- Optional SAM3 promptable-concept segmentation. User phrases provide semantic
  identity, SAM3 provides masks, and automatic SAM2 regions still fill omissions.
- Optional CLIPSeg + SAM2 fallback for small or missed semantic details.
- Correlation-aware multi-source evidence fusion.
- Cross-source physical-part association before Part-ID creation.
- A candidate-to-physical fusion layer. Colour, shading, texture, and raw SAM2
  regions remain internal evidence; editable Group IDs are emitted only after
  connectivity cleanup, semantic inventory constraints, and structural fusion.
- Two public ID levels: fine Part IDs for audit and local detail, plus
  conservative editable Group IDs. Character anatomy shares a stable body
  group, one upper garment includes its sleeves, and repeated scene objects
  remain distinct.
- Character grouping audits the central head region against independently
  supported face/hand skin before assigning an overbroad hair mask. Boundary
  closing is clipped back to the original foreground, so grouping cannot add
  subject pixels.
- Character clothing zones are measured along a head-to-feet pose axis instead
  of the image y-axis. The upper/lower boundary is derived from the detected
  head end and footwear start; paired skin-consistent lower-limb regions can
  return from an overbroad footwear mask without changing the shoe pixels.
- Inventory-constrained knife decomposition uses a long-axis width bottleneck
  for the blade boundary and relative axial material changes for a handle wrap.
  It does not use a fixed hue, image coordinate, or file-name rule.
- Firearm decomposition uses axis topology to establish a seven-part inventory
  and trusted interior seeds. A supported fore-end seed supplies a smoothed Lab
  material reference with lightness downweighted, allowing highlighted and
  shadowed portions of one handguard to reconnect before an edge-preserving
  watershed aligns public boundaries. Appearance cannot create a Group ID.
- Coherence-aware strong-containment association: a local support mask can join
  one connected physical part without collapsing disconnected repeated parts.
- Area-adaptive GrabCut boundary refinement with an image-gradient acceptance
  gate and detail-preserving cleanup for thin visual regions.
- Weak-evidence border-surface rejection for primary-asset routing.
- A second, image-only relational appearance pass. Stable first-pass parts act
  as anchors for configured local detail proposals, which are then sent through
  the same ownership and identity fusion instead of being copied to the output.
- Parent-constrained ownership, residual recovery, cleanup, and remainder
  attachment.
- Stable `part_id`, `semantic_parent`, and resolved `assembly_parent_id` fields.
- Optional evidence-gated hidden-region completion with LaMa appearance
  inpainting and SAM2 mask recovery.
- Candidate audit masks, full provenance, SHA-256 file manifest, and an
  independent package validator.

Hidden completion is an inferred amodal hypothesis. It is not physical interior
reconstruction, depth recovery, or proof of unseen geometry.

## Installation

Python 3.10 or newer is required. A CUDA GPU is strongly recommended for the
foundation-model path.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[foundation,ui]"
```

On Windows PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

To enable hidden-region completion, install the isolated LaMa adapter and write
a machine-local configuration:

```bash
hpid-split setup-completion
```

The generated config and model files are stored under `~/.hpid-split` by
default. Use `--package-root`, `--model-cache`, and `--config` to place them on
another drive. A clone-local `configs/lama_sam2_evidence.local.json` is also
recognized and ignored by Git.

## Upload UI

```bash
hpid-split-web
```

Open `http://127.0.0.1:7860`. In automatic mode, `Fast` uses the reusable SAM2
proposal pool, shape proposals, cross-cue fusion, and a bounded one-iteration
boundary pass. Grounding DINO is lazy during root discovery, but profiles that
explicitly require semantic refinement, including humanoid characters, may
load it for one bounded part-query pass.
`Ensemble` runs the broader detector/semantic stack and is intentionally slower.
For `Entire scene`, Fast prioritizes distinct object masks and marks every
natural-language scene label as provisional. Ensemble additionally runs the
SigLIP 2 scene-ontology consensus when semantic object names are required.

`Automatic` needs no part names. `Prompt-guided` prefers the gated
[`facebook/sam3`](https://huggingface.co/facebook/sam3) weights:

```bash
hf auth login
```

Accept the model terms on its Hugging Face page before logging in. HPID-Split
does not substitute CLIP similarity labels when SAM3 is unavailable. The
default `auto` backend records an explicit Grounded-SAM2 fallback in the status
line and diagnostics; use `--guided-backend sam3` to require SAM3.

## CLI

Fast split:

```bash
hpid-split auto \
  --image input.png \
  --output runs/example \
  --prompt-bank configs/general_asset_prompts.json \
  --dense-semantic-fallback \
  --decomposition-mode automatic \
  --proposal-first-fast \
  --no-isolated-profile-resolution \
  --no-profile-refinement \
  --grabcut-iterations 1 \
  --device auto
```

Retrieval-augmented automatic split:

```bash
hpid-split build-retrieval-index \
  --manifest data/references.json \
  --output models/hpid_retrieval_v1 \
  --device auto

hpid-split auto \
  --image input.png \
  --output runs/example_retrieved \
  --retrieval-index models/hpid_retrieval_v1 \
  --dense-semantic-fallback \
  --device auto
```

Only references explicitly marked `reviewed: true` can enter the index.
Unknown objects below the open-set threshold receive no retrieved labels. See
[docs/RETRIEVAL_LEARNING.md](docs/RETRIEVAL_LEARNING.md) for the manifest,
training, auditing, and incremental-update workflow.

Prompt-guided split:

```bash
hpid-split auto \
  --image input.png \
  --output runs/example_guided \
  --prompt-bank configs/general_asset_prompts.json \
  --dense-semantic-fallback \
  --decomposition-mode prompt-guided \
  --guided-backend auto \
  --part-prompts "stock=buttstock|rear stock, magazine, receiver, trigger" \
  --visual-crop-layers 1 \
  --device auto
```

Simple comma/newline lists are accepted. The optional
`id=phrase|alias` syntax keeps an ASCII ID while allowing multiple search
phrases. Prompt-guided mode remains open-set: unnamed SAM2 regions are exported
as `*_visual_panel_*`, `*_visual_strip_*`, or `*_visual_detail_*` instead of
being discarded.

Ensemble split with hidden-region completion:

```bash
hpid-split auto \
  --image input.png \
  --output runs/example_complete \
  --prompt-bank configs/general_asset_prompts.json \
  --grounding-model IDEA-Research/grounding-dino-tiny \
  --additional-grounding-model IDEA-Research/grounding-dino-base \
  --dense-semantic-fallback \
  --completion-config configs/lama_sam2_evidence.local.json \
  --device auto
```

Validate an exported package:

```bash
hpid-split validate-package --package runs/example_complete
```

Rebuild only the editable physical-group layer after a grouping-algorithm
upgrade, while preserving the frozen fine Part-ID map and candidate evidence:

```bash
python scripts/regroup_package.py \
  --package runs/example_complete \
  --output runs/example_regrouped
```

## Output package

Each run includes:

- `part_id_map.tiff`: lossless instance-index map.
- `parts.json`: stable IDs, semantics, hierarchy, geometry, and completion state.
- `group_id_map.tiff` and `groups.json`: conservative physical editing groups.
- `group_id_preview.png` and `group_overlay.png`: default editing and edge views.
- `masks_visible/` and `crops/`: visible per-ID assets.
- `masks_full/` and `crops_completed/`: optional inferred amodal assets.
- `candidates.json` and `candidate_masks/`: accepted proposal audit trail.
- `inference_diagnostics.json`: model, prompt, fusion, and rejection diagnostics.
- `package_manifest.json`: algorithm provenance and SHA-256 for every payload file.

See [docs/ALGORITHM.md](docs/ALGORITHM.md) and
[docs/OUTPUT_FORMAT.md](docs/OUTPUT_FORMAT.md) for the exact contract.

## Current evidence and limits

The checked-in development record is
[benchmarks/development_audit_v0.1.json](benchmarks/development_audit_v0.1.json).
It contains exact values copied from frozen outputs plus their source hashes.
The controlled before/after record for the relational appearance stage is
[benchmarks/relational_appearance_audit_v0.1.json](benchmarks/relational_appearance_audit_v0.1.json).
The targeted edge, identity, and root-routing regression is
[benchmarks/edge_identity_regression_v0.1.json](benchmarks/edge_identity_regression_v0.1.json).

- The 10-character fusion audit is internal, not a public benchmark.
- Ground truth is used only after inference for metric computation.
- On the internal 10-character dual-source audit, relational appearance raised
  fine mIoU from 0.3297 to 0.3473 and fine boundary F1 from 0.3410 to 0.3783.
  These are development results, not public-benchmark or external-domain claims.
- Small-part recall remains low despite that improvement and is still an active
  algorithmic limitation.
- The synthetic amodal audit tests self-occlusion recovery of predicted masks;
  it is not natural amodal ground truth.
- The public non-character study originally separated 65 PACO-LVIS oracle-crop
  development cases from 226 object instances with no source-image or
  object-instance overlap. The 226 cases cover 60 categories across devices,
  furniture, containers, vehicles, daily objects, and tools/props. They were
  subsequently inspected during physical-group debugging and must therefore be
  reported as a cross-category regression set, not an untouched independent
  test set or unseen-category generalization.
- On the frozen 226-case candidates, the `0.3.1` editable-group pass reports
  precision 0.4602, recall 0.4149, Part F1@0.25 0.4043, matched IoU 0.4669,
  boundary F1 0.4938, semantic recall 0.2107, and oversegmentation ratio 1.1523.
  Relative to the preceding physical-group pass, five cases improved and none
  regressed in Part F1@0.25, while the mean improvement was small. All 226
  packages passed manifest and structural validation. A second run under a
  different Python hash seed reproduced 1,356 core output files byte for byte.
  These results support deterministic packaging and a narrower fusion repair;
  the absolute F1 remains modest and does not establish error-free automatic
  decomposition.
- On the 226 frozen-candidate test cases, the A0-to-A3 fusion ablation raises
  Part F1@0.25 from 0.1511 to 0.3133 and recall from 0.1101 to 0.3656.  The
  absolute strict-overlap and semantic scores remain modest.  Every variant is
  predicted before test part labels or masks are loaded, but all methods receive
  the ground-truth object crop/root and category; this is not a full-image
  detection result.
- The current local reviewed-prototype seed contains ten stylized characters.
  It retrieved an unseen character while rejecting the tested rifle as
  out-of-index. This is an execution and rejection check, not a cross-domain
  accuracy benchmark; reviewed prop, object, and scene references are still
  required.
- Recorded core-pipeline stage totals on the development Windows/CUDA machine,
  excluding export and package validation, were 34.60 s for the goggle
  character (18 fine Parts, 7 Groups), 36.62 s for the T-pose character
  (15 Parts, 6 Groups), 35.78 s for the knife (15 Parts, exactly 3 Groups),
  32.23 s for the firearm (15 Parts, 7 Groups), 39.42 s for the globe (7 Parts,
  4 physical Groups), 27.85 s for the repeated-rock scene (22 Parts, 22 Groups),
  and 25.17 s for the two-building scene (6 Parts, 6 provisional Groups).
  All seven packages passed structural, hierarchy,
  foreground-conservation, byte-count, and recomputed SHA-256 validation with
  `ground_truth_used=false`. These are unlabelled regression checks: part count
  is not an accuracy metric, and timings depend on hardware, resolution, model
  cache state, and route selection.
- The default humanoid inventory keeps a unified editable body while separating
  supported hair, upper clothing, lower clothing, and accessories. Optional
  headwear and eyewear are not forced in automatic mode because a single text
  detector can confuse hair or empty eye sockets with those categories; they
  remain available in prompt-guided mode. This is not evidence of error-free
  decomposition on arbitrary characters or domains.
- Prompt-guided semantic quality has not been evaluated because the local test
  machine does not have access to the gated SAM3 production weights. The real
  SAM3 API path was exercised with Hugging Face's public random tiny model; this
  is an integration test only. The auditable Grounded-SAM2 fallback ran, but on
  the AK smoke test none of ten requested semantic IDs passed both ambiguity and
  independent dense-semantic evidence gates; generic structural IDs remained.
- Exact source hashes, stage timings, structural counts, and review items are
  recorded in
  [benchmarks/cross_domain_structural_regression_v0.7.json](benchmarks/cross_domain_structural_regression_v0.7.json).
- Scene Fast mode rejects canvas background masks, preserves a terrain fallback
  layer, prevents selected roots from reappearing as child regions, and keeps
  repeated rocks/trees as distinct IDs. Public Groups use the neutral
  `scene_object` semantic and require review; use Ensemble for the full ontology
  route.
- The low-contrast side bag in the current goggle-character check is omitted by
  automatic Fast mode because its detector mask also covered the upper garment.
  `character_bag` remains available in prompt-guided mode with a stricter area
  gate. In the two-building check, a thin display platform or attached
  shadow can enter an object envelope. Both remain review cases rather than
  evidence of general error-free decomposition.
- The system deliberately rejects unsupported hidden completions instead of
  filling every part.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

## Third-party models and licensing

HPID-Split does not rebrand external models as its own algorithm. Grounding
DINO, SAM2, SAM3, CLIPSeg, and LaMa provide proposals or appearance completion; the
hierarchy-constrained fusion, identity association, ownership, ID assignment,
completion gates, export, and audit logic live in this repository.

Read [THIRD_PARTY.md](THIRD_PARTY.md) before redistribution. In particular, the
official CLIPSeg repository states that its MIT code license does not cover the
model weights.

## Repository license

The original HPID-Split code in this repository is released under the
[Apache License 2.0](LICENSE). Third-party code, models, and checkpoints remain
subject to their own licenses and are not relicensed by this repository.
