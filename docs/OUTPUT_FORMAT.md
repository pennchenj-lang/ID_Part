# HPID package format

## Required files

| Path | Meaning |
| --- | --- |
| `source.png` | RGB source copied into the package |
| `semantic_ids.png` | Semantic class index per pixel |
| `semantic_preview.png` | Color visualization only |
| `part_id_map.tiff` | Unsigned 16-bit instance index per pixel |
| `parts.json` | Per-ID metadata and asset paths |
| `group_id_map.tiff` | Unsigned 16-bit editable physical-group index per pixel |
| `groups.json` | Per-group semantics, member Part IDs, geometry, and evidence |
| `group_id_preview.png` | Colour visualization of editable physical groups |
| `group_overlay.png` | Physical-group colours over source-image edges |
| `taxonomy.json` | Fine classes, parent classes, and detail classes |
| `package_manifest.json` | Format version, algorithm provenance, and file hashes |

Foundation-fusion runs also export `candidates.json`, `candidate_masks/`, and
`inference_diagnostics.json`. Completion runs export `masks_full/`,
`crops_completed/`, and `occlusion_edges.json`.

## Part record

The principal fields are:

- `part_id`: stable hierarchical string.
- `semantic_name`: class assigned to this physical part.
- `semantic_parent`: taxonomy relationship.
- `assembly_parent_id`: concrete exported Part ID used for assembly.
- `instance_index`: integer stored in `part_id_map.tiff`.
- `group_id`: editable physical group containing this fine Part ID. Readers of
  format `0.2.0` packages may fall back to `part_id` when this field is absent.
- `side`: `left`, `right`, or `center`, measured relative to the assembly parent.
- `bbox_visible`, `centroid_xy`, `area_px`: visible geometry.
- `mask_visible_path`, `crop_path`, `crop_offset`: visible asset files.
- `mask_full_path`, `crop_completed_path`, `bbox_full`: optional inferred asset.
- `completion_metadata`: accepted, rejected, or budget-skipped state and evidence.

`semantic_parent` is a class relationship. `assembly_parent_id` is an instance
reference. They are intentionally not interchangeable.

## Physical group record

Format `0.3.0` adds a conservative editing layer without deleting fine Part
IDs. Each `groups.json` record includes `group_id`, `group_index`,
`semantic_name`, `asset_id`, `member_part_ids`, visible geometry, evidence, and
`review_required`. Appearance regions such as `visual_panel_*` are candidate
evidence and cannot independently create a physical group. The validator
requires `group_id_map.tiff` to cover exactly the same foreground as
`part_id_map.tiff`.

Fast multi-object scene packages prioritize independent object masks. Because
that low-latency route skips the full scene-ontology model, every exported Group
uses `semantic_name=scene_object`, sets `review_required=true`, and prefixes its
evidence with `provisional_scene_label/`. The ID and mask are usable without
presenting an uncertain category as verified. Ensemble scene packages retain
the full ontology-consensus route.

## Provenance

`package_manifest.json` records:

- HPID-Split version and inference mode;
- zero-ground-truth inference assertion;
- prompt-bank SHA-256;
- exact detector, SAM2, and dense fallback model IDs;
- decomposition mode, guided prompt count, and resolved SAM3 or experimental
  Grounded-SAM2 text backend;
- root-domain arbitration and multiscale visual-region diagnostics;
- the relational appearance algorithm identifier when the second pass ran;
- fusion switches and thresholds;
- completion backend name;
- SHA-256 and byte count for each payload file.

Run `hpid-split validate-package --package PATH` before consuming a package.
The validator checks file hashes, map dimensions, fine and group ID uniqueness,
group membership, parent resolution, visible masks, amodal superset constraints,
and pixel-exact visible locking. It accepts legacy `0.2.0` packages and enforces
the group products for `0.3.0` packages.
