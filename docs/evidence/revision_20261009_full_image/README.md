# Full-image evaluation of the original 42-object holdout

This is the retrospective full-image extension requested in reviewer comment 4. All 42 original materialized targets, from 42 distinct source images, remain in the denominator. It is an added analysis of the original cohort, not a new untouched holdout.

The frozen release `be543003632fa739d0b63e9acbffb8d91d99b84a` processes complete archived normalized source images at anonymous paths. It receives no annotated crop, root, category, target point or object identifier. The original primary-object policy chooses the output. Commands change only image and output paths. There was no fitting, prompt adjustment, threshold tuning or outcome-based retry. All 42 attempts completed and yielded valid packages.

## Principal result

| Metric | Cropped input | Full image | Full minus crop (paired 95% CI) |
|---|---:|---:|---:|
| Group F1 at IoU 0.25 (primary) | 0.3951 | 0.2532 | −0.1419 [−0.2378, −0.0552] |
| Root mask IoU | 0.7287 | 0.4600 | −0.2686 [−0.3855, −0.1606] |

The contrast quantifies the combined effect of target selection, context/scale, root construction and downstream grouping when the annotation-derived crop is removed. It does not isolate detector error and is not all-object scene detection AP. Selecting another object counts as an error against the original fixed target. It does not demonstrate recovery from occlusion or superiority over segmentation baselines.

The primary interval uses 10,000 paired image bootstrap draws, seed 20261009, with cases sorted by case_id. Secondary intervals are descriptive, without multiplicity adjustment. All metrics and counts are retained in the accompanying CSV and JSON files.

## Recompute the reported scores

With Python, NumPy and SciPy installed, run from this directory:

```text
python recompute_results.py --evidence . --output recomputed.json
```

This rebuilds Hungarian matches and F1 scores from the included float32 pairwise IoU matrices, root/foreground IoUs from integer pixel counts, and all paired intervals. It performs 1,246 per-case and statistical comparisons with the reported files. The recorded run passed with maximum per-case numerical difference 1.11e-16.

This portable check reproduces scoring from saved overlaps; it does not rerun neural inference. Raw RGB images are not redistributed here. Image/object identifiers, original image URLs, crop coordinates and image hashes are included in `overlap_evidence.json`. Neural replay requires separately obtaining the original images and frozen model assets. Historical local paths in the execution/audit records document the run and must be adapted for another installation.

## Controls and provenance

- The local protocol was fixed before full-image inference/scoring. It was not publicly preregistered. Its original bytes and SHA-256 are preserved.
- All seven original frozen assets matched their archived hashes. All 42 new commands were independently checked against their corresponding original commands.
- One original crop was rerun as an environment control, with identical Part/Group pixel maps and ordered identity/semantic records. The other 41 crops were not rerun on GPU.
- CPU replay of all 42 archived cropped predictions reproduced the historical Group F1 at IoU 0.25, 0.395109915134312. All full/crop object and part reference masks agreed pixel-for-pixel after translation; there was no reference truncation.
- Four historical cross-category replacement slots retain their actual materialized categories when scoring domain/profile accuracy. Stale slot labels were never inference inputs.
- `input_output_fingerprints.json` records anonymous input hashes and all output hashes. `commands_as_executed.json` retains the actual invocation records. `manifest_sha256.json` covers the files in this evidence release.

The revised manuscript reports the primary result in Section VI-G and the full protocol and secondary results in Supplementary Section S19.12, Tables S47–S48.
