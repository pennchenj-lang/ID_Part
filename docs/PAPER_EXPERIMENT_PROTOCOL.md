# Locked Paper Experiment Protocol

Protocol version: 1.0, paired with HPID-Split v0.3.0.

## Questions

The evaluation separates five questions that must not be collapsed into one
score:

1. Does the system recover the primary object silhouette?
2. Does it discover physically editable parts rather than visual fragments?
3. Does the serial semantic-structure-appearance gate improve precision without
   destroying useful part recall?
4. Are IDs and editable groups internally consistent and isolated under edits?
5. What are the latency, failure, and human-review costs across object domains?

## Locked datasets

### Public quantitative holdout

Use the PACO-LVIS holdout manifest whose SHA-256 is
`3d16426e4389d59c9fc2f820e8efd3f12c06aa6b881c28e5914479cb1f42dbcb`.
It lists 67 categories, but the frozen local materialization contains 65
executable cases; `sponge` and `pliers` have no case path. Report both counts and
do not silently treat the two incomplete entries as inference failures or remove
them from the manifest description.
The development manifest and test holdout must never be pooled. The object crop
uses the public ground-truth bounding box; this is an oracle-root-crop setting and
must be named as such. Part masks are hidden during inference and loaded only by
the evaluator.

### Cross-domain regression set

Use the seven source hashes in `benchmarks/frozen_cross_domain_v0.3.0.json` for
qualitative failure analysis, runtime profiling, and deterministic package
regression. These cases do not provide quantitative segmentation ground truth.

### Internal character data

The existing ten-character LOCO experiment may be reported separately. It is an
internal-domain supervised experiment and cannot be presented as external
generalization to arbitrary objects.

## Conditions

Run all conditions on identical inputs and seeds.

| ID | Condition | Purpose |
| --- | --- | --- |
| C0 | root mask only | object-silhouette baseline |
| C1 | semantic candidates only | tests inventory evidence |
| C2 | semantic then structure | tests topology and physical-boundary filtering |
| C3 | full serial gate | proposed semantic-structure-appearance method |
| C4 | full gate without profile refinement | profile-refinement ablation |
| C5 | full gate without retrieval/router aid | learned-routing ablation |

C1-C3 are strict cumulative ablations: a candidate rejected in an earlier stage
cannot re-enter later. Any implementation used only to expose intermediate gate
outputs must be checked against the frozen full-gate path so that C3 is byte-for-
byte equivalent to v0.3.0.

## Metrics

Report macro averages over independent cases and category-stratified results:

- root-object IoU, precision, and recall;
- part discovery precision, recall, and F1 at IoU 0.25 and 0.50;
- mean matched-mask IoU;
- boundary F1 with a fixed 3-pixel tolerance and a scale-normalized sensitivity
  check;
- semantic part recall;
- over-segmentation and under-segmentation ratios;
- editable-group count and fine-Part-ID count;
- package validity, unresolved-review rate, and failed-run rate;
- end-to-end latency, stage latency, peak RAM, and peak VRAM when available.

For paired comparisons, report the per-case difference, bootstrap 95% confidence
interval, and a paired non-parametric test. With small internal samples, emphasize
effect size and interval width rather than p-values.

## No-leakage rules

- No ground-truth part name, mask, count, or boundary may enter inference.
- A public object-category prompt is a separate prompted condition, never the
  automatic condition.
- Train-only prototype indexes must record source split and SHA-256.
- Failed cases stay in the denominator and are reported explicitly.
- Development results cannot be relabeled as holdout results.
- Human corrections are reported as human-assisted results, not automatic output.

## Reporting boundary

ID-map consistency and edit-isolation scores measure ownership contracts. They do
not establish perceptual quality, amodal correctness, 3D geometry, physical
depth, contact quality, or multiview Gaussian reconstruction. The current output
is a 2D part/group contract with downstream asset interfaces.
