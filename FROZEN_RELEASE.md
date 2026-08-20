# HPID-Split v0.3.0 Frozen Research Release

Freeze date: 2026-08-20 (Asia/Taipei)

This snapshot freezes the current HPID-Split implementation before paper-scale
evaluation. The source algorithm is identified as
`hpid-semantic-structure-appearance-verification-v2`.

## Frozen behavior

Final editable groups are produced by a serial evidence gate:

1. semantic and object-inventory eligibility;
2. structural and topological support;
3. appearance, edge, and shading consistency.

A later stage cannot rescue a candidate rejected by an earlier stage. Color,
material, brightness, texture, and foundation-model masks are proposal evidence;
none is a Part-ID by itself. The public package format is `0.3.0` and includes
fine Part-IDs plus editable group IDs.

## Verification at freeze

- Unit and integration tests: 493 passed.
- Static checks: Ruff passed.
- Seven cross-domain output packages passed the package validator.
- Every recorded inference package reports `ground_truth_used=false`.
- The real-image runs cover a phone, microwave, firearm, knife, globe,
  stylized character, and multi-rock scene.

The machine-readable records are in
`benchmarks/frozen_cross_domain_v0.3.0.json`.

## Evidence boundary

Package validation proves schema, file-integrity, ID-map, and relationship
consistency. It does not prove perceptual segmentation accuracy. The seven
cross-domain runs are regression and qualitative audit cases without dense
ground truth; their part counts, group counts, quality grades, and timings must
not be reported as mIoU, boundary F1, or accuracy.

Quantitative paper claims must come from the locked public benchmark protocol in
`docs/PAPER_EXPERIMENT_PROTOCOL.md`. Ground truth is available only to the
post-inference evaluator unless a condition is explicitly labeled as supervised
training or oracle-root evaluation.

## Release contents

The GitHub release contains source code, tests, configurations, documentation,
small audit summaries, and SHA-256 checksums. Model weights, caches, user images,
private run directories, and third-party datasets are intentionally excluded.

## Reproducing the snapshot

Install the project in a clean environment, run `pytest -q`, and run
`ruff check .`. Verify release files with `SHA256SUMS_RELEASE_v0.3.0.txt`.

The Git commit and tag are added only when the repository is published. Until
then, the dated source archive and its checksum are the immutable local record.
