# HPID-Split v0.3.0 PACO-LVIS Holdout Results

These files report the first frozen automatic-condition evaluation used for
paper analysis. The source manifest SHA-256 is
`3d16426e4389d59c9fc2f820e8efd3f12c06aa6b881c28e5914479cb1f42dbcb`.

The manifest lists 67 categories. Sixty-five cases are materialized and were
evaluated; `sponge` and `pliers` have no case path in this frozen materialization.
All 65 inference processes completed. The crop uses the public ground-truth
object bounding box, so this is an oracle-root-crop condition. Ground-truth part
masks are unavailable during inference and are loaded only after package export.

The candidate-gate ablation evaluates proposal retention after semantic,
semantic-plus-structure, and full serial verification. It is not the final
grouped end-to-end score. The files intentionally retain negative results,
including zero-score cases and confidence intervals that cross zero.

Raw image packages and PACO-LVIS images are excluded from this repository. The
portable CSV and JSON summaries contain case identifiers, metrics, configuration
scope, and hashes only.
