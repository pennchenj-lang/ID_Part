# Correct identity continuity under controlled input perturbations

This evidence package tests whether frozen HPID-Split keeps finding the same **correct** part more reliably than the paper's fixed same-candidate DBSCAN and greedy NMS baselines. It contains the complete results, including unfavorable comparisons. The primary analysis does **not** establish a stability advantage for HPID-Split.

The analysis plan was frozen locally before inspecting comparative results. It was not publicly preregistered. Historical algorithm code, published parameters, and original results were not changed. These are controlled diagnostics on an already analyzed cohort, not a new independent validation set or a comparison with learned state-of-the-art segmentation systems.

## Design and result

All 226 cases, 1,951 frozen candidates and 990 annotated targets were retained. Three methods received identical inputs for each of nine prespecified perturbations: the original three root-mask perturbations, score jitter at three fixed seeds, and candidate dropout at three fixed seeds. Including the unperturbed condition, 6,780 predictions completed without failure. All 226 original HPID maps and all original per-case metrics for the three methods reproduced the published results.

The primary cohort contains 126 targets in 90 cases that all three methods initially found correctly at semantic-compatible IoU ≥ 0.25. A target is retained only when the **same identity key** still maps to that same GT target correctly after perturbation. The three families receive equal weight within each case, followed by an equal-case average. The remaining 136 cases contain 523 GT targets and have no common correct target; they are omitted only from this conditional endpoint and remain in all-GT and segmentation analyses.

| Primary correct-ID retention | HPID-Split | DBSCAN | Greedy NMS |
|---|---:|---:|---:|
| Nine-condition case mean | 93.9918% | 96.0082% | 93.9815% |

The independently recomputed HPID-minus-baseline differences are −2.0165 percentage points versus DBSCAN (97.5% case-bootstrap interval −4.0123 to −0.0822) and +0.0103 points versus NMS (−2.6337 to +2.6544). These two individual intervals provide Bonferroni familywise 95% coverage. Category-cluster sensitivity is also supplied; the DBSCAN contrast spans zero in that sensitivity analysis. Neither analysis establishes an HPID advantage. Intervals spanning zero do not establish equivalence.

All family/condition results, IoU 0.50 results, pairwise-common cohorts, target-pooled estimates, all-GT correctness, Part F1 and native-ID sensitivity are included. Candidate dropout uses the fixed `round(0.10 × N)` rule: 94/226 cases, including 26/90 primary-cohort cases, lose zero candidates because of small candidate counts. Each seed removes 162 candidates overall. This discretization was preserved rather than adjusted after seeing results.

## Identity scope

The common adapter preserves each method's output masks, semantic labels and instance count. It assigns the same semantic/rank rule to every method from final-mask geometry, independently in every condition. It removes differences due to HPID's hierarchy/side/identity-mask naming and the baseline adapter's root-center side labels. Ground truth is used only for scoring; it never selects outputs or assigns IDs.

The prespecified native-ID sensitivity restores HPID's actual exported `part_id` and each baseline's existing semantic/side/rank evaluation-adapter ID. The original HPID ID, semantic label and mask triplets match all 226 saved packages exactly. This sensitivity compares the complete method-plus-naming rules and cannot isolate fusion quality; it also does not establish superiority. Neither evaluation measures all provenance, hierarchy or package-validation functionality.

## Files and recomputation

- `protocol.json`: locally frozen design, with machine paths replaced by relative provenance namespaces.
- `target_retention.csv`: all 59,400 target/method/condition/threshold observations; GT-compatible matching and identity continuity remain separately inspectable.
- `case_metrics.csv`: all 6,780 per-case prediction evaluations, including segmentation and diagnostic runtime.
- `case_metadata.json`: non-image cohort metadata for independent statistics.
- `audit/statistics/`: independently recomputed estimates, contrasts, cohorts, strata, native-ID sensitivity and segmentation results.
- `summary_crosscheck.json` and `native_summary.json`: a separate producer-side point-estimate cross-check. All 48 checked method/scope estimates agree within 4.45 × 10⁻¹⁶; small bootstrap-endpoint differences arise from independently seeded sampling streams. Use `audit/statistics/` as the reporting source.
- `reproduction_audit.json`, `native_identity_reproduction.json`, `audit/independent_mask_check.json`: original-result checks and local mask-level verification certificates. The independent check reconstructed 360 prediction sets from 12 deterministically selected cases and verified all 6,780 cache-container hashes.
- `audit/input_bindings.json`: hashes of 4,073 input files. Relative `runtime/` paths identify external inputs; those images and masks are not included.
- `release_transformations.json`: original executed-source and portable-copy hashes. Only paths and the metadata-reading adapter were changed for release.
- `SHA256SUMS.json`: hashes of the released files.

With Python and NumPy installed, the public tables can be independently recomputed without images or private paths, from this directory:

```text
python -B audit/independent_statistics.py --targets target_retention.csv --protocol protocol.json --out recomputed_statistics
python -B audit/independent_f1_summary.py
python -B audit/check_native_and_crosschecks.py
```

The latter two commands refresh the corresponding tables in `audit/statistics/`. The first command includes the paired 10,000-resample case bootstrap, category-cluster sensitivity and disclosed multiplicity adjustments. Target and perturbation rows from the same case remain clustered.

`run_stability.py` is the portable inference runner. Full prediction replay additionally requires the original licensed input masks/packages, the frozen `hpid_split_be54300_holdout` source tree, and the unchanged original baseline/evaluation helpers. Set `HPID_RUNTIME_ROOT` and `HPID_PROJECT_ROOT` to those local roots, then run `python -B run_stability.py --workers 4`. Source and input hashes identify the required versions. Prediction caches and original images/masks are deliberately excluded from this public evidence package; the local mask audit cannot be reproduced from CSVs alone.

Runtime values were measured during a concurrent four-worker diagnostic rerun, excluding loading and scoring. They are not a controlled speed benchmark. The scope is fixed-candidate root changes, score jitter and limited candidate dropout, not natural-image changes, regenerated proposals, unseen occlusion or interactive user performance.
