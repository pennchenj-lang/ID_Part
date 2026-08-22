# Object-Conditioned Evaluation v0.3.1

This directory contains the compact, machine-readable evidence package used by
the object-conditioned HPID-Split manuscript. It does not contain PACO/LVIS
images or annotations; those remain subject to their original licenses.

## Scope and isolation

- Development: 65 PACO-LVIS object crops.
- Independent test: 226 unique images and object annotations, 60 categories,
  six domains.
- Development and test have zero source-image and object-instance overlap.
- Categories are shared. The experiment measures independent-instance
  generalization, not unseen-category generalization.
- Every method receives the ground-truth object crop/root and category. This is
  an object-conditioned part-decomposition test, not full-image detection.
- Test part masks, labels, and counts are loaded only after predictions for an
  evaluation variant have been completed.

## Headline results

On the 226 frozen-candidate test cases, the final-fusion ablation changes:

| Variant | Part F1@0.25 | Recall@0.25 | Part F1@0.50 | Semantic F1@0.25 | Object IoU |
|---|---:|---:|---:|---:|---:|
| A0 independent maximum response | 0.1511 | 0.1101 | 0.0856 | 0.0912 | 0.2967 |
| A1 cross-source consensus | 0.2012 | 0.1575 | 0.1203 | 0.1222 | 0.3314 |
| A2 consensus and hierarchy | 0.2831 | 0.3128 | 0.1458 | 0.1378 | 0.4716 |
| A3 full fusion and specificity-aware ownership | 0.3133 | 0.3656 | 0.1509 | 0.1396 | 0.5180 |

The paired A3-A0 change is +0.1622 [0.1325, 0.1913] for Part F1@0.25 and
+0.2555 [0.2248, 0.2874] for recall. Strict-overlap and semantic scores remain
modest; this package does not support a claim of error-free or unrestricted
open-world segmentation.

## Contents

- `manifests/`: the exact 65-case development and 226-case test manifests.
- `test226_public_baselines/`: SAM2 and Grounded-SAM2 summaries.
- `test226_clipseg_ovparts/`: CLIPSeg object-part prompting baseline.
- `test226_fusion_strict/`: A0-A3 case, domain, summary, and paired-effect data.
- `test226_quality_exit/`: operational review-state audit.
- `test226_sensitivity/`: frozen one-factor sensitivity audit.
- `runtime_audit/`: measured isolated-process stage timings and environment.
- `parameter_registry/`: global settings, inventory rules, prompt-bank hash.
- `dev65_*`: development-only gate, fusion, and quality analyses.
- `paper_facts/`: manuscript fact bundle and source hashes.
- `paper_figures/`: PNG/SVG figures generated from the stored tables.

`SHA256SUMS.txt` binds every evidence file in this directory. The v0.3.0 tag
identifies the algorithm release; evaluation scripts and this evidence package
were first frozen in commit `2d538397674c890aeef5889a7ff21a0075235dce`.
