Review-option evidence audit — 9 October 2026

This package contains a complete statistical audit of all 37 previously frozen
controlled requests and all five originally compared methods. It is an explicit
post-hoc analysis: the previously reported cohort means were already known. No
new independent cases were collected, no algorithm was retuned, no request was
excluded, and no favorable subgroup replaces the full cohort.

Supported result
HPID exposes 2.0541 review options per request, versus 3.6757 for raw proposals,
3.7838 for greedy NMS, 2.8108 for DBSCAN, and 4.3243 for local pairwise CRF.
The HPID-minus-baseline option-count differences remain below zero under the
four-comparison adjustment for raw proposals, NMS and CRF. For NMS, the observed
reduction is 45.7% (1.7297 fewer options; adjusted 98.75% paired bootstrap interval
for the signed difference [-3.3243, -0.5269]). These counts characterize output
compactness and are not a measurement of human time or effort.

The DBSCAN adjusted interval [-1.7704, +0.0677] includes zero; superiority over
DBSCAN is not established by this adjusted comparison. In the 14 requests where
both methods have a correct semantic option at IoU 0.25, their per-request option
counts are identical. This finding is retained alongside the favorable results.

Correctness and availability are retained as guardrails. At IoU 0.25 HPID and
NMS have 14 and 9 uniquely correct requests, respectively; these are observed
counts, not a confirmatory accuracy-superiority claim. Actual exposed sets,
including fallback sets for unresolved semantics, contain geometrically correct
options for 19 HPID requests and 21 NMS/DBSCAN requests. Wrong-unique counts are
6, 4 and 5 for HPID, NMS and DBSCAN. Thus there is no claim of universally
preserved recall, equivalent correctness, or unconditional superiority.

Definitions and analysis
If a normalized semantic match exists, the review set contains those matches;
otherwise it contains all exported instances. Geometry in a fallback set does
not establish semantic resolution. Semantic correctness, exposed-set geometric
coverage and geometry among all exports are therefore tabulated separately at
IoU 0.25 and 0.50, together with unique/ambiguous/unresolved query states.

All 37 requests have distinct source image identifiers and source-image hashes.
The independent unit is a paired request/image (n=37), not a method-request
record (185) or an individual candidate. The fixed audit uses 10,000 paired case
bootstrap draws, NumPy default_rng seed 20261009, cases sorted by case_id, and
linear percentile quantiles. The same full-cohort resample matrix is used across
comparisons. All metrics have descriptive 95% intervals; the four option-count
comparisons additionally have Bonferroni 98.75% individual intervals, giving
nominal familywise coverage of 95%.

An exact two-sided McNemar check was added after the fixed audit for sparse
binary guardrails. All such checks are included, unadjusted and descriptive.
The 5-versus-0 uniquely-correct discordances for HPID versus NMS give p=0.0625;
the 4-versus-0 semantic-availability discordances give p=0.125. These guardrails
are not described as confirmed accuracy gains. Truth-conditioned auxiliary
subsets remain exploratory mechanism checks; they do not prove causal savings.

Reproduce the complete statistical results
Requires Python 3.10+ and NumPy; the checked environment used Python 3.12.2 and
NumPy 2.5.1. From this directory run:

    python recompute.py --output-dir recomputed

This independently written script reads data/review_cases.csv, recomputes every
method summary, paired contrast, planned auxiliary subset and exact McNemar
guardrail, and verifies 1,202 expected numeric values to tolerance 1e-12. Output
goes to the specified directory; the packaged source and expected tables are
unchanged. No model, image, network connection or machine-specific path is
required for this statistical recomputation.

Full inference replay with external frozen inputs
replay_with_external_inputs.py preserves the original replay/scoring/statistics
implementation while making input/output roots configurable. Set the environment
variables HPID_RUNTIME_ROOT, HPID_PROJECT_ROOT and HPID_REPLAY_OUTPUT to your
copies of the frozen runtime, original project and a separate output directory,
then run the script without --recompute-only. The expected runtime contains:

  code_snapshots/hpid_split_be54300_holdout/src
  experiments/paper_v031_untouched_group_holdout_42_20260825_r5
  experiments/paper_v031_identity_frontend_20260828

The expected project contains hpid_split/src and hpid_split/scripts. The replay
requires the original project's dependencies, including Pillow and OpenCV.
The original benchmark uses controlled targets from annotated object crops;
reference masks freeze and score targets, not HPID inference. This package does
not redistribute the source photos or masks and does not claim that full-image
inference can be reproduced from the CSVs alone. All 645 replay inputs were
checked against their recorded SHA256 values; none changed during packaging.

Contents
  protocol.json: audit plan fixed before the additional uncertainty computation.
  data/: all 185 request-method records; every exported instance's semantic,
    exposed status, confidence, target IoU, area and mask hash; source units and
    all input fingerprints. These are numerical/metadata records, not images.
  source/: unchanged original completion/routing case CSVs and target manifest.
    The manifest's machine-specific input-root field alone is replaced with a
    relative source role; its original file hash is retained in provenance.json.
  results/: all five methods, all full-cohort comparisons, all auxiliary subsets,
    discordant requests, exact binary guardrails and manuscript-ready values.
  audit/: independent replay/statistics results, original-endpoint assertions,
    cross-checks, provenance and an explicit interpretation audit.
  recompute.py: independent, portable full statistical recomputation.
  replay_with_external_inputs.py: configurable full replay adapter.
  provenance.json: source fingerprints, adapter scope and unchanged-input checks.
  manifest.json: every packaged file's byte count, SHA256 and Git blob SHA1.

Independent verification
Two separate implementations reproduced all 185 original routing endpoints and
agreed on 3,700 case-level values and 344 full-cohort statistics to 1e-12 before
the portable package was prepared. The independent portable computation then
checked all 1,202 table values, including the added exact tests and all auxiliary
subsets. No empty export or zero-option set was observed. All losing, tied and
uncertain comparisons remain visible in the included data and tables.

This evidence is separate from the frozen historical algorithm releases. It
does not establish superior segmentation, cross-input identity stability,
full-image detection, amodal completion, or measured user productivity.
