# HPID-Split post-review experimental evidence — 2026-10-08

This additive evidence bundle accompanies the constrained IEEE Access revision. It preserves the measured results, including adverse findings. It does not replace frozen releases/tags, create independent new test cases, claim upstream model retraining, or establish a learned-baseline accuracy comparison.

## Contents and scope

- `evidence/`: full numerical CSV/JSON results, protocols and logs for the 50-setting fine-ID sensitivity, 42-case paired root diagnostic, shared-input/stage audit, 37-request provenance, six unique failures, taxonomy fixtures, score provenance and eight-subset Group gate experiment.
- `evidence/gate_ablation/`: 2,144 condition rows. Earlier226 Group results use recovered `da7d236`; holdout42 uses `be54300`. All268 baseline maps and ID/semantic sequences reproduce exactly. Candidate/fine-map inputs are fixed. Two counterfactual cases have one-pixel repeat differences of unassigned cause despite a fixed hash seed; reported metrics and intervals agree across full repeated runs. Read the report before interpreting changed-map counts.
- `evidence/qa_group_cohorts.json` and its audit script distinguish the earlier 1,951-candidate test226 package used by these fixed-pool analyses (Group F1@0.25=0.3661978203) from the later 2,188-candidate v0.3.1 Group regression package (0.4042812032). The 226 RGB inputs match, but only 19 fine maps and 20 Group maps are identical. These are separate prediction packages on the same cases; the results are not pooled. The additional later historical package needed to rerun this cross-package audit is not included.
- `sources/`: frozen algorithm/configuration sources, historical Group source, and unchanged comparison/evaluation helpers. `code/original/` retains analysis code with machine paths redacted; `code/analysis/` contains path-adapted runnable copies. The adapter changes I/O paths and recursively resolves manifest tokens; it does not retune algorithms.
- `inputs/prediction_packages.zip`: 268 sets of generated candidate masks/scores/metadata, fine/Group maps and records, numerical frontend tables and manifest metadata. No source photographs or reference-mask pixels are included.
- `inputs/case_reconstruction_manifest.json`: original PACO image/object/part annotation IDs, crop boxes, source URLs and hashes. `omitted_input_requirements.json` lists every externally supplied file and expected SHA256. This is an explicit availability boundary, not a self-contained image-level replay archive.
- `FILE_MANIFEST.json`, `SOURCE_COPY_AUDIT.json`: public checksums/sizes and original-to-public path-redaction provenance. Numerical values are not beautified. Local paths become `${RUNTIME_ROOT}`, `${PROJECT_ROOT}` and related markers.
- `PREDICTION_PIXEL_AUDIT.json` confirms that all 3,047 image members are binary generated masks or integer ID maps, with no RGB members. `sources/project/hpid_split/` is the canonical helper location used by the runner; retained source aliases contain the same path-sanitized material.
- Gate `backups/` retain complete first-pass, integrity-pass and final-run CSV/JSON/log records. Four supplementary NPZ files contain only 32 uint16 counterfactual ID maps documenting the repeat limitation; they contain no RGB or reference pixels. `code/packaging/prepare_public_evidence.py` records the staging procedure and requires the original local source archives.

## Verify the public numerical package without images

File integrity uses only the Python standard library. The independent numerical checker additionally requires NumPy; it imports no algorithm modules, images or models:

```sh
python tools/verify_bundle.py
python tools/verify_public_metrics.py . --report ../metric_verification.json
```

The optional `--privacy` flag also requires Pillow and decodes every published prediction image to check its single-channel binary-mask or integer-ID-map format. The saved `verification/public_metrics_and_privacy.json` records 275,151 numeric consistency checks, 1,283 independently recomputed intervals and the filename/text/pixel audit. Write fresh reports outside this bundle if retaining exact manifest integrity.

The metric checker uses saved CSV/JSON values. It does not infer new masks or independently validate image annotations. Its report distinguishes what was recomputed from what is an archived execution assertion.

The optional cross-package audit can be repeated when the additional later historical package is separately available:

```sh
python tools/compare_prediction_cohorts.py --earlier /path/to/earlier226 --later /path/to/later226 --output /path/to/results/qa_group_cohorts.json
```

The saved export-contract QA records checks against original package manifests, visible-part masks and linked diagnostics beyond the minimal replay inputs. Its original audit source is included; these extra archived package files must be supplied to repeat that file-link audit.

`evidence/qa_psnr_intervals.json` verifies that the original and post-review PSNR intervals use the same 37 paired values. They differ because of bootstrap seed and case order, not changed outcomes. After extracting the generated archive, `python tools/verify_psnr_intervals.py --data-root /path/to/replay_data --output /path/to/results/psnr_interval_audit.json` repeats both intervals using only the included numerical tables. No external image files are required for this check.

## Prepare image-level replay

Recorded environment: Python3.12.2, NumPy2.5.1, SciPy1.18.0, OpenCV5.0.0, Pillow12.3.0; exact recorded versions are in `evidence/stage_run_environment.json`. The replay requirement file records these versions. GPU checkpoints and detector inference are unnecessary for fixed-pool analyses. The complete old model-generation workflow is outside this bundle.

```sh
python tools/prepare_inputs.py --data-root /path/to/replay_data --extract --check
```

This first command must report missing external files. On Windows, use a short absolute data-root path (for example `D:/hpid_replay`) because archived case filenames are long. Obtain the source images and PACO-LVIS annotations under their applicable terms and reproduce the crops/reference masks specified by the reconstruction manifest. Place them at the listed relative paths and verify their hashes. Users with the original local archive may copy only the omitted files, without altering that archive:

```sh
python tools/prepare_inputs.py --data-root /path/to/replay_data --local-runtime /path/to/original_runtime --check
```

PACO's official [README](https://github.com/facebookresearch/paco) provides annotation/image acquisition instructions and describes the repository license as applying to source code. The [MIT license](https://github.com/facebookresearch/paco/blob/main/LICENSE) and [dataset description](https://github.com/facebookresearch/paco/blob/main/docs/PACO_DATASET.md) did not establish a blanket permission to redistribute the separate images/reference-mask files in this audit; these files are therefore omitted. Included prediction masks/maps are generated HPID outputs. No image license is fabricated. Existing HPID source-license notices remain unchanged.

## Run analyses into a new output directory

```sh
python tools/run_analysis.py taxonomy --output /path/to/results
python tools/run_analysis.py score --data-root /path/to/replay_data --output /path/to/results
python tools/run_analysis.py routing --data-root /path/to/replay_data --output /path/to/results
python tools/run_analysis.py sensitivity --data-root /path/to/replay_data --output /path/to/results --workers 2
python tools/run_analysis.py holdout --data-root /path/to/replay_data --output /path/to/results
python tools/run_analysis.py gate --data-root /path/to/replay_data --output /path/to/results --workers 2
python tools/run_analysis.py stage --data-root /path/to/replay_data --output /path/to/results --workers 2
python tools/run_analysis.py request --data-root /path/to/replay_data --output /path/to/results
```

Use `--limit 3` for gate/stage/request/sensitivity smoke checks; trial output must be separate from complete runs. `--check-inputs` reports missing external files before image-level analysis. The runner writes outside `evidence/` by default and does not modify original experiments. External PartCATSeg/HOPS entrypoint failure logs describe the recorded local environment; they are not trained accuracy runs and are not promised to recur on a different installation.

Root42 and parameter sweeps reuse the same archived cases. Gate nominees are not all exported candidates. Stage diagnostics include unions/intermediate argmax proxies and must not be mistaken for interventions or added samples. Routing uniqueness is not calibrated confidence; retain the six failures and adverse PSNR outcomes. See each protocol for matching rules, denominators, paired units, confidence intervals and limitations.

## Maintenance and static checks

The local `ruff.toml` preserves the current Ruff rule selection for maintained replay code (`code/analysis/`) and tools. It excludes `sources/`, `code/original/`, `code/packaging/` and `evidence/` from style rewriting because those are checksum-bound archived records. This configuration is scoped to this evidence directory and does not change the repository's production rules; undefined-name and fatal checks remain enabled for the maintained code.

The post-publication integration cleanup sorts imports, removes unused bindings, makes subprocess error handling explicit, binds loop-local callbacks, and uses an equivalent adjacent-pair iterator. The gate adapter imports its path bootstrap before archived helpers. Three narrowly annotated exception catches intentionally record audit failures or API fixture exception types; failed checks remain failures. The original packaging source records the initial staging procedure, while `verification/lint_integration.json` records this subsequent maintained-code cleanup and `verification/lint_portability_recheck.json` verifies the same nine replay checks. Archived source, input and numerical-evidence bytes remain unchanged.
