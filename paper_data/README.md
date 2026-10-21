# Current manuscript data

**Object-Conditioned HPID-Split: Deterministic Part-Identity Fusion for Editable Assets**  
Jianpeng Chen and Yu Huang  
Data organization updated: **21 October 2026**

This directory brings together the numerical source records used in the current manuscript and supplementary material, including the additional experiments and diagnoses reported in the revision. Start with the [study-to-manuscript index](DATA_INDEX.md).

| Directory | Contents |
| --- | --- |
| [original](original) | Cohort definitions, fine segmentation, identity audit, original sensitivity, Group regression, 42-object holdout, original 37-request handoff, and supporting diagnostics |
| [revision](revision) | Full-image evaluation, learned PartCATSeg comparison, expanded 253-request routing, taxonomy, reliability, ownership, root, and resolver analyses |
| [review_burden37](review_burden37) | Review-option analysis of the original 37 requests |
| [stability](stability) | Input-perturbation and target-retention evaluation |
| [auxiliary](auxiliary) | Character/Gaussian/PAGA studies and v0.3.8 engineering audit retained in the supplement |

Only studies used in the current paper are included here. Older unreported development comparisons and superseded release/editing records have been removed from the current repository tree. Experiment dates and implementation versions remain in the records to identify the exact analyses; they are not alternative versions of this data release. Numerical outcomes were not changed during this reorganization.

The 1,951-candidate fine-ID benchmark and the later 2,188-candidate Group regression are separate prediction pools on the same 226 images. Their results must be read with the version and endpoint recorded in each protocol. The 65-case development cohort is labeled as development throughout this index.

## Supplementary material

[Supplementary Material for Review (PDF)](Supplementary_Material_for_Review.pdf) contains the supplementary methods, tables and figures for this manuscript.

## Files and provenance

[MANIFEST.json](MANIFEST.json) lists every included evidence file, its byte count and SHA-256 digest. [PROVENANCE.json](PROVENANCE.json) records source and published digests; portable path substitutions are identified where applicable. [DATASET.json](DATASET.json) identifies the manuscript files used for the scope check. CSV files use their header rows as field names; JSON files retain the experiment-specific schema and units.

To verify the downloaded evidence files, run:

```bash
python paper_data/verify_data.py
```

This is a numerical evidence and analysis-code deposit. Original photographs, third-party annotations, complete candidate-mask pools and model checkpoints are obtained from their original sources under the applicable terms. Frozen analysis scripts may require these inputs, their recorded implementation versions, dependencies and local path rebinding. The manifest verifier checks deposited-file integrity; it does not rerun model inference.

Repository code remains under the existing [Apache-2.0 license](../LICENSE). Third-party datasets, images and weights retain their own terms; see [third-party notices](../THIRD_PARTY.md). No new license for third-party materials is granted by this deposit.

For a fixed citation, use the GitHub commit permalink for this directory. A commit identifies the deposited version; it is not a DOI. Earlier fixed commit links remain historical records.
