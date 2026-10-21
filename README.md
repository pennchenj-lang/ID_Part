# HPID-Split

**Object-Conditioned HPID-Split: Deterministic Part-Identity Fusion for Editable Assets**  
Jianpeng Chen and Yu Huang

## Manuscript data and supplementary material

This repository contains the data, supplementary material and implementation for the current manuscript. The data below correspond to the [21 October 2026 submission snapshot](https://github.com/pennchenj-lang/ID_Part/tree/96416d8a5c42497b33540f0b620eaebb7eeab821/paper_data).

**[Supplementary Material for Review (PDF)](paper_data/Supplementary_Material_for_Review.pdf)** · **[Study-to-manuscript index](paper_data/DATA_INDEX.md)** · [Data documentation](paper_data/README.md) · [中文说明](README.zh-CN.md)

| Data directory | Contents |
| --- | --- |
| [Original experiments](paper_data/original) | Cohort definitions, 226-case fine segmentation, identity audit, sensitivity, Group regression, 42-object holdout, original 37-request handoff, and supporting diagnostics |
| [Revision experiments](paper_data/revision) | Full-image evaluation, learned PartCATSeg comparison, expanded 253-request routing, taxonomy, reliability, ownership, root and resolver analyses |
| [Original 37-request review options](paper_data/review_burden37) | Review-option counts, paired comparisons and correctness checks |
| [Input perturbation stability](paper_data/stability) | Target-retention and segmentation results under controlled input changes |
| [Auxiliary studies](paper_data/auxiliary) | Character/Gaussian/PAGA experiments and the v0.3.8 engineering audit reported in the supplement |

The [data index](paper_data/DATA_INDEX.md) maps all 25 study groups to their manuscript sections, tables and source files. Each experiment retains its cohort and implementation version.

The 1,951-candidate fine-ID benchmark and the later 2,188-candidate Group regression use different prediction pools on the same 226 images. The 253-request evaluation comprises 37 historical and 216 additional requests.

## File verification and access

- [File manifest](paper_data/MANIFEST.json): file sizes and SHA-256 checksums.
- [Provenance](paper_data/PROVENANCE.json): source and published-file checksums.
- [Dataset metadata](paper_data/DATASET.json): manuscript versions and study mapping.

After downloading the repository, verify the deposited files with:

```bash
python paper_data/verify_data.py
```

Original photographs, third-party annotations, complete candidate-mask pools and model checkpoints are obtained from their original sources under the applicable terms. Frozen analysis scripts require the inputs and dependencies specified in their protocols.

## Implementation

HPID-Split combines object-conditioned evidence fusion with deterministic Part/Group identity export. The package records hierarchy, source coordinates and provenance for downstream editing requests. Frozen experimental versions are identified in the corresponding data protocols.

The implementation is in [src](src), with [configuration files](configs), [scripts](scripts) and [tests](tests). See [algorithm documentation](docs/ALGORITHM.md) and [output format](docs/OUTPUT_FORMAT.md).

Python 3.10 or newer is required. Install from the repository root:

```bash
python -m pip install -e ".[foundation,ui]"
hpid-split-web
```

See the [Apache-2.0 license](LICENSE) and [third-party notices](THIRD_PARTY.md).
