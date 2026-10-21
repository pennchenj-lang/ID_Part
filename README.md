# HPID-Split

**Object-Conditioned HPID-Split: Deterministic Part-Identity Fusion for Editable Assets**

## Data for the current manuscript

**[Current paper data — updated 21 October 2026](paper_data/README.md)** · [Study-to-manuscript index](paper_data/DATA_INDEX.md) · [中文说明](paper_data/README.zh-CN.md)

[Supplementary Material for Review (PDF)](paper_data/Supplementary_Material_for_Review.pdf)

The data directory contains the numerical source records used in the current manuscript and supplement, covering the original evaluations and the revision experiments. Older unreported development datasets and superseded publishing records have been removed from the current tree. Historical experiments still cited in the paper remain clearly identified by cohort and implementation version.

## Implementation

HPID-Split combines object-conditioned evidence fusion with deterministic Part/Group identity export. The package records hierarchy, source coordinates and provenance for downstream editing requests. Frozen experimental versions are identified in the corresponding data protocols.

The implementation is in [src](src), with [configuration files](configs), [scripts](scripts) and [tests](tests). See [algorithm documentation](docs/ALGORITHM.md) and [output format](docs/OUTPUT_FORMAT.md).

Python 3.10 or newer is required. Install from the repository root:

```bash
python -m pip install -e ".[foundation,ui]"
hpid-split-web
```

Model inference requires the appropriate third-party dependencies and weights. See [third-party notices](THIRD_PARTY.md) and the [existing license](LICENSE).
