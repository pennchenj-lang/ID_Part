# v0.3.1 untouched editable-group holdout evidence

This directory archives the records for the 42-object holdout used in the manuscript.
The v0.3.1 code/configuration and case slots were frozen locally before materialization;
all blind inference completed before the sealed reference manifest was opened.

## Registration wording

This was a **prespecified single-pass blind holdout**, not an independently registered preregistration.
The evidence commit was created after execution and is an archival record, not a retroactive timestamp claim.

## Strict-threshold results

- Group F1@.25: 0.3951 (95% case-level bootstrap CI [0.3160, 0.4757])
- Group F1@.50: 0.2417 (95% case-level bootstrap CI [0.1618, 0.3303])
- Group F1@.75: 0.1035 (95% case-level bootstrap CI [0.0435, 0.1777])
- Mean Group F1 over .25:.05:.75: 0.2350 (95% case-level bootstrap CI [0.1653, 0.3136])

## Materialization audit

- Prior manifests were excluded by their **union**.
- 15/42 slots used the prespecified source-integrity object fallback; 4 crossed category within the same domain.
- 30/42 RGB sources were resampled to annotation dimensions with PIL Lanczos after <=1% aspect-ratio agreement.
- Object and part masks were decoded at annotation dimensions and cropped in the same coordinate system.
- Fallback and resize decisions occurred before inference and did not use predictions or mask quality.

See `07_evidence_timeline.json` for the event order and every JSON/CSV file in this directory for the auditable records.
