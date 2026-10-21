# Data index for the current manuscript

Updated 21 October 2026. The locations below refer to the current English manuscript and supplementary material. Historical experiment versions are retained only where their results are still reported in these documents.

| Study | Cohort or unit | Manuscript location | Files |
| --- | --- | --- | --- |
| Development cohort | 65 development cases | Main V-A, Table III; Supplement S1/S7, Tables S10–S11 | [original/cohorts/development_65_manifest.json](original/cohorts/development_65_manifest.json), [original/development65](original/development65) |
| Fine segmentation and external baselines | 226 cases, earlier 1,951-candidate pool | Main Tables IV/VI; Supplement S4–S5 | [original/fine226/external_baselines](original/fine226/external_baselines), [original/fine226/clipseg](original/fine226/clipseg), [original/fine226/paired_baselines](original/fine226/paired_baselines), [original/fine226/fusion_a0_a3](original/fine226/fusion_a0_a3) |
| Identity export audit | 226 packages; 678 candidate-order trials | Main Table V; Supplement S12 | [original/fine226/identity_audit](original/fine226/identity_audit) |
| Quality exit | 226 cases | Supplement S8 | [original/fine226/quality_exit](original/fine226/quality_exit) |
| Initial sensitivity and parameter inventory | 25 conditions; global and inventory registries | Supplement S9 | [original/fine226/original_sensitivity](original/fine226/original_sensitivity), [original/parameter_registry](original/parameter_registry) |
| Runtime | Stage and environment records | Supplement S10 | [original/runtime](original/runtime) |
| Group regression | Two version-matched results on the same 226 images; later pool of 2,188 candidates | Main Table VII; Supplement S6/S10 | [original/group_regression226_prior2188](original/group_regression226_prior2188), [original/group_regression226_later2188](original/group_regression226_later2188) |
| Original Group holdout | 42 objects | Main Table VII; Supplement S2 | [original/group_holdout42](original/group_holdout42) |
| Character localization and Gaussian/PAGA auxiliary studies | 10 characters; 182 localization parts; 153 Gaussian/selection parts; 200 transform cases | Supplement S11 | [auxiliary/character_gaussian](auxiliary/character_gaussian) |
| Post-freeze engineering audit | v0.3.8: 226 packages and 1,356 core files | Main Fig. 6; Supplement S12.1, Table S22A | [auxiliary/engineering_v038](auxiliary/engineering_v038) |
| Same-candidate post-processing | 226 cases; 1,951 accepted proposals; five methods | Main Table VIII; Supplement S13 | [original/same_candidate1951](original/same_candidate1951) |
| Original restoration and routing | 37 paired requests | Main V-E/Fig. 8; Supplement S14/S19.3–S19.4 | [original/handoff37](original/handoff37), [revision/R2C5](revision/R2C5), [revision/R2C6](revision/R2C6) |
| Automatic and reference roots | 226 cases | Main Fig. 9; Supplement S15 | [revision/R2C8_Root226](revision/R2C8_Root226) |
| Historical learned-method execution probes | Execution provenance | Supplement S16 | [original/post_review_diagnostics_20261008/external_entrypoint_probes.json](original/post_review_diagnostics_20261008/external_entrypoint_probes.json) |
| Expanded ownership sensitivity | 226 cases × 50 settings; 16 joint corners | Supplement S19.1, Tables S32–S33 | [revision/R2C4](revision/R2C4) |
| Group-gate ablation | 226 earlier-pool cases + 42 holdout cases; eight subsets | Supplement S19.6, Tables S40–S41 | [original/post_review_diagnostics_20261008/gate_ablation](original/post_review_diagnostics_20261008/gate_ablation) |
| Holdout fine-ID root replacement | 42 cases | Supplement S19.5, Table S39 | [original/post_review_diagnostics_20261008/holdout_root_cases.csv](original/post_review_diagnostics_20261008/holdout_root_cases.csv), [original/post_review_diagnostics_20261008/holdout_root_summary.json](original/post_review_diagnostics_20261008/holdout_root_summary.json) |
| Stagewise spatial-support diagnosis | 226 cases; 990 reference parts | Supplement S19.2/S19.7 | [revision/R2C2](revision/R2C2) |
| Unknown taxonomy behavior | 43 deterministic interface fixtures | Supplement S19.8 | [revision/R2C1_Taxonomy](revision/R2C1_Taxonomy) |
| Source reliability and noisy-OR | Candidate audit, 226 clean cases and 150 stress cases | Main Eq. 2; Supplement S19.9 | [revision/R2C3](revision/R2C3) |
| Original review-option burden | 37 requests; five methods | Supplement S19.10, Table S45 | [review_burden37](review_burden37) |
| Input perturbation stability | 226 cases; 990 targets; primary common-target subset reported separately | Supplement S19.11, Table S46 | [stability](stability) |
| Full-image discovery | 42 full images paired with the holdout crops | Supplement S19.12, Tables S47–S48 | [revision/R1C4_FullImage42](revision/R1C4_FullImage42) |
| Expanded routing | 253 requests = 37 original + 216 additional; five methods | Main Table IX; Supplement S19.13 | [revision/R1C6_Handoff253](revision/R1C6_Handoff253) |
| Learned PartCATSeg comparison | 42 crops; frozen 456-label interface | Supplement S19.14, Table S53 | [revision/R1C5_PartCATSeg](revision/R1C5_PartCATSeg) |

Case-level and aggregate files can represent the same observations. In particular, 42 PartCATSeg case files and their aggregate JSON are two views of the same cohort; duplicated routing tables support different analysis scripts and are not additional observations. The two Group-regression summaries use the same 226 images. The 253-request evaluation reuses 37 historical requests and 216 additional object requests; it does not contain 253 newly executed restoration runs.
