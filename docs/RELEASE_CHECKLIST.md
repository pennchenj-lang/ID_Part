# GitHub release checklist

- [x] One-image CLI and upload UI call the same inference path.
- [x] Inference API has no ground-truth input.
- [x] Fast and ensemble foundation-model paths run locally.
- [x] Two-pass relational appearance fusion runs by default and has a frozen
  before/after audit with no ground-truth inference input.
- [x] Evidence-gated LaMa + SAM2 completion runs locally.
- [x] Exported packages have candidate audits, diagnostics, and SHA-256 hashes.
- [x] 486 unit tests and full static checks pass on the current release tree.
- [x] Fast automatic inference reuses one SAM2 proposal pool and keeps
  Grounding DINO unloaded.
- [x] Character, knife, firearm, globe, repeated-rock scene, and dual-building
  Fast packages pass structural and recomputed SHA-256 validation.
- [x] Internal character and synthetic amodal evidence is labeled honestly.
- [x] Third-party source and checkpoint terms are separated.
- [ ] Copyright owner selects the HPID-Split repository license.
- [ ] Add freely redistributable example images or document download steps.
- [ ] Add user-supplied prop/object/scene data and annotations.
- [ ] Run cross-domain benchmark and domain-specific ablations.
- [ ] Decide repository name, owner, citation metadata, and release tag.
- [ ] Publish only after inspecting the staged Git file list for private data.

Do not upload `.runtime/`, model caches, local completion configs, internal source
images, or frozen experiment directories unless their redistribution rights are
explicitly confirmed.
