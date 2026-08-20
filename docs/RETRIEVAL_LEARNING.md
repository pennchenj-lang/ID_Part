# Reviewed prototype learning

HPID-Split can learn how an object category is decomposed without adding a new
Python rule for every test image. The learned layer is deliberately separated
from the segmentation backends:

1. A frozen CLIPSeg vision encoder represents each reviewed object and each
   reviewed semantic part.
2. A lightweight feature metric is fitted from repeated labels across
   independent assets.
3. At inference, each routed root retrieves similar reviewed objects.
4. Their recurrent part inventory, aliases, geometry, and visual prototypes
   become automatic queries and priors.
5. Grounding DINO and SAM2 must still produce a mask. Prototype similarity and
   normalized geometry rerank that mask before it enters HPID fusion.
6. An object below the open-set threshold receives no retrieved semantic label.

This is retrieval-augmented part parsing, not one-shot memorization. A new image
never edits the source code. Adding a reviewed batch updates the index once and
benefits later images from the same or a visually related object family.

## Reference manifest

Start from
[`configs/retrieval_references.example.json`](../configs/retrieval_references.example.json).
Every entry must contain:

- a unique `asset_id`;
- an `asset_label` such as `rifle`, `chair`, or `stylized character`;
- an `asset_domain` used by the HPID root hierarchy;
- the literal flag `reviewed: true`;
- either a reviewed HPID `package`, or `image`, `label_map`, and `taxonomy`.

Paths are resolved relative to the manifest. A top-level `part_name_mapping`
can map dataset labels to the release ontology. `part_aliases` adds phrases that
the open-vocabulary detector can understand. When `prompt_bank` is supplied,
known canonical labels inherit its semantic and assembly parents. Labels that
are not in the prompt bank remain learnable and retain the reference parent.
`exclude_parts` removes source labels that are valid for the original dataset
but do not denote one editable physical part in the release ontology. For
example, a single `skin` label covering face, hands, and legs must not be mapped
to `head`; excluding it is safer than inventing a false equivalence.

Automatic predictions are not accepted as teachers merely because they exist
in an output folder. The manifest must explicitly mark a human-reviewed
reference. Generic names such as `visual_panel_03` are rejected by default.

## Build and inspect

```bash
hpid-split build-retrieval-index \
  --manifest data/references.json \
  --output models/hpid_retrieval_v1 \
  --device auto

hpid-split inspect-retrieval-index \
  --index models/hpid_retrieval_v1
```

The index contains normalized object embeddings, part embeddings, learned
feature weights, normalized geometry, source hashes, annotation hashes, and the
review status. It does not copy ground-truth masks into inference outputs.

## Automatic inference

```bash
hpid-split auto \
  --image input.png \
  --output runs/example \
  --retrieval-index models/hpid_retrieval_v1 \
  --dense-semantic-fallback \
  --device auto
```

The default object threshold is conservative. A different-domain root is only
relabeled when at least two reviewed references support the retrieved domain.
The complete decision, nearest assets, similarities, prompt inventory,
candidate reranking, and domain correction are written to
`inference_diagnostics.json`.

For the upload UI, set `HPID_RETRIEVAL_INDEX` to `index.json` before launching
`hpid-split-web`. Automatic mode uses the index; prompt-guided mode continues to
obey the user's explicit part names.

## What the current seed does not prove

A character-only index can retrieve character parts and reject an unrelated
prop. It cannot learn rifle, furniture, vehicle, or scene decomposition until
reviewed examples from those families are added. Structural output on an
unlabelled object is not training truth and is not an accuracy result.

For each new family, retain a held-out set and report semantic mIoU, boundary
F1, small-part recall, instance consistency, and open-set false acceptance.
Those measurements determine whether the learned memory is useful; the number
of exported IDs does not.
