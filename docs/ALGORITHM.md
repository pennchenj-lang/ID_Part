# HPID-Split algorithm

## Problem

Given one RGB image, produce a non-overlapping instance map and stable Part-ID
records. Automatic mode requires no user part names. Prompt-guided mode also
accepts a list of part phrases. Candidate generators may disagree, overlap,
duplicate a physical part, or miss details; none may assign final IDs directly.

## Pipeline

1. In automatic Fast mode, run one global SAM2 automatic-mask pool. Score its
   masks with boundary alignment, saliency, shape, area, centre support, and
   nested-region evidence. For a clear isolated asset, estimate an additional
   foreground root from border-connected background and refine it once with
   GrabCut. Grounding DINO stays lazy and is used only when the image-only route
   cannot form a coherent root. Ensemble and prompt-guided modes retain
   detector-based proposals.
2. Select either one primary root or several scene roots. A large low-contrast
   four-border canvas mask is rejected; when its complement is coherent, that
   complement becomes the foreground or terrain layer. No ground truth is
   available to this decision.
3. In scene mode, retain broad layers and contained physical objects as separate
   roots. A selected root mask is never reintroduced as a child of the broad
   layer. Any remaining visual region is routed to the smallest compatible root,
   while the broad terrain layer acts only as an unclaimed-pixel fallback.
   When a non-frame panel background surrounds an object but has a narrow gap,
   close only the bounded panel rim, derive the enclosed object envelope, and
   deduplicate overlapping envelopes by rejecting rectangular panel fills.
4. Reuse the SAM2 pool for part proposals. Add category-independent
   distance-transform/watershed proposals only for silhouette lobes or
   appendages that have strong outer-boundary support and are not already
   represented by a SAM2 mask.
5. Route the root to a category profile and select its admissible semantic part
   inventory. Profiles that explicitly require category-guided discovery run
   one bounded Grounded-SAM2 refinement pass; other profiles reuse the Fast
   proposal pool. The three checks are serial rather than additive: semantic
   inventory evidence must pass first, geometry and topology are evaluated
   second, and appearance is evaluated only after both earlier gates pass. A
   high score in a later stage cannot rescue an earlier failure.
6. Build scale-balanced appearance proposals and a cross-cue region graph.
   Candidate slots are reserved across detail, small, medium, and large scales.
   Chromatic contrast, luminance contrast, texture appearance, boundary
   alignment, boundary closure, multi-view consensus, shape support, and
   optional semantic support jointly determine whether a region is useful
   candidate evidence. Closed contours and enclosed interiors recover lens,
   display, door, and
   panel candidates. RGB texture evidence is an appearance proxy, not physical
   material recognition. Shading-only and laminar-conflict penalties suppress
   highlights, shadows, and overlapping fragments.
   Public Group IDs cannot be created from colour, shading, or texture alone.
7. Give unresolved fine diagnostic regions deterministic spatial names
   (`visual_panel`, `visual_strip`, or `visual_detail`). Infer visual parent
   relations from stable containment. Repeated roots receive separate child
   namespaces and retain distinct `asset_id` values.
8. In prompt-guided mode, run SAM3 Promptable Concept Segmentation for each user
   phrase and constrain returned masks to routed roots. Semantic SAM3 masks and
   label-free SAM2 regions enter the same fusion. If gated SAM3 is unavailable,
   the default `auto` backend explicitly records a sequential Grounded-SAM2
   fallback. That route rejects near-identical masks claimed by different
   phrases and retains generic visual IDs when semantic evidence is weak.
9. Apply prompt-specific spatial, parent-area, and parent-containment
   constraints. A semantic child that falls mostly outside its claimed parent
   is rejected rather than kept under a confident-looking but incorrect name.
   Record every accepted candidate with model, prompt, score, parent, and source
   family.
10. Refine candidate boundaries in an area-adaptive narrow band, fill small
   holes, and remove unsupported fragments. Area, overlap, and Lab-gradient
   edge gates reject updates that drift or move away from image boundaries. Fast
   mode limits GrabCut to one iteration and a bounded candidate budget.
11. Deduplicate candidates within one source family. Stages from the same model
   stack are correlated and therefore contribute at most one evidence surface.
12. Combine independent source families with noisy-OR evidence. Parent support,
   direct-detail gates, and transitive residual rules restrict where descendants
   may own pixels. Broad scene layers receive fallback weight so overlapping
   object roots own their pixels first.
13. Resolve a preliminary ownership map. For every prompt rule with an
   `appearance_anchor`, split the predicted anchor into physical components and
   search only its configured parent support. Local contrast and geometry form
   auditable `above` or `upper_boundary` detail candidates. These candidates
   receive one correlated source family and cannot become output directly.
14. Run evidence aggregation and ownership resolution again with the relational
   candidates included. This makes the second pass compete with all original
   evidence and prevents a local heuristic from overwriting unrelated parts.
15. Suppress generic regions nested inside an independently supported named
   semantic part, and suppress texture-only marks nested in a coherent large
   structural surface. Repeated same-semantic candidates may jointly relabel a
   larger connected region when their combined evidence supports one physical
   part.
16. Associate masks that describe the same physical part when semantic labels
   match and IoU or containment-plus-centroid affinity passes the configured
   thresholds. A strongly contained local proposal may also support one
   coherent connected identity; a disconnected broad proposal cannot use this
   path, so repeated instances such as left and right shoes remain separable.
   Association happens before ID creation.
17. Resolve per-pixel ownership, iteratively remove scale-dependent fragments,
   and attach nearby unsupported remainder components to an existing identity.
   Detached components remain separate identities.
18. Assign `left`, `right`, or `center` relative to the actual assembly parent,
   then create deterministic Part IDs and resolve every assembly reference to an
   existing ID.
19. Fuse fine candidates into public physical Groups. Semantic inventory seeds
   define candidate groups and geometry attaches unresolved visual fragments.
   When the semantic mask and physical surface are different proposals, match
   them by containment and topology before assigning the public name. A phone
   display, for example, requires a verified screen inventory slot, an
   independently inset and predominantly rectangular surface, and finally a
   closed non-shading boundary. The structural surface supplies the pixels;
   the semantic proposal supplies only the admissible name.
   For an inventory-constrained object such as a firearm, eroded structural
   regions become fixed semantic markers. A semantically and spatially valid
   fore-end seed may supply a smoothed Lab material reference with lightness
   downweighted; this reconnects one handguard across illumination changes
   before an edge-preserving watershed aligns boundaries to visible seams.
   Character grouping estimates a head-to-feet pose axis, derives its garment
   boundary from the head end and footwear start, audits compact face-interior
   islands topologically, and uses paired lower-limb skin evidence only after
   semantic lower-body routing. It then clips every morphological update to the
   existing root.
   Appearance evidence cannot add a Group ID. Fast scene output uses distinct
   neutral `scene_object` Groups when ontology labels are provisional.
20. Build occlusion hypotheses. Optional completion removes a hypothesized
   occluder with LaMa, asks SAM2 for an amodal mask, and accepts the result only
   when direction, contact, size, visible-lock, and false-addition gates pass.

## Reference pseudocode

```text
function HPID_SPLIT(
    image, prompt_bank, quality, mode, user_phrases=None, prototype_index=None
):
    if quality == FAST and mode == AUTOMATIC:
        pool = sam2_global_proposal_pool(image)
        roots = proposal_first_roots(image, pool, prompt_bank)
        visual = route_to_smallest_compatible_root(pool, roots)
        visual += silhouette_bottleneck_proposals(roots, pool)
        candidates = roots + cross_cue_region_graph(image, visual)
    else:
        candidates = detector_and_sam2_proposals(image, prompt_bank)
        roots = masked_domain_arbitration_and_routing(image, candidates)
        candidates += multiscale_sam2_regions(image, roots, ground_truth=None)

    if mode == "automatic" and prototype_index is not None:
        plans = retrieve_reviewed_object_and_part_prototypes(image, roots)
        plans = reject_unknown_or_ambiguous_roots(plans)
        retrieved = grounded_sam2_masks(image, roots, plans.part_queries)
        retrieved = rerank_by_visual_prototype_and_geometry(retrieved, plans)
        candidates += retrieved

    if mode == "prompt-guided":
        semantic_masks = sam3_text_masks(image, user_phrases)
        candidates += constrain_to_roots(semantic_masks, roots)

    preliminary = hierarchy_constrained_fusion(candidates)
    relational = propose_details_from_anchors(
        image, preliminary, prompt_bank, ground_truth=None
    )
    candidates += relational
    semantic_map = hierarchy_constrained_fusion(candidates)

    identity_groups = associate_same_physical_parts(candidates)
    instances = assign_identity_groups(semantic_map, identity_groups)
    instances = attach_nearby_remainders(instances)
    records = make_stable_ids(instances, parent_relative_sides=True)
    assert all_assembly_parents_resolve(records)
    return semantic_map, instances, records, audit(candidates, evidence)
```

## Invariants

- Inference APIs do not accept ground-truth masks.
- The user annotation/reference-image path is not an inference input.
- Only entries explicitly marked as reviewed can build the prototype index;
  prediction packages do not become teachers automatically.
- Prototype retrieval has an open-set rejection path. A nearest neighbour is
  not forced into the output when absolute similarity is insufficient.
- A retrieved root domain needs support from at least two reviewed references
  before it can replace the detector's original domain.
- Root-domain evidence chooses an ontology but does not become a final mask.
- Prompt-guided output records the requested and resolved semantic backend.
- A missing gated SAM3 model is recorded before the default `auto` path falls
  back to Grounded-SAM2; `--guided-backend sam3` raises an actionable error.
  No CLIP-only semantic labels are silently substituted.
- Correlated stages from one model stack cannot inflate consensus.
- Relational candidates are derived only from the image, prompt rules, and a
  first-pass prediction; their API has no ground-truth argument.
- All relational candidates share one source family and must compete in the
  second fusion pass before receiving an ID.
- Every visible foreground pixel has at most one final instance owner.
- The public Group map has exactly the same foreground support as the fine
  Part-ID map; boundary regularization may reassign ownership but cannot add or
  delete subject pixels.
- Every `part_id` and `instance_index` is unique.
- Every non-null `assembly_parent_id` resolves to an exported Part ID.
- A completed mask must contain its visible mask.
- Pixels in the visible region are copied exactly from the source image.
- Rejected completion attempts remain visible-only and retain rejection metadata.

## What belongs to HPID-Split

The implementation surface owned by HPID-Split is the source-family-aware
proposal-first root selection, cross-cue region graph, silhouette-bottleneck
supplement, scene/object ownership rules, hierarchy gating, relation-guided
appearance proposals, two-pass fusion, identity association, stable ID
assignment, completion gates, and auditable package contract. External
foundation models supply candidate boxes, masks, text-concept masks, or
inpainted appearance.

## Known failure modes

- Tiny low-contrast parts may never be proposed.
- Fast mode favours stable structural masks and latency; its automatic character
  output can be substantially coarser than a reviewed manual decomposition.
- Retrieval cannot supply an object family that is absent from the reviewed
  index. A narrow index should reject unrelated assets rather than inventing a
  semantic transfer.
- A biased or incorrectly reviewed reference set can bias object retrieval and
  part priors. Source hashes make this auditable but do not repair the data.
- Root-domain arbitration can still confuse visually adjacent categories such
  as a machine and a handheld prop.
- Similar adjacent instances can be over-associated when their masks overlap
  heavily.
- A detector may still assign the wrong natural-language domain to an otherwise
  useful structural mask; generic visual IDs are retained when evidence is weak.
- Dense fallback can inherit texture bias from CLIPSeg.
- Relation-guided details fail when their anchor is missing or when the prompt
  rule's appearance polarity does not match the asset. The current released
  rules cover character eyebrow and eyelash recovery; other domains still need
  data-backed rules.
- Hidden completion can preserve plausible appearance while inventing the wrong
  topology; evidence gates reduce this risk but do not eliminate it.
- Scene-scale images with many independent objects require a dedicated benchmark
  and may exceed the current root and hierarchy budgets.
- Primary-asset routing can miss thin attached props when the detector does not
  return a coherent standalone root.
- A low-contrast attached accessory can be omitted when no independent proposal
  supports it. Conversely, a display platform or attached shadow can enter a
  scene object envelope when it is topologically inseparable in the available
  proposal pool.
- Automatic humanoid routing does not force a bag label from one broad text
  detector mask. Bag semantics remain prompt-guided until independent region
  evidence supports a narrower automatic route.
- Prompt-guided semantic quality requires production SAM3 weights and labelled
  evaluation; the public random tiny model only validates API integration.
