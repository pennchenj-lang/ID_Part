# Third-party components

HPID-Split downloads or calls external models but does not vendor their source
or checkpoints in this repository. Users are responsible for checking the terms
that apply to their use and redistribution.

| Component | Role | Upstream status checked on 2026-08-11 |
| --- | --- | --- |
| [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | Open-vocabulary boxes | Official repository and [tiny model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny) identify Apache-2.0 |
| [SAM2](https://github.com/facebookresearch/sam2) | Box-to-mask refinement and amodal mask evidence | Official repository and [SAM2.1 tiny model card](https://huggingface.co/facebook/sam2.1-hiera-tiny) identify Apache-2.0; optional bundled components may carry their own notices |
| [SAM3](https://huggingface.co/facebook/sam3) | Prompt-guided concept segmentation | The official weights are gated and use Meta's separate SAM license; users must accept the terms and authenticate before use. Do not redistribute the checkpoint in an HPID-Split release archive |
| [CLIPSeg](https://github.com/timojl/clipseg) | Dense fallback for missed details | Official source repository is MIT, but it explicitly states that MIT does not apply to model weights; verify the exact `CIDAS/clipseg-rd64-refined` weight terms before redistribution |
| [LaMa](https://github.com/advimman/lama) | Hidden-region appearance inpainting | Official LaMa repository is Apache-2.0 |
| [simple-lama-inpainting](https://github.com/enesmsahin/simple-lama-inpainting) | Isolated LaMa wrapper | Wrapper repository is Apache-2.0; its release downloads `big-lama.pt`, whose checkpoint terms must also be checked |
| [Transformers](https://github.com/huggingface/transformers) | Model loading and inference API | Apache-2.0 library; model-card terms remain separate |

The repository-level license for HPID-Split itself is intentionally pending the
copyright owner's choice. Adding a license here does not override any model,
checkpoint, dataset, or image-asset terms.
