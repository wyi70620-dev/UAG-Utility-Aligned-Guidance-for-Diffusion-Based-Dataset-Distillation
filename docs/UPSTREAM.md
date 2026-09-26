# Upstream dependencies

`scripts/setup_upstream.py` downloads the exact Git revisions recorded in `vendor/versions.json`:

- [MinimaxDiffusion](https://github.com/vimar-gu/MinimaxDiffusion): DiT implementation, diffusion utilities, student architectures and Minimax fine-tuning.
- [MGD3](https://github.com/jachansantiago/mode_guidance): mode discovery and mode-guided sampling.
- [IMS3](https://github.com/Westlake-AGI-Lab/IMS3): inversion matching and subgroup selection.
- [DiT](https://github.com/facebookresearch/DiT): original reference implementation.

Downloaded upstream directories are ignored by Git. Preserve their original copyright notices and licenses. DiT is distributed under CC BY-NC 4.0; consult each upstream repository for its own terms.

`uag/upstream.py` applies two runtime adaptations: global ImageNet label mapping for subsets, and optional UAG wrapping of the CFG prediction. Base-method fine-tuning and evaluation are retained. See `REPRODUCTION_NOTES.md` for method-specific assumptions.
