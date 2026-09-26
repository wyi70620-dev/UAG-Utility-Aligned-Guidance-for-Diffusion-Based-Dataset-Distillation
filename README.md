# UAG Reproduction

A code implementation of **Utility-Aligned Guidance for Diffusion-Based Dataset Distillation**. It includes sampling-time entropy guidance, reference and student training, experiment schedules, ablations, and research diagnostics.

This repository contains source code and configuration only. It does not include datasets, model weights, generated samples, experiment results, or deployment records. It is a reproduction implementation, not an author-provided official release. Implementation choices left unspecified by the manuscript are documented in [Reproduction notes](docs/REPRODUCTION_NOTES.md).

## Structure

| Path | Purpose |
|---|---|
| `uag/guidance.py` | UAG noise correction with frozen denoiser and differentiable VAE/classifier path |
| `uag/sample.py` | Paired DiT/UAG generation, DDPM 50 steps, CFG 4, original-timestep guidance gate |
| `uag/train.py` | Reference classifiers, hard-label students, and ImageNet-1K soft-label training |
| `uag/networks.py` | ConvNet-6, ResNetAP-10, ResNet-18, WRN-28-10 and ViT-Tiny/16 |
| `uag/soft_labels.py` | Region partitioning and teacher targets |
| `uag/experiments.py` | Main experiments, hyperparameter/reference ablations, baseline and plug-in schedules |
| `uag/diagnostics.py` | Paper's generation–utility analysis and permutation statistics |
| `uag/redundancy.py` | Paper's five-subset training-signal redundancy experiment |
| `uag/select.py` | Random, K-center and herding baselines |
| `uag/upstream.py`, `uag/finetune.py` | Pinned Minimax, MGD3 and IMS3 integration |
| `uag/report.py`, `uag/cost.py` | Result aggregation, plots and runtime measurement |
| `configs/` | Five benchmark configurations and WNID lists |
| `scripts/` | Environment setup, pinned upstream download and experiment launchers |

## Installation

The dependency pins target Python 3.8 and PyTorch 1.13. Use a dedicated environment:

```bash
conda create -n uag python=3.8 -y
conda activate uag
pip install -r requirements-core.txt
python scripts/setup_upstream.py
source scripts/env.sh
bash scripts/install_extras.sh
```

Upstream code is downloaded at fixed commits and stays outside this repository's tracked files. Source attribution is in [Upstream dependencies](docs/UPSTREAM.md).

## Data and checkpoints

Edit `configs/<dataset>.json` before running. Relative asset paths resolve from the repository root, independently of the current directory.

```text
data/<dataset>/train/<WNID>/image.JPEG
data/<dataset>/val/<WNID>/image.JPEG
checkpoints/DiT-XL-2-256x256.pt
artifacts/reference/<dataset>/resnet18/final.pt
```

Benchmarks: `imagewoof`, `imagenette`, `imageidc`, `imagenet100`, `imagenet1k`. Supply the pretrained [DiT-XL/2 256 checkpoint](https://github.com/facebookresearch/DiT#sampling--). The VAE and ImageNet-pretrained reference can be obtained from their upstream providers during execution; set `offline` in the configuration when using cached assets.

For the four smaller benchmarks, the main experiment plan includes reference training if the configured checkpoint is absent. ImageNet-1K uses a torchvision pretrained ResNet-18. A custom reference checkpoint must preserve the configuration's ordered classes.

## Experiment schedules

Commands are planned by default. Add `--execute` only when ready to run.

```bash
source scripts/env.sh
bash scripts/plan_all.sh
bash scripts/run_suite.sh main --datasets imagewoof
bash scripts/run_suite.sh main --datasets imagewoof --execute
```

Available suites:

- `main`: five benchmarks, reported IPC/architecture settings, paired DiT/UAG, three seeds.
- `ablation`: the 9 × 9 guidance-strength/cutoff grid.
- `references`: five reference architectures at IPC 10/20/50.
- `plugins`: Minimax, MGD3 and IMS3 with and without UAG.
- `diagnostics`: frozen real-data partitions, gradient-flow removal utility and joint permutations.
- `redundancy`: five disjoint subsets and the full cross-evaluation matrix.
- `baselines`: random selection, K-center and herding.

```bash
bash scripts/run_suite.sh ablation --execute
bash scripts/run_suite.sh references --execute
bash scripts/run_suite.sh diagnostics --execute
bash scripts/run_suite.sh redundancy --execute
```

The manuscript does not specify which main-table rows use Minimax checkpoints. Select the generator explicitly with `--main-checkpoint dit` (default) or `--main-checkpoint minimax`.

### Base-method fine-tuning and plug-ins

Prepare the relevant base checkpoints before the plug-in suite. These commands print their recipes unless `--execute` is supplied:

```bash
python -m uag.finetune --config configs/imagewoof.json --method minimax
python -m uag.finetune --config configs/imagewoof.json --method ims3
bash scripts/run_suite.sh plugins --execute
```

### Individual stages

```bash
python -m uag.train --config configs/imagewoof.json --role reference --out artifacts/reference/imagewoof/resnet18
python -m uag.sample --config configs/imagewoof.json --ipc 10 --seed 0 --scale 10 --out artifacts/manual/uag
python -m uag.sample --config configs/imagewoof.json --ipc 10 --seed 0 --scale 0 --out artifacts/manual/dit
python -m uag.train --config configs/imagewoof.json --syn artifacts/manual/uag --out artifacts/manual/student --ipc 10 --arch resnetap10
python -m uag.report summary --root artifacts/evaluation --out artifacts/summary.json
python -m uag.report heatmap --root artifacts/ablation --out artifacts/sensitivity.pdf
python -m uag.report grid --base artifacts/manual/dit --guided artifacts/manual/uag --out artifacts/qualitative.png
python -m uag.cost --out artifacts/cost
```

The runtime harness covers DiT, UAG, Minimax, MGD3 and IMS3. Other published methods in the paper's comparison tables remain external comparisons; this repository does not substitute invented results or implementations for them.
