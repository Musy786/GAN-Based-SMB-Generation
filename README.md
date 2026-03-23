# GAN-Based Super Mario Level Generation (Academic Research Project)

An academic research prototype exploring procedural level generation for 2D platformers using a Generative Adversarial Network (GAN) trained on ASCII-based Super Mario level data.

Built to demonstrate practical machine learning workflow skills:
- custom dataset preprocessing for tile-based level representations
- PyTorch GAN implementation (Generator + Discriminator)
- multi-dataset experimentation and qualitative output analysis

## Project Purpose

The goal of this project was to test whether a GAN can learn tile placement patterns from existing Mario levels and generate new playable-looking level layouts.

This was an experimental research prototype rather than a production game tool.

## Features

- **ASCII Level Dataset Loader** - Loads `.txt` levels and converts tiles to indexed tensors
- **Legend-Based Encoding** - Uses JSON tile legends to map symbols to model classes
- **GAN Training Pipeline** - Trains Generator and Discriminator with BCE loss and Adam optimizers
- **Tile-Based One-Hot Inputs** - Converts real level data into one-hot channel format for discriminator training
- **Single-Run Visualisation** - Generates a sample level and displays discrete tile output
- **Multi-Dataset Training Script** - Runs training across `ge`, `pattern_count`, and `smb` datasets

## Try it!

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional, CPU fallback is supported)

### Installation and Running

```bash
# Clone repository
git clone [insert repository url]
cd "GAN-Based-SMB-Generation"

# Create a virtual environment
python -m venv .venv

# If `python` is not found
# py -3.10 -m venv .venv

# Install dependencies
python -m pip install -r requirements.txt

# If `python` is not found
# py -m pip install -r requirements.txt

# Extract datasets (required)
# IMPORTANT: After extraction, your final structure should be:
# <root>/data/ge
# <root>/data/pattern_count
# <root>/data/smb
# <root>/data/marioai_legend.json

# Some unzip tools create an extra nested folder: <root>/data/data/...
# If that happens, move the inner data contents up one level so legend.json is at:
# <root>/data/smb/legend.json

# Quick smoke test: one epoch, SMB dataset, no visual window
python train_gan.py --epochs 1 --batch-size 8 --no-vis
# Or py

# Standard SMB training run
python train_gan.py --epochs 100
# Or py

# Train on all datasets (ge, pattern_count, smb)
python train_all.py
# Or py
```

### PyTorch Install Note

`requirements.txt` uses standard PyPI package names for portability.

- CPU-only machines: the default install command above works.
- NVIDIA GPU users: if you need a specific CUDA build, reinstall torch/torchvision from the official PyTorch selector for your CUDA version.

### Common First-Run Issue

If you see `Dataset directory not found` or `Legend file not found`, extract `data.zip` so a top-level `data` directory exists next to `train_gan.py`.

## Technology Stack

**Machine Learning:**
- PyTorch 2.6
- Torchvision 0.21

**Data and Visualisation:**
- NumPy
- Matplotlib
- JSON / ASCII level formats

## What I Tested

I tested the model in several practical ways:
- Trained the GAN on different datasets (`ge`, `pattern_count`, `smb`) to compare behavior
- Used consistent latent size (`z_dim = 100`) and repeated 100-epoch runs for baseline comparison
- Observed Generator/Discriminator loss trends per epoch to monitor adversarial balance
- Visualized generated outputs and converted predicted tiles back to ASCII for inspection

## Results Summary

The model was able to produce tile grids that resemble Mario-like structural patterns (ground, gaps, and mixed tile regions), but output quality varied between datasets and training runs.

This supports the project objective as an initial proof-of-concept for GAN-based level generation, while also highlighting the typical instability of GAN training in discrete tile domains.

## Evaluation Snapshot (Qualitative)

| Check | Observation |
|---|---|
| Structural coherence | Partial success (some logical terrain, some noisy regions) |
| Tile diversity | Moderate |
| Dataset sensitivity | High (results changed by dataset choice) |
| Training stability | Mixed (expected GAN oscillation) |
| Production readiness | Low (research prototype) |

## Dissertation Reflection

Key lessons from this project:
- GANs can capture recognizable level patterns from symbolic tile data
- Data encoding and legend consistency have strong impact on output quality
- Evaluating generated game levels requires both visual and rule-based checks
- Research prototypes need clear write-up and evaluation framing, not only working code

## Limitations

- No formal playability validator (for reachability, jump feasibility, completion path)
- No checkpointing/model export pipeline for repeatable comparisons
- Primarily qualitative evaluation rather than robust quantitative metrics
- Single architecture baseline (no direct comparison against alternatives like VAE or diffusion)

## Future Improvements

- Add rule-based playability checks and validity scoring
- Save model checkpoints and generated samples per epoch
- Track metrics across runs (diversity, novelty, tile distribution similarity)
- Compare against additional procedural generation baselines
- Build a lightweight interface to batch-generate and review levels