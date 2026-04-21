# Vendored upstream code

Third-party source copied into this repo so the experiments are reproducible
from a single clone.

## `conavgpt/`

Minimal Co-NavGPT upstream — https://github.com/yuyang-J/Co-NavGPT — containing
the subset needed to run MindNav and the Co-NavGPT baseline:

| Path | Purpose |
|---|---|
| `agents/` | LLM/VLM brain wrappers + original Co-NavGPT Helicase modules |
| `envs/habitat/` | Multi-agent Habitat environment wrapper |
| `utils/` | FMM planner, pose math, visualization, semantic prediction |
| `constants.py` | Category mappings, color palettes |
| `arguments.py` | CLI argparse definitions |
| `local_vlm.py` | Local VLM server (swapped to Qwen2.5-VL-7B) |
| `exp_main_brain.py` | Entry point — MindNav brain |
| `exp_main_original.py` | Entry point — Co-NavGPT text-LLM baseline |
| `exp_main_original_minimax.py` | Entry point — Co-NavGPT with MiniMax API |

## Not vendored (install separately)

| Dep | Why not vendored | How to get it |
|---|---|---|
| `detectron2` | 15 MB, third-party library with its own license; pip-installable | `pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'` |
| `RedNet/` | 627 MB (includes checkpoint); large binary | Clone upstream Co-NavGPT and symlink `RedNet/`, or download the RedNet checkpoint from the upstream README |
| `habitat-sim / habitat-lab` | Pinned in `requirements.txt` | `pip install habitat-sim==0.2.1 habitat-lab==0.2.1` |
| HM3D scenes | Academic license | See `data/README.md` |

## License notes

- Co-NavGPT: upstream license applies to the vendored files. See upstream LICENSE.
- This repo's original code (under `code/src/` and `paper/`) is MIT; see `LICENSE`.
