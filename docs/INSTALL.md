# Installation

Tested configuration:

| Component    | Version |
|--------------|---------|
| Python       | 3.9 (Co-NavGPT) / 3.10 (MindNav brain) |
| CUDA         | 12.1 |
| torch        | 2.1.0 |
| transformers | 4.51.3 |
| habitat-sim  | 0.2.1 |
| habitat-lab  | 0.2.1 |
| detectron2   | installed from source |

## Steps

```bash
conda create -n mindnav python=3.10 -y
conda activate mindnav

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r code/requirements.txt

# Detectron2 (matches Co-NavGPT)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
```

Point this repo at the upstream Co-NavGPT perception stack:

```bash
ln -s /tf/notebooks/Co-NavGPT code/upstream
```

## Models

- `Qwen2.5-7B-Instruct` — MindNav and Co-NavGPT text brain
- `Qwen2.5-VL-7B-Instruct` — MCoCoNav and EfficientNav VLM brains

Under `/tf/notebooks/models/` on the lab machine.

## Known issues

- `bitsandbytes` incompatible with torch 2.1. Do not install.
- `env.py` patched for Habitat 0.2.1 compatibility (adds `cur_mapping`
  shim); see `baselines/mcoconav.md`.
- `Perception_weight_decision` returning "Neither" string caused a
  TypeError in MCoCoNav; patched locally.
