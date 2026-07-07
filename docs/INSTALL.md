# Local Setup

This checkout is configured for the local workspace:

```bash
PROJECT=/home/huaziheng/project/objectnav-kg-toolcalling
CONAVGPT=$PROJECT/code/vendor/conavgpt
```

Use the `mindnav38` conda environment for Habitat runs:

```bash
conda run -n mindnav38 python code/scripts/check_repro_env.py \
  --model-path /home/huaziheng/models/Qwen2.5-7B-Instruct
```

The preflight script checks Python imports, CUDA, RedNet, HM3D episode files,
HM3D scene files, and local model weights.

## Assets

The run scripts default to the vendored Co-NavGPT subset:

```text
/home/huaziheng/project/objectnav-kg-toolcalling/code/vendor/conavgpt
```

Current local assets:

```text
code/vendor/conavgpt/RedNet/model/rednet_semmap_mp3d_40.pth
code/vendor/conavgpt/data/datasets/objectnav_hm3d_v2/
code/vendor/conavgpt/data/hm3d-v0.2-full/
code/vendor/conavgpt/data/scene_datasets/hm3d_v0.2 -> data/hm3d-v0.2-full/scene_datasets/hm3d
```

Full HM3D val is available through:

```text
code/vendor/conavgpt/data/scene_datasets/hm3d_v0.2/val/
```

The full `val` split has 1000 ObjectNav episodes and 36 referenced scenes.

## Models And API

Local model paths:

```text
/home/huaziheng/models/Qwen2.5-7B-Instruct
/home/huaziheng/models/Qwen2.5-VL-7B-Instruct
```

DeepSeek API settings live in the project-root `.env` file. The file is
gitignored and should not be committed. Current run scripts load it
automatically.

Useful overrides:

```bash
MODEL_PATH=/home/huaziheng/models/Qwen2.5-7B-Instruct
VLM_PATH=/home/huaziheng/models/Qwen2.5-VL-7B-Instruct
BRAIN_BACKEND=deepseek
BRAIN_MODEL=deepseek-v4-flash
DEEPSEEK_THINKING=disabled
```

## Rebuilding HM3D Val

If the full HM3D val assets need to be downloaded again:

```bash
PROJECT=/home/huaziheng/project/objectnav-kg-toolcalling
CONAVGPT=$PROJECT/code/vendor/conavgpt
HM3D_FULL=$CONAVGPT/data/hm3d-v0.2-full

mkdir -p "$HM3D_FULL"
conda run -n mindnav38 python -m habitat_sim.utils.datasets_download \
  --username "<matterport-token-id>" \
  --password "<matterport-token-secret>" \
  --uids hm3d_val \
  --data-path "$HM3D_FULL"

cd "$CONAVGPT"
rm -f data/scene_datasets/hm3d_v0.2
ln -s "$HM3D_FULL/scene_datasets/hm3d" data/scene_datasets/hm3d_v0.2
```

Verify:

```bash
find -L "$CONAVGPT/data/scene_datasets/hm3d_v0.2/val" -maxdepth 2 -name '*.basis.glb' | wc -l
test -f "$CONAVGPT/data/scene_datasets/hm3d_v0.2/val/hm3d_val_basis.scene_dataset_config.json"
```

## Known Issues

- `bitsandbytes` is incompatible with this pinned torch stack. Do not install it.
- Habitat 0.2.1 emits Gym deprecation warnings; they are non-fatal.
- `DeepSeek` is OpenAI-compatible but must be run with `DEEPSEEK_THINKING=disabled` for these prompt-format baselines.
