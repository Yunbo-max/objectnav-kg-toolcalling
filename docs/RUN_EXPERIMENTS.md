# Running Experiments

All commands below assume:

```bash
cd /home/huaziheng/project/objectnav-kg-toolcalling
```

Run the environment check first:

```bash
conda run -n mindnav38 python code/scripts/check_repro_env.py \
  --model-path /home/huaziheng/models/Qwen2.5-7B-Instruct
```

## Smoke Test

Use one full HM3D `val` episode to verify scene loading, DeepSeek connectivity,
KG tool-calling, and JSONL logging:

```bash
SPLIT=val MAX_EPISODES=1 MAX_EPISODE_LENGTH=60 \
EXP_NAME=mindnav_kg_deepseek_val_smoke \
LOG=results/runs/mindnav_kg_deepseek_val_smoke.log \
JSONL=results/runs/mindnav_kg_deepseek_val_smoke.jsonl \
BRAIN_BACKEND=deepseek BRAIN_MODEL=deepseek-v4-flash DEEPSEEK_THINKING=disabled \
bash code/scripts/run_mindnav.sh --method_name mindnav_kg_deepseek_val_smoke
```

## Full HM3D Val

The full HM3D ObjectNav `val` split has 1000 episodes. Start with 100 episodes
before launching the full run.

```bash
SPLIT=val MAX_EPISODES=100 \
EXP_NAME=mindnav_kg_deepseek_val_100 \
LOG=results/runs/mindnav_kg_deepseek_val_100.log \
JSONL=results/runs/mindnav_kg_deepseek_val_100.jsonl \
BRAIN_BACKEND=deepseek BRAIN_MODEL=deepseek-v4-flash DEEPSEEK_THINKING=disabled \
bash code/scripts/run_mindnav.sh --method_name mindnav_kg_deepseek
```

Full 1000-episode run:

```bash
SPLIT=val MAX_EPISODES=0 \
EXP_NAME=mindnav_kg_deepseek_val_full \
LOG=results/runs/mindnav_kg_deepseek_val_full.log \
JSONL=results/runs/mindnav_kg_deepseek_val_full.jsonl \
BRAIN_BACKEND=deepseek BRAIN_MODEL=deepseek-v4-flash DEEPSEEK_THINKING=disabled \
bash code/scripts/run_mindnav.sh --method_name mindnav_kg_deepseek
```

## Co-NavGPT Baselines

Qwen local flat-text Co-NavGPT:

```bash
SPLIT=val MAX_EPISODES=100 \
MODEL_PATH=/home/huaziheng/models/Qwen2.5-7B-Instruct \
BRAIN_BACKEND=local \
EXP_NAME=co_navgpt_qwen_val_100 \
LOG=results/runs/co_navgpt_qwen_val_100.log \
JSONL=results/runs/co_navgpt_qwen_val_100.jsonl \
bash code/scripts/run_co_navgpt.sh --method_name co_navgpt_qwen
```

DeepSeek flat-text Co-NavGPT:

```bash
SPLIT=val MAX_EPISODES=100 \
BRAIN_BACKEND=deepseek BRAIN_MODEL=deepseek-v4-flash DEEPSEEK_THINKING=disabled \
EXP_NAME=co_navgpt_deepseek_val_100 \
LOG=results/runs/co_navgpt_deepseek_val_100.log \
JSONL=results/runs/co_navgpt_deepseek_val_100.jsonl \
bash code/scripts/run_co_navgpt.sh --method_name co_navgpt_deepseek
```

## Legacy Val Mini

Existing quick comparisons in `results/runs/*val_mini*.jsonl` were run on the
small `val_mini` split with 30 episodes. They are useful for fast debugging,
but full-paper claims should use `SPLIT=val`.

## Aggregation

Most current comparisons are JSONL files under `results/runs/`. To aggregate a
new table, either run:

```bash
python code/scripts/aggregate_results.py
```

or compute directly from JSONL:

```bash
python - <<'PY'
import json, pathlib
for p in sorted(pathlib.Path("results/runs").glob("*.jsonl")):
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    if not rows:
        continue
    sr = sum(r["success"] for r in rows) / len(rows)
    spl = sum(r["spl"] for r in rows) / len(rows)
    print(f"{p.name}: n={len(rows)} SR={sr:.3f} SPL={spl:.3f}")
PY
```
