# MCoCoNav

Upstream: https://github.com/FrankZxShen/MCoCoNav.git

## What It Is

MCoCoNav is a VLM extension of the Co-NavGPT perception stack. It uses the same
2D semantic map and frontier/planner infrastructure, but asks a VLM to make the
frontier decision from visual/map context.

## Local Implementation

Local path:

```text
/home/huaziheng/project/objectnav-kg-toolcalling/code/vendor/mcoconav/
```

Local VLM path:

```text
/home/huaziheng/models/Qwen2.5-VL-7B-Instruct
```

Run script:

```text
code/scripts/run_mcoconav.sh
```

## Run

```bash
cd /home/huaziheng/project/objectnav-kg-toolcalling
SPLIT=val MAX_EPISODES=100 \
VLM_PATH=/home/huaziheng/models/Qwen2.5-VL-7B-Instruct \
EXP_NAME=mcoconav_qwen_vl_val_100 \
LOG=results/runs/mcoconav_qwen_vl_val_100.log \
JSONL=results/runs/mcoconav_qwen_vl_val_100.jsonl \
bash code/scripts/run_mcoconav.sh
```

## Current Legacy Numbers

These are small `val_mini` sanity results:

| Method | Episodes | SR |
|---|---:|---:|
| MCoCoNav, Qwen2.5-VL-7B | 30 | 0.000 |

The observed failure mode is spatial frontier-assignment collapse under the
VLM prompt. Full-val runs should be treated as confirmation runs, not as the
first debugging target.
