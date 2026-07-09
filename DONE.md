# MindNav 改进记录 7.10

## 已完成

### 1. Agent 状态管理

已在 `agent` 分支完成第一版 Agent 状态管理，当前只作为本地 frontier 分配保护层使用，暂未接入 LLM prompt/决策记忆。

实现位置：
- `code/vendor/conavgpt/exp_main_brain.py`

核心改动：
- 为每个 robot 维护运行状态：当前目标 frontier、目标网格坐标、上次距离、停滞计数、是否到达、最近 step 等。
- 在 LLM/KG 每 25 steps 给出新 frontier 后，先经过 `finalize_frontier_assignments()` 做本地状态约束，再真正写入 `goal_points`。
- 如果 agent 还在向上一个 frontier 正常前进，则保留旧目标，避免每轮 LLM 决策都打断正在执行的局部规划。
- 如果 agent 已到达、卡住、目标 frontier 消失或新目标明显更优，则允许切换 frontier。
- 对两个 robot 被分到同一个 frontier 的情况做去重重分配，减少重复探索。
- `target_found` 分支不阻止切换，仍优先让 agent 朝已发现目标方向移动。

开关和主要参数：
- `AGENT_STATE_GUARD=1`：启用状态保护，默认启用。
- `AGENT_STATE_ARRIVE_DIST`：判断到达 frontier 的距离阈值。
- `AGENT_STATE_STUCK_STEPS`：连续未有效靠近多少轮后认为卡住。
- `AGENT_STATE_PROGRESS_EPS`：判断是否有进展的最小距离变化。
- `AGENT_STATE_FRONTIER_MATCH_DIST`：新旧 frontier 匹配距离阈值。
- `AGENT_STATE_NEW_PRIOR_MARGIN`：新 frontier 先验明显更优时允许切换的 margin。


### 2. 目标决策确认

已在 MindNav 链路完成第一版 `found_goal` 证据修复。STOP 最终判定已恢复为 legacy 逻辑：底层 FMM 返回 `stop=True` 且 `found_goal=1` 时发出 STOP；已移除额外的当前帧确认 gate。

实现位置：
- `code/vendor/conavgpt/agents/llm_agents.py`
- `code/vendor/conavgpt/arguments.py`
- `code/vendor/conavgpt/exp_main_brain.py`
- `code/scripts/run_mindnav.sh`

核心改动：
- `found_goal` 不再只看目标语义通道是否非零，而是先计算 target map evidence：目标通道阈值、最大连通域面积、component mass、总 mass、最大 score 和 centroid。
- STOP 判定回到 legacy：`planner_stop && found_goal` 即发出 `action=0`；FMM 仍只负责几何上的 short-term goal 和 `stop` 信号。
- COCO/Mask R-CNN 单独语义不作为额外 STOP 特权来源；它只进入统一语义图和 KG/frontier context，最终是否 STOP 仍由 FMM `stop` 与 `found_goal` 决定。
- GT 和非 GT 语义共用同一套 `found_goal` 证据逻辑，用于排查 MP3D GT 与预测语义差异。

开关和主要参数：
- `TARGET_STOP_MODE=enforce|legacy`：默认 `enforce`。
- `TARGET_MAP_SCORE_THR`：target map 二值化阈值，默认 `0.10`。
- `TARGET_MAP_MIN_AREA`：最大连通域最小面积，默认 `3`。
- `TARGET_MAP_MIN_MASS`：最大连通域最小语义质量，默认 `0.5`。

新增记录字段：
- `target_first_seen`：首次触发 `found_goal` 时记录 `found_goal_reason` 和 `target_map_*`。
- `stop_action`：真正发出 STOP 时记录完整 target evidence。
- `episode_end` 和 episode JSONL：记录 `target_seen_ever`、`first_target_seen_step`、`min_distance_to_goal`、`target_stop_mode` 等字段。

### 3.MindNav 支持单个Agent

在HM3D上验证效果 SR=0.6，SPL=0.25
低于two agent的效果

对比其他zero-shot的objectnav（单个robot）







