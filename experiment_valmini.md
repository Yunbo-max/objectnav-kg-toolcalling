# MindNav 在 HM3D `val_mini` 上的补充实验计划

**状态**：PLANNED（仅规划，尚未启动 Habitat/vLLM）

**制定日期**：2026-07-19

**评测集**：HM3D ObjectNav `val_mini`，30 episodes
**方法**：MindNav；预测语义；默认 2 agents；高层决策间隔 25 steps

## 1. 本轮目标与结论边界

本轮只回答以下三个问题：

1. MindNav 对本地 Qwen2.5-7B-Instruct 与 Qwen2.5-3B-Instruct 的模型规模是否敏感？
2. 在 DeepSeek API 下，固定同步环境步数时，1/2/3 agents 的端到端导航收益与成本如何变化？
3. DeepSeek API 决策时，提供近期决策 memory 是否有可测收益？

本轮不复用 `val_mini` 的历史正式结果作对照。已有 `val_mini` KG-format 批次早于当前统一的 tool-calling/fixed-order 实现，且其日志字段、配置快照与当前协议不完全一致；因此所有本轮比较均从同一当前代码版本重新运行。历史结果只能作为调试参考，不能混入统计表。

### Claim map

| Claim | 最小可信证据 | 反例/限制 |
|---|---|---|
| C1：MindNav 可在 7B 与 3B 本地 Qwen 后端上完成任务，且模型规模的性能—成本 trade-off 可量化 | 相同 30 个 compound episode keys、相同配置下报告 SR/SPL/DTG、tokens、calls、runtime、格式失败诊断 | 30 episodes 的一次运行只能说明本协议下的 episode 不确定性；不能声称跨模型统计等价或普遍可泛化 |
| C2：增加 agents 改变团队覆盖和导航效率 | DeepSeek 下 `N={1,2,3}` 使用同一 episode 顺序，报告 paired 1−2、2−3、1−3 差值与成本 | 固定 500 个同步 team steps 会让总 robot-actions 随 N 增长；结果只能支持 fleet-size scaling，不能单独归因于协调算法 |
| C3：近期决策 memory 是否有效 | DeepSeek 2-agent 的 memory-on/off 仅改变 memory 字段，报告 paired SR/SPL/DTG 与 calls/tokens | 若区间跨 0，只能说在本评测规模下未观察到稳定收益，不能证明 memory 无效 |

## 2. 冻结的评测协议

### 2.1 数据与配对

- Split：`val_mini`；`EPISODE_SHUFFLE=0`、`START_EPISODE_INDEX=0`、`MAX_EPISODES=30`。
- 在首个 smoke 前生成并冻结 `results/manifests/valmini_20260719_selection_manifest.json`：顺序记录 30 个 `(scene, episode_id, goal)`。`episode_id` 可跨 scene 重复，因此比较和审计一律使用 `scene::episode_id` compound key。
- 当前工作树的 `val_mini` 是到 `/home/huaziheng/project/objectnav-kg-toolcalling/.../val_mini` 的符号链接，已解析。正式运行前仍需确认 30 个 episode 可被当前 Habitat 环境加载，并将数据路径、文件 hash/mtime 写入 manifest。
- 不得按结果删除 episode。中断或崩溃只补跑相同 index；合并时按连续 `episode_index` 与 compound key 去重，smoke 和无效尝试不进入正式 JSONL。

### 2.2 固定系统变量

除明确消融项外，所有行固定：

| 项目 | 固定值 |
|---|---|
| KG | `text` serialization；同一 KG 构造、截断、tools、prompt 与 fallback |
| 语义 | RedNet + Mask R-CNN 预测语义；不使用 GT semantic |
| episode horizon | 最多 500 个同步 team steps |
| 高层规划 | 每 25 steps 一次 |
| 成功口径 | Habitat success distance 0.2 m；DTG 为团队到目标有效 viewpoint 的最小 geodesic distance |
| 随机性 | Python/NumPy/PyTorch/Habitat seed=1，`--reset_seed_each_episode`；Qwen 请求 seed=1；DeepSeek 服务端不设请求 seed |
| LLM decoding | adapter 固定 `temperature=0`；DeepSeek thinking disabled |
| 输出与可追溯性 | 每行独立 `EXP_NAME`、`METHOD_NAME`、log、JSONL、mapping JSONL、dump 目录；记录 commit、backend/model、GPU IDs、时间戳和 provider usage |

### 2.3 “无 memory”的精确定义

本轮的 memory 指当前代码已接线并有单元测试保护的 **决策历史包**，不是删除场景 KG：

- memory-on：向第二阶段决策 packet 传入 `recent_assignments`，并在 current rooms 中提供 `recent_history_summary` 和 coverage/history rules。
- memory-off：使用 `DECISION_HISTORY=off`；两个 LLM 阶段都不得看到上述历史衍生字段，内部不继续积累近期 assignment history。
- memory-off **保留**当前观测、当前机器人位姿、当前 frontiers、当前 KG 中的环境事实、工具 schema、planner、fallback、动作预算和语义输入。
- `HelicaseBrain.reflexion_memory` 当前没有被导航主循环消费，因此不是本实验变量；不得把它写成已被移除的输入。

## 3. 最小正式运行矩阵

DeepSeek 2-agent / memory-on 是 C2 与 C3 的公共对照。因此需要 6 个唯一全量配置、共 180 formal episodes；不存在需要重复运行的 DeepSeek 2-agent 行。

| Run ID | 目的 | LLM | Agents | Memory | Episodes | 主要比较 | 状态 |
|---|---|---|---:|---|---:|---|---|
| `V-Q7B-N2-MON` | C1 Qwen 模型规模 | Qwen2.5-7B-Instruct / vLLM | 2 | on | 30 | vs `V-Q3B-N2-MON` | TODO |
| `V-Q3B-N2-MON` | C1 Qwen 模型规模 | Qwen2.5-3B-Instruct / vLLM | 2 | on | 30 | vs `V-Q7B-N2-MON` | TODO |
| `V-DS-N1-MON` | C2 agent 数量 | DeepSeek API | 1 | on | 30 | vs N=2, N=3 | TODO |
| `V-DS-N2-MON` | C2/C3 公共对照 | DeepSeek API | 2 | on | 30 | vs N=1, N=3, memory-off | TODO |
| `V-DS-N3-MON` | C2 agent 数量 | DeepSeek API | 3 | on | 30 | vs N=1, N=2 | TODO |
| `V-DS-N2-MOFF` | C3 memory 消融 | DeepSeek API | 2 | **off** | 30 | vs `V-DS-N2-MON` | TODO |

输出路径固定为 `results/valmini_20260719/runs/<run_id>.{log,jsonl,mapping.jsonl}`；正式汇总输出为 `results/valmini_20260719/aggregated/`。新路径不得覆盖既有 `results/runs/*valmini*` 历史文件。

## 4. 实验块

### B1：Qwen2.5-7B 与 Qwen2.5-3B

- **测试 claim**：C1。
- **比较**：`V-Q7B-N2-MON` vs `V-Q3B-N2-MON`；仅替换 vLLM 服务的模型路径、served model name 与对应 backend env 文件。
- **主指标**：SR ↑、SPL ↑、DTG ↓；同时记录每 episode input/output tokens、API calls、planning rounds、invalid outputs、repair calls、fallback decisions 与 runtime。
- **成功判断**：两行均为 30/30 合法、完全配对的正式记录；以 Q3B−Q7B 的 paired bootstrap CI 表示性能差异。成本较低但导航 CI 明显退化时，结论只能写成 trade-off。
- **失败解释**：若 3B 失败率升高，先检查 JSON/tool schema 与 output truncation；若失败率不高但导航差，才讨论模型推理能力差异。
- **论文位置**：主实验/附录中的 LLM-backend scalability 小表。
- **优先级**：MUST-RUN。

### B2：DeepSeek 的 1/2/3-agent 扩展

- **测试 claim**：C2。
- **比较**：`V-DS-N1-MON`、`V-DS-N2-MON`、`V-DS-N3-MON`；唯一设计变量为 `num_agents`。
- **实现边界**：N=2 必须走原 `run_mindnav.sh` golden path；N=1/3 必须通过 `run_mindnav_robot_ablation.sh`，后者显式拒绝 N=2。N=1/3 使用已有动态 robot-count 分支和 prompt wrapper，运行前需复跑相关单元测试与 1-episode Habitat smoke。
- **主指标**：SR、SPL、DTG；报告 2−1、3−2、3−1 的 paired bootstrap 差值及 95% CI。
- **成本诊断**：team steps、robot-actions、累计路径长度（若日志已有）、input/output tokens、API calls、planning rounds、wall-clock runtime。N=2 的 JSONL 目前不写 `num_agents/team_steps/robot_actions` 字段，正式前须以 `num_agents=2` 和 `team_steps=steps` 补充到聚合输入或让聚合器明确处理该固定基线；不能静默丢掉 N=2 成本列。
- **成功判断**：三行各 30/30、compound keys 相同；至少一个主导航指标的 paired CI 支持差异后，才作“规模收益”的强表述。若仅点估计单调，写为趋势。
- **论文位置**：agent-count ablation 表；表注必须写明固定 500 team steps、robot-actions 随 N 线性扩大。
- **优先级**：MUST-RUN。

### B3：DeepSeek 决策 memory-off

- **测试 claim**：C3。
- **比较**：`V-DS-N2-MON` vs `V-DS-N2-MOFF`；唯一变量为 `DECISION_HISTORY=on/off`。
- **主指标**：SR、SPL、DTG 及 paired `on−off` CI；诊断项为 tokens、calls、planning rounds、重复 room/frontier 分配率（可从 mapping JSONL 审计）、invalid/repair/fallback。
- **验收**：正式前运行现有 `test_history_off_omits_all_history_derived_packet_fields`，并抽查 memory-off mapping/prompt audit：packet 没有 `recent_assignments`、room 没有 `recent_history_summary`、prompt 没有 coverage/history rules。
- **失败解释**：无差异表示当前持久化 KG 已覆盖多数累计信息的可能性；memory-on 变差则检查历史陈旧或过度抑制重访。两种情形均不能误写成“删除 KG”。
- **论文位置**：核心消融表。
- **优先级**：MUST-RUN。

## 5. 运行前验收（M0）与 smoke（M1）

### M0：不启动 Habitat 的预检

- [ ] 记录 `git rev-parse HEAD` 与 `git status --short`；当前工作树已有用户未提交改动，实验期间不得混入无关代码修改。
- [ ] 以 `mindnav38` 运行完整单元测试，至少包括 history-off、robot-count、LLM API 与多 agent DTG 测试。
- [ ] 验证 `val_mini` 可读、共 30 episodes，并冻结 selection manifest；确认所有运行使用同一 key 集与顺序。
- [ ] 确认 `models/Qwen2.5-7B-Instruct`、`models/Qwen2.5-3B-Instruct` 均可读；7B 使用已有 `code/configs/qwen_vllm.env`，3B 需要一个内容相同但 `BRAIN_MODEL=qwen2.5-3b` 的独立 env 文件。
- [ ] 验证 `.env` 中 DeepSeek backend/model/thinking 配置可用，但不在日志或本文档写入 API key。
- [ ] 确认 JSONL metadata 含 `scene`、`episode_id`、`backend`、`model`、`decision_history`、usage、success threshold 与 commit；为 N=2 成本字段制定明确的聚合兼容处理。

### M1：关键路径 smoke

每个 Run ID 先运行 index 0 的 1 episode smoke（共 6 episodes），输出到 `results/valmini_20260719/smoke/`，绝不与 formal 数据合并。

- 每个 smoke 只接受恰好一条 JSONL record，且 SR/SPL ∈ [0,1]、DTG 非负有限、success threshold=0.2、backend/model 与 Run ID 一致。
- Qwen smoke：确认 vLLM health、model 名称、provider usage 以及 Qwen request seed 被记录。
- N=1/3 smoke：确认实际 Habitat `NUM_AGENTS`、每个 robot 有 assignment/action、无重复 frontier（可用时）、地图融合与 `robot_actions=N×team_steps` 合法。
- memory-off smoke：执行上述 prompt/mapping 审计；任一历史字段泄漏即标为 INVALID，不进入 formal。
- 任何 smoke 失败时只修复并重跑对应路径；不启动该路径的 30-episode formal run。

## 6. GPU、环境与分轮执行

每次启动 Habitat **前**先运行 `nvidia-smi`，记录 GPU 使用情况和可用显存。所有 Habitat/MindNav Python 进程使用 `mindnav38`；Qwen vLLM 服务使用 `vllm`，并独占一张 LLM GPU。每个 Habitat run 需要一张 sim GPU 与一张 semantic-model GPU；不得让 vLLM 与 Habitat/semantic 模型共用同一 GPU。

默认三轮正式排程如下；实际 GPU 编号仅在通过 `nvidia-smi` 预检后填入。H/S 采用交叉映射以避免同一进程的 sim 与语义模型争抢同一张卡。

| Round | Slot A | Slot B | 资源约束 |
|---|---|---|---|
| R1 | `V-DS-N2-MON`（H0/S1） | `V-Q7B-N2-MON`（H1/S0/L2） | 7B vLLM 独占 L2 |
| R2 | `V-DS-N2-MOFF`（H0/S1） | `V-Q3B-N2-MON`（H1/S0/L2） | 先停止 7B，再启动 3B vLLM；3B 独占 L2 |
| R3 | `V-DS-N1-MON`（H0/S1） | `V-DS-N3-MON`（H1/S0） | N=3 会加载三份语义模型；预检显存不足时改为串行 |

任一槽位完成或失败不影响另一个已完成的 Run ID；重跑必须使用新的 shard/raw 路径，再经严格去重合并。DeepSeek 的服务端不确定性不由 episode bootstrap 覆盖，论文中需将其列为限制。

### 运行模板（正式，参数占位）

```bash
# Before every Habitat launch. Inspect manually before assigning H/S/L IDs.
nvidia-smi

# Common formal settings; replace RUN_ID/ENV_FILE/GPUs per the matrix.
EPISODE_SHUFFLE=0 SPLIT=val_mini MAX_EPISODES=30 START_EPISODE_INDEX=0 \
MAX_EPISODE_LENGTH=500 SIM_GPU_ID=<H> SEM_GPU_ID=<S> LLM_GPU_ID=<L-or--1> \
EXP_NAME=<run_id> METHOD_NAME=<run_id> \
LOG=results/valmini_20260719/runs/<run_id>.log \
JSONL_LOG=results/valmini_20260719/runs/<run_id>.jsonl \
DUMP_LOCATION=results/valmini_20260719/dump/<run_id> \
KG_SERIALIZATION=text DECISION_HISTORY=<on-or-off> USE_GTSEM=0 \
ENV_FILE=<backend-env> \
bash code/scripts/run_mindnav.sh --reset_seed_each_episode --seed 1
```

For N=1/3, replace the last line with `NUM_AGENTS=<1-or-3> bash code/scripts/run_mindnav_robot_ablation.sh --reset_seed_each_episode --seed 1`. Do not pass `NUM_AGENTS=2` to that launcher.

For Qwen, start exactly one server at a time in a separate terminal, then wait for its local health endpoint before launching Habitat:

```bash
GPU_ID=<L> MODEL_PATH=models/Qwen2.5-7B-Instruct \
SERVED_MODEL_NAME=qwen2.5-7b CONDA_ENV=vllm \
bash code/scripts/run_vllm_qwen.sh
```

For 3B, change `MODEL_PATH` and `SERVED_MODEL_NAME=qwen2.5-3b`, and select the separate 3B backend env file. Stop the prior vLLM server before switching models.

## 7. 聚合、统计与可交付表

### 7.1 审计规则

- 每个 Run ID 必须有 30 条正式 records，`episode_index=0..29` 连续且 30 个 compound keys 与冻结 manifest 一致。
- 拒绝缺失/重复 key、非有限或负 DTG、非 `complete` status、错误模型/后端、错误 history flag、错误 success threshold、或来自 smoke 的记录。
- 不聚合 MCoCoNav 字段；主指标仅使用 Habitat `habitat_success`、`habitat_spl`、`distance_to_goal`。
- 所有 paired comparisons 按 compound key 对齐，而不是按 JSONL 行号假定一致。

### 7.2 统计口径

- 每行报告 mean 与 95% episode-bootstrap CI（10,000 draws，固定 bootstrap seed `20260719`）。
- C1：报告 `Q3B−Q7B` paired ΔSR/ΔSPL/ΔDTG CI。
- C2：报告 `2−1`、`3−2`、`3−1` paired Δ；正 ΔSR/ΔSPL、负 ΔDTG 表示左侧规模较好。
- C3：报告 `memory-on−memory-off` paired Δ；DTG 也可等价报告 `off−on` 以使“减少距离”为正，但必须在表头写清方向。
- 另外报告 tokens/calls 总量和每 episode 均值。tokens 必须来自 vLLM/API 的 `usage` 返回值；calls 必须为底层实际 `chat.completions.create` 请求数，包含 repair/retry。

### 7.3 目标表

1. **Qwen backend/model-size table**：7B、3B 的 SR/SPL/DTG CI、tokens/episode、calls/episode、runtime/episode、invalid/repair/fallback。
2. **DeepSeek agent-count table**：N=1/2/3 的 SR/SPL/DTG CI 及 team steps、robot-actions、tokens、calls、runtime；紧接 paired delta 表。
3. **DeepSeek memory ablation table**：on/off 的 SR/SPL/DTG CI 和 paired delta；附 tokens/calls、重复分配/repair/fallback 诊断。

现有 `aggregate_results.py` 的固定 `PAIRS` 和默认 `expected_n=60` 不可直接作为本轮唯一聚合入口；正式前应新增或参数化一个专用聚合器，使其接受 N=30、上述 Run IDs/比较对、N=2 robot-cost 兼容规则和独立输出目录。先用一份合成 30-row fixture 验证 audit/pairing，再聚合正式数据。

## 8. 里程碑、预算与决策门

| Milestone | 目标 | 工作量 | Go/No-Go | 主要风险与处理 |
|---|---|---:|---|---|
| M0 | 冻结代码、数据 manifest、backend env、聚合规则 | 0 episodes | 单元测试与预检通过 | 数据链接/metadata 不一致：先修复，不启动仿真 |
| M1 | 六个关键路径 smoke | 6 episodes | 每个 smoke 审计通过 | memory 泄漏、vLLM 不可用、N=3 OOM：只修对应路径 |
| M2/R1 | 公共 DeepSeek N=2 基线 + Qwen 7B | 60 episodes | 两行均 30/30 | vLLM 独占 L2、输出隔离 |
| M3/R2 | memory-off + Qwen 3B | 60 episodes | history-off audit 与 Qwen health 通过 | 先停 7B vLLM 后启动 3B |
| M4/R3 | DeepSeek N=1/3 | 60 episodes | N=1/3 robot/mapping/cost audit 通过 | N=3 的多份语义模型显存不足则串行 |
| M5 | 严格聚合、paired bootstrap、案例抽查 | 0 episodes | 三张表全部 N=30、key 完全配对 | 不把历史 valmini 或 smoke 混入 |

按现有 `val_60` 运行时长粗估，formal 部分需要约 3 个并行轮次；Qwen 7B 和 DeepSeek 2-agent 约各 1–1.5 小时/30 episodes，N=3 可能更慢，Qwen 3B 的实际用时以 smoke 测得的每 episode runtime 为准。预留约 15–20 GPU-hours（Habitat/semantic/vLLM 合计）和 DeepSeek API 调用预算；该数是排程估计，不能当作实验结果。

## 9. 最终检查清单

- [ ] 30 个 `val_mini` keys 已冻结且六行完全相同
- [ ] 六个 smoke 均通过，且未进入 formal JSONL
- [ ] Qwen 7B/3B 均通过独立 vLLM env 和实际 model-name 审计
- [ ] N=1/3 真实进入 Habitat、brain、map、action 与 cost 日志；N=2 保持 golden path
- [ ] memory-off 无历史字段泄漏，但不删除当前 KG
- [ ] `nvidia-smi` 在每次 Habitat 启动前已检查并记录；`mindnav38`/`vllm` 环境严格分离
- [ ] 所有 formal records 的 success threshold 均为 0.2 m，SR/SPL/DTG 合法
- [ ] 聚合器通过 N=30 合成 fixture，所有差值按 compound key paired bootstrap
- [ ] 报告了 token/call/runtime 与 invalid/repair/fallback；未将其与导航指标混淆
- [ ] 结果表明确限制：N=30、单次完整运行、DeepSeek 服务端非确定性、agent-count 的总动作预算随 N 改变
