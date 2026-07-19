# MindNav 在 HM3D `val_60` 上的实验方案与结果

**状态**：PARTIAL（E1–E4 已完成：6 个配置、360/360 个正式 episode；E5 机器人数量消融已规划、尚未实现/运行）  
**制定日期**：2026-07-18  
**E1–E4 完成日期**：2026-07-18  
**E5 规划日期**：2026-07-19  
**主评测集**：HM3D ObjectNav `val_60`  
**主方法**：MindNav；E1–E4 固定 2 个机器人，E5 比较 1/2/3 个机器人；无需训练，高层决策每 25 步调用一次

> 当前内容包含 1 组主实验和 4 组消融/诊断实验。本文档将 GT 语义实验视为 oracle perception study（感知上界诊断），不把它作为可部署方法参与主结果比较。

## 1. 实验目标与 Claim Map

| Claim ID | 待验证结论 | 最小可信证据 | 对应实验 |
|---|---|---|---|
| C1 | MindNav 在云端 DeepSeek 与本地 Qwen2.5-7B 上均能完成 `val_60` ObjectNav，方法不依赖单一 LLM | 两个模型在完全相同的 60 个 episode 上报告 SR、SPL、DTG，并给出置信区间 | E1 主实验 |
| C2 | KG 的组织形式会影响决策效果和调用效率 | 在同一 DeepSeek、同一 KG 事实、同一 episode 上，仅改变 text/JSON/triples 序列化；同时报告性能、token 和 API 调用数 | E2 KG 形式消融 |
| C3 | 决策历史包能减少无效重复探索并改善导航 | DeepSeek 完整方法与 no-history 版本的 paired SR/SPL/DTG 差异 | E3 历史消融 |
| C4 | 语义感知误差是 MindNav 性能上限的重要来源 | 仅将预测语义替换为 Habitat GT 语义，测量 SR/SPL/DTG 相对提升 | E4 GT 语义上界 |
| C5 | 在其余设置和同步 episode horizon 不变时，增加机器人数量能提升团队覆盖与导航成功率，并呈现可量化的收益/成本曲线 | 在相同 60 个 episode、同一代码提交和同一 DeepSeek 配置下比较 1/2/3 robots 的 paired SR/SPL/DTG，同时报告动作、token、调用和时间成本 | E5 机器人数量消融 |

需要排除的替代解释：

- E2 的差异不能来自三种格式携带不同 KG 事实、不同截断范围或不同 max tokens。
- E3 只能移除历史衍生字段，不能同时移除持久化 KG、当前机器人状态或当前 frontier 信息。
- E4 只能替换语义来源，不能改变 episode、LLM、KG、规划器、成功阈值或动作预算。
- E5 中每个机器人数量必须使用相同 episode、单机器人最大 500 个同步环境步和相同规划间隔；机器人增加会自然增加总 robot-actions，因此该实验只能直接支持“fleet-size scaling”，不能单独证明增益来自协作机制。
- 所有比较必须使用相同 `(scene, episode_id)` compound key 和顺序，不能把 `val_mini` 与 `val_60` 混在同一张表中。

## 2. 固定评测协议

### 2.1 数据集

使用仓库中的 `code/vendor/conavgpt/data/datasets/objectnav_hm3d_v2/val_60`：

- 60 个 episode，4 个 scene，每个 scene 15 个 episode；
- 类别分布：chair 10、bed 10、plant 9、toilet 14、tv_monitor 5、sofa 12；
- 固定 `EPISODE_SHUFFLE=0`、`START_EPISODE_INDEX=0`、`MAX_EPISODES=60`；
- 每次运行前保存并核对 `selection_manifest.json`、60 个 `(scene, episode_id)` compound key 及其顺序；单独的 `episode_id` 会在不同 scene 内重复，不能作为全局唯一键；
- 不根据运行结果删 episode。崩溃的 episode 必须补跑同一 ID，并在日志中保留异常原因。

### 2.2 固定系统设置

除被消融的变量外，所有运行固定为：

| 项目 | 固定值 |
|---|---|
| 机器人数量 | E1–E4 固定为 2；E5 的唯一自变量为 1/2/3 |
| 最大 episode 步数 | 500 |
| 高层规划间隔 | 25 步 |
| Habitat split | `val_60` |
| 主 KG 格式 | `text` |
| 主历史设置 | 最近最多 4 次分配/探索反馈，启用 |
| 主语义设置 | RedNet + Mask R-CNN 预测语义，沿用当前融合逻辑 |
| DeepSeek 设置 | `temperature=0`、thinking disabled；记录实际 `base_url`、model 字符串和运行时间。当前默认 model 为 `deepseek-v4-flash` |
| Qwen 设置 | 本地 `Qwen2.5-7B-Instruct`，通过 vLLM OpenAI-compatible 服务，`temperature=0` |
| 随机性 | Python/NumPy/PyTorch/Habitat seed 固定为 1；Qwen 请求固定 seed；DeepSeek 不设请求 seed，每个配置只完整运行 1 次，不做重复实验 |
| 输出预算 | 两阶段 LLM 调用使用相同 max output tokens；三种 KG 格式不得单独调大预算 |
| 地图与控制 | 相同深度、地图分辨率、frontier 提取、FMM、STOP 逻辑及回退策略 |

所有实验固定采用当前 `multi_objectnav_hm3d.yaml` 的默认成功距离 **0.2 m**，距离定义为到目标有效 view point 的 geodesic distance。本文档、聚合脚本和论文表格均只使用 0.2 m 口径，不再评估或报告其他成功距离。

### 2.3 GPU 与环境约定

每次启动 Habitat 前先运行 `nvidia-smi` 检查 GPU 使用情况。正式实验每轮同时并行两个运行，采用交叉映射：

- 并行槽位 A：Habitat 使用 GPU 0，预测语义模型使用 GPU 1；
- 并行槽位 B：Habitat 使用 GPU 1，预测语义模型使用 GPU 0；
- GPU 0/1 各自同时承载一个 Habitat 进程和另一个实验的语义模型；启动前必须确认叠加后的显存足够；
- 本地 Qwen2.5-7B/vLLM 独占 GPU 2，使用 `vllm` conda 环境；Qwen 实验与一个 DeepSeek 实验并行，DeepSeek 不占本地 LLM GPU；
- Habitat/MindNav Python 进程使用 `mindnav38` conda 环境；
- GT 语义运行不初始化或调用预测语义模型，因此该槽位的 `SEM_GPU_ID` 仅作占位、不产生语义模型负载；如果 `mindnav38` 无法加载 GT 语义数据，则该实验改用 `mindnav38_hm3d022`；
- 两个并行进程必须使用不同的 `EXP_NAME`、日志、JSONL、dump 目录及后端配置文件，禁止共享可覆盖的输出路径；
- 第一轮同时运行 DeepSeek 和 Qwen，必须支持按进程选择独立 env 配置（建议为 `run_mindnav.sh` 增加 `ENV_FILE`），不能让两个进程共同读取同一份固定 `BRAIN_BACKEND` 配置。

同一轮的两个运行同时启动；任一运行失败只补跑该 Run ID，不要求已经完成的配对运行重跑。

## 3. 指标与统计口径

### 3.1 导航指标

主表统一读取 per-episode JSONL 的 Habitat 字段：

- **SR ↑**：`mean(habitat_success)`，以百分比报告；
- **SPL ↑**：`mean(habitat_spl)`；
- **DTG ↓**：episode 结束时当前团队内所有机器人到任一目标有效 view point 的最小 geodesic distance，即 `mean(distance_to_goal)`，单位为米；

E5 中上述团队指标按实际 `num_agents` 动态计算：SR 为任一机器人成功，SPL 为成功机器人各自路径效率的最大值，DTG 为所有机器人中到目标的最小 geodesic distance。用户提出的 **SRL** 在当前仓库和 Habitat 指标中没有对应实现；本方案按现有上下文将其视为 **SPL** 的笔误。若 SRL 是另一个自定义指标，必须在全量运行前先冻结公式并增加日志与测试，不能事后从聚合值推断。

由于 SR/SPL/DTG 都会随团队规模产生“更多尝试机会”，E5 还应报告以下成本诊断，但不替代三个主指标：`robot_actions = num_agents × executed_team_steps`、每 episode token/API calls、wall-clock time；如实现方便，再报告所有机器人累计路径长度。主表必须明确其预算口径是固定 **500 个同步 team steps**，而不是固定总 robot-actions。

本实验不统计、不聚合、不报告任何 MCoCoNav 指标；正式运行已移除相关 tracker/JSONL 字段。

聚合时拒绝缺失、负数或非有限 DTG，不得把默认占位值 `-1` 纳入平均值。正式全量运行的每行均满足 `N=60` 且 compound episode key 集完全相同。

### 3.2 Token 与 API 调用指标

E2 和 E5 对每个 episode 记录并汇总：

- `input_tokens`：所有实际 LLM 请求的 `response.usage.prompt_tokens` 之和，包含 system 与 user 消息；
- `output_tokens`：所有实际 LLM 请求的 `response.usage.completion_tokens` 之和；
- `api_calls`：底层 `chat.completions.create` 的真实请求次数；正常一轮高层决策通常为两次请求，但校正请求和外层重试都必须计数；
- 同时报告总量与每 episode 均值；如 provider 返回 cached/reasoning token，再单独落盘，不能替代上述主口径；
- token 必须使用 API/vLLM 返回的 usage，不能把本地估算 token 与 provider usage 混合。

### 3.3 不确定性与显著性

- 每个单项结果报告 95% episode-bootstrap CI（10,000 次重采样，bootstrap seed `20260718`）；
- 消融差值使用相同 episode 的 paired bootstrap，报告 `Δ = 完整方法 - 消融方法`；
- E5 另外固定报告 `2−1`、`3−2`、`3−1`，正的 ΔSR/ΔSPL 和负的 ΔDTG 表示左侧机器人规模更优；
- SR 同时可报告 Wilson 95% CI；主文至少保留 bootstrap CI；
- 每个配置只完整运行 1 次，再对该次运行的 60 个 episode 做 bootstrap；DeepSeek 不因服务端不确定性增加重复运行。

## 4. 最小运行矩阵

DeepSeek-text 完整运行 `M-DS-TEXT` 同时作为 E1 主实验、E2 text 对照、E3 history-on 对照和 E4 predicted-semantics 对照，因此 E1–E4 最小只需 **6 个唯一全量配置，共 360 episodes**。六个配置分成 3 轮，每轮并行 2 个实验。

| Run ID | 实验 | LLM | KG 格式 | 决策历史 | 语义 | 主要统计 | 状态 |
|---|---|---|---|---|---|---|---|
| M-DS-TEXT | E1/E2/E3/E4 公共对照 | DeepSeek | text | on | predicted | SR, SPL, DTG, tokens, calls | COMPLETE |
| M-Q7B-TEXT | E1 主实验 | Qwen2.5-7B-Instruct | text | on | predicted | SR, SPL, DTG | COMPLETE |
| A-KG-JSON | E2 KG 格式 | DeepSeek | JSON | on | predicted | SR, SPL, tokens, calls | COMPLETE |
| A-KG-TRIPLES | E2 KG 格式 | DeepSeek | triples | on | predicted | SR, SPL, tokens, calls | COMPLETE |
| A-NO-HISTORY | E3 历史 | DeepSeek | text | **off** | predicted | SR, SPL, DTG | COMPLETE |
| A-GT-SEM | E4 oracle 感知 | DeepSeek | text | on | **GT** | SR, SPL, DTG | COMPLETE |

E5 推荐在完成动态机器人实现后，用同一代码提交、同一时间窗口重新运行完整的 1/2/3 robots 矩阵，而不是把旧的 2-robot 结果与新代码下的 1/3-robot 结果直接拼接。这样新增 **3 个配置、180 episodes**：

| Run ID | 实验 | Robots | LLM | KG / history / semantics | 主要统计 | 状态 |
|---|---|---:|---|---|---|---|
| A-ROBOTS-1 | E5 数量消融 | 1 | DeepSeek | text / on / predicted | SR, SPL, DTG, costs | TODO |
| A-ROBOTS-2 | E5 数量消融的同期对照 | 2 | DeepSeek | text / on / predicted | SR, SPL, DTG, costs | TODO |
| A-ROBOTS-3 | E5 数量消融 | 3 | DeepSeek | text / on / predicted | SR, SPL, DTG, costs | TODO |

若计算预算严格受限，可在 2-robot 动态化回归测试证明 prompt、地图融合、动作和指标路径均未变化后，复用 `M-DS-TEXT`，只新增 1/3 robots 共 120 episodes；但论文优先采用同期重跑的 180-episode 方案，以减小代码版本和远端 DeepSeek 服务时间漂移的混杂。

若后续决定 Qwen2.5-7B 是论文默认主干，可在附录补跑 Qwen 的 no-history 与 GT-semantic；当前不是 must-run，避免把 LLM 差异与核心消融混在一起。

## 5. 实验块定义

### E1：DeepSeek 与 Qwen2.5-7B 主实验

- **目的**：给出 MindNav 在远程强 LLM 和可本地部署 7B LLM 下的核心结果。
- **比较项**：`M-DS-TEXT`、`M-Q7B-TEXT`。
- **控制变量**：相同 text KG、完整历史、预测语义、episode 顺序、动作预算、prompt、JSON 校验与 fallback。
- **指标**：SR、SPL、DTG；建议附录另报 fallback rate 与 API calls/episode 作为可执行性诊断。
- **可信标准**：两行均完成 60/60 episode 且日志通过数据审计。若要声称“跨模型稳健”，应同时给出 paired 差值与置信区间，而不能只比较点估计。
- **失败解释**：若 Qwen 明显落后，结论应限定为 MindNav 对 LLM 能力敏感，并结合 invalid-output、repair、fallback 比例区分推理能力与格式遵循问题。
- **论文目标**：主结果表。
- **优先级**：MUST-RUN。

### E2：KG 组织形式消融（text / JSON / triples）

- **目的**：比较 KG wire format 对导航表现、上下文长度和请求稳定性的影响。
- **比较项**：`M-DS-TEXT`、`A-KG-JSON`、`A-KG-TRIPLES`。
- **唯一自变量**：第一次 LLM 调用中的 `SERIALIZED_KG` 表达形式；第二次调用的 decision packet、工具 schema、提示规则和输出预算完全不变。
- **内容等价要求**：三种序列化必须来自同一个 bounded KG view，当前可分配 rooms、历史 rooms、objects、relations、frontiers 和数值精度一致。正式运行前用单元测试对三种格式提取 canonical facts 并比较。
- **指标**：SR、SPL、总/平均输入 token、总/平均输出 token、总/平均 API calls；可附 invalid response、schema repair、outer retry、fallback 次数。
- **可信标准**：若某种格式在 paired CI 下性能不降且显著减少 token/calls，可支持效率结论；若性能差异不确定，只报告 trade-off，不作格式优越性结论。
- **失败解释**：调用数升高而 SR/SPL 降低通常意味着格式遵循或输出截断问题；应先检查序列化内容等价和 provider usage，再解释为 KG 格式效应。
- **论文目标**：主文消融表；详细调用诊断放附录。
- **优先级**：MUST-RUN。

### E3：不向 LLM 提供决策历史包

- **目的**：隔离最近决策/探索反馈是否真正帮助 LLM 避免重复探索。
- **比较项**：`M-DS-TEXT`（history on）与 `A-NO-HISTORY`（history off）。
- **off 的严格定义**：两个 LLM 阶段都不能看到 `recent_assignments`、`recent_history_summary`、room-level progress/stagnation 等历史衍生字段；仍保留持久化 KG 的当前累积环境事实、当前 robot pose、当前 frontier 和本轮观测。
- **指标**：SR、SPL、DTG，paired delta；诊断项为重复 room/frontier 分配率、fallback rate 和平均规划轮数。
- **可信标准**：history-on 的 paired SR/SPL 提升或 DTG 降低且 CI 不跨 0，才能声称历史包有效。
- **失败解释**：若无差异，历史可能与持久化 KG 信息冗余；若 history-on 更差，应检查陈旧历史和过度惩罚重复房间的问题。
- **论文目标**：主文消融表。
- **优先级**：MUST-RUN。

### E4：GT 语义替换预测语义（Oracle Perception）

- **目的**：估计导航性能被语义预测误差限制的程度。
- **比较项**：`M-DS-TEXT`（predicted）与 `A-GT-SEM`（GT）。
- **唯一自变量**：送入 semantic mapping、目标检测和 KG 更新的逐像素语义来源。GT 实验使用 Habitat semantic sensor 的 instance ID，经场景 annotation/mapping 转成与预测分支相同的 15 类通道。
- **禁止泄漏**：不得把目标 view point、goal position、最短路径或 distance-to-goal 输入导航策略；GT 只替换语义类别掩码。
- **控制变量**：DeepSeek/text/history、RGB/depth、地图、frontier、FMM、STOP、episode 和成功距离不变。
- **指标**：SR、SPL、DTG，paired delta；可附目标首次发现步数。
- **可信标准**：GT 优于 predicted 的差值量化 perception headroom；该行必须明确标记 `Oracle / not deployable`。
- **失败解释**：GT 不升反降时优先判定为类别映射、通道索引、resize 或 STOP 接线错误，不能直接得出“预测语义优于 GT”。
- **论文目标**：感知瓶颈分析/附录；若差距很关键可进入主文。
- **优先级**：MUST-RUN。

### E5：机器人数量消融（1 / 2 / 3 robots）

- **目的**：量化团队规模从 1 增至 3 时的导航收益、边际收益和推理/执行成本，回答 2 robots 是否是合理默认点。
- **比较项**：`A-ROBOTS-1`、`A-ROBOTS-2`、`A-ROBOTS-3`。
- **唯一自变量**：`num_agents ∈ {1, 2, 3}`。DeepSeek/text KG/history on/predicted semantics、60 个 episode、seed、500 个同步 team steps、25-step 高层规划间隔、成功距离和所有导航逻辑均固定。
- **初始状态约束**：继续沿用当前多机器人协议——所有机器人共享 dataset start position，`robot_0` 使用 episode rotation，其余机器人使用 seeded random rotation；必须为三种规模保存 initial-pose manifest，并核对 `robot_0` 以及 2/3-robot 设置中的 `robot_1` 初始状态完全一致。
- **主指标**：SR、SPL、DTG；分别报告 2−1、3−2、3−1 的 paired delta 与 95% bootstrap CI。
- **成本诊断**：team steps、robot-actions、累计路径长度（若接线）、input/output tokens、API calls、planning rounds、wall-clock/episode。机器人数量会改变 LLM 输出中的 assignment 数量，这部分 token/call 成本属于规模扩展的真实成本，不应人为对齐。
- **预算解释**：固定 500 个同步 team steps 意味着总可用 robot-actions 随机器人数量线性增长。结果应写成“团队规模扩展的端到端效果”；若要进一步声称协作本身有效，需要另做固定总 robot-actions 或 independent-agent baseline，不纳入本轮 must-run。
- **可信标准**：至少 `3 robots − 1 robot` 的 SR/SPL 改善或 DTG 降低有一项 paired CI 不跨 0，且其余指标不出现明显反向退化，才支持“更多机器人带来可靠收益”。只有点估计单调时写成趋势；`3−2` 小于 `2−1` 可作为收益递减的描述，但不能在区间重叠时写成显著结论。
- **失败解释**：3 robots 无提升时，依次检查第三机器人是否真正参与地图融合/动作执行、frontier 数是否不足、LLM assignment 是否退化、同起点碰撞、语义吞吐和 500-step 终止逻辑；接线通过后，结果可说明任务在 2 robots 已饱和或协调开销抵消额外覆盖。
- **论文目标**：主文/附录的机器人规模曲线表；建议用 3 行小表而非只画点图，保留 CI 与成本列。
- **优先级**：MUST-RUN（新增）。

## 6. 正式运行前的实现与验收项

截至 2026-07-18，E1–E4 的实现与审计项均已完成；E5 仍需先完成 6.4，当前不能直接启动 1/3-robot 正式运行。

### 6.1 已具备

- `run_mindnav.sh` 已支持 `SPLIT=val_60`、episode 范围、GPU ID、唯一日志路径和 `KG_SERIALIZATION`；
- `HelicaseBrain` 已支持 `text`、`json`、`triples`；
- per-episode JSONL 已记录 Habitat success、SPL、final distance-to-goal 和 KG 格式；
- `val_60` split 与 selection manifest 已存在。

### 6.2 已完成的补充实现

- [x] **Token 日志**：记录 provider/vLLM 返回的 prompt、completion、cached 和 reasoning tokens，并按 episode 做 usage snapshot 差分。
- [x] **API 请求计数**：最低层统计每次 `chat.completions.create`，包含 repair 与 retry。
- [x] **历史开关**：`DECISION_HISTORY=off` 同时屏蔽两个阶段的所有历史衍生字段，并停止内部历史积累；单元测试和正式 mapping audit 均通过。
- [x] **GT 语义接线**：Habitat instance ID 经跟踪在 `code/configs/matterport_category_mappings.tsv` 的完整 Matterport mapping 映射为 15 类通道；GT 分支不初始化 RedNet/Mask R-CNN。由于 Habitat-Sim 0.2.1 不加载该批 HM3D annotation，GT 使用 `mindnav38_hm3d022`。
- [x] **聚合器**：支持 SR/SPL/DTG、token/calls、N、10,000 次 bootstrap CI、paired delta、类别诊断和严格数据审计。
- [x] **移除无关指标**：运行记录与聚合器均不包含 MCoCoNav 字段。
- [x] **配置元数据**：每条记录含 backend/model、history、semantic source、scene、compound episode key 所需字段、seed、0.2 m 阈值、commit、时间和诊断计数。
- [x] **并行后端隔离**：`ENV_FILE` 按进程选择；Qwen 使用 `code/configs/qwen_vllm.env`。
- [x] **指标审计**：团队 SPL 改为成功机器人各自实际路径的最佳 SPL，并增加多机器人成功路径单元测试。
- [x] **成功阈值检查**：360 条正式记录均为 0.2 m。
- [x] **可恢复合并**：`merge_episode_jsonl.py` 按连续 `episode_index` 和 `(scene, episode_id)` 去重、排序和规范化零值诊断字段。

### 6.3 Smoke test 验收标准

六个关键路径各运行 1 个 episode，共 6 episodes；仍按正式排程分 3 轮、每轮并行 2 个：

- 第 1 轮：DeepSeek text（H0/S1）与 Qwen text（H1/S0/L2）；
- 第 2 轮：DeepSeek JSON（H0/S1）与 DeepSeek triples（H1/S0）；
- 第 3 轮：no-history（H0/S1）与 GT semantic（H1/GT）。

验收要求：每个 JSONL 恰好一条有效记录；SR、SPL、DTG 合法；0.2 m 成功距离元数据正确；token/calls 可复算；三种 KG 格式事实等价；no-history prompt 无历史字段；GT 分支未调用预测语义模型；两个并行进程的 backend、GPU 和输出路径互不串扰。

验收结果：6/6 初始关键路径通过；三种 KG 格式的 canonical current rooms/frontiers 相同，history-off 无字段泄漏。GT 还额外经历两次阻断性审计：Habitat-Sim 0.2.1 无 annotation 的 smoke 被判 INVALID；第一次全量运行又发现仓库中的 category mapping 文件只是 placeholder，造成 tv/couch 通道缺失，该批也被判 INVALID。补入完整 mapping 后，sofa/tv targeted smoke 的目标通道均非零且 SR=1，随后才重新执行有效 GT 全量。

### 6.4 E5 动态机器人数量：当前可行性与待实现项

**可行性结论：可以实现，但当前代码不可直接运行该消融。** 仓库中的 `HelicaseBrain`、KG assignment、Habitat `DistanceToGoal` 和团队 SPL 主体已按 `num_agents` 循环，具备扩展基础；阻断点集中在 launcher、Habitat agent 配置和 `exp_main_brain.py` 的双机器人硬编码。

必须在 smoke 前完成：

- [ ] **统一参数入口**：`run_mindnav.sh` 新增 `NUM_AGENTS=${NUM_AGENTS:-2}`，只接受 1–3，并显式传入 `--num_agents "$NUM_AGENTS"`。当前 CLI 虽已有 `--num_agents`，但 launcher 未传，且程序未将它写回 Habitat 配置。
- [ ] **动态 Habitat 配置**：在创建环境前设置 `config_env.SIMULATOR.NUM_AGENTS=args.num_agents` 和 `SIMULATOR.AGENTS=["AGENT_0", ...]`；对缺失的 `AGENT_2` 从 `AGENT_0` clone 相同 height/radius/sensors。不得只改 `NUM_AGENTS` 而保留长度为 2 的 `AGENTS`。
- [ ] **移除主循环双机器人硬编码**：把 `action=[0, 0]` 改为按 `num_agents` 初始化；把 `torch.cat((full_map[0], full_map[1]))` 改为对全部 `full_map` 做 `torch.stack(...).max(0)`，确保 1 台不越界、3 台不漏图。
- [ ] **动态化活动 prompt/schema**：`code/src/brain.py` 当前仍包含“two frontiers / other robot”规则和只列 `robot_0/robot_1` 的 exact JSON wrapper。按 `self.num_agents` 生成 robot keys、示例 JSON 和覆盖策略：N=1 不要求跨机器人 diversity；N=3 在候选不足时允许受约束复用，在候选充足时要求三台完整且唯一的合法 room/frontier pair。否则 N=3 会持续缺 key 并进入 repair/fallback，N=1 会被示例诱导输出多余 key。
- [ ] **动态历史与 probe**：将 history 中固定的 `r0/r1`、`chosen_0/chosen_1` 和二元打印改为 `assignments={robot_i: frontier_i}`、动态 chosen list、duplicate count；否则第三台决策无法完整审计，单机器人字段也会产生伪值。
- [ ] **共享冻结感知模型**：当前每个 `LLM_Agent` 都各自构造一份 Mask R-CNN 和 RedNet，N=3 会在同一 `SEM_GPU_ID` 上加载三份权重。优先让多个 agent 共享同一份 eval-only perception model bundle、仍逐机器人顺序推理并保留各自地图状态；不得为了规避 OOM 只给 3-robot 行降低分辨率、换模型或改阈值。若暂不重构，N=3 smoke 必须证明显存确实足够。
- [ ] **元数据和成本日志**：per-episode JSONL 增加 `num_agents`、`initial_agent_poses`、`team_steps`、`robot_actions`；聚合器必须审计同一 Run ID 内 `num_agents` 恒定、跨规模初始姿态满足嵌套约束，并加入 E5 三个 paired comparison。
- [ ] **动态单元测试**：至少覆盖 N=1/2/3 的 action/map shape、Habitat `AGENTS` 配置、brain assignment 完整性/唯一性、deterministic fallback、DTG 和团队 SPL；保留现有 N=2 测试作为回归门。
- [ ] **2-robot 行为兼容审计**：在不调用远端 LLM 的 deterministic fixture 下，动态化前后的 N=2 地图融合、prompt robot keys、最终 frontier assignments、action list 和指标必须一致。未通过则不得复用旧 `M-DS-TEXT`。

实现后的 smoke 使用每种规模相同的 1 个 episode，按 N=1→2→3 顺序执行；N=3 先单独检查显存，再允许进入并行排程。每条记录必须满足 `len(observations/actions/semantic_pixel_counts_by_agent)==num_agents`，prompt 与最终 assignments 恰有 `robot_0...robot_{N-1}`，融合地图包含所有机器人，`robot_actions=num_agents×team_steps`，SR/SPL/DTG 合法。任何条件失败都应修复并重跑 smoke，不能进入全量。


## 7. 推荐运行顺序与决策门

E1–E4 的 M0–M5 均已完成。为减少空闲，实际执行在保持两个固定 GPU 槽位和全部配置不变的前提下采用滚动衔接；有效 GT 的 mapping recovery 则使用两个 30-episode shard 并行重跑。E5 新增 M6–M9，尚未开始。

| Milestone | 内容 | 运行量 | Go/No-Go 条件 | 风险与处理 |
|---|---|---:|---|---|
| M0 | 实现 token/calls/history/GT/aggregation；运行单元测试 | 0 Habitat episode | 所有相关测试通过 | 不完整的 instrumentation 会使整批实验作废 |
| M1 | 6 个关键路径各 1-episode smoke test；每次并行 2 个 | 6 episodes | 通过 6.3 全部验收项 | 出错只重跑对应 smoke，不进入全量 |
| M2 / 第 1 轮 | 并行：`M-DS-TEXT`（H0/S1）与 `M-Q7B-TEXT`（H1/S0/L2） | 120 episodes | 两行均 60/60，指标与元数据完整 | Qwen 使用 GPU 2；两个进程使用独立 backend env |
| M3 / 第 2 轮 | 并行：`A-KG-JSON`（H0/S1）与 `A-KG-TRIPLES`（H1/S0） | 120 episodes | 与 text 对照 episode 完全配对 | 启动前检查 GPU 0/1 交叉负载后的显存 |
| M4 / 第 3 轮 | 并行：`A-NO-HISTORY`（H0/S1）与 `A-GT-SEM`（H1/GT） | 120 episodes | 消融变量审计通过 | GT 映射错误时禁止解释结果，先回到 smoke |
| M5 | 聚合、bootstrap、paired analysis、失败案例抽查 | 0 episodes | 表格 N=60、无重复/缺失/非法值 | 保留 raw JSONL，聚合表可重建 |
| M6 / E5 实现 | 动态化 1–3 agents，补单元测试和 JSONL/聚合字段 | 0 episodes | `mindnav38` 下全量单测通过，N=2 deterministic regression 一致 | 双机器人硬编码遗漏会使消融无效 |
| M7 / E5 smoke | N=1/2/3 各跑同一 1 episode；N=3 先独占检查显存 | 3 episodes | 通过 6.4 的 shape、assignment、map、metric、cost 验收 | 启动 Habitat 前先 `nvidia-smi`；失败只重跑对应规模 |
| M8 / E5 全量 | 推荐同期跑 1/2/3 robots，共 180 episodes | 180 episodes | 三行均 60/60，同 commit、episode keys、模型配置和阈值 | 第一轮 N=1 与 N=3 并行；第二轮 N=2；N=3 OOM 时降低并发而非改模型 |
| M9 / E5 分析 | 计算 2−1、3−2、3−1 paired CI 和成本曲线 | 0 episodes | `num_agents` 审计通过，主表同时呈现效果与成本 | 不把固定 team-step 结果误写成固定总动作预算 |



## 8. 运行配置模板

正式执行时，每个 Run ID 均使用独立 `ENV_FILE`、`EXP_NAME`、`METHOD_NAME`、`LOG`、`JSONL_LOG` 和 dump 目录。公共参数如下：

```bash
EPISODE_SHUFFLE=0 \
SPLIT=val_60 MAX_EPISODES=60 START_EPISODE_INDEX=0 \
MAX_EPISODE_LENGTH=500 SIM_GPU_ID=<SIM_GPU> SEM_GPU_ID=<SEM_GPU> \
bash code/scripts/run_mindnav.sh --reset_seed_each_episode --seed 1
```

三轮全量实验的 GPU/后端排程固定如下：

| 轮次 | 槽位 A | GPU 映射 | 槽位 B | GPU 映射 |
|---|---|---|---|---|
| 1 | M-DS-TEXT / DeepSeek | Habitat 0, Sem 1 | M-Q7B-TEXT / Qwen | Habitat 1, Sem 0, vLLM 2 |
| 2 | A-KG-JSON / DeepSeek | Habitat 0, Sem 1 | A-KG-TRIPLES / DeepSeek | Habitat 1, Sem 0 |
| 3 | A-NO-HISTORY / DeepSeek | Habitat 0, Sem 1 | A-GT-SEM / DeepSeek | Habitat 1, GT semantic（无预测模型） |

正式运行使用两个隔离命令并行/滚动启动；`ENV_FILE` 支持已完成并用于隔离 DeepSeek 与 Qwen 后端。

各运行的差异参数：

| Run ID | 后端/服务配置 | 运行参数 |
|---|---|---|
| M-DS-TEXT | 独立 DeepSeek `ENV_FILE` | `KG_SERIALIZATION=text --decision_history on --use_gtsem 0` |
| M-Q7B-TEXT | 独立 Qwen `ENV_FILE`：`BRAIN_BACKEND=vllm`, `BRAIN_MODEL=qwen2.5-7b` | `KG_SERIALIZATION=text --decision_history on --use_gtsem 0` |
| A-KG-JSON | 独立 DeepSeek `ENV_FILE` | `KG_SERIALIZATION=json --decision_history on --use_gtsem 0` |
| A-KG-TRIPLES | 独立 DeepSeek `ENV_FILE` | `KG_SERIALIZATION=triples --decision_history on --use_gtsem 0` |
| A-NO-HISTORY | 独立 DeepSeek `ENV_FILE` | `KG_SERIALIZATION=text --decision_history off --use_gtsem 0` |
| A-GT-SEM | 独立 DeepSeek `ENV_FILE` | `KG_SERIALIZATION=text --decision_history on --use_gtsem 1` |

E5 完成 6.4 后，三行只改变 `NUM_AGENTS` 和唯一输出路径：

```bash
# 每次启动 Habitat 前先检查 GPU；Python 使用 mindnav38。
nvidia-smi

# 修改 N、SIM_GPU、SEM_GPU 后启动对应行；DeepSeek 配置默认读取 .env。
N=1
SIM_GPU=0
SEM_GPU=1
ENV_FILE=.env NUM_AGENTS="$N" \
METHOD_NAME="A-ROBOTS-$N" EXP_NAME="a_robots_${N}_val60" \
LOG="results/runs/a_robots_${N}_val60.log" \
JSONL_LOG="results/runs/a_robots_${N}_val60.jsonl" \
EPISODE_SHUFFLE=0 SPLIT=val_60 MAX_EPISODES=60 START_EPISODE_INDEX=0 \
MAX_EPISODE_LENGTH=500 SIM_GPU_ID="$SIM_GPU" SEM_GPU_ID="$SEM_GPU" \
KG_SERIALIZATION=text DECISION_HISTORY=on USE_GTSEM=0 \
bash code/scripts/run_mindnav.sh --reset_seed_each_episode --seed 1
```

推荐排程：第一轮 `A-ROBOTS-1` 用 H0/S1、`A-ROBOTS-3` 用 H1/S0 并行；第二轮 `A-ROBOTS-2` 用 H0/S1。DeepSeek 不占本地 LLM GPU，E5 不需要启动 vLLM。根据现有 2-robot 运行约 2.5–3 小时/60 episodes 的记录，预计 E5 同期重跑墙钟约 6–8 小时；N=3 的实际吞吐和显存必须以 smoke 为准。

`--decision_history` 与 `--use_gtsem 1` 均已完成接线和审计。Qwen 服务在独立终端和 `vllm` 环境启动：

```bash
GPU_ID=2 \
MODEL_PATH=models/Qwen2.5-7B-Instruct \
SERVED_MODEL_NAME=qwen2.5-7b \
CONDA_ENV=vllm bash code/scripts/run_vllm_qwen.sh
```

## 9. 论文实验结果

所有均值与 CI 由 `code/scripts/aggregate_results.py` 从正式 JSONL 生成。单项 CI 为 10,000 次 episode bootstrap；差值 CI 为相同 compound episode key 的 paired bootstrap；seed 均为 `20260718`。数值保留三位小数。

### 表 1：MindNav 主实验（HM3D `val_60`, 2 robots, predicted semantics, success distance 0.2 m）

| Method | LLM | N | SR ↑ | 95% CI | SPL ↑ | 95% CI | DTG (m) ↓ | 95% CI |
|---|---|---:|---:|---|---:|---|---:|---|
| MindNav | DeepSeek (`deepseek-v4-flash`) | 60 | 0.717 | [0.600, 0.833] | 0.353 | [0.277, 0.430] | 0.947 | [0.527, 1.439] |
| MindNav | Qwen2.5-7B-Instruct (`qwen2.5-7b`) | 60 | 0.667 | [0.550, 0.783] | 0.307 | [0.232, 0.382] | 1.476 | [0.790, 2.319] |

Qwen − DeepSeek 的 paired 差值：ΔSR = −0.050，CI [−0.150, 0.050]；ΔSPL = −0.046，CI [−0.109, 0.012]；ΔDTG = +0.529 m，CI [−0.070, 1.307]。三个区间均跨 0。

### 表 2：KG 组织形式消融（DeepSeek, HM3D `val_60`, success distance 0.2 m）

| KG format | N | SR ↑ | SPL ↑ | Input tokens (total) ↓ | Input / ep ↓ | Output tokens (total) ↓ | Output / ep ↓ | API calls (total) ↓ | Calls / ep ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | 60 | 0.717 | 0.353 | 2,129,754 | 35,495.900 | 288,862 | 4,814.367 | 946 | 15.767 |
| JSON | 60 | 0.683 | 0.337 | 2,699,694 | 44,994.900 | 335,986 | 5,599.767 | 1,070 | 17.833 |
| Triples | 60 | 0.700 | 0.361 | 3,729,940 | 62,165.667 | 291,744 | 4,862.400 | 939 | 15.650 |

相对 text 的 paired 性能差值：

| Comparison | ΔSR | 95% CI | ΔSPL | 95% CI | ΔDTG (m) | 95% CI |
|---|---:|---|---:|---|---:|---|
| JSON − Text | −0.033 | [−0.133, 0.067] | −0.016 | [−0.070, 0.034] | −0.032 | [−0.428, 0.383] |
| Triples − Text | −0.017 | [−0.100, 0.067] | +0.008 | [−0.037, 0.054] | +0.260 | [−0.153, 0.736] |

调用与解析诊断如下：

| KG format | Invalid outputs | Repair calls | Outer retries | Fallback decisions | Cached input tokens（如有） |
|---|---:|---:|---:|---:|---:|
| Text | 68 | 12 | 0 | 28 | 569,472 |
| JSON | 52 | 18 | 0 | 17 | 678,272 |
| Triples | 35 | 9 | 0 | 13 | 773,632 |

### 表 3：决策历史包消融（DeepSeek, text KG, success distance 0.2 m）

| History to LLM | N | SR ↑ | 95% CI | SPL ↑ | 95% CI | DTG (m) ↓ | 95% CI |
|---|---:|---:|---|---:|---|---:|---|
| On (MindNav full) | 60 | 0.717 | [0.600, 0.833] | 0.353 | [0.277, 0.430] | 0.947 | [0.527, 1.439] |
| Off | 60 | 0.700 | [0.583, 0.817] | 0.352 | [0.274, 0.433] | 1.090 | [0.603, 1.654] |

Paired `on − off`：ΔSR = +0.017，CI [−0.083, 0.117]；ΔSPL = +0.001，CI [−0.055, 0.060]；ΔDTG = −0.143 m，CI [−0.668, 0.341]。若按“DTG 改善”写成 `off − on`，点估计为 +0.143 m，CI [−0.341, 0.668]。

### 表 4：预测语义与 GT 语义上界（DeepSeek, text KG, history on, success distance 0.2 m）

| Semantic source | Deployable | N | SR ↑ | 95% CI | SPL ↑ | 95% CI | DTG (m) ↓ | 95% CI |
|---|---|---:|---:|---|---:|---|---:|---|
| RedNet + Mask R-CNN prediction | Yes | 60 | 0.717 | [0.600, 0.833] | 0.353 | [0.277, 0.430] | 0.947 | [0.527, 1.439] |
| Habitat GT semantic (Oracle) | No | 60 | 0.783 | [0.683, 0.883] | 0.425 | [0.345, 0.506] | 0.667 | [0.320, 1.088] |

Paired `GT − predicted`：ΔSR = +0.067，CI [−0.050, 0.183]；ΔSPL = +0.073，CI [0.009, 0.140]；ΔDTG = −0.281 m，CI [−0.828, 0.258]。因此只有 SPL 的提升在本次 60-episode paired bootstrap 下区间不跨 0。

### 可选附录表：按目标类别诊断

样本少（特别是 tv_monitor 仅 5 个），该表只作诊断，不作显著性结论。

| Variant | chair SR | bed SR | plant SR | toilet SR | tv_monitor SR | sofa SR | Macro SR |
|---|---:|---:|---:|---:|---:|---:|---:|
| M-DS-TEXT | 0.900 | 1.000 | 0.556 | 0.643 | 0.400 | 0.667 | 0.694 |
| M-Q7B-TEXT | 0.900 | 0.900 | 0.667 | 0.429 | 0.600 | 0.583 | 0.680 |
| A-NO-HISTORY | 0.900 | 1.000 | 0.667 | 0.500 | 0.800 | 0.500 | 0.728 |
| A-GT-SEM | 0.900 | 1.000 | 0.556 | 0.643 | 0.800 | 0.833 | 0.789 |

## 10. 实验追踪表

| Run ID | Round | GPU mapping | Backend/model | Output JSONL | Start time | End time | N complete | Status | Notes |
|---|---:|---|---|---|---|---|---:|---|---|
| M-DS-TEXT | 1A | H0/S1 | DeepSeek / deepseek-v4-flash | `results/runs/m_ds_text_val60.jsonl` | 06:35:40 | 09:08:51 | 60/60 | COMPLETE | episode 0 后触发 fallback 日志变量缺陷；保留 ep0，修复后从 ep1 续跑并严格合并 |
| M-Q7B-TEXT | 1B | H1/S0/L2 | vLLM / qwen2.5-7b | 06:42:32 | 09:50:30 | 60/60 | COMPLETE | 修复 fallback 缺陷后从 ep0 重启 |
| A-KG-JSON | 2A | H0/S1 | DeepSeek / deepseek-v4-flash | 09:14:14 | 12:13:48 | 60/60 | COMPLETE | 与 Qwen 尾段滚动重叠 |
| A-KG-TRIPLES | 2B | H1/S0 | DeepSeek / deepseek-v4-flash | 09:55:14 | 12:38:21 | 60/60 | COMPLETE | — |
| A-NO-HISTORY | 3A | H0/S1 | DeepSeek / deepseek-v4-flash | 12:24:46 | 15:03:32 | 60/60 | COMPLETE | 正式 mapping audit 无 history 泄漏 |
| A-GT-SEM | recovery | H0/GT + H1/GT | DeepSeek / deepseek-v4-flash | 16:29:19 | 17:41:31 | 60/60 | COMPLETE | `mindnav38_hm3d022`；修正 mapping 后按 0–29/30–59 两 shard 重跑并合并 |
| A-ROBOTS-1 | E5-1A | H0/S1 | DeepSeek / deepseek-v4-flash | `results/runs/a_robots_1_val60.jsonl` | — | — | 0/60 | TODO | 等待 M6/M7 |
| A-ROBOTS-2 | E5-2A | H0/S1 | DeepSeek / deepseek-v4-flash | `results/runs/a_robots_2_val60.jsonl` | — | — | 0/60 | TODO | 推荐同期重跑；仅在严格 N=2 回归通过且预算受限时复用 M-DS-TEXT |
| A-ROBOTS-3 | E5-1B | H1/S0 | DeepSeek / deepseek-v4-flash | `results/runs/a_robots_3_val60.jsonl` | — | — | 0/60 | TODO | N=3 smoke 先独占验证显存 |

状态仅使用：`TODO`、`SMOKE_PASS`、`RUNNING`、`COMPLETE`、`INVALID`。中断后续跑时必须记录 episode 范围和合并方式。

## 11. 最终数据检查清单

- [x] 每个主运行恰好有 60 个唯一 `(scene, episode_id)` compound key，且六行 key 集与顺序相同；
- [x] scene、goal、episode 顺序与 `val_60/selection_manifest.json` 一致；
- [x] 没有把 smoke、失败重试的重复 episode 或其他 split 混入聚合；
- [x] SR、SPL、DTG 使用 Habitat 主字段，口径和成功距离一致；
- [x] 所有运行的 success distance 都是 0.2 m，且未聚合或报告任何 MCoCoNav 指标；
- [x] token/calls 来自 provider usage 和最低层 transport 计数；
- [x] 三种 KG 格式仅改变序列化；history-off prompt 审计通过；GT 无导航信息泄漏；
- [x] 报告 fallback/repair/retry，outer retry 六组均为 0；
- [x] 使用 paired bootstrap 计算消融差值与 CI；
- [x] 表格中的 model、commit、seed、运行时间、GPU、配置和 raw log 路径均可追溯；
- [x] 保留 raw JSONL、无效批次和 shard，所有表格均可由脚本重建。
- [ ] E5 的 `num_agents` 已真正进入 Habitat、brain、地图融合、动作和日志，而不只是 CLI 元数据；
- [ ] E5 三个规模各有 60 个相同 compound episode keys，2−1、3−2、3−1 paired CI 与成本列可重建；
- [ ] E5 结果明确标注固定 500 team steps、总 robot-actions 随 N 增长，不把 fleet scaling 误解为纯协调增益。

## 12. 结果分析与结论

### 12.1 Claim verdict

- **C1：部分支持。** DeepSeek 与本地 Qwen2.5-7B 都完成 60/60，点估计上 DeepSeek 高 5.0 pp SR、0.046 SPL，DTG 低 0.529 m；但三项 paired CI 都跨 0。因此可以声称方法能在两个后端执行并取得相近量级结果，不能声称两者统计等价，也不能断言 DeepSeek 显著更优。
- **C2：支持“格式影响效率形态”，不支持某一格式导航性能显著更优。** JSON/Triples 相对 text 的 SR、SPL、DTG paired CI 均跨 0。JSON 比 text 多 26.8% input tokens、16.3% output tokens、13.1% calls；Triples 多 75.1% input tokens、约 1.0% output tokens，calls 只少 0.7%。结构化格式减少了 invalid/fallback（text 68/28，JSON 52/17，Triples 35/13），但在当前 prompt 下并未转化为显著导航收益，Triples 还显著扩大输入上下文。
- **C3：不支持历史包带来可测导航增益。** History-on 相对 off 的 ΔSR=+0.017、ΔSPL=+0.001、ΔDTG=−0.143 m，CI 全跨 0；history-on 的 input tokens 反而高 16.6%。当前证据更符合“持久化 KG 已包含足够累积信息，短期历史包大多冗余”，而不是历史包有效改善导航。
- **C4：部分支持感知瓶颈。** 修正类别映射后的 GT 把 SR 从 0.717 提至 0.783、SPL 从 0.353 提至 0.425、DTG 从 0.947 降至 0.667 m。只有 paired SPL 增益的 CI 不跨 0；SR 与 DTG 仍不确定。因此可信结论是 GT semantic 显著改善路径效率，提供约 +0.073 SPL 的 perception headroom，而不是宣称 SR 已显著提高。
- **C5：待验证。** 当前只有 2-robot 数据，且代码仍含双机器人硬编码；在完成 M6/M7 和 1/2/3 同期全量前，不对机器人数量收益作结论。

### 12.2 诊断观察

- Text、Triples、no-history 的规划轮数分别为 465、465、468，说明它们的 token 差异不是简单由运行轮数造成；Triples 的主要成本来自 wire format 本身。
- Qwen 的 invalid outputs/fallback 为 32/10，少于 DeepSeek text 的 68/28，但 Qwen 规划轮数和 calls 更多（526 轮、1,071 calls），最终性能点估计略低。格式遵循更好不等同于导航推理更好。
- GT 的类别诊断提升集中在 sofa（0.667→0.833）与 tv_monitor（0.400→0.800）；这与修正 `couch→sofa`、`tv/monitor→tv_monitor` 映射后的目标通道恢复一致。类别样本很小，只作为接线与误差来源诊断。
- Bootstrap 只刻画这一次完整运行中的 episode 不确定性，不包含 DeepSeek 服务端非确定性或跨 seed 方差。所有“显著/不显著”表述均限于本协议、这 60 个 episodes 和该 bootstrap 口径。

### 12.3 异常处理与有效数据边界

- DeepSeek text 首次在 episode 1 遇到 `fallback_reason` 后触发未赋值局部变量；代码修复后从 episode 1 续跑。episode 0 原记录与 1–59 shard 按连续索引合并，未重复执行或择样删除。
- Habitat-Sim 0.2.1 的 GT smoke 未加载 semantic annotations，原始文件保存在 `results/smoke/a_gt_sem.invalid_no_annotations.*`，未进入聚合。
- 第一次 GT 全量暴露 placeholder mapping：tv_monitor 5/5 目标通道为零、sofa 12 个中 9 个为零。该批保存在 `results/runs/a_gt_sem_val60.invalid_placeholder_mapping.*` 及 `results/aggregated_invalid_placeholder_mapping/`，状态为 INVALID，未进入最终结果。
- 有效 GT 使用完整 Matterport mapping，targeted sofa/tv smoke 均有非零目标像素且成功；正式 60 条记录全部有非零前景通道，5/5 tv_monitor 和 11/12 sofa episodes 观察到非零目标通道。

### 12.4 可复现产物

- 最终机器可读汇总：`results/aggregated/experiment_summary.json`
- 表格：`results/aggregated/run_summary.csv`、`results/aggregated/paired_comparisons.csv`
- 数据与配置审计：`results/aggregated_audit.txt`、`results/aggregated/semantic_and_config_audit.txt`
- 正式 per-episode 数据：`results/runs/{m_ds_text,m_q7b_text,a_kg_json,a_kg_triples,a_no_history,a_gt_sem}_val60.jsonl`
- 单元测试：`conda run --no-capture-output -n mindnav38 python -m unittest discover -s tests -p 'test_*.py' -q`，结果 47/47 通过。
