# MindNav KG 形式汇总：HM3D val_mini + val_60

**统计日期**：2026-07-19  
**统计范围**：每种 KG 形式合并 `val_mini` 30 episodes 与 `val_60` 60 episodes，因此 text / JSON / triples 各 `N=90`，共 270 条运行记录。  
**主指标**：Habitat SR、SPL、DTG（成功距离 0.2 m）；steps 作为执行成本诊断。  
**不确定性**：10,000 次按 split 分层的 episode bootstrap，固定保留 30:60 权重；paired 差值在各 split 内对齐相同 episode 后合并。

## 1. 数据审计

- 三种格式均为 `30 val_mini + 60 val_60 = 90` 条，无缺失。
- `val_mini` 与 `val_60` 共 90 个互不重复的 compound episode keys；每个 split 内三种格式的 episode、scene、goal 与顺序一致。
- 全部记录使用两个 robots、DeepSeek `deepseek-v4-flash`、预测语义和 0.2 m Habitat 成功口径；val_60 固定 history on。
- val_mini 的旧 JSONL 缺少 scene/episode_id 字段，compound keys 从同名 `.log` 的 30 条 `[EPISODE_START]` 恢复并交叉核对。
- 旧 `dualmetrics` 中 MCoCoNav 辅助字段不可比较：JSON/triples 的 `mcoconav_success/mcoconav_spl` 异常为全零；本报告只采用与 DTG 阈值一致的 Habitat 指标。

## 2. 每种 KG 形式的 90-episode 合并结果

| KG format | N | Success | SR ↑ | 95% CI | SPL ↑ | 95% CI | DTG (m) ↓ | 95% CI | Steps / ep ↓ |
|---|---:|---:|---:|---|---:|---|---:|---|---:|
| Text | 90 | 63 | **0.700** | [0.600, 0.789] | **0.352** | [0.290, 0.416] | 1.192 | [0.764, 1.677] | **173.389** |
| JSON | 90 | 61 | 0.678 | [0.578, 0.778] | 0.338 | [0.273, 0.403] | **1.051** | [0.646, 1.508] | 188.489 |
| Triples | 90 | 61 | 0.678 | [0.578, 0.767] | 0.348 | [0.283, 0.415] | 1.424 | [0.895, 2.004] | 184.244 |

点估计上，text 的 SR、SPL 和 steps 最好；JSON 的最终 DTG 最低；triples 的 SPL 接近 text，但 DTG 最高。三者差距都较小，必须结合 paired CI 判断。

## 3. Split 分层原始均值

| KG format | Split | N | Success | SR ↑ | SPL ↑ | DTG (m) ↓ | Steps / ep ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Text | val_mini | 30 | 20 | 0.667 | 0.351 | 1.682 | 132.133 |
| Text | val_60 | 60 | 43 | 0.717 | 0.353 | 0.947 | 194.017 |
| JSON | val_mini | 30 | 20 | 0.667 | 0.339 | 1.324 | 136.933 |
| JSON | val_60 | 60 | 41 | 0.683 | 0.337 | 0.915 | 214.267 |
| Triples | val_mini | 30 | 19 | 0.633 | 0.323 | 1.856 | 170.433 |
| Triples | val_60 | 60 | 42 | 0.700 | 0.361 | 1.208 | 191.150 |

格式排序在两个 split 上不完全一致：triples 在 val_60 的 SPL 最高，但在 val_mini 的 SR/SPL 最低；JSON 在两个 split 上都有最低 DTG。这种异质性不支持简单宣称某种格式普遍最优。

## 4. 合并后的 paired 差值

正的 ΔSR/ΔSPL 表示左侧更好；负的 ΔDTG/ΔSteps 表示左侧更好。

| Comparison | ΔSR | 95% CI | ΔSPL | 95% CI | ΔDTG (m) | 95% CI | ΔSteps | 95% CI |
|---|---:|---|---:|---|---:|---|---:|---|
| JSON − Text | −0.022 | [−0.100, 0.056] | −0.015 | [−0.061, 0.032] | −0.141 | [−0.463, 0.195] | +15.100 | [−2.778, 34.167] |
| Triples − Text | −0.022 | [−0.100, 0.056] | −0.004 | [−0.051, 0.044] | +0.231 | [−0.211, 0.756] | +10.856 | [−8.056, 31.156] |
| Triples − JSON | 0.000 | [−0.078, 0.078] | +0.010 | [−0.026, 0.048] | +0.372 | [−0.008, 0.845] | −4.244 | [−26.834, 18.134] |

所有 paired CI 均包含 0。当前 90-episode 合并证据不支持 text、JSON 或 triples 在 SR、SPL、DTG 或 steps 上具有统计可靠的整体优势。

## 5. 可解释范围与限制

这是一份历史结果汇总，不是严格的 KG-only 因果消融。val_mini text 于 09:30 运行，而工具调用改动提交 `d0d718a` 的时间为 13:56；带 `fixedorder` 的 JSON/triples 在 15:43/16:31 运行。val_mini mapping 日志中的 `invalid_after_repair` 分别为 text `56/168`、JSON `6/173`、triples `5/215`，说明 text 与结构化格式还混入了代码/固定工具顺序差异。val_60 三行是后续统一协议下的受控比较。

因此可报告的安全结论是：在现有两批 HM3D 结果合并后，三种 KG 形式的导航效果处于相近区间，text 的综合点估计略优、JSON 的 DTG 点估计最低，但所有 paired 区间均跨 0。若论文需要声称序列化形式的因果影响，应以统一代码下的 val_60 消融为主，或在当前代码上重跑 val_mini text/JSON/triples。

## 6. 可复现产物

- 聚合脚本：`code/scripts/aggregate_kg_formats_90.py`
- 机器可读汇总：`results/aggregated_kg_formats_90/experiment_summary.json`
- 合并主表：`results/aggregated_kg_formats_90/combined_run_summary.csv`
- Split 分层表：`results/aggregated_kg_formats_90/split_run_summary.csv`
- Paired 比较：`results/aggregated_kg_formats_90/paired_comparisons.csv`
- 类别诊断：`results/aggregated_kg_formats_90/category_summary.csv`

