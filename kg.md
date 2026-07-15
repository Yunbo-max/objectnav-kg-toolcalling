# MindNav 中的两种 LLM 推理决策方式

当前代码和实验中出现过两种高层决策方式：

1. Python 根据语义先验给 frontier 打分，再由 LLM 分配机器人；
2. Python 构建知识图谱（KG），由 LLM 给 room 打分并分配机器人。

两种方式最终都不会直接控制 Habitat 的离散动作。它们只负责为两个机器人选择高层探索目标，底层的 `LLM_Agent` 再使用语义地图和局部规划器完成移动、目标追踪与停止。

## 1. 语义先验 Frontier 决策

### 1.1 整体流程

```text
两个机器人的语义地图
        |
        v
融合地图并提取 frontier、物体和墙体
        |
        v
Python 推断每个 frontier 附近的房间类型
        |
        v
Python 计算每个 frontier 的 target_likelihood
        |
        v
将当前 frontier 及其先验分数写入 prompt
        |
        v
LLM 为 robot_0 和 robot_1 分配不同的 frontier
        |
        v
frontier_idx -> frontier 中心点 -> 底层导航
```

主要实现位于 `code/vendor/conavgpt/exp_main_brain.py`：

- `Frontiers()` 从融合地图中提取当前可探索边界；
- `enrich_frontiers()` 为每个 frontier 补充房间类型、附近物体和目标先验；
- `form_prompt_for_chatgpt()` 将这些信息转换成文本 prompt；
- `parse_answer()` 解析 LLM 返回的机器人分配。

### 1.2 Python 如何计算先验

每个 frontier 的信息包括：

```text
frontier_idx
centroid
area
nearby_objects
room_type
room_confidence
target_prior
```

其中 `target_prior` 主要来自三类人工语义知识：

- `TARGET_ROOM_PRIOR`：目标与房间类型的关系，例如床更可能位于卧室；
- `OBJECT_COOCCURRENCE`：目标与附近物体的共现关系，例如沙发附近更可能出现电视；
- `ROOM_ADJACENCY`：相邻房间先验，例如卧室附近可能存在浴室。

例如，目标为 `bed` 时，交给 LLM 的信息可能是：

```text
frontier_0: <centroid: (120, 260), number: 183,
             room_type: bedroom, nearby: chest_of_drawers,
             target_likelihood: 0.90>

frontier_1: <centroid: (310, 180), number: 246,
             room_type: living_room, nearby: sofa, chair,
             target_likelihood: 0.05>
```

### 1.3 LLM 的职责

这种方式中，LLM不需要从零估计目标出现概率。Python已经给出了 `target_likelihood`，LLM主要负责：

- 优先覆盖高先验 frontier；
- 让两个机器人选择不同 frontier；
- 综合机器人距离、探索覆盖范围和此前运动方向；
- 避免没有必要的频繁目标切换。

LLM输出格式为：

```text
robot_0: frontier_0
robot_1: frontier_2
```

需要注意，Python先验在该实现中只是写进自然语言 prompt，并没有直接参与最终的 `argmax`。代码不会强制某一台机器人选择最高分 frontier，也不会验证选择结果必须包含最高分 frontier。因此这是一种“Python提供先验，LLM做多目标分配”的软约束方法，并非由Python直接完成分配。

在旧3B前15个 episode 的记录中，排除所有先验完全相同的步骤后，共有60次可比较决策，其中52次（86.7%）至少有一台机器人覆盖了Python最高分 frontier。这说明它通常能利用先验，但不保证 `robot_0` 一定选择最高分。

### 1.4 优点与局限

优点：

- 输入对象就是当前有效 frontier，输出可以直接转换成导航坐标；
- 目标先验明确，小模型也容易使用；
- 不存在持久 room 节点与当前 frontier 之间的二次映射；
- 一台机器人利用高先验，另一台机器人可以保持探索多样性。

局限：

- 房间和目标关系依赖人工规则；
- `target_likelihood` 是启发式分数，不是真实概率；
- LLM可能为了距离、方向或历史一致性而忽略高分 frontier；
- 当多个 frontier 的先验相同时，Python规则无法提供有效排序；
- 没有充分利用跨时间的房间、物体和空间关系。

## 2. KG Room 决策

### 2.1 整体流程

```text
两个机器人的语义地图
        |
        v
提取当前 frontier、物体、墙体和机器人位置
        |
        v
更新持久 Knowledge Graph
        |
        v
将 KG 转换为文本
        |
        v
LLM 估计每个未探索 room 的 P(target | room)
        |
        v
LLM 为两个机器人分配不同 room
        |
        v
room.properties["frontier_idx"] -> frontier 中心点
        |
        v
底层导航
```

当前主程序使用：

- `code/src/kg_construction.py` 构建和更新 KG；
- `code/src/brain.py` 读取 KG、调用 LLM 并解析决策；
- `code/vendor/conavgpt/exp_main_brain.py` 将 room 决策映射回当前 frontier 并调用底层控制器。

### 2.2 KG 中包含的信息

当前 KG 的主要节点为：

- `robot`：机器人及其当前位置；
- `room`：未探索或已探索的空间区域；
- `object`：语义地图中发现的物体。

主要边包括：

- `robot -> in -> room`；
- `robot -> explored -> room`；
- `robot -> path_to -> room`；
- `room -> contains -> object`；
- `room -> connected_to -> room`；
- `room -> separated_by_wall -> room`；
- `object -> next_to/near -> object`；
- `object -> suggests -> room_type`。

当前 `room` 并不是真实的房间实例分割结果，而是按地图位置划分的50像素网格代理：

```text
room_id = room_{y // 50}_{x // 50}
```

每个当前 frontier 根据中心点关联到一个 room，并通过下面的属性保留返回路径：

```text
room.properties["frontier_idx"] = frontier_idx
```

### 2.3 LLM 的职责

KG会被转换成类似下面的文本：

```text
robot_0: in=room_4_3
  reachable_unexplored: room_5_4(80px), room_6_2(150px)

UNEXPLORED:
  room_5_4 [frontier=1]: type=likely_bedroom, certainty=30%
    objects: bed(90%), chest_of_drawers(70%)
    connected: room_5_5

  room_6_2 [frontier=3]: type=likely_living_room, certainty=30%
    objects: sofa(90%), chair(70%)
```

如果目标为 `bed`，LLM需要同时完成概率估计和机器人分配：

```text
P(bed|room_5_4) = 0.85
P(bed|room_6_2) = 0.10

robot_0: room_5_4
robot_1: room_6_2
```

Brain解析房间ID后，通过 `frontier_idx` 将其转换成底层可以执行的 frontier：

```text
robot_0 -> room_5_4 -> frontier_1 -> centroid -> local planner
```

因此，这种方式可以概括为“KG组织证据，LLM估计房间概率并分配机器人”。

### 2.4 当前实现的关键特点

当前KG提示词没有把第一种方式已经计算好的 `target_prior` 传给LLM。LLM看到的是简化后的房间类型、物体和空间关系，需要自行估计 `P(target | room)`。因此，即使模型更大，输入信息不充分时也不一定比直接使用Python先验更准确。

此外，当前实现还存在以下限制：

- 多个 frontier 可能落入同一个50像素 room，导致 `frontier_idx` 被覆盖；
- KG跨决策步骤保留room节点，而当前 frontier 编号会重新生成，可能产生陈旧映射；
- LLM可能给持久KG中的room打高分，但该room不再对应当前最合适的 frontier；
- 房间概率和机器人分配由一次文本调用同时完成，没有结构化工具调用约束；

原有的KG `helicase_target_found` 快捷分支已经删除。现在KG和LLM只负责目标尚未确认时的探索；当底层语义地图检测到目标后，`LLM_Agent` 会用目标语义连通区域覆盖高层 frontier 目标，导航到目标附近，并在局部规划器到达目标且 `found_goal=1` 时执行STOP。

### 2.5 优点与局限

优点：

- 能表达跨时间积累的房间、物体和空间关系；
- LLM可以使用物体共现、房间连接、墙体和机器人可达性进行推理；
- 比固定目标—房间表具有更强的扩展能力；
- 更接近MindNav所强调的KG推理流程。

局限：

- 当前room只是粗粒度网格代理；
- LLM需要从不完整证据中自行估计概率；
- room与当前frontier之间存在映射损失；
- 持久KG与动态frontier的生命周期尚未完全对齐；
- 自然语言输出解析失败时仍需要启发式回退。

## 3. 两种方式的核心区别

| 对比项 | 语义先验 Frontier 决策 | KG Room 决策 |
|---|---|---|
| LLM直接选择的对象 | 当前 frontier | KG中的 room |
| 概率由谁计算 | Python人工规则 | LLM根据KG估计 |
| Python先验是否进入prompt | 是 | 当前没有 |
| LLM的主要任务 | 双机器人frontier分配 | 房间概率估计和机器人分配 |
| 最终导航目标 | frontier中心点 | room映射到的frontier中心点 |
| 状态是否跨步骤积累 | 主要使用当前地图和上次决策 | KG在episode内持续更新 |
| 是否存在room到frontier映射 | 否 | 是 |
| 主要优势 | 简单、直接、先验稳定 | 关系表达能力和可扩展性强 |
| 主要风险 | 人工先验错误、LLM不服从软约束 | 证据不足、room碰撞和陈旧映射 |

第一种方式可以概括为：

```text
Python判断“哪里更可能”，LLM决定“两台机器人怎么分配”。
```

第二种方式可以概括为：

```text
KG组织环境证据，LLM同时判断“哪个房间更可能”和“两台机器人怎么分配”。
```

## 4. 可考虑的融合方式

两种方式并不冲突。较稳妥的融合方案是把Python先验作为基础概率，把KG推理作为有界修正：

```text
final_score(room) = clamp(Python_prior(room) + KG_adjustment(room), 0, 1)
```

其中LLM不再从零生成任意概率，而是根据KG额外证据输出有限范围的修正量，例如：

```text
KG_adjustment in [-0.15, +0.15]
```

最终由Python根据 `final_score` 完成可验证的分配：

- `robot_0` 选择最终分数最高的有效 frontier；
- `robot_1` 在剩余 frontier 中综合最终分数和空间距离；
- 强制两个机器人选择不同且当前有效的 frontier；
- 结构化记录Python prior、LLM修正量、最终分数和实际选择。

这种融合保留第一种方式稳定、直接的优点，同时允许KG提供第一种人工规则未覆盖的空间关系和上下文证据。

## 5. 删除 `helicase_target_found` 后的前15个episode测评

### 5.1 实验设置

本次实验用于评估删除KG中的 `helicase_target_found` 快捷路径后，MindNav在HM3D Val上的表现，并同步审计 `room_id` 与动态frontier之间的映射。

- 数据集：HM3D v0.2 Val；
- episode：固定顺序的前15个，`EPISODE_SHUFFLE=0`；
- 随机种子：1；
- LLM：Qwen2.5-7B-Instruct，通过vLLM提供OpenAI兼容服务；
- LLM上下文长度：32768；
- 每个episode最多500 steps；
- Habitat、语义模型和LLM分别使用GPU 0、1、2；
- 对照组和删除后的实验使用相同episode顺序与目标类别。

删除前的快捷路径在9/15个episode中共触发41次；删除后的运行日志中不再出现 `helicase_target_found` 或 `check_target`，因此本次对比覆盖到了被删除的实际运行分支。

### 5.2 总体结果

| 指标 | 删除前 | 删除后 | 变化 |
|---|---:|---:|---:|
| SR | 0.400（6/15） | 0.400（6/15） | 0 |
| SPL | 0.1946 | 0.1764 | -0.0182（-9.4%） |
| 平均distance-to-goal | 3.4126 m | 4.0709 m | +0.6583 m |
| 平均steps | 185.27 | 187.73 | +2.47 |
| 总steps | 2779 | 2816 | +37 |

删除前成功的episode为：1、6、9、11、14、15。

删除后成功的episode为：1、2、6、9、11、14。

因此总体SR没有变化，但成功episode发生了交换：

- Episode 2（toilet）：从500 steps失败变为162 steps成功，SPL从0提升至0.2710。旧版本从step 124起连续触发16次 `helicase_target_found`，却没有完成episode；删除快捷路径后，正常KG+LLM决策继续运行并最终成功。
- Episode 15（bed）：从125 steps成功变为500 steps失败，SPL从0.5448降为0。旧版本在step 124触发快捷路径后立即成功；删除后，决策受到陈旧room/frontier映射影响并重复至超时。

这说明 `helicase_target_found` 不是单向有益或单向有害的：它可能因为语义误检而让探索陷入错误目标，也可能在目标检测正确时快速把机器人引向目标。删除它能够消除与底层 `found_goal` 重复的目标检测职责，但不能自动提高导航效果，因为删除后KG映射问题会完全暴露出来。

### 5.3 `room_id` 与frontier映射的审计结果

本次运行额外记录了117次LLM决策的只读映射审计。该审计在决策完成后写入单独的JSONL，不参与导航决策。

| 映射现象 | 统计结果 |
|---|---:|
| 存在当前frontier合并的决策 | 45/117（38.5%） |
| 第一次决策就发生合并的episode | 10/15（66.7%） |
| 被同room覆盖的活跃frontier实例 | 54/415（13.0%） |
| 包含陈旧KG room候选的决策 | 97/117（82.9%） |
| 陈旧room候选实例 | 408/730（55.9%） |
| LLM选择非当前room的决策 | 90/117（76.9%） |
| LLM分配到陈旧KG room | 136/234（58.1%） |
| LLM分配到无有效frontier的room | 3/234（1.3%） |
| 非当前room分配合计 | 139/234（59.4%） |
| 最终经过索引截断或重复重定向的分配 | 52/234（22.2%） |

这里的分母117表示规划决策次数，分母234表示117次决策中的两台机器人分配，分母415和730表示跨决策累计的frontier与room候选实例，而不是互相独立的episode样本。

### 5.4 映射问题的形成机制

当前room与frontier的绑定同时混用了“持久空间实体”和“当前规划列表索引”：

1. `room_id` 根据 `room_{y//50}_{x//50}` 生成。在5 cm地图分辨率下，50 px约等于2.5 m，因此同一网格内经常同时存在多个frontier。
2. KG的每个room节点只保存一个 `frontier_idx`。当多个frontier映射到同一room时，后写入节点的 `properties.update()` 会覆盖前一个frontier索引，使部分当前frontier失去独立身份。
3. KG在一个episode内持续积累，但frontier列表会在每次规划时重新检测、排序和编号。`frontier_0` 到 `frontier_3` 是瞬时索引，不是稳定ID。
4. 已经不属于当前frontier集合的历史room仍保留旧 `frontier_idx`，并继续作为未探索room出现在KG文本中。
5. LLM会对这些历史room估计概率并进行分配。代码随后把旧索引截断到当前frontier范围；如果两台机器人得到同一索引，再根据Python先验给 `robot_1` 选择替代frontier。

因此，最终执行的frontier可能已经不对应LLM推理时选择的room。索引即使仍处于0到3的合法范围内，也可能在新一轮规划中代表完全不同的位置，不能仅靠范围检查发现。

Episode 15是最明显的案例。step 174之后，当前 `frontier_1` 的bed先验达到0.9，但LLM长期选择历史room `room_5_4` 和 `room_2_5`，二者保存的旧索引均为2。两个room选择经过重复处理后被转换为frontier `(2, 1)`，该模式持续到500 steps。此时LLM的“room推理”与最终执行的“frontier导航”已经脱节。

### 5.5 本次结论

删除 `helicase_target_found` 后，前15个episode上的SR保持0.4，但SPL下降、平均distance-to-goal增大。由于样本只有15个episode和单一seed，这些数值只能作为定位性结果，不能据此判断完整HM3D上的泛化效果。

更重要的结论是：当前性能瓶颈不是是否保留 `helicase_target_found`，而是持久KG room与瞬时frontier索引之间缺少有效的生命周期管理。只删除快捷路径会减少一条错误检测路径，但无法保证KG+LLM输出能够正确落到当前frontier。

因此暂不建议重新加入KG层的目标检测快捷路径。下一步应在不改变LLM核心推理规则的前提下，先修复room/frontier接口：

- room节点继续作为持久KG实体，但不永久保存瞬时 `frontier_idx`；
- 每次决策单独构建 `current_room_to_frontiers` 映射；
- 历史room可保留用于关系推理，但必须标记为当前不可分配，并从LLM可选目标中排除；
- 同一room中的多个frontier必须保留为独立候选，不能通过覆盖只留下一个；
- LLM输出room后，只能使用当次映射转换为有效frontier；
- 如果一个room对应多个当前frontier，可以使用当前Python prior、距离和覆盖度选择该room内的具体frontier；
- 修复后使用同一前15个episode重新进行配对验证，再扩展到更多episode和多个seed。

本次结果文件：

- `results/runs/main_mindnav_hm3d_val_first15_qwen25_7b_no_kg_target_20260715.jsonl`：删除后的15个episode结果；
- `results/runs/main_mindnav_hm3d_val_first15_qwen25_7b_no_kg_target_20260715.mapping.jsonl`：117次room/frontier映射审计；
- `results/runs/main_mindnav_hm3d_val_first15_qwen25_7b_no_kg_target_20260715.log`：完整运行日志；
- `results/runs/main_mindnav_hm3d_val_first15_qwen25_7b_32k_20260715.jsonl`：删除前的7B对照结果。
