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

## 6. 第一阶段修复实现与验证（`kg` 分支，2026-07-16）

### 6.1 阶段目标与完成状态

第一阶段只处理第5节定位出的room/frontier生命周期与执行映射问题，不恢复 `helicase_target_found`，也不改变底层语义目标检测和局部规划器的STOP规则。

当前状态：**实现完成、单元验证完成、历史审计重放完成、前15个episode配对运行完成**。

核心修改如下：

1. 在 `code/src/kg_construction.py` 中增加每次决策独立构建的 `CurrentFrontierView`。它同时保存当前frontier实体和 `room_id -> [frontier]` 的一对多映射，避免同room中的frontier互相覆盖。
2. 持久KG room节点不再写入 `frontier_idx`、`size`、`target_prior` 等瞬时字段；更新时还会清理旧版本遗留的瞬时属性。
3. KG提示词明确分成 `CURRENT ASSIGNABLE ROOMS` 和 `HISTORICAL ROOM CONTEXT — NOT ASSIGNABLE`。历史room只用于空间关系推理，不能被LLM分配。
4. 在 `code/src/brain.py` 中只接受当次白名单内的 `room_id`，并只通过当次 `CurrentFrontierView` 把room转换成frontier。非法room、非法概率或不完整响应进入确定性当前frontier回退。
5. 当一个room对应多个当前frontier时，使用目标先验、距离、面积和索引进行稳定排序；frontier足够时给两台机器人分配不同候选，候选不足时才允许复用。
6. 在 `code/vendor/conavgpt/exp_main_brain.py` 中删除索引截断、随机回退和事后重复重定向。LLM与回退结果共用同一条严格执行校验路径。
7. 映射审计升级到schema v2，记录提示词白名单、LLM原始/接受room、最终frontier、持久KG瞬时属性泄漏及room/frontier一致性。

第一阶段形成的接口边界为：

| 数据 | 生命周期 | 是否可被分配 | 保存位置 |
|---|---|---:|---|
| room实体、位置、相邻关系、已见物体 | episode内持久 | 仅当它同时属于当前视图 | KG |
| frontier索引、面积、当前目标先验 | 单次规划决策 | 是 | `CurrentFrontierView` |
| 历史room上下文 | episode内持久 | 否 | KG提示词的历史区域 |

### 6.2 无GPU验证

所有Python命令均按项目约定使用 `mindnav38` conda环境。

```bash
conda run -n mindnav38 python -m unittest discover -s tests -v
conda run -n mindnav38 python -m py_compile \
  code/src/kg_construction.py \
  code/src/brain.py \
  code/vendor/conavgpt/exp_main_brain.py \
  code/scripts/replay_frontier_mapping.py \
  tests/test_current_frontier_mapping.py
git diff --check
```

结果：10/10个单元测试通过，Python编译检查和diff空白检查通过。测试覆盖同room多frontier保留、frontier过期、确定性回退、候选不足时复用、旧瞬时属性清理、提示词隔离、陈旧room拒绝、异常LLM响应及主程序使用canonical `KGUpdater` 等路径。

对删除快捷路径后的旧审计文件执行离线重放：

```bash
conda run -n mindnav38 python code/scripts/replay_frontier_mapping.py \
  results/runs/main_mindnav_hm3d_val_first15_qwen25_7b_no_kg_target_20260715.mapping.jsonl
```

| 重放指标 | 旧接口 | 第一阶段接口模拟 |
|---|---:|---:|
| 决策数 | 117 | 117 |
| 当前frontier实例 | 415 | 415 |
| 同room碰撞决策 | 45 | 45，全部走一对多映射 |
| 被同room覆盖的实例 | 54 | 0 |
| LLM非当前room分配 | 139/234 | 139个被拒绝 |
| 当前接口模拟后的重复frontier决策 | 不适用 | 0 |
| 最终执行非当前frontier | 无严格保证 | 0 |

重放还确认旧日志可以无歧义地重建当次room ID，重建不一致数为0。这说明修复针对的是已观测到的真实失败模式，而不是仅对人工样例有效。

### 6.3 定点仿真回归

启动Habitat前已检查GPU，按约定使用GPU 0运行Habitat、GPU 1运行语义模型、GPU 2运行Qwen2.5-7B的vLLM服务。

| 定点运行 | Success | SPL | steps | distance-to-goal | 映射审计 |
|---|---:|---:|---:|---:|---|
| Episode 2（toilet） | 1 | 0.3593 | 96 | 0.0434 m | 3次决策，全部错误项为0 |
| Episode 15（bed） | 1 | 0.4479 | 195 | 0.0538 m | 8次决策，全部错误项为0 |

结果文件为 `results/runs/kg_phase1_ep2_20260716.jsonl` 和 `results/runs/kg_phase1_ep15_20260716.jsonl`。定点运行跳过了前序episode，仿真随机状态与完整顺序运行不同，因此这里只把它们用作运行接口与异常路径回归，不作为性能配对样本。

### 6.4 完整前15个episode设置

- 数据集：HM3D ObjectNav v2，`val` split；
- 场景与episode顺序：与第5节相同，`start_episode_index=0`，共15个episode；
- 随机种子：1；
- 最大步数：500；
- 两台机器人；
- 决策模型：本地Qwen2.5-7B-Instruct，vLLM，32K上下文，`temperature=0`；
- 方法：删除 `helicase_target_found` 后叠加第一阶段room/frontier修复；
- 总运行时间：35分08秒。

逐episode原始配对结果如下。左侧“删除后”是第5节的未修复版本，右侧“第一阶段”是本次版本。

| Ep | 目标 | 删除后 S/SPL/steps | 第一阶段 S/SPL/steps | 第一阶段DTG |
|---:|---|---:|---:|---:|
| 1 | bed | 1 / 0.4369 / 122 | 1 / 0.5013 / 127 | 0.0412 m |
| 2 | toilet | 1 / 0.2710 / 162 | 0 / 0 / 500 | 0.0471 m |
| 3 | tv_monitor | 0 / 0 / 34 | 0 / 0 / 40 | 11.0383 m |
| 4 | chair | 0 / 0 / 155 | 0 / 0 / 500 | 0.0808 m |
| 5 | sofa | 0 / 0 / 500 | 0 / 0 / 500 | 5.0105 m |
| 6 | toilet | 1 / 0.8153 / 43 | 1 / 0.7041 / 54 | 0.0354 m |
| 7 | sofa | 0 / 0 / 500 | 0 / 0 / 500 | 2.8852 m |
| 8 | tv_monitor | 0 / 0 / 52 | 0 / 0 / 92 | 3.3552 m |
| 9 | toilet | 1 / 0.4939 / 141 | 1 / 0.3139 / 174 | 0.0030 m |
| 10 | chair | 0 / 0 / 165 | 1 / 0.9685 / 270 | 0.0516 m |
| 11 | sofa | 1 / 0.5521 / 136 | 1 / 0.4774 / 161 | 0.0113 m |
| 12 | tv_monitor | 0 / 0 / 29 | 0 / 0 / 57 | 5.1425 m |
| 13 | chair | 0 / 0 / 76 | 1 / 0.4904 / 174 | 0.0539 m |
| 14 | toilet | 1 / 0.0766 / 201 | 1 / 0.1735 / 111 | 0.0298 m |
| 15 | bed | 0 / 0 / 500 | 1 / 0.2486 / 453 | 0.0446 m |

### 6.5 汇总与配对分析

| 指标 | 删除前快捷路径 | 删除后未修复 | 第一阶段修复 | 第一阶段相对未修复 |
|---|---:|---:|---:|---:|
| SR | 0.4000（6/15） | 0.4000（6/15） | **0.5333（8/15）** | +0.1333（相对+33.3%） |
| SPL | 0.1946 | 0.1764 | **0.2585** | +0.0821（相对+46.6%） |
| 平均distance-to-goal | 3.4126 m | 4.0709 m | **1.8554 m** | -2.2155 m（-54.4%） |
| 平均steps | 185.27 | 187.73 | 247.53 | +59.80 |
| 总steps | 2779 | 2816 | 3713 | +897 |

成功episode：

- 删除前快捷路径：1、6、9、11、14、15；
- 删除后未修复：1、2、6、9、11、14；
- 第一阶段修复：1、6、9、10、11、13、14、15。

相对未修复版本，第一阶段新增成功Episode 10、13、15，丢失Episode 2。使用episode配对bootstrap（200,000次，seed 20260716）得到95%区间：SR差值 `[-0.1333, 0.4000]`，SPL差值 `[-0.0456, 0.2458]`，steps差值 `[5.73, 126.13]`，DTG差值 `[-3.9639, -0.6410]`。成功/失败的discordant pair为1个丢失、3个新增，双侧exact McNemar `p=0.625`。

因此可以观察到SR、SPL和DTG方向上的改善，但15个episode、单场景顺序和单一seed不足以支持统计显著性或完整HM3D泛化结论。步数明显上升，一部分来自新增成功episode继续探索到目标，另一部分来自下面讨论的两个“到达但未STOP”episode。

### 6.6 在线映射审计验收

完整运行共记录152次决策、304个机器人分配和576个当前frontier实例。

| 验收项 | 第5节未修复版本 | 第一阶段修复 |
|---|---:|---:|
| 决策数 | 117 | 152 |
| 同room多frontier决策 | 45 | 85 |
| 同room多frontier组 | 未完整保留 | 87组全部保留 |
| 被覆盖的当前frontier | 54 | **0** |
| 陈旧候选room决策 | 97 | **0** |
| LLM对非当前room评分 | 有 | **0** |
| LLM非当前room分配 | 139/234 | **0/304** |
| 持久room瞬时属性泄漏 | 未清理 | **0** |
| 最终执行非当前frontier | 可能发生 | **0** |
| room与最终frontier不一致 | 可能发生 | **0** |
| 最终重复frontier | 52次分配受重定向 | **0个决策** |
| 当前room未进入提示词白名单 | 未约束 | **0** |

152次决策中有85次（55.9%）实际包含room碰撞，说明一对多映射不是少数边缘情况。新接口保留了576/576个当前frontier。298/304个分配直接来自合法LLM room，剩余6个使用 `current_frontier_reuse`；这6次决策都只有一个当前frontier，符合“候选不足才复用”的设计。

**第一阶段的映射正确性验收通过。** 性能提升仍应视为初步信号，但第5节的三类关键错误——覆盖、陈旧候选和错误执行映射——在本次完整运行中均降为0。

### 6.7 Episode 2和4：已进入成功距离但没有结束

完整顺序运行中，Episode 2最终DTG为0.0471 m，Episode 4为0.0808 m，均小于任务配置 `TASK.SUCCESS.SUCCESS_DISTANCE=0.2 m`，但两者都运行到500 steps且Success为0。

这不是“两台机器人必须同时到达”的问题。当前多机器人任务的实际判定链如下：

1. `DistanceToGoal` 遍历两台机器人并取最小距离，即只要其中一台接近目标即可；
2. `env.step([action_0, action_1])` 只要动作列表中包含一个 `0`，就调用任务级全局 `STOP`；
3. `Success` 只有在“已调用STOP”且最小DTG小于0.2 m时才为1；
4. 底层 `LLM_Agent` 只在本地语义地图存在目标类别（`found_goal=1`）且FMM对该语义目标返回 `stop=True` 时输出动作0。

因此，**只需要一个Agent触发STOP，且两台Agent中的任意一台处于成功距离内即可成功；单纯进入0.2 m范围不会自动结束。** Episode 2和4跑满500步说明两台Agent都没有输出STOP。

Habitat的DTG是相对数据集目标view point计算的oracle地理距离，而策略的STOP来自RedNet语义预测、建图和FMM局部目标。这两个坐标/信号并不等价：机器人可以进入oracle成功区域，但因目标漏检、视角不合适、语义地图目标偏移或局部FMM尚未判定到达而不STOP。现有完整运行日志没有逐step保存 `found_goal` 和FMM `stop`，所以仅凭这次旧日志不能进一步区分具体是哪一个内部条件失败。

另一个配置陷阱是命令行参数 `--success_dist=1.0` 目前只在参数解析器中定义，在本执行路径没有被使用；真正的评测阈值来自 `multi_objectnav_hm3d.yaml`，为0.2 m。

该问题与本阶段已修复的room/frontier映射相互独立，建议作为第二阶段的第一个诊断项：

- 先记录每台机器人每步的动作、`found_goal`、FMM `stop`、目标语义面积和仅用于评测诊断的per-agent oracle DTG；
- 单独重跑Episode 2和4，定位“首次进入0.2 m”到episode结束之间是哪一个STOP条件未满足；
- 不应在策略中直接读取Habitat oracle DTG并据此自动STOP，否则会引入评测信息泄漏；
- 在不使用oracle信息的前提下，增加基于当前视觉目标置信度、连续帧确认和到语义目标地图距离的STOP仲裁；任意一个Agent确认后发出全局STOP；
- 统一或删除未生效的 `--success_dist` 参数，避免实验配置误读。

### 6.8 第一阶段结论与下一步

第一阶段达到了预定目标：持久room与瞬时frontier已经解耦，LLM只能选择当前room，同room中的多个frontier不再丢失，最终执行与LLM选择保持一致。前15个episode的SR从未修复版本的0.4000上升到0.5333，SPL从0.1764上升到0.2585，平均DTG从4.0709 m下降到1.8554 m；但由于样本量和seed有限，这些是后续扩大验证的依据，不是最终性能结论。

下一阶段建议按以下顺序进行：

1. 先补齐STOP链路可观测性并复现Episode 2和4，避免有效到达被计为超时失败；
2. 保持当前严格映射接口不变，再调整LLM room评分与Python先验融合，减少低价值长时间探索；
3. 扩展到更多episode和至少3个seed，报告均值、标准差、bootstrap区间和成功交换情况；
4. 在扩大实验前继续保留schema v2映射审计作为回归门槛，所有正确性错误项必须维持为0。

本次主要结果文件：

- `results/runs/main_mindnav_hm3d_val_first15_qwen25_7b_kg_phase1_20260716.jsonl`：第一阶段15个episode结果；
- `results/runs/main_mindnav_hm3d_val_first15_qwen25_7b_kg_phase1_20260716.mapping.jsonl`：152次schema v2映射审计；
- `results/runs/main_mindnav_hm3d_val_first15_qwen25_7b_kg_phase1_20260716.log`：完整运行日志；
- `results/runs/kg_phase1_ep2_20260716.jsonl`、`results/runs/kg_phase1_ep15_20260716.jsonl`：定点回归结果。

### 6.9 Episode 2和4的STOP诊断复现（2026-07-16）

为判断第6.7节的近目标超时是否为偶然现象，本次增加了可选的逐step STOP诊断日志。该日志记录每台机器人的动作、目标语义地图质量、`found_goal`、FMM `planner_stop` 和per-agent oracle DTG。oracle距离只用于事后诊断，不参与动作选择或STOP控制。

复现分为两种设置：

- **顺序复现**：从Episode 1运行到Episode 4，`seed=1`、`EPISODE_SHUFFLE=0`，保持与15-episode实验相同的前序随机数消耗；
- **隔离复现**：通过 `start_episode_index` 直接跳到Episode 2或4。跳过操作只调用 `env.reset()`，不会执行前序episode，因此不会消耗其导航过程中的Torch/NumPy随机数。

这里的“前序RNG/轨迹影响”不表示上一episode的机器人位置或KG被带入下一episode；它们都会reset。它表示程序只在进程启动时设一次seed，前序导航会推进随机数生成器状态，从而改变目标episode早期的随机目标、动作、观测、地图、frontier和后续LLM提示，最终形成不同轨迹。

| 运行方式 | Episode | 结果 | steps | DTG | 终止原因 |
|---|---:|---:|---:|---:|---|
| 原15集顺序运行 | 2 | 失败 | 500 | 0.0471 m | timeout，未STOP |
| 本次顺序复现 | 2 | 失败 | 500 | 0.0471 m | timeout，未STOP |
| 先前隔离定点 | 2 | 成功 | 96 | 0.0434 m | robot_0 STOP |
| 本次隔离诊断 | 2 | 成功 | 96 | 0.0434 m | robot_0 STOP |
| 原15集顺序运行 | 4 | 失败 | 500 | 0.0808 m | timeout，未STOP |
| 本次顺序复现 | 4 | 失败 | 149 | 8.2426 m | 远距离语义误检后误STOP |
| 本次隔离诊断 | 4 | 成功 | 275 | 0.0628 m | robot_0 STOP，SPL 0.6079 |

#### Episode 2：原顺序下可以稳定复现，不是一次偶然超时

本次顺序复现与原15集运行得到完全相同的Episode 2终值：500 steps、Success 0、DTG 0.047094 m；两次运行在所有共同决策step上的最终frontier分配也完全一致。

逐step诊断显示：

- step 110首次进入0.2 m，之后共有391步处于成功距离内；
- robot_0最小DTG为0.0327 m，并有395步 `found_goal=1`；
- robot_0有26步 `planner_stop=true`，但都发生在它具有有效目标语义证据之前，因此 `found_goal && planner_stop` 的同step交集为0；
- robot_1有451步 `planner_stop=true`，但500步内始终 `found_goal=0`；
- 两台机器人均没有输出动作0。

在step 110，具体状态为：robot_0 `found_goal=1, planner_stop=false`，robot_1 `found_goal=0, planner_stop=true`。该互补状态持续到最后一步。当前实现要求**同一台机器人**同时满足两个条件才STOP，不能把两台机器人的信号合并，因此Episode 2即使长时间位于成功区域也不会结束。

隔离Episode 2则在step 90首次进入0.2 m；step 96时robot_0同时满足 `found_goal=1, planner_stop=true` 并输出STOP，因此成功。这说明Episode 2对进入episode时的RNG状态和早期轨迹敏感，但在论文主实验采用的原顺序设置下，超时路径是可复现的。

#### Episode 4：原始“近目标超时”没有稳定复现，但STOP链路仍不稳定

本次顺序复现没有重复原来的500步近目标超时。它在step 99首次与原运行发生frontier分配分叉：

| step 99 | 原运行 | 本次顺序复现 |
|---|---|---|
| LLM概率 | `room_2_2=0.15, room_2_3=0.20, room_3_2=0.25` | `room_2_2=0.20, room_2_3=0.30, room_3_2=0.10` |
| 最终分配 `(robot_0, robot_1)` | `(frontier_0, frontier_1)` | `(frontier_3, frontier_2)` |

分叉后，本次顺序运行在step 149由robot_1触发STOP，但STOP机器人距离目标仍为8.2426 m，当前两台机器人的最小重新计算DTG也为4.8581 m。诊断显示robot_1出现chair语义误检并同时满足FMM停止条件，因此这是一次远距离误STOP，而不是近目标超时。

隔离Episode 4走了第三条轨迹：step 270首次进入0.2 m，step 275时robot_0的目标检测和FMM停止条件在同一步成立，最终成功。

因此不能把Episode 4原来的500步超时视为稳定、确定性的单episode结果；它对LLM概率输出和早期轨迹高度敏感。不过三条轨迹共同说明当前STOP链路存在系统性脆弱点：既可能出现“真实接近目标但语义目标与FMM停止条件不重合”的漏停，也可能出现“语义误检与FMM停止恰好重合”的误停。

#### 诊断结论和修复约束

1. Episode 2的近目标timeout在相同顺序条件下精确复现，不能归为偶然情况。
2. Episode 4的具体近目标timeout未复现，属于轨迹敏感结果；但复现暴露了同一机制的远距离误STOP风险。
3. 不能简单改成“任意Agent检测到目标且任意Agent的FMM停止就全局STOP”。Episode 2正好存在这种跨Agent互补信号，而它在其他场景会把一个Agent的语义误检和另一个Agent到达普通frontier错误组合，增加误STOP。
4. 下一步应让每台Agent的FMM停止判定明确对应它自己的、当前且新鲜的目标语义组件，并增加连续帧/面积/置信度确认；全局任务仍可在任一Agent通过本地一致性检查后结束。
5. 为提高配对实验复现性，建议在每个episode开始时使用可记录的派生seed，并向vLLM请求显式传入seed；同时保留逐step STOP诊断作为回归日志。

本次复现新增文件：

- `results/runs/kg_phase1_stopdiag_seq_ep1_4_20260716.jsonl`：Episode 1–4顺序复现结果；
- `results/runs/kg_phase1_stopdiag_seq_ep1_4_20260716.stop.jsonl`：顺序复现的逐step STOP诊断；
- `results/runs/kg_phase1_stopdiag_isolated_ep2_20260716.jsonl`：隔离Episode 2结果；
- `results/runs/kg_phase1_stopdiag_isolated_ep2_20260716.stop.jsonl`：隔离Episode 2诊断；
- `results/runs/kg_phase1_stopdiag_isolated_ep4_20260716.jsonl`：隔离Episode 4结果；
- `results/runs/kg_phase1_stopdiag_isolated_ep4_20260716.stop.jsonl`：隔离Episode 4诊断。

三组运行共49次KG决策，schema v2映射审计中的覆盖、陈旧候选、非当前room分配、瞬时属性泄漏、执行非当前frontier、room/frontier不一致及重复分配均保持为0。


