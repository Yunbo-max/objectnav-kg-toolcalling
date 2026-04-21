# MindNav Module Specifications

## Overview
MindNav replaces Co-NavGPT's flat-text frontier assignment with a Knowledge Graph (KG) driven autonomous reasoning system. It keeps the same perception pipeline (Detectron2 + RedNet + FMM) but adds KG construction and LLM-based KG reasoning.

## Module 1: KG Construction (`helicase_kg.py`)

### Node Types
| Type | ID Format | Fields | Example |
|------|-----------|--------|---------|
| Room | `room_{y}_{x}` (grid-discretized) | name, certainty, position, explored, frontier_idx, observation_count | `room_5_4` |
| Object | `{category}_in_{room_id}` | category, certainty, detections list [(conf, pos)], position | `sink_in_room_5_4` |
| Robot | `robot_{k}` | position, heading | `robot_0` |
| Door | `door_{id}` | position | `door_1` |

### Edge Types
| Relation | Source → Target | Condition | Properties |
|----------|----------------|-----------|------------|
| `contains` | Room → Object | Object detected in room | — |
| `next_to` | Object → Object | Distance < 20px (~0.2m) | distance |
| `near` | Object → Object | Distance < 60px (~0.6m) | distance |
| `connected_to` | Room → Room | Traversable boundary exists | distance |
| `separated_by_wall` | Room → Room | Wall detected between rooms | — |
| `in` | Robot → Room | Robot currently in room | — |

### Object Merging Algorithm
```
For each new detection (category, confidence, position):
  1. Find existing nodes of same category in same room
  2. If any within 30px distance → MERGE:
     - Update certainty: cert = 1 - (1-cert_old)(1-conf_new)
     - Update position: pos = 0.7*pos_old + 0.3*pos_new (exponential moving average)
     - Append to detections list
  3. If none nearby → CREATE new node
  4. Cap certainty at 0.99
```

### Certainty Formula (Noisy-OR)
```
cert(object) = 1 - ∏_{i=1}^{n} (1 - c_i)
```
Where c_i is the i-th detection confidence from Detectron2.

### Room ID Stability
Rooms use position-based IDs: `room_{floor(y/scale)}_{floor(x/scale)}`
- Same physical room → same ID across timesteps
- No re-clustering needed (unlike methods that segment rooms per frame)

## Module 2: KG-Driven Brain (`helicase.py → HelicaseBrain`)

### Input
- Serialized KG text (rooms, objects with certainty, connections, robot positions)
- Target object category
- Frontier list with room associations

### Reasoning Process
1. **Read KG**: Parse room-object associations
2. **Estimate P(target|room)**: LLM uses world knowledge (e.g., sink+towel → bathroom → toilet likely)
3. **Assign frontiers**: Each robot to a different frontier, maximizing coverage
4. **Diversity**: Include inter-frontier distances to prevent both robots going to same area

### Output Format
```json
{"robot_0": "frontier_X", "robot_1": "frontier_Y"}
```

### Priority Target Detection
If target detected with certainty > 0.5 → bypass reasoning, send nearest robot directly.

### No Hard-Coded Heuristics
- No room-object lookup tables
- No pre-computed target_prior scores
- LLM reasons from KG structure using its parametric world knowledge

## Module 3: Perception Pipeline (inherited from Co-NavGPT)

### Components (unchanged)
1. **Detectron2**: Instance segmentation (16 categories)
2. **RedNet**: Semantic segmentation from RGB-D
3. **2D Semantic Map**: Occupancy + semantic channels, shared between robots
4. **FMM Planner**: Fast Marching Method for shortest paths
5. **Frontier Extraction**: Identify unexplored boundaries every 25 local steps

### Integration Point
At each global planning step (every 25 local steps):
1. Perception updates semantic map
2. KG Builder reads semantic map → constructs/updates KG
3. Brain reads KG → assigns frontiers
4. FMM plans paths to assigned frontiers
