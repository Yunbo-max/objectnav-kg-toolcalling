# Data Setup

Habitat is launched from:

```text
/home/huaziheng/project/objectnav-kg-toolcalling/code/vendor/conavgpt
```

Dataset paths are therefore relative to `code/vendor/conavgpt`, not to this
repository-root `data/` directory.

## Current HM3D v0.2 Layout

```text
code/vendor/conavgpt/data/
├── datasets/
│   └── objectnav_hm3d_v2/
│       ├── val/
│       │   ├── val.json.gz
│       │   └── content/
│       └── val_mini/
│           ├── val_mini.json.gz
│           └── content/
├── hm3d-v0.2-full/
│   └── scene_datasets/hm3d -> versioned_data/hm3d-1.0/hm3d
└── scene_datasets/
    └── hm3d_v0.2 -> ../hm3d-v0.2-full/scene_datasets/hm3d
```

Full HM3D `val` scenes are available at:

```text
code/vendor/conavgpt/data/scene_datasets/hm3d_v0.2/val/
```

The full-val scene config is:

```text
code/vendor/conavgpt/data/scene_datasets/hm3d_v0.2/val/hm3d_val_basis.scene_dataset_config.json
```

## Splits

| Split | Episodes | Scene files | Use |
|---|---:|---:|---|
| `val` | 1000 | 36 referenced scenes, 100 downloaded val scene dirs | main evaluation |
| `val_mini` | 30 | 2 referenced scenes | legacy quick debugging |

## Habitat Task Config

The task config is:

```text
code/vendor/conavgpt/envs/habitat/configs/tasks/multi_objectnav_hm3d.yaml
```

Relevant fields:

```yaml
DATASET:
  DATA_PATH: "data/datasets/objectnav_hm3d_v2/{split}/{split}.json.gz"
  SCENES_DIR: "data/scene_datasets"
SIMULATOR:
  SCENE_DATASET: "data/scene_datasets/hm3d_v0.2/{split}/hm3d_{split}_basis.scene_dataset_config.json"
```

## Verification

```bash
cd /home/huaziheng/project/objectnav-kg-toolcalling
readlink -f code/vendor/conavgpt/data/scene_datasets/hm3d_v0.2
find -L code/vendor/conavgpt/data/scene_datasets/hm3d_v0.2/val -maxdepth 2 -name '*.basis.glb' | wc -l
test -f code/vendor/conavgpt/data/scene_datasets/hm3d_v0.2/val/hm3d_val_basis.scene_dataset_config.json
```

Expected `.basis.glb` count for the downloaded val assets: `100`.
