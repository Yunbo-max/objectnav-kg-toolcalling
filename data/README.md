# Data setup

## HM3D v0.2

We use `val_mini` (30 episodes, 10 scenes, 6 goal categories) from HM3D v0.2.

Expected directory (symlink or copy):

```
data/hm3d_v0.2/
├── val_mini/
│   └── 00800-TEEsavR23oF/.../*.glb
└── val/          # optional for full eval
```

Lab checkout at `/tf/notebooks/Co-NavGPT/data/scene_datasets/hm3d_v0.2/val/`.

## Episodes

- 30 episodes on `val_mini`
- Goal categories: `chair`, `bed`, `sofa`, `toilet`, `plant`, `tv_monitor`
- Geodesic success threshold: 1.0 m
- Max steps per episode: 500
