# EfficientNav

Upstream: NeurIPS 2025 method; public code release was not vendored into this
checkout.

## What It Is

EfficientNav is a VLM-head navigation variant originally reported with a
larger VLM. In this project it is tracked as an external baseline candidate,
not as a first-class local implementation.

## Local Status

There is no local EfficientNav checkout under:

```text
/home/huaziheng/project/objectnav-kg-toolcalling/code/vendor/
```

The existing VLM failure analysis is represented by the local MCoCoNav
reproduction in:

```text
code/vendor/mcoconav/
```

## Current Legacy Numbers

The old `val_mini` table recorded the same collapse pattern as MCoCoNav:

| Method | Episodes | SR |
|---|---:|---:|
| EfficientNav-style Qwen2.5-VL-7B head | 30 | 0.000 |

Do not treat this as a full local EfficientNav reproduction unless the upstream
code is added to the repository and rerun on HM3D `val`.
