# MindNav: Knowledge-Graph Tool Calling for Training-Free Multi-Robot Object Navigation

Reference code and paper for a training-free multi-robot ObjectNav system that
replaces Co-NavGPT's flat-text frontier assignment with LLM tool-calling over
a dynamically constructed knowledge graph (KG) of rooms, objects, doors, and
robots.

Paper: [`paper/main.pdf`](paper/main.pdf) (EMNLP submission).

## Headline result (HM3D `val_mini`, N=2, Qwen2.5-7B-Instruct)

| Method | SR ↑ | SPL ↑ |
|---|---|---|
| Co-NavGPT (random frontier) | 0.760 | — |
| **MindNav (ours)** | **0.733** | **0.416** |
| Co-NavGPT (LLM, flat text) | 0.700 | 0.306 |
| MCoCoNav (VLM 7B) | 0.000 | 0.000 |
| EfficientNav (VLM 7B) | 0.000 | 0.000 |

At matched perception + LLM scale, KG tool calling gives **+36% SPL**
(relative) over Co-NavGPT's text-LLM frontier assignment.

## Contents

| Path               | What it is |
|--------------------|------------|
| `paper/`           | LaTeX source + references.bib + sections |
| `code/src/`        | KG construction + MindNav brain modules |
| `code/configs/`    | Launch configs for MindNav and each baseline |
| `code/scripts/`    | Batch runners and result aggregators |
| `baselines/`       | Upstream repos + how we ran them |
| `data/`            | HM3D setup (no data committed) |
| `results/`         | Raw CSVs, episode logs, plotting scripts |
| `docs/`            | `INSTALL.md`, `RUN_EXPERIMENTS.md` |

## Quickstart

Create the local API configuration (the real `.env` is git-ignored):

```bash
cp .env.example .env
# Set DEEPSEEK_API_KEY in .env
```

```bash
pip install -r code/requirements.txt

# Run MindNav on val_mini with the DeepSeek API brain
bash code/scripts/run_mindnav.sh
```

MindNav's per-episode JSONL preserves Habitat's `success`/`spl` fields and
also records `mcoconav_success`, `mcoconav_spl`, the original order-dependent
MCoCoNav deciding robot, and per-robot detection/path audits. The MCoCoNav
tracker is read-only and does not feed Habitat ground-truth semantics back into
navigation.

## Citation

```bibtex
@inproceedings{mindnav2026,
  title     = {MindNav: Knowledge-Graph Tool Calling for Training-Free Multi-Robot Object Navigation},
  author    = {Anonymous},
  booktitle = {EMNLP},
  year      = {2026}
}
```

## License

Code: MIT (see `LICENSE`). Paper: CC-BY-4.0.
