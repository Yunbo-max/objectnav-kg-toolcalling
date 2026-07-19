#!/usr/bin/env python3
"""Build a reproducible HM3D ObjectNav validation split.

The default construction keeps every episode in ``val_mini`` and adds four
previously unseen validation scenes with 15 episodes per scene.  Category
counts are matched to the integer allocation closest to the full ``val``
distribution.  Episodes within each scene/category are selected at evenly
spaced geodesic-distance quantiles to avoid taking only easy or hard cases.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


JsonDict = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    code_root = Path(__file__).resolve().parents[1]
    default_dataset_root = (
        code_root / "vendor/conavgpt/data/datasets/objectnav_hm3d_v2"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--base-split", default="val_mini")
    parser.add_argument("--source-split", default="val")
    parser.add_argument("--output-split", default="val_balanced_90")
    parser.add_argument("--new-scenes", type=int, default=4)
    parser.add_argument("--episodes-per-new-scene", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace only the named output split."
    )
    return parser.parse_args()


def read_gzip_json(path: Path) -> JsonDict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_gzip_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="", fileobj=raw_handle, mode="wb", mtime=0
        ) as gzip_handle:
            gzip_handle.write(payload)


def content_files(dataset_root: Path, split: str) -> List[Path]:
    return sorted((dataset_root / split / "content").glob("*.json.gz"))


def scene_name(path: Path) -> str:
    suffix = ".json.gz"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def count_categories(
    episodes: Iterable[Mapping[str, Any]], categories: Sequence[str]
) -> Tuple[int, ...]:
    counts = Counter(ep["object_category"] for ep in episodes)
    return tuple(counts[category] for category in categories)


def largest_remainder_target(
    source_counts: Sequence[int], target_total: int
) -> Tuple[int, ...]:
    source_total = sum(source_counts)
    raw = [target_total * count / source_total for count in source_counts]
    target = [math.floor(value) for value in raw]
    order = sorted(
        range(len(raw)), key=lambda index: (-(raw[index] - target[index]), index)
    )
    for index in order[: target_total - sum(target)]:
        target[index] += 1
    return tuple(target)


def generate_allocations(
    availability: Sequence[int], remaining: Sequence[int], total: int
) -> List[Tuple[int, ...]]:
    limits = [min(have, need) for have, need in zip(availability, remaining)]
    allocations: List[Tuple[int, ...]] = []

    def visit(index: int, left: int, prefix: List[int]) -> None:
        if index == len(limits) - 1:
            if 0 <= left <= limits[index]:
                allocations.append(tuple(prefix + [left]))
            return
        minimum = max(0, left - sum(limits[index + 1 :]))
        maximum = min(limits[index], left)
        for value in range(minimum, maximum + 1):
            visit(index + 1, left - value, prefix + [value])

    visit(0, total, [])
    return allocations


def choose_scenes_and_allocations(
    candidates: Sequence[Tuple[Path, JsonDict]],
    categories: Sequence[str],
    required: Sequence[int],
    scene_count: int,
    episodes_per_scene: int,
    seed: int,
) -> List[Tuple[Path, JsonDict, Tuple[int, ...]]]:
    ordered = list(candidates)
    random.Random(seed).shuffle(ordered)
    availability = [
        count_categories(dataset["episodes"], categories) for _, dataset in ordered
    ]

    def enough_capacity(start: int, slots: int, remaining: Sequence[int]) -> bool:
        for category_index, need in enumerate(remaining):
            possible = sorted(
                (counts[category_index] for counts in availability[start:]),
                reverse=True,
            )
            if sum(possible[:slots]) < need:
                return False
        return True

    def search(
        start: int,
        slots: int,
        remaining: Tuple[int, ...],
    ) -> List[Tuple[int, Tuple[int, ...]]] | None:
        if slots == 0:
            return [] if all(value == 0 for value in remaining) else None
        if sum(remaining) != slots * episodes_per_scene:
            return None
        if len(ordered) - start < slots or not enough_capacity(start, slots, remaining):
            return None

        for candidate_index in range(start, len(ordered) - slots + 1):
            allocations = generate_allocations(
                availability[candidate_index], remaining, episodes_per_scene
            )
            ideal = [value / slots for value in remaining]
            allocations.sort(
                key=lambda allocation: (
                    sum(
                        (allocation[index] - ideal[index]) ** 2
                        for index in range(len(categories))
                    ),
                    allocation,
                )
            )
            for allocation in allocations:
                next_remaining = tuple(
                    need - selected for need, selected in zip(remaining, allocation)
                )
                result = search(candidate_index + 1, slots - 1, next_remaining)
                if result is not None:
                    return [(candidate_index, allocation)] + result
        return None

    solution = search(0, scene_count, tuple(required))
    if solution is None:
        raise RuntimeError(
            "No exact scene/category allocation exists for the requested construction."
        )
    return [(*ordered[index], allocation) for index, allocation in solution]


def quantile_sample(
    episodes: Sequence[JsonDict],
    categories: Sequence[str],
    allocation: Sequence[int],
) -> List[JsonDict]:
    original_index = {id(episode): index for index, episode in enumerate(episodes)}
    selected: List[JsonDict] = []
    for category, count in zip(categories, allocation):
        candidates = sorted(
            (ep for ep in episodes if ep["object_category"] == category),
            key=lambda ep: (
                ep.get("info", {}).get("geodesic_distance", math.inf),
                str(ep.get("episode_id", "")),
            ),
        )
        if count > len(candidates):
            raise ValueError(
                f"Requested {count} {category} episodes from {len(candidates)}"
            )
        positions = (
            [
                ((2 * index + 1) * len(candidates)) // (2 * count)
                for index in range(count)
            ]
            if count
            else []
        )
        selected.extend(candidates[position] for position in positions)
    return sorted(selected, key=lambda episode: original_index[id(episode)])


def goals_for_episodes(
    goals_by_category: Mapping[str, Any], episodes: Sequence[Mapping[str, Any]]
) -> JsonDict:
    required_keys = {
        f"{Path(ep['scene_id']).name}_{ep['object_category']}" for ep in episodes
    }
    missing = required_keys.difference(goals_by_category)
    if missing:
        raise KeyError(f"Missing goals_by_category entries: {sorted(missing)}")
    return {key: goals_by_category[key] for key in sorted(required_keys)}


def category_dict(categories: Sequence[str], counts: Sequence[int]) -> Dict[str, int]:
    return dict(zip(categories, counts))


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = dataset_root / args.output_split
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}")
        if output_dir.resolve().parent != dataset_root:
            raise ValueError("Refusing to replace an output outside dataset_root")
        shutil.rmtree(output_dir)

    source_index = read_gzip_json(
        dataset_root / args.source_split / f"{args.source_split}.json.gz"
    )
    categories = tuple(source_index["category_to_task_category_id"].keys())

    source_data = [
        (path, read_gzip_json(path))
        for path in content_files(dataset_root, args.source_split)
    ]
    base_data = [
        (path, read_gzip_json(path))
        for path in content_files(dataset_root, args.base_split)
    ]
    source_episodes = [
        episode for _, dataset in source_data for episode in dataset["episodes"]
    ]
    base_episodes = [
        episode for _, dataset in base_data for episode in dataset["episodes"]
    ]
    target_total = len(base_episodes) + args.new_scenes * args.episodes_per_new_scene
    source_counts = count_categories(source_episodes, categories)
    base_counts = count_categories(base_episodes, categories)
    target_counts = largest_remainder_target(source_counts, target_total)
    required_counts = tuple(
        target - base for target, base in zip(target_counts, base_counts)
    )
    if any(count < 0 for count in required_counts):
        raise ValueError(
            "The fixed base split already exceeds the proportional target for a "
            "category."
        )

    base_scene_names = {scene_name(path) for path, _ in base_data}
    candidates = [
        (path, dataset)
        for path, dataset in source_data
        if scene_name(path) not in base_scene_names
    ]
    chosen = choose_scenes_and_allocations(
        candidates,
        categories,
        required_counts,
        args.new_scenes,
        args.episodes_per_new_scene,
        args.seed,
    )

    output_content = output_dir / "content"
    output_content.mkdir(parents=True)
    manifest_scenes: List[JsonDict] = []
    output_episodes: List[JsonDict] = []

    for path, dataset in base_data:
        episodes = dataset["episodes"]
        output_dataset = dict(dataset)
        output_dataset["goals_by_category"] = goals_for_episodes(
            dataset["goals_by_category"], episodes
        )
        write_gzip_json(output_content / path.name, output_dataset)
        counts = count_categories(episodes, categories)
        output_episodes.extend(episodes)
        manifest_scenes.append(
            {
                "scene": scene_name(path),
                "source_split": args.base_split,
                "episodes": len(episodes),
                "category_counts": category_dict(categories, counts),
            }
        )

    for path, dataset, allocation in chosen:
        episodes = quantile_sample(dataset["episodes"], categories, allocation)
        output_dataset = dict(dataset)
        output_dataset["episodes"] = episodes
        output_dataset["goals_by_category"] = goals_for_episodes(
            dataset["goals_by_category"], episodes
        )
        write_gzip_json(output_content / path.name, output_dataset)
        output_episodes.extend(episodes)
        manifest_scenes.append(
            {
                "scene": scene_name(path),
                "source_split": args.source_split,
                "episodes": len(episodes),
                "category_counts": category_dict(categories, allocation),
            }
        )

    output_index = dict(source_index)
    output_index["episodes"] = []
    write_gzip_json(output_dir / f"{args.output_split}.json.gz", output_index)

    actual_counts = count_categories(output_episodes, categories)
    manifest = {
        "split": args.output_split,
        "construction": {
            "base_split": args.base_split,
            "source_split": args.source_split,
            "new_scenes": args.new_scenes,
            "episodes_per_new_scene": args.episodes_per_new_scene,
            "seed": args.seed,
            "episode_sampling": (
                "even_geodesic_distance_quantiles_within_scene_and_category"
            ),
            "category_target": "largest_remainder_from_full_source_split",
        },
        "episodes": len(output_episodes),
        "scenes": len(manifest_scenes),
        "source_category_counts": category_dict(categories, source_counts),
        "target_category_counts": category_dict(categories, target_counts),
        "actual_category_counts": category_dict(categories, actual_counts),
        "scene_selection": manifest_scenes,
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if actual_counts != target_counts:
        raise AssertionError(f"Actual counts {actual_counts} != target {target_counts}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
