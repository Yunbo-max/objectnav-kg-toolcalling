#!/usr/bin/env python3

import sys
from pathlib import Path


CONAVGPT_DIR = Path(__file__).resolve().parents[1] / "vendor" / "conavgpt"
sys.path.insert(0, str(CONAVGPT_DIR))

from arguments import get_args  # noqa: E402


def main():
    args = get_args()
    print(args.dataset)
    print(args.num_sem_categories)
    print(args.target_categories)
    print(args.semantic_categories)
    print(args.category_to_channel)

    assert args.num_sem_categories == len(args.semantic_categories)
    assert "toilet" in args.category_to_channel
    assert "sink" in args.category_to_channel


if __name__ == "__main__":
    main()
