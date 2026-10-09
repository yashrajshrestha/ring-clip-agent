"""CLI: score one clip or a folder, print JSON lines, and group duplicates.

    python -m vision.cli clips/ > facts.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .checks import analyze, is_duplicate

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi"}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    target = Path(argv[1])
    paths = sorted(p for p in target.rglob("*") if p.suffix.lower() in VIDEO_EXT) if target.is_dir() else [target]

    facts = [analyze(p) for p in paths]

    # Greedy duplicate grouping: first clip of a group is the keeper.
    group_of: dict[str, int] = {}
    for i, a in enumerate(facts):
        if a.path in group_of:
            continue
        group_of[a.path] = i
        for b in facts[i + 1:]:
            if b.path not in group_of and is_duplicate(a.hashes, b.hashes):
                group_of[b.path] = i

    for f in facts:
        d = f.to_dict()
        d["dup_group"] = group_of[f.path]
        print(json.dumps(d))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
