"""Synchronize derived datasets to the MIMIC-III 11-class type taxonomy.

Only the top-level ``class`` field is synchronized.  Dataset-specific
normalization statistics and the finer ``sub_class`` field are preserved.
Every target variable must have an exact same-name entry in the MIMIC-III
metadata; the script deliberately refuses fuzzy or partial mappings.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "mimic3" / "var_mean_std.json"
DEFAULT_TARGETS = (
    PROJECT_ROOT / "data" / "mimic-iv" / "var_mean_std.json",
    PROJECT_ROOT / "data" / "mimic-iv" / "var_mean_std_mimic4.json",
    PROJECT_ROOT / "data" / "eicu" / "var_mean_std.json",
)
EXPECTED_CLASS_COUNT = 11


def _load_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Semantic metadata must be a non-empty object: {path}")
    return payload


def synchronize_taxonomy(source_path: Path, target_paths: Iterable[Path]) -> None:
    source = _load_metadata(source_path)
    source_classes = {item["class"] for item in source.values()}
    if len(source_classes) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLASS_COUNT} MIMIC-III classes in {source_path}, "
            f"found {len(source_classes)}: {sorted(source_classes)}"
        )

    for target_path in target_paths:
        target = _load_metadata(target_path)
        missing = sorted(set(target) - set(source))
        if missing:
            raise KeyError(
                f"{target_path} contains variables absent from the MIMIC-III "
                f"taxonomy: {missing}"
            )

        for variable, item in target.items():
            item["class"] = source[variable]["class"]

        temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
        try:
            with temporary_path.open("w", encoding="utf-8") as handle:
                json.dump(target, handle, ensure_ascii=False, indent=4)
                handle.write("\n")
            os.replace(temporary_path, target_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        target_classes = {item["class"] for item in target.values()}
        if target_classes != source_classes:
            raise ValueError(
                f"{target_path} does not cover all MIMIC-III classes after "
                f"synchronization: {sorted(target_classes)}"
            )
        print(
            f"Synchronized {target_path}: {len(target)} variables, "
            f"{len(target_classes)} classes"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the MIMIC-III 11-class taxonomy to MIMIC-IV/eICU metadata"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--targets", type=Path, nargs="+", default=list(DEFAULT_TARGETS))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    synchronize_taxonomy(arguments.source, arguments.targets)
