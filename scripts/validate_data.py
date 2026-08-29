#!/usr/bin/env python3
"""Validate the structural integrity of the offline and online data under data/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from offline.data import OFFLINE_DATA_ROOT, OFFLINE_SAMPLES_FILE, load_reference, load_samples  # noqa: E402

ONLINE_DATA_ROOT = ROOT / "data" / "online_data"
ONLINE_SAMPLES_FILE = ROOT / "data" / "online_samples.jsonl"


def _check_image(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing image: {path}")
        return
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as error:  # noqa: BLE001
        errors.append(f"corrupt image: {path}: {error}")


def validate_offline(errors: list[str]) -> None:
    samples = load_samples(OFFLINE_SAMPLES_FILE)
    if len(samples) != 500:
        errors.append(f"offline_samples.jsonl should have 500 samples, found {len(samples)}")
    dirs = {p.name for p in OFFLINE_DATA_ROOT.iterdir() if p.is_dir()}
    if dirs != set(samples):
        errors.append(
            f"offline_data does not match offline_samples.jsonl: "
            f"extra {sorted(dirs - set(samples))[:3]}, missing {sorted(set(samples) - dirs)[:3]}"
        )
        return
    total_steps = 0
    for sample_id, sample in samples.items():
        sample_dir = OFFLINE_DATA_ROOT / sample_id
        try:
            reference = load_reference(sample_dir)
        except Exception as error:  # noqa: BLE001
            errors.append(f"{sample_id}: cannot parse the reference trajectory: {error}")
            continue
        if sample.get("n_steps") != len(reference):
            errors.append(f"{sample_id}: n_steps={sample.get('n_steps')} does not match {len(reference)} trajectory lines")
        if [row.get("step_id") for row in reference] != list(range(len(reference))):
            errors.append(f"{sample_id}: step_id is not consecutive")
        _check_image(sample_dir / "initial.png", errors)
        for step, row in enumerate(reference):
            if not isinstance(row.get("semantic_action"), dict) or not row["semantic_action"].get("type"):
                errors.append(f"{sample_id} step {step}: missing semantic_action")
            _check_image(sample_dir / f"step_{step:03d}_after.png", errors)
        total_steps += len(reference)
    print(f"offline: {len(samples)} samples, {total_steps} transitions")


def validate_online(errors: list[str]) -> None:
    try:
        rows = [
            json.loads(line)
            for line in ONLINE_SAMPLES_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"cannot parse online_samples.jsonl: {error}")
        return
    task_ids = [row.get("task_id") for row in rows]
    if len(rows) != 200:
        errors.append(f"online_samples.jsonl should have 200 tasks, found {len(rows)}")
    if len(set(task_ids)) != len(task_ids):
        errors.append("duplicate task_id in online_samples.jsonl")
    dirs = {p.name for p in ONLINE_DATA_ROOT.iterdir() if p.is_dir()}
    if dirs != set(task_ids):
        errors.append(
            f"online_data does not match online_samples.jsonl: "
            f"extra {sorted(dirs - set(task_ids))[:3]}, missing {sorted(set(task_ids) - dirs)[:3]}"
        )
    for task_id in task_ids:
        _check_image(ONLINE_DATA_ROOT / task_id / "initial.png", errors)
    print(f"online: {len(rows)} tasks")


def main() -> int:
    errors: list[str] = []
    validate_offline(errors)
    validate_online(errors)
    if errors:
        print(f"\n{len(errors)} problem(s):")
        for error in errors[:50]:
            print(f"  - {error}")
        return 1
    print("data validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
