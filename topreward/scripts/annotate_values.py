"""Compute MC value estimates from TOPReward predictions.

Walks a results directory where each subdirectory is a dataset containing
a *_predictions.jsonl. Writes value_annotations.json into a parallel
output directory.

V(t) = r(t) + γ * r(t+1) + γ² * r(t+2) + ...

where r(t) = log P("True" | frames_0:t, instruction)

When rewards are computed at a stride (e.g. every 10 frames), use --impute
to fill in intermediate timesteps before computing values:
  zero:        r=0 for in-between steps
  interpolate: linearly interpolate rewards between stride points
  duplicate:   repeat each stride reward for the next N steps

Usage:
    python -m topreward.scripts.annotate_values \
        --results-dir results_value \
        --output-dir datasets \
        --gamma 0.99 \
        --impute duplicate
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger


def impute_rewards(
    prefix_rewards: list[float],
    original_frames_indices: list[int],
    total_frames: int,
    method: str,
) -> list[float]:
    """Expand strided rewards to per-frame rewards.

    Args:
        prefix_rewards: Rewards at stride points.
        original_frames_indices: Frame indices where rewards were computed.
        total_frames: Total number of frames in the episode.
        method: 'zero', 'interpolate', or 'duplicate'.

    Returns:
        Rewards for every frame from 0 to total_frames-1.
    """
    rewards = np.zeros(total_frames, dtype=np.float64)
    indices = original_frames_indices

    if method == "zero":
        for idx, r in zip(indices, prefix_rewards):
            rewards[idx] = r

    elif method == "interpolate":
        # Fill before first stride point
        rewards[:indices[0] + 1] = prefix_rewards[0]
        # Interpolate between stride points
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            rewards[start:end + 1] = np.linspace(
                prefix_rewards[i], prefix_rewards[i + 1], end - start + 1
            )
        # Fill after last stride point
        rewards[indices[-1]:] = prefix_rewards[-1]

    elif method == "duplicate":
        # Each reward fills forward from its index to the next stride point
        for i, (idx, r) in enumerate(zip(indices, prefix_rewards)):
            end = indices[i + 1] if i + 1 < len(indices) else total_frames
            rewards[idx:end] = r
        # Fill before first stride point with first reward
        rewards[:indices[0]] = prefix_rewards[0]

    else:
        raise ValueError(f"Unknown impute method: {method}")

    return rewards.tolist()


def compute_values(rewards: list[float], gamma: float) -> list[float]:
    """Discounted sum of future rewards, computed via backward pass."""
    values = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        values[t] = running
    return values.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MC values from TOPReward predictions.")
    parser.add_argument("--results-dir", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--gamma", "-g", default=0.99, type=float)
    parser.add_argument(
        "--impute", default=None, choices=["zero", "interpolate", "duplicate"],
        help="Impute rewards for intermediate frames between stride points.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.is_dir():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    dataset_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and list(d.glob("*predictions.jsonl"))
    )

    if not dataset_dirs:
        logger.error(f"No datasets with *predictions.jsonl found in {results_dir}")
        sys.exit(1)

    logger.info(f"Found {len(dataset_dirs)} dataset(s)")

    for ds_dir in dataset_dirs:
        ds_name = ds_dir.name
        pred_file = sorted(ds_dir.glob("*predictions.jsonl"))[-1]

        episodes = []
        skipped = 0
        with pred_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                prefix_rewards = record.get("prefix_rewards")
                original_frames_indices = record.get("original_frames_indices")
                total_frames = record.get("num_frames")
                if prefix_rewards is None or len(prefix_rewards) < 2:
                    skipped += 1
                    continue

                ep_idx = record.get("episode_index", 0)

                if args.impute and original_frames_indices and total_frames:
                    rewards = impute_rewards(
                        prefix_rewards, original_frames_indices, total_frames, args.impute
                    )
                else:
                    rewards = prefix_rewards

                episodes.append({
                    "episode_index": ep_idx,
                    "instruction": record.get("instruction"),
                    "rewards": rewards,
                    "values": compute_values(rewards, args.gamma),
                })

        out_ds_dir = output_dir / ds_name
        out_ds_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_ds_dir / "value_annotations.json"

        with out_file.open("w", encoding="utf-8") as f:
            json.dump({"gamma": args.gamma, "impute": args.impute, "episodes": episodes}, f, indent=2)

        logger.info(f"[{ds_name}] {len(episodes)} episodes ({skipped} skipped) -> {out_file}")

    logger.success(f"Done. {len(dataset_dirs)} dataset(s) -> {output_dir}")


if __name__ == "__main__":
    main()
