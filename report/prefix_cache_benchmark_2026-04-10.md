# Prefix Cache Benchmark

## Summary

Measured the speed gain from Qwen prefix KV-cache reuse using the repo's existing benchmark harness:

- Script: `scripts/benchmark_cache.py`
- Command: `uv run python scripts/benchmark_cache.py`
- Timestamp: `2026-04-10 00:35:49 UTC`
- Python: `3.11.14`
- Torch: `2.8.0+cu128`
- GPUs visible: `4x NVIDIA H100 80GB HBM3`
- Model: `Qwen/Qwen3-VL-8B-Instruct`

## Benchmark Setup

The benchmark script:

- initializes `QwenClient(model_name="Qwen/Qwen3-VL-8B-Instruct")`
- generates `45` random `224x224` RGB frames
- evaluates instruction reward prefixes for `15` evenly spaced prefix lengths
- runs once with `TOPREWARD_QWEN_PREFIX_CACHE=0`
- runs once with `TOPREWARD_QWEN_PREFIX_CACHE=1`

Instruction used by the script:

- `Pick up the block.`

## Results

| Metric | Value |
|---|---:|
| Prefix count computed | `15` |
| Uncached time | `3.43s` |
| Cached time | `2.50s` |
| Speedup | `1.37x` |
| Max absolute reward difference | `1.000000` |

## Interpretation

On this run, enabling prefix KV-cache reuse reduced end-to-end benchmark time from `3.43s` to `2.50s`, which is a `1.37x` speedup.

The first invocation also downloaded the model weights into the local Hugging Face cache before execution. That download time is not included in the benchmark numbers above; the reported times are the script's internal uncached vs cached timing measurements.
