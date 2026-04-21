# OXE Prefix Cache Benchmark

## Summary

This benchmark uses a real OXE dataset example instead of synthetic frames.

- Dataset family: `OXE`
- Dataset: `lerobot/berkeley_mvp`
- Config alias: `berkeleymvp`
- Episode index: `0`
- Instruction: `pick fruit`
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- FPS: `5.0`
- Selected frames: `15`
- Selected frame indices: `[1, 8, 16, 23, 31, 38, 46, 53, 61, 68, 76, 83, 91, 98, 106]`

## Method

Benchmark procedure:

1. Load one real episode from `lerobot/berkeley_mvp`.
2. Use the same selected `15` trajectory frames for both conditions.
3. Warm up once on a minimal real prefix.
4. Run `compute_instruction_rewards_for_prefixes(...)` three times with `TOPREWARD_QWEN_PREFIX_CACHE=0`.
5. Run the same workload three times with `TOPREWARD_QWEN_PREFIX_CACHE=1`.

The timing below is compute time for the prefix-reward pass only. Dataset loading and model initialization are excluded from the comparison.

## Performance Comparison

| Condition | Run 1 | Run 2 | Run 3 | Mean |
|---|---:|---:|---:|---:|
| Uncached | `1.909s` | `1.750s` | `1.747s` | `1.802s` |
| Cached | `1.122s` | `1.075s` | `1.112s` | `1.103s` |

Derived comparison:

- Mean speedup: `1.63x`
- Mean latency reduction: `0.699s`
- Relative latency reduction: `38.8%`

## Output Comparison

| Metric | Uncached | Cached |
|---|---:|---:|
| Final reward | `-6.375` | `-6.250` |
| VOC | `0.9429` | `0.9302` |
| Max absolute prefix-reward difference | `1.125` | `1.125` |

Prefix lengths evaluated:

- `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`

Uncached prefix rewards:

- `[-18.75, -17.125, -11.0, -12.0, -11.625, -12.125, -10.75, -11.25, -9.5, -9.0, -9.25, -8.75, -8.0, -7.625, -6.375]`

Cached prefix rewards:

- `[-18.75, -17.125, -12.125, -12.875, -12.25, -12.5, -11.5, -12.125, -9.625, -9.75, -9.75, -9.875, -9.0, -8.625, -6.25]`

## Takeaway

On this real OXE workload, prefix KV-cache reuse improved prefix-reward throughput by about `1.63x` on average.

The cached path is faster, but it is not numerically identical to the uncached path on this example. The largest per-prefix reward delta was `1.125`, and VOC changed from `0.9429` to `0.9302`.
