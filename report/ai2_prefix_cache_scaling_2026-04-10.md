# Prefix Cache Scaling

## Setup

- Dataset: `ai2_lerobot`
- Episode index: `14`
- Instruction: `push the can to the middle and put the cube on it`
- Raw video length: `984` frames
- FPS: `30.0`
- Model: `Qwen/Qwen3-VL-8B-Instruct`
- Prefix samples per run: `15`
- Repeats per point: `2`

This uses one real long ManiRewardBench-style AI2 video and measures end-to-end `compute_instruction_rewards_for_prefixes(...)` runtime with:

- `TOPREWARD_QWEN_PREFIX_CACHE=0`
- `TOPREWARD_QWEN_PREFIX_CACHE=1`

for increasing video prefix lengths.

## Scaling Results

![Prefix cache speedup vs video length](/weka/oe-training-default/shiruic/inst_reward/TOPReward/report/ai2_prefix_cache_scaling_2026-04-10.png)

| Raw frames | Uncached mean | Cached mean | Speedup | Latency reduction |
|---|---:|---:|---:|---:|
| `64` | `2.34s` | `1.13s` | `2.06x` | `51.6%` |
| `128` | `3.81s` | `1.26s` | `3.03x` | `66.9%` |
| `256` | `6.79s` | `1.66s` | `4.08x` | `75.5%` |
| `384` | `9.87s` | `1.93s` | `5.10x` | `80.4%` |
| `512` | `12.79s` | `2.28s` | `5.61x` | `82.2%` |
| `768` | `19.08s` | `3.20s` | `5.96x` | `83.2%` |
| `984` | `26.03s` | `3.82s` | `6.82x` | `85.3%` |

## Takeaway

The speedup scales strongly with video length on this real long clip.

- At `64` frames the cache is already worth `2.06x`.
- By `256` frames it reaches `4.08x`.
- At `984` frames it reaches `6.82x`.

So your intuition was correct: longer videos make the prefix-cache benefit materially larger.

## VOC Comparison

The cache mostly preserves VOC on this long video. The main visible drift happens on shorter prefixes; in the long-video regime, VOC is effectively unchanged.

| Raw frames | Uncached VOC | Cached VOC | VOC delta |
|---|---:|---:|---:|
| `64` | `0.9382` | `0.9159` | `-0.0223` |
| `128` | `0.9955` | `0.9946` | `-0.0009` |
| `256` | `1.0000` | `0.9893` | `-0.0107` |
| `384` | `0.9964` | `0.9964` | `0.0000` |
| `512` | `0.9929` | `0.9929` | `0.0000` |
| `768` | `0.9714` | `0.9714` | `0.0000` |
| `984` | `0.9669` | `0.9669` | `0.0000` |

Interpretation:

- The speedup grows strongly with length.
- VOC drift is small overall and disappears on the longest prefixes in this experiment.
- At the full `984`-frame clip, the cache gave `6.82x` speedup with identical measured VOC (`0.9669` vs `0.9669`).

## Artifacts

- Main plot: [report/ai2_prefix_cache_scaling_2026-04-10.png](/weka/oe-training-default/shiruic/inst_reward/TOPReward/report/ai2_prefix_cache_scaling_2026-04-10.png)
- Combined plot: [report/ai2_prefix_cache_scaling_with_voc_2026-04-10.png](/weka/oe-training-default/shiruic/inst_reward/TOPReward/report/ai2_prefix_cache_scaling_with_voc_2026-04-10.png)
- Raw data: [report/ai2_prefix_cache_scaling_2026-04-10.json](/weka/oe-training-default/shiruic/inst_reward/TOPReward/report/ai2_prefix_cache_scaling_2026-04-10.json)
- VOC data: [report/ai2_prefix_cache_voc_2026-04-10.json](/weka/oe-training-default/shiruic/inst_reward/TOPReward/report/ai2_prefix_cache_voc_2026-04-10.json)
