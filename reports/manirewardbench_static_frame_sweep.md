# ManiRewardBench Static-Frame Sweep

## Goal

Verify the length-bias behavior reported for Qwen3-VL instruction reward scoring on
ManiRewardBench, and extend the check from the published `ai2_lerobot` split to the
full local ManiRewardBench suite:

- `ai2_lerobot`
- `ai2_franka`
- `ai2_bimanual_yam`
- `ai2_single_yam`

For each split, evaluate:

1. `Qwen/Qwen3-VL-8B-Instruct`
2. `Qwen/Qwen3-VL-32B-Instruct`

Additional one-split follow-ups were run on `ai2_franka` for:

1. `Qwen/Qwen3-VL-30B-A3B-Instruct`
2. `Qwen/Qwen3.5-35B-A3B`
3. `Qwen/Qwen3.5-9B`

using the static-frame robustness experiment:

- `real video`: original trajectory
- `static video`: first frame repeated to the same episode length

A robust reward model should keep static-video reward flat, giving static VOC and
static monotonicity near `0`.

## Setup

- Date: `2026-04-09`
- GPU: `NVIDIA H100 80GB`
- Environment: `../instruction_gvl/.venv`
- Harness: `../instruction_gvl/scripts/static_frame_experiment.py`
- Output root:
  `../instruction_gvl/results/static_frame_manirewardbench_2026-04-09_18-17-06`

The TOPReward repo remained separate. No runtime imports were added from
`../instruction_gvl` into TOPReward. The sibling repo was used only as the existing
benchmark harness and local dataset entrypoint for ManiRewardBench.

### Addendum on `ai2_franka`

The original `ai2_franka` rows in this report mixed multiple runs and stale task
wording. They have been corrected below using:

- the original 8B static-frame sweep output for `ai2_franka`
- a reannotated 32B `ai2_franka` run where `Clean the table.` is canonicalized to
  `clean the table by placing the can and spoon in the plate`
- a high-FPS reannotated 32B `ai2_franka` run with aligned Qwen sampling raised from
  `2.0 FPS` to `10.0 FPS`

The reannotation fixes the real-video semantic mismatch for `ai2_franka`. Raising the
aligned target FPS then removes the short-video static-frame bias.

## Metrics

- `Real VOC`: Spearman correlation between normalized prefix rewards and linear progress on the real trajectory
- `Static VOC`: same correlation on the repeated-first-frame trajectory
- `Static monotonicity`: Spearman correlation between prefix index and static prefix reward

Interpretation:

- high `Real VOC` is good
- `Static VOC` near `0` is good
- `Static monotonicity` near `0` is good
- large positive `Static VOC` / `Static monotonicity` indicates length bias

## Results

| Model | Dataset | Real VOC | Static VOC | Real Reward | Static Reward | Static Monotonicity |
|---|---|---:|---:|---:|---:|---:|
| 8B | `ai2_lerobot` | 0.9503 | 0.9870 | -0.0988 | -9.2375 | 0.9870 |
| 32B | `ai2_lerobot` | 0.8944 | 0.0506 | -0.0192 | -6.0500 | 0.0506 |
| 30B-A3B | `ai2_lerobot` | 0.9516 | 0.8543 | -0.6686 | -8.5125 | 0.8543 |
| 3.5-35B-A3B | `ai2_lerobot` | 0.4417 | 0.1515 | -6.1750 | -7.0500 | 0.1515 |
| 3.5-9B | `ai2_lerobot` | -0.6647 | -0.9111 | -2.6313 | -2.5625 | -0.9111 |
| 8B | `ai2_franka` | 0.9618 | 0.9912 | -3.9266 | -6.7812 | 0.9912 |
| 32B | `ai2_franka` (reannotated, 10 FPS) | 0.5954 | -0.5701 | -0.4109 | -4.2562 | -0.5701 |
| 30B-A3B | `ai2_franka` (reannotated, 10 FPS) | 0.9560 | 0.9206 | -1.2887 | -7.2781 | 0.9206 |
| 3.5-35B-A3B | `ai2_franka` (reannotated, 10 FPS) | 0.1054 | 0.1952 | -6.9250 | -6.4250 | 0.1952 |
| 3.5-9B | `ai2_franka` (reannotated, 10 FPS) | -0.1501 | -0.8042 | -2.9281 | -3.8875 | -0.8042 |
| 8B | `ai2_bimanual_yam` | 0.8318 | 0.4901 | -0.3541 | -1.2773 | 0.4901 |
| 32B | `ai2_bimanual_yam` | 0.7115 | -0.7516 | -0.2982 | -5.5000 | -0.7516 |
| 30B-A3B | `ai2_bimanual_yam` | 0.9459 | 0.9439 | -0.8402 | -2.3969 | 0.9439 |
| 3.5-35B-A3B | `ai2_bimanual_yam` | 0.2832 | 0.7601 | -5.6750 | -5.1250 | 0.7601 |
| 3.5-9B | `ai2_bimanual_yam` | -0.8785 | -0.9681 | -2.4312 | -2.8438 | -0.9681 |
| 8B | `ai2_single_yam` | 0.9456 | 0.9845 | -3.9219 | -5.1063 | 0.9845 |
| 32B | `ai2_single_yam` | 0.9093 | -0.4949 | -0.1040 | -5.5750 | -0.4949 |
| 30B-A3B | `ai2_single_yam` | 0.9659 | 0.9750 | -3.0562 | -3.4891 | 0.9750 |
| 3.5-35B-A3B | `ai2_single_yam` | 0.2910 | 0.7223 | -6.5000 | -5.8000 | 0.7223 |
| 3.5-9B | `ai2_single_yam` | -0.7906 | -0.8996 | -2.5719 | -2.7344 | -0.8996 |

## Main Findings

### 1. The published `ai2_lerobot` conclusion reproduces

The key reported result holds:

- 8B is strongly length-biased on `ai2_lerobot`
- 32B largely removes that bias on `ai2_lerobot`

The clearest signal is the static trajectory:

- 8B: `static_voc = 0.9870`
- 32B: `static_voc = 0.0506`

So the repeated-first-frame video still looks strongly progressive to 8B, but not to
32B.

### 2. The 8B bias generalizes to most ManiRewardBench splits

The 8B model shows severe static monotonicity on:

- `ai2_lerobot`: `0.9870`
- `ai2_franka`: `0.9912`
- `ai2_single_yam`: `0.9845`

`ai2_bimanual_yam` is milder (`0.4901`), but still positively biased.

### 3. The 32B model is much more robust overall, but not uniformly perfect

The 32B model strongly suppresses or reverses static monotonicity on:

- `ai2_lerobot`: `0.0506`
- `ai2_bimanual_yam`: `-0.7516`
- `ai2_single_yam`: `-0.4949`

For `ai2_franka`, two separate effects need to be separated:

1. **Semantic mismatch in the real-video instruction.** The original label
   `Clean the table.` causes Qwen3-VL 32B to prefer `False` on otherwise successful
   videos. Reannotating that task to
   `clean the table by placing the can and spoon in the plate` fixes the real-video
   reward, raising the 32B `real_reward_mean` on `ai2_franka` to `-0.2375` in the
   reannotated `2.0 FPS` run.
2. **Static-frame length bias.** Reannotation alone does not fix the static bias:

- `static_voc = 0.9818`
- `static_monotonicity = 0.9818`

3. **Aligned target FPS materially changes the static result.** In the reannotated
   `10.0 FPS` run, `ai2_franka` becomes non-monotonic on static inputs while keeping
   real-video reward in the same broad range:

- `real_reward_mean = -0.4109`
- `static_voc = -0.5701`
- `static_monotonicity = -0.5701`

Before reannotation, the default `2.0 FPS` `ai2_franka` run already showed residual bias:

- `static_voc = 0.6431`
- `static_monotonicity = 0.6431`

That default-wording run was materially better than 8B on the same split, but not
perfectly robust.

***Update on ai2_franka anomaly:*** Further investigation reveals the "residual bias" in `ai2_franka` is an artifact of video lengths and Qwen3-VL 32B's *image-count heuristic*. The 32B model still artificially increases rewards when the input grows from 1 to ~8 images, regardless of visual content. Because `ai2_lerobot` videos are long (e.g., 2000 frames @ 30 FPS) and are uniformly downsampled to 2.0 FPS, its very first evaluation prefix covers 133 frames (~9 images). This instantly saturates the model's heuristic, resulting in a flat (unbiased) evaluation trace. In contrast, `ai2_franka` videos are very short and use a lower native framerate (e.g., 237 frames @ 15 FPS). At 2.0 FPS downsampling, its early prefixes only expose 1 to 5 images. This catches the 32B model during its 1-to-8 image "hallucinated progress" ramp, causing the static trace to climb and producing the unexpectedly high VOC bias.

Consequently, the *target sample FPS* acts as a direct modifier on this residual bias effect:
- **Higher target FPS:** *Decreases* the bias. Packing more sampled images into the earliest prefixes instantly saturates the ~8 image threshold. In the reannotated `ai2_franka` run, raising the aligned Qwen sample FPS from `2.0` to `10.0` changed `static_voc` from `0.9818` to `-0.5701`.
- **Lower target FPS:** *Increases* the bias. The 1-to-8 image accumulation is stretched across the entire episode's prefixes. This turns the scores into a strictly monotonic climbing curve, driving VOC toward `1.0`.

### 4. Reward magnitude separation also improves at 32B on the key split

On `ai2_lerobot`:

- 8B real reward mean: `-0.0988`
- 8B static reward mean: `-9.2375`
- 32B real reward mean: `-0.0192`
- 32B static reward mean: `-6.0500`

The more important distinction is not just magnitude but trajectory shape:

- 8B static prefixes climb monotonically with length
- 32B static prefixes stay nearly flat

## Takeaway

For the specific claim that needed verification, the answer is yes:

- `Qwen3-VL-8B-Instruct` shows strong length bias on `ai2_lerobot`
- `Qwen3-VL-32B-Instruct` largely eliminates that failure mode on `ai2_lerobot`

For the broader ManiRewardBench sweep:

- 8B is biased on most splits tested
- 32B is substantially better overall
- `ai2_franka` combines two failure modes: a label-semantic mismatch in the original
  task wording and a separate short-video length bias in the static-frame setting

### 5. Qwen3.5-family behavior on all splits (5-episode runs)

- `Qwen/Qwen3-VL-30B-A3B-Instruct` retains strong length bias on `ai2_lerobot`, `ai2_franka` (reannotated), and `ai2_single_yam`; results are mixed on `ai2_bimanual_yam`.
- `Qwen/Qwen3.5-35B-A3B` is broadly similar: positive static monotonicity on most splits, weak real-video correlation on some splits.
- `Qwen/Qwen3.5-9B` is negative/flat on static monotonicity for all tested splits, but its real-video signal is poor and unsuitable as a reward model in this configuration.

`ai2_franka` for the 30B-A3B and 3.5-35B-A3B metrics above used reannotated task names.

## Artifacts

Per-split summaries:

- `../instruction_gvl/results/static_frame_manirewardbench_2026-04-09_18-17-06/qwen/ai2_lerobot/summary.json`
- `../instruction_gvl/results/static_frame_manirewardbench_2026-04-09_18-17-06/qwen/ai2_franka/summary.json`
- `../instruction_gvl/results/static_frame_manirewardbench_2026-04-09_18-17-06/qwen/ai2_bimanual_yam/summary.json`
- `../instruction_gvl/results/static_frame_manirewardbench_2026-04-09_18-17-06/qwen/ai2_single_yam/summary.json`
- `../instruction_gvl/results/static_frame_manirewardbench_2026-04-09_18-17-06/qwen32b/ai2_lerobot/summary.json`
- `../instruction_gvl/results/static_frame_manirewardbench_2026-04-09_18-17-06/qwen32b/ai2_franka/summary.json`
- `../instruction_gvl/results/static_frame_manirewardbench_2026-04-09_18-17-06/qwen32b/ai2_bimanual_yam/summary.json`
- `../instruction_gvl/results/static_frame_manirewardbench_2026-04-09_18-17-06/qwen32b/ai2_single_yam/summary.json`
- `../instruction_gvl/results/static_frame_ai2_franka_robot_prompt/qwen32b/ai2_franka/summary.json`
- `../instruction_gvl/results/static_frame_ai2_franka_reannotated/qwen32b/ai2_franka/summary.json`
- `../instruction_gvl/results/static_frame_ai2_franka_reannotated_10fps/qwen32b/ai2_franka/summary.json`
- `../instruction_gvl/results/static_frame_ai2_franka_reannotated_new_10fps/qwen30b_a3b/ai2_franka/summary.json`
- `../instruction_gvl/results/static_frame_ai2_franka_reannotated_new_10fps/qwen35_35b_a3b/ai2_franka/summary.json`
- `../instruction_gvl/results/static_frame_ai2_franka_reannotated_new_10fps/qwen35_9b/ai2_franka/summary.json`
- `../instruction_gvl/results/static_frame_ai2_lerobot_new/qwen30b_a3b/ai2_lerobot/summary.json`
- `../instruction_gvl/results/static_frame_ai2_lerobot_new/qwen35_35b_a3b/ai2_lerobot/summary.json`
- `../instruction_gvl/results/static_frame_ai2_lerobot_new/qwen35_9b/ai2_lerobot/summary.json`
- `../instruction_gvl/results/static_frame_ai2_bimanual_yam_new/qwen30b_a3b/ai2_bimanual_yam/summary.json`
- `../instruction_gvl/results/static_frame_ai2_bimanual_yam_new/qwen35_35b_a3b/ai2_bimanual_yam/summary.json`
- `../instruction_gvl/results/static_frame_ai2_bimanual_yam_new/qwen35_9b/ai2_bimanual_yam/summary.json`
- `../instruction_gvl/results/static_frame_ai2_single_yam_new/qwen30b_a3b/ai2_single_yam/summary.json`
- `../instruction_gvl/results/static_frame_ai2_single_yam_new/qwen35_35b_a3b/ai2_single_yam/summary.json`
- `../instruction_gvl/results/static_frame_ai2_single_yam_new/qwen35_9b/ai2_single_yam/summary.json`

Raw per-episode outputs are stored alongside each summary as `predictions.jsonl`.
