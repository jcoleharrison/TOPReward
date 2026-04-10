import os
import time
import numpy as np
from topreward.clients.qwen import QwenClient
import torch

def main():
    print("Initializing QwenClient (8B) for benchmarking...")
    client = QwenClient(model_name="Qwen/Qwen3-VL-8B-Instruct")
    
    # 45 raw frames @ 2 FPS alignment -> more meaningful testing.
    frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(45)]
    instruction = "Pick up the block."
    
    print("\n[Warming up models & cache...]")
    os.environ["TOPREWARD_QWEN_PREFIX_CACHE"] = "0"
    client.compute_instruction_rewards_for_prefixes(frames[:5], instruction, num_samples=1)
    
    # We must explicitly call 15 prefixes to see the loop scaling.
    print("\n[Running Uncached...]")
    os.environ["TOPREWARD_QWEN_PREFIX_CACHE"] = "0"
    start_uncached = time.time()
    # By passing frames to prefix lengths automatically evenly spanned
    # We force the model to evaluate incrementally larger chunks.
    res_uncached = client.compute_instruction_rewards_for_prefixes(
        frames=frames, 
        instruction=instruction, 
        num_samples=15
    )
    t_uncached = time.time() - start_uncached
    
    print("\n[Running Cached...]")
    os.environ["TOPREWARD_QWEN_PREFIX_CACHE"] = "1"
    start_cached = time.time()
    res_cached = client.compute_instruction_rewards_for_prefixes(
        frames=frames, 
        instruction=instruction, 
        num_samples=15
    )
    t_cached = time.time() - start_cached
    
    print(f"\n--- Benchmark Results ---")
    print(f"Prefix count computed: {len(res_uncached.prefix_lengths)}")
    print(f"Uncached Time: {t_uncached:.2f}s")
    print(f"Cached Time:   {t_cached:.2f}s")
    speedup = t_uncached / t_cached
    print(f"Speedup:       {speedup:.2f}x")
    
    print(f"\nMax Absolute Difference: {max(abs(u - c) for u, c in zip(res_uncached.prefix_rewards, res_cached.prefix_rewards)):.6f}")

if __name__ == "__main__":
    main()
