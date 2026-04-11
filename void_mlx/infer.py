#!/usr/bin/env python3
"""VOID inference on Apple Silicon via MLX.

Usage:
    python -m void_mlx.infer \
        --sample sample/BigBen \
        --base-model /path/to/CogVideoX-Fun-V1.5-5b-InP-mlx \
        --pass1 weights/void_pass1.safetensors \
        --output output_bigben.gif
"""

import argparse
import time
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="VOID MLX inference")
    parser.add_argument("--sample", type=str, required=True,
                        help="Path to sample directory (with input_video.mp4, trimask_quadmask.mp4, prompt.json)")
    parser.add_argument("--base-model", type=str,
                        default="/Users/dgrauet/Work/mlx-forge/models/cogvideox-fun-v1.5-5b-inp-mlx")
    parser.add_argument("--pass1", type=str, default="weights/void_pass1.safetensors")
    parser.add_argument("--pass2", type=str, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--max-frames", type=int, default=85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="void_output.gif")
    args = parser.parse_args()

    from void_mlx.mask_utils import load_sample
    from void_mlx.pipeline import VOIDPipeline

    total_t0 = time.monotonic()

    # Load sample
    print(f"Loading sample from {args.sample}...")
    video, mask, prompt = load_sample(
        args.sample, height=args.height, width=args.width, max_frames=args.max_frames,
    )
    print(f"  Video: {video.shape}, Mask: {mask.shape}")
    print(f"  Prompt: \"{prompt}\"")
    print(f"  Mask values: {np.unique(mask.round(2))}")

    # Load pipeline
    print(f"\nLoading VOID pipeline...")
    pipe = VOIDPipeline(
        base_model_path=args.base_model,
        pass1_checkpoint=args.pass1,
        pass2_checkpoint=args.pass2,
    )

    # Run inference
    two_pass = args.pass2 is not None
    mode = "two-pass" if two_pass else "pass 1 only"
    print(f"\nRunning VOID inference ({mode}, {args.steps} steps)...")
    output = pipe(
        video=video,
        mask=mask,
        prompt=prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        two_pass=two_pass,
    )

    # Save output
    print(f"\nSaving to {args.output}...")
    output_uint8 = (output * 255).clip(0, 255).astype(np.uint8)

    from PIL import Image
    frames = [Image.fromarray(output_uint8[i]) for i in range(output_uint8.shape[0])]
    if args.output.endswith(".gif"):
        frames[0].save(args.output, save_all=True, append_images=frames[1:],
                       duration=83, loop=0)  # ~12fps
    elif args.output.endswith(".mp4"):
        import imageio
        imageio.mimsave(args.output, output_uint8, fps=12)
    else:
        # Save individual frames
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(out_dir / f"frame_{i:04d}.png")

    total = time.monotonic() - total_t0
    print(f"\nDone! {len(frames)} frames saved to {args.output} ({total:.1f}s total)")


if __name__ == "__main__":
    main()
