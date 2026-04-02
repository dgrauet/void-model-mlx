"""
Inference with Pre-generated Pass 1 Warped Noise

This script:
1. Takes one or more input video names (e.g., "ball-play" "car-crash")
2. Loads the model ONCE (efficient for batch processing)
3. For each video:
   - Finds the corresponding pass 1 video
   - Generates warped noise from pass 1 video (or loads cached version)
   - Runs inference with your trained model using that warped noise

Usage (single video):
    python inference_with_pass1_warped_noise.py \
        --video_name ball-play \
        --model_checkpoint path/to/your/trained/model.safetensors \
        --output_dir ./inference_results

Usage (multiple videos - model loaded only once!):
    python inference_with_pass1_warped_noise.py \
        --video_names ball-play car-crash dog-frisbee \
        --model_checkpoint path/to/your/trained/model.safetensors \
        --output_dir ./inference_results
"""

import torch
import cv2
import numpy as np
import imageio
import sys
import os
import argparse
import subprocess
from pathlib import Path
from loguru import logger

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLCogVideoX, CogVideoXTransformer3DModel,
                                T5EncoderModel, T5Tokenizer)
from videox_fun.pipeline import CogVideoXFunInpaintPipeline
from videox_fun.utils.utils import get_video_mask_input
from diffusers import CogVideoXDDIMScheduler


def find_pass1_video(video_name, pass1_dir="./pass1_outputs"):
    """
    Find the pass 1 video for a given video name.

    Args:
        video_name: Base name (e.g., "ball-play")
        pass1_dir: Directory containing pass 1 videos

    Returns:
        Path to pass 1 video or None if not found
    """
    # Look for pattern: {video_name}-fg=-1-*.mp4 (without _tuple)
    pass1_path = Path(pass1_dir)

    if not pass1_path.exists():
        logger.error(f"Pass 1 directory not found: {pass1_dir}")
        return None

    # Find matching videos (exclude _tuple versions)
    candidates = list(pass1_path.glob(f"{video_name}-fg=-1-*.mp4"))
    candidates = [c for c in candidates if "_tuple" not in c.name]

    if not candidates:
        logger.error(f"No pass 1 video found for: {video_name}")
        logger.error(f"Looked in: {pass1_dir}")
        logger.error(f"Pattern: {video_name}-fg=-1-*.mp4")
        return None

    # Return first match
    pass1_video = candidates[0]
    logger.info(f"✓ Found pass 1 video: {pass1_video}")
    return str(pass1_video)


def generate_warped_noise_from_video(video_path, output_dir):
    """
    Generate warped noise from a video using make_warped_noise.py.

    Args:
        video_path: Path to input video
        output_dir: Directory to save warped noise

    Returns:
        Path to generated warped noise .npy file
    """
    os.makedirs(output_dir, exist_ok=True)

    video_stem = Path(video_path).stem
    output_subdir = Path(output_dir) / video_stem
    noise_file = output_subdir / "noises.npy"

    # Check if already generated
    if noise_file.exists():
        logger.info(f"✓ Warped noise already exists: {noise_file}")
        return str(noise_file)

    logger.info("Generating warped noise...")
    logger.info(f"  Input video: {video_path}")
    logger.info(f"  Output dir: {output_subdir}")

    script_dir = Path(__file__).parent
    gwf_script = script_dir / "make_warped_noise.py"
    if not gwf_script.exists():
        logger.error(f"make_warped_noise.py not found: {gwf_script}")
        return None

    # Run make_warped_noise.py using the current Python interpreter
    cmd = [
        sys.executable,
        str(gwf_script),
        os.path.abspath(video_path),
        os.path.abspath(output_subdir),
    ]

    logger.info("  Running Go-with-the-Flow (this may take a few minutes)...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            logger.error("Go-with-the-Flow failed!")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            return None

        if not noise_file.exists():
            logger.error(f"Warped noise file not created: {noise_file}")
            logger.error(f"Files in output dir: {list(output_subdir.glob('*'))}")
            return None

        logger.info(f"✓ Generated warped noise: {noise_file}")
        return str(noise_file)

    except subprocess.TimeoutExpired:
        logger.error("Timeout! Go-with-the-Flow took too long (>10 minutes)")
        return None
    except Exception as e:
        logger.error(f"Error running Go-with-the-Flow: {str(e)}")
        return None


def load_and_resize_warped_noise(noise_path, target_shape, device, dtype):
    """
    Load warped noise and resize to match latent dimensions.

    Args:
        noise_path: Path to warped noise .npy file
        target_shape: (latent_T, latent_H, latent_W, latent_C)
        device: torch device
        dtype: torch dtype

    Returns:
        Warped noise tensor (1, T, C, H, W)
    """
    latent_T, latent_H, latent_W, latent_C = target_shape

    # Load noise
    warped_noise_np = np.load(noise_path)
    logger.info(f"Loaded noise: {warped_noise_np.shape}, dtype: {warped_noise_np.dtype}")

    # Convert float16 to float32 if needed
    if warped_noise_np.dtype == np.float16:
        warped_noise_np = warped_noise_np.astype(np.float32)

    # Convert TCHW to THWC if needed
    if warped_noise_np.ndim == 4 and warped_noise_np.shape[1] == 16:
        warped_noise_np = warped_noise_np.transpose(0, 2, 3, 1)
        logger.info(f"Converted from TCHW to THWC: {warped_noise_np.shape}")

    # Resize if needed
    if warped_noise_np.shape != target_shape:
        logger.info(f"Resizing noise from {warped_noise_np.shape} to {target_shape}")

        # Temporal resize
        if warped_noise_np.shape[0] != latent_T:
            indices = np.linspace(0, warped_noise_np.shape[0]-1, latent_T).astype(int)
            warped_noise_np = warped_noise_np[indices]

        # Spatial resize (per channel)
        resized_frames = []
        for t in range(latent_T):
            frame = warped_noise_np[t]
            channels_resized = []
            for c in range(frame.shape[2]):
                channel = frame[:, :, c]
                channel_resized = cv2.resize(channel, (latent_W, latent_H),
                                            interpolation=cv2.INTER_LINEAR)
                channels_resized.append(channel_resized)
            frame_resized = np.stack(channels_resized, axis=2)
            resized_frames.append(frame_resized)

        warped_noise_np = np.stack(resized_frames, axis=0)
        logger.info(f"Resized to: {warped_noise_np.shape}")

    # Convert to torch: (T, H, W, C) → (1, T, C, H, W)
    warped_noise_np = warped_noise_np.transpose(0, 3, 1, 2)  # (T, C, H, W)
    warped_noise = torch.from_numpy(warped_noise_np).float().unsqueeze(0)
    warped_noise = warped_noise.to(device, dtype=dtype)

    logger.info(f"✓ Warped noise ready: {warped_noise.shape}")
    logger.info(f"  Mean: {warped_noise.mean():.4f}, Std: {warped_noise.std():.4f}")

    return warped_noise


def main():
    parser = argparse.ArgumentParser(description="Inference with Pass 1 Warped Noise")

    # Input/Output
    parser.add_argument("--video_name", type=str, default=None,
                       help="Single video name (e.g., 'ball-play')")
    parser.add_argument("--video_names", type=str, nargs='+', default=None,
                       help="Multiple video names (e.g., 'ball-play' 'car-crash' ...)")
    parser.add_argument("--data_rootdir", type=str, default="./data",
                       help="Root directory containing test videos")
    parser.add_argument("--pass1_dir", type=str, default="./pass1_outputs",
                       help="Directory containing pass 1 videos")
    parser.add_argument("--output_dir", type=str, default="./inference_with_warped_noise",
                       help="Output directory for results")

    # Model
    parser.add_argument("--model_name", type=str, default="./CogVideoX-Fun-V1.5-5b-InP",
                       help="Base model path")
    parser.add_argument("--model_checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint (.safetensors)")

    # Video settings
    parser.add_argument("--max_video_length", type=int, default=197,
                       help="Maximum video length")
    parser.add_argument("--temporal_window_size", type=int, default=85,
                       help="Temporal window size")
    parser.add_argument("--height", type=int, default=384,
                       help="Video height")
    parser.add_argument("--width", type=int, default=672,
                       help="Video width")

    # Generation settings
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--guidance_scale", type=float, default=6.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of denoising steps")

    # Warped noise settings
    parser.add_argument("--warped_noise_cache_dir", type=str, default="./pass1_warped_noise_cache",
                       help="Cache directory for warped noise")
    parser.add_argument("--skip_noise_generation", action="store_true",
                       help="Skip warped noise generation (use existing cache)")

    # Misc
    parser.add_argument("--use_quadmask", action="store_true", default=True,
                       help="Use quadmask format")

    args = parser.parse_args()

    # Determine video list
    if args.video_names:
        video_names = args.video_names
    elif args.video_name:
        video_names = [args.video_name]
    else:
        logger.error("Must provide either --video_name or --video_names")
        sys.exit(1)

    # Setup
    device = torch.device("cuda")
    weight_dtype = torch.bfloat16
    sample_size = (args.height, args.width)

    logger.info("="*80)
    logger.info("INFERENCE WITH PASS 1 WARPED NOISE")
    logger.info("="*80)
    logger.info(f"Videos to process: {len(video_names)}")
    logger.info(f"Model checkpoint: {args.model_checkpoint}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*80)

    os.makedirs(args.output_dir, exist_ok=True)

    # ==================== LOAD MODEL (ONCE) ====================
    logger.info("\n[1/3] Loading model...")

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.model_name, subfolder="transformer", torch_dtype=weight_dtype,
        use_vae_mask=True, stack_mask=False,
    )

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.model_checkpoint}")
    from safetensors.torch import load_file
    state_dict = load_file(args.model_checkpoint)
    state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

    # Handle channel mismatch if needed
    param_name = 'patch_embed.proj.weight'
    if param_name in state_dict and param_name in transformer.state_dict():
        if state_dict[param_name].size(1) != transformer.state_dict()[param_name].size(1):
            latent_ch, feat_scale = 16, 8
            feat_dim = int(latent_ch * feat_scale)
            new_weight = transformer.state_dict()[param_name].clone()
            new_weight[:, :feat_dim] = state_dict[param_name][:, :feat_dim]
            new_weight[:, -feat_dim:] = state_dict[param_name][:, -feat_dim:]
            state_dict[param_name] = new_weight

    m, u = transformer.load_state_dict(state_dict, strict=False)
    transformer = transformer.to(device)

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.model_name, subfolder="vae"
    ).to(weight_dtype).to(device)

    text_encoder = T5EncoderModel.from_pretrained(
        args.model_name, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(device)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name, subfolder="tokenizer")
    scheduler = CogVideoXDDIMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    pipeline = CogVideoXFunInpaintPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        transformer=transformer, scheduler=scheduler,
    )
    pipeline.enable_model_cpu_offload(device=device)
    logger.info("✓ Model loaded successfully")

    # Calculate latent dimensions (same for all videos)
    video_length = int((args.max_video_length - 1) // vae.config.temporal_compression_ratio *
                      vae.config.temporal_compression_ratio) + 1
    latent_T = (args.temporal_window_size - 1) // 4 + 1
    latent_H = args.height // 8
    latent_W = args.width // 8
    latent_C = 16

    # ==================== PROCESS VIDEOS ====================
    logger.info("\n[2/3] Processing videos...")
    logger.info("="*80)

    total_videos = len(video_names)
    succeeded = 0
    failed = 0

    for idx, video_name in enumerate(video_names, 1):
        logger.info("\n" + "="*80)
        logger.info(f"VIDEO {idx}/{total_videos}: {video_name}")
        logger.info("="*80)

        try:
            # Find pass 1 video
            logger.info(f"  [a] Finding pass 1 video...")
            pass1_video_path = find_pass1_video(video_name, args.pass1_dir)

            if pass1_video_path is None:
                logger.error(f"  ✗ Cannot find pass 1 video for {video_name}")
                failed += 1
                continue

            # Generate/load warped noise
            logger.info(f"  [b] Generating/loading warped noise...")

            if args.skip_noise_generation:
                video_stem = Path(pass1_video_path).stem
                noise_file = Path(args.warped_noise_cache_dir) / video_stem / "noises.npy"
                if not noise_file.exists():
                    logger.error(f"  ✗ Noise file not found: {noise_file}")
                    failed += 1
                    continue
                warped_noise_path = str(noise_file)
            else:
                warped_noise_path = generate_warped_noise_from_video(
                    pass1_video_path,
                    args.warped_noise_cache_dir,
                )

                if warped_noise_path is None:
                    logger.error(f"  ✗ Failed to generate warped noise for {video_name}")
                    failed += 1
                    continue

            # Load input video & mask
            logger.info(f"  [c] Loading input video and mask...")
            input_video, input_video_mask, prompt, _ = get_video_mask_input(
                video_name, sample_size=sample_size, keep_fg_ids=[-1],
                max_video_length=video_length, temporal_window_size=args.temporal_window_size,
                data_rootdir=args.data_rootdir, use_trimask=False, use_quadmask=args.use_quadmask,
            )
            logger.info(f"    Input video: {input_video.shape}")
            logger.info(f"    Prompt: {prompt}")

            # Load and resize warped noise
            warped_noise = load_and_resize_warped_noise(
                warped_noise_path,
                (latent_T, latent_H, latent_W, latent_C),
                device,
                weight_dtype
            )

            # Run inference
            logger.info(f"  [d] Running inference...")
            generator = torch.Generator(device=device).manual_seed(args.seed)

            with torch.no_grad():
                output = pipeline(
                    prompt, num_frames=args.temporal_window_size, negative_prompt="",
                    height=args.height, width=args.width,
                    generator=generator, guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    video=input_video, mask_video=input_video_mask, strength=1.0,
                    use_trimask=False, zero_out_mask_region=False,
                    use_vae_mask=True, stack_mask=False,
                    latents=warped_noise,
                ).videos

            logger.info(f"    Generated: {output.shape}")

            # Save results
            logger.info(f"  [e] Saving results...")
            if output.min() < -0.5:
                video_np = (output[0] + 1.0) / 2.0
            else:
                video_np = output[0]
            video_np = video_np.clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()
            video_np = (video_np * 255).astype(np.uint8)

            output_path = Path(args.output_dir) / f"{video_name}_warped_noise_inference.mp4"
            imageio.mimsave(output_path, video_np, fps=12, codec='libx264',
                           quality=8, pixelformat='yuv420p')

            logger.info(f"  ✓ Success: {output_path}")
            succeeded += 1

        except Exception as e:
            logger.error(f"  ✗ Error processing {video_name}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            failed += 1

    # ==================== SUMMARY ====================
    logger.info("\n" + "="*80)
    logger.info("[3/3] BATCH PROCESSING COMPLETE")
    logger.info("="*80)
    logger.info(f"Total videos:  {total_videos}")
    logger.info(f"Succeeded:     {succeeded}")
    logger.info(f"Failed:        {failed}")
    logger.info(f"Output dir:    {args.output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
