"""VOID pipeline for MLX.

Two-pass video inpainting:
  Pass 1: Standard quadmask inpainting with void_pass1 weights
  Pass 2: Refinement with warped noise from pass 1 output + void_pass2 weights

Uses the VideoX-Fun-mlx pipeline as the inference engine.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

# Add VideoX-Fun-mlx to path
_VIDEOX_FUN_MLX = os.environ.get(
    "VIDEOX_FUN_MLX_PATH",
    str(Path(__file__).parent.parent.parent / "VideoX-Fun-mlx"),
)
if _VIDEOX_FUN_MLX not in sys.path:
    sys.path.insert(0, _VIDEOX_FUN_MLX)

from videox_fun_mlx.models.cogvideox_vae import AutoencoderKLCogVideoX
from videox_fun_mlx.models.cogvideox_transformer3d import CogVideoXTransformer3DModel
from videox_fun_mlx.models.t5_encoder import T5Encoder
from videox_fun_mlx.models.tokenizer import T5Tokenizer
from videox_fun_mlx.pipeline.pipeline_cogvideox_fun_inpaint import CogVideoXFunInpaintPipeline
from videox_fun_mlx.pipeline.scheduler import DDIMScheduler


def load_void_weights(transformer: CogVideoXTransformer3DModel, checkpoint_path: str):
    """Load VOID checkpoint weights into an existing transformer.

    Handles the patch_embed.proj.weight channel mismatch:
    VOID checkpoints may have different channel counts than the base model.
    The first and last 128 channels are copied from the checkpoint,
    middle channels keep the base model's pretrained weights.
    """
    from videox_fun_mlx.utils import convert_pytorch_weights

    weights = mx.load(checkpoint_path)
    weights = convert_pytorch_weights(weights)

    # Handle patch_embed.proj.weight channel mismatch
    pe_key = "patch_embed.proj.weight"
    if pe_key in weights:
        void_w = weights[pe_key]
        import mlx.nn
        model_leaves = dict(mlx.nn.utils.tree_flatten(transformer.trainable_parameters()))
        if pe_key in model_leaves:
            model_w = model_leaves[pe_key]
            if void_w.shape != model_w.shape:
                print(f"  Adapting {pe_key}: {void_w.shape} -> {model_w.shape}")
                feat_dim = 128  # 16 latent_ch * 8 feat_scale
                new_w = mx.array(model_w)
                if void_w.ndim == 2:
                    # Copy first feat_dim and last feat_dim from checkpoint
                    first = void_w[:, :feat_dim]
                    last = void_w[:, -feat_dim:]
                    new_w = mx.concatenate([
                        first,
                        new_w[:, feat_dim:-feat_dim],
                        last,
                    ], axis=1)
                weights[pe_key] = new_w

    transformer.load_weights(list(weights.items()), strict=False)


class VOIDPipeline:
    """VOID two-pass video inpainting pipeline for MLX."""

    def __init__(
        self,
        base_model_path: str,
        pass1_checkpoint: str,
        pass2_checkpoint: Optional[str] = None,
    ):
        self.base_model_path = base_model_path
        self.pass1_checkpoint = pass1_checkpoint
        self.pass2_checkpoint = pass2_checkpoint

        # Load shared components
        print("Loading VAE...")
        self.vae = AutoencoderKLCogVideoX.from_pretrained(base_model_path)

        print("Loading T5 text encoder...")
        self.t5 = T5Encoder.from_pretrained(base_model_path)
        self.tokenizer = T5Tokenizer(base_model_path)

        # Scheduler config
        sched_config = {}
        sched_file = os.path.join(base_model_path, "scheduler_scheduler_config.json")
        if os.path.exists(sched_file):
            import json
            with open(sched_file) as f:
                sched_config = json.load(f)
            for k in ("_class_name", "_diffusers_version", "trained_betas"):
                sched_config.pop(k, None)
        self.sched_config = sched_config

        # Load transformer with VOID pass1 weights
        # VOID uses 48 input channels (16 latent + 16 VAE-mask + 16 VAE-video)
        # vs base model's 33 channels (16 + 17)
        print("Loading transformer (in_channels=48 for VOID)...")
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(
            base_model_path,
            transformer_additional_kwargs={"in_channels": 48},
        )
        # Base model weights have patch_embed for 33ch — load with strict=False
        # then VOID weights overwrite everything including the correct patch_embed
        print("  Loading VOID pass1 weights...")
        load_void_weights(self.transformer, pass1_checkpoint)
        print("  VOID pass1 loaded.")

    def _make_pipeline(self) -> CogVideoXFunInpaintPipeline:
        scheduler = DDIMScheduler(**self.sched_config)
        return CogVideoXFunInpaintPipeline(
            vae=self.vae,
            transformer=self.transformer,
            scheduler=scheduler,
            text_encoder=self.t5,
            tokenizer=self.tokenizer,
        )

    def run_pass1(
        self,
        video: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: int = 42,
    ) -> np.ndarray:
        """Run VOID pass 1 (base inpainting with quadmask).

        VOID uses VAE-encoded mask (16ch) instead of raw 1ch mask.
        The mask is inverted (1-mask), tiled to 3ch RGB, and encoded by VAE.
        The masked_video is the original video (not zeroed out).
        guidance_scale defaults to 1.0 (VOID doesn't use CFG).

        Args:
            video: (F, H, W, 3) float32 in [0, 1].
            mask: (F, H, W, 1) float32 quadmask (0=remove, 0.5=affected, 1=keep).
            prompt: Text prompt describing the background.

        Returns:
            (F_out, H, W, 3) float32 output video in [0, 1].
        """
        pipe = self._make_pipeline()

        video_mx = mx.array(video[None])  # (1, F, H, W, 3)
        mask_1ch = mx.array(mask[None])   # (1, F, H, W, 1)

        # VOID mask processing:
        # 1. Invert mask: 1-mask (so remove=1, keep=0)
        # 2. Tile to 3ch (RGB) for VAE encoding
        # 3. VAE-encode the inverted mask -> 16ch mask latent
        # 4. VAE-encode the original video (unmasked) -> 16ch video latent
        inverted_mask_3ch = mx.repeat(1.0 - mask_1ch, 3, axis=-1)  # (1, F, H, W, 3)

        # Encode mask through VAE
        mask_encoded = self.vae.encode(inverted_mask_3ch).mode() * self.vae.scaling_factor
        # Encode original video through VAE
        video_encoded = self.vae.encode(video_mx).mode() * self.vae.scaling_factor

        # Convert to channels-first: (1, F, H, W, C) -> (1, F, C, H, W)
        mask_cf = mask_encoded.transpose(0, 1, 4, 2, 3)
        video_cf = video_encoded.transpose(0, 1, 4, 2, 3)

        # Inpaint conditioning: 16ch mask + 16ch video = 32ch
        inpaint_cf = mx.concatenate([mask_cf, video_cf], axis=2)

        # Latent noise
        if seed is not None:
            mx.random.seed(seed)

        # Encode prompt
        prompt_embeds = pipe.encode_prompt(prompt, "", do_cfg=False)
        mx.eval(prompt_embeds)

        # Start from noise
        latent_shape = video_cf.shape  # (1, F_lat, 16, H_lat, W_lat)
        noise = mx.random.normal(latent_shape)

        pipe.scheduler.set_timesteps(num_inference_steps)
        current = pipe.scheduler.add_noise(video_cf, noise, pipe.scheduler.timesteps[0])

        # RoPE
        B, F_vid, H_vid, W_vid, _ = video_mx.shape
        F_lat = video_cf.shape[1]
        image_rotary_emb = pipe._prepare_rotary_embeddings(H_vid, W_vid, F_lat)

        # Denoising loop (no CFG — guidance_scale=1.0)
        t0 = time.monotonic()
        for i, t in enumerate(pipe.scheduler.timesteps):
            t_input = mx.array([float(t)])
            noise_pred = self.transformer(
                hidden_states=current,
                encoder_hidden_states=prompt_embeds,
                timestep=t_input,
                inpaint_latents=inpaint_cf,
                image_rotary_emb=image_rotary_emb,
            )
            current = pipe.scheduler.step(noise_pred, t, current)
            mx.eval(current)

        # Decode
        decoded = current.transpose(0, 1, 3, 4, 2) / self.vae.scaling_factor
        output = self.vae.decode(decoded)
        output = mx.clip(output / 2 + 0.5, 0, 1)
        mx.eval(output)

        elapsed = time.monotonic() - t0
        print(f"  Pass 1: {elapsed:.1f}s ({elapsed/num_inference_steps:.1f}s/step)")

        return np.array(output[0].astype(mx.float32))

    def __call__(
        self,
        video: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        seed: int = 42,
    ) -> np.ndarray:
        """Run full VOID inference (pass 1 only for now).

        Args:
            video: (F, H, W, 3) float32 in [0, 1].
            mask: (F, H, W, 1) float32 quadmask.
            prompt: Text prompt.

        Returns:
            (F_out, H, W, 3) float32 output in [0, 1].
        """
        return self.run_pass1(
            video, mask, prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
