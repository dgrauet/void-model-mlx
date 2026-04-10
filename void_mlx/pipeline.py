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

        # Load transformer with pass1 weights
        print("Loading transformer + VOID pass1 weights...")
        self.transformer = CogVideoXTransformer3DModel.from_pretrained(base_model_path)
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
        guidance_scale: float = 6.0,
        seed: int = 42,
    ) -> np.ndarray:
        """Run VOID pass 1 (base inpainting with quadmask).

        Args:
            video: (F, H, W, 3) float32 in [0, 1].
            mask: (F, H, W, 1) float32 quadmask.
            prompt: Text prompt.

        Returns:
            (F_out, H, W, 3) float32 output video in [0, 1].
        """
        pipe = self._make_pipeline()

        video_mx = mx.array(video[None])
        mask_mx = mx.array(mask[None])

        t0 = time.monotonic()
        output = pipe(
            video=video_mx,
            mask=mask_mx,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
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
