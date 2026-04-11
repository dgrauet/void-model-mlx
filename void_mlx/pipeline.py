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

    Supports both PyTorch (.safetensors with 3D+ conv weights) and
    MLX-converted (already transposed, possibly quantized) formats.
    """
    from videox_fun_mlx.utils import convert_pytorch_weights

    weights = mx.load(checkpoint_path)

    # Detect if already MLX format: quantized weights have .scales keys
    has_scales = any(k.endswith(".scales") for k in weights)
    has_3d_weights = any(w.ndim >= 3 for w in weights.values() if isinstance(w, mx.array))

    if has_3d_weights and not has_scales:
        # PyTorch format — needs conv transposition
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
                feat_dim = 128
                new_w = mx.array(model_w)
                if void_w.ndim == 2:
                    first = void_w[:, :feat_dim]
                    last = void_w[:, -feat_dim:]
                    new_w = mx.concatenate([first, new_w[:, feat_dim:-feat_dim], last], axis=1)
                weights[pe_key] = new_w

    # If quantized, convert Linear -> QuantizedLinear before loading
    if has_scales:
        from videox_fun_mlx.utils import quantize_model_from_weights, get_quantize_config
        # Build a fake path for quantize config
        ckpt_dir = os.path.dirname(checkpoint_path)
        quantize_model_from_weights(transformer, weights, ckpt_dir, "transformer")

    transformer.load_weights(list(weights.items()), strict=False)


def _create_and_load_void_transformer(base_model_path: str, checkpoint_path: str):
    """Create transformer with VOID config and load weights directly.

    Avoids loading base model weights (saves ~6GB RAM).
    """
    import json

    # Read transformer config from base model
    tf_config_file = os.path.join(base_model_path, "transformer_config.json")
    if not os.path.exists(tf_config_file):
        cfg_root = os.path.join(base_model_path, "config.json")
        with open(cfg_root) as f:
            cfg = json.load(f)
        tf_config = cfg.get("transformer", cfg)
    else:
        with open(tf_config_file) as f:
            tf_config = json.load(f)

    # Override in_channels for VOID (48 instead of 33)
    init_keys = {
        "num_attention_heads", "attention_head_dim", "in_channels", "out_channels",
        "flip_sin_to_cos", "freq_shift", "time_embed_dim", "text_embed_dim",
        "num_layers", "dropout", "attention_bias", "sample_width", "sample_height",
        "sample_frames", "patch_size", "patch_size_t", "temporal_compression_ratio",
        "max_text_seq_length", "activation_fn", "timestep_activation_fn",
        "norm_elementwise_affine", "norm_eps", "spatial_interpolation_scale",
        "temporal_interpolation_scale", "use_rotary_positional_embeddings",
        "use_learned_positional_embeddings", "patch_bias",
    }
    filtered = {k: v for k, v in tf_config.items() if k in init_keys}
    filtered["in_channels"] = 48

    # Create empty model
    model = CogVideoXTransformer3DModel(**filtered)

    # Load VOID weights directly (no base model weights loaded first)
    load_void_weights(model, checkpoint_path)

    import mlx.nn
    leaves = mlx.nn.utils.tree_flatten(model.trainable_parameters())
    param_count = sum(v.size for _, v in leaves)
    print(f"  Transformer: {param_count / 1e6:.1f}M parameters")

    return model


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

        # Create transformer architecture (48ch for VOID) and load VOID weights
        # directly — skip loading base model weights to save memory.
        # VOID checkpoints contain ALL transformer weights.
        print("Loading transformer (in_channels=48 for VOID)...")
        self.transformer = _create_and_load_void_transformer(
            base_model_path, pass1_checkpoint,
        )
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

    def _encode_inputs(self, video: np.ndarray, mask: np.ndarray, batch_frames: int = 9):
        """Encode video and mask to latent space for VOID conditioning.

        Encodes in batches of frames to avoid OOM on long videos.

        Returns:
            (video_shape, video_cf, inpaint_cf)
        """
        F = video.shape[0]
        video_mx = mx.array(video[None])
        mask_1ch = mx.array(mask[None])
        inverted_mask_3ch = mx.repeat(1.0 - mask_1ch, 3, axis=-1)

        # Encode in temporal batches to save memory
        # VAE temporal compression = 4x, so batch_frames should be 4k+1
        mask_latents = []
        video_latents = []
        for start in range(0, F, batch_frames):
            end = min(start + batch_frames, F)
            m_batch = self.vae.encode(inverted_mask_3ch[:, start:end]).mode() * self.vae.scaling_factor
            v_batch = self.vae.encode(video_mx[:, start:end]).mode() * self.vae.scaling_factor
            mx.eval(m_batch, v_batch)
            mask_latents.append(m_batch)
            video_latents.append(v_batch)

        mask_encoded = mx.concatenate(mask_latents, axis=1)
        video_encoded = mx.concatenate(video_latents, axis=1)

        mask_cf = mask_encoded.transpose(0, 1, 4, 2, 3)
        video_cf = video_encoded.transpose(0, 1, 4, 2, 3)
        inpaint_cf = mx.concatenate([mask_cf, video_cf], axis=2)

        mx.eval(video_cf, inpaint_cf)
        return video_mx.shape, video_cf, inpaint_cf

    def _denoise_window(
        self, latents, inpaint, prompt_embeds, rope, scheduler, timesteps,
    ):
        """Run denoising on a single temporal window."""
        current = latents
        for t in timesteps:
            t_input = mx.array([float(t)])
            noise_pred = self.transformer(
                hidden_states=current,
                encoder_hidden_states=prompt_embeds,
                timestep=t_input,
                inpaint_latents=inpaint,
                image_rotary_emb=rope,
            )
            current = scheduler.step(noise_pred, t, current)
            mx.eval(current)
        return current

    def run_pass1(
        self,
        video: np.ndarray,
        mask: np.ndarray,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: int = 42,
        temporal_window: int = 85,
        temporal_stride: int = 16,
    ) -> np.ndarray:
        """Run VOID pass 1 with temporal multidiffusion for long videos.

        Args:
            video: (F, H, W, 3) float32 in [0, 1].
            mask: (F, H, W, 1) float32 quadmask.
            prompt: Text prompt.
            temporal_window: Temporal window size in video frames (default 85).
            temporal_stride: Overlap stride in video frames (default 16).

        Returns:
            (F_out, H, W, 3) float32 output video in [0, 1].
        """
        pipe = self._make_pipeline()

        if seed is not None:
            mx.random.seed(seed)

        # Encode inputs
        video_shape, video_cf, inpaint_cf = self._encode_inputs(video, mask)
        B, F_vid, H_vid, W_vid, _ = video_shape
        F_lat = video_cf.shape[1]

        # Prompt
        prompt_embeds = pipe.encode_prompt(prompt, "", do_cfg=False)
        mx.eval(prompt_embeds)

        # Noise
        noise = mx.random.normal(video_cf.shape)
        pipe.scheduler.set_timesteps(num_inference_steps)
        current = pipe.scheduler.add_noise(video_cf, noise, pipe.scheduler.timesteps[0])

        # Latent temporal window size
        latent_window = (temporal_window - 1) // 4 + 1
        latent_stride = temporal_stride // 4

        t0 = time.monotonic()

        if F_lat <= latent_window:
            # Short video: process all at once
            rope = pipe._prepare_rotary_embeddings(H_vid, W_vid, F_lat)
            current = self._denoise_window(
                current, inpaint_cf, prompt_embeds, rope, pipe.scheduler,
                pipe.scheduler.timesteps,
            )
        else:
            # Temporal multidiffusion: sliding window with linear blending
            print(f"  Temporal multidiffusion: {F_lat} latent frames, "
                  f"window={latent_window}, stride={latent_stride}")

            for i, t in enumerate(pipe.scheduler.timesteps):
                t_input = mx.array([float(t)])

                # Accumulate predictions with blending weights
                canvas = mx.zeros_like(current).astype(mx.float32)
                weights = mx.zeros((1, F_lat, 1, 1, 1), dtype=mx.float32)

                time_beg = 0
                while time_beg < F_lat:
                    time_end = min(time_beg + latent_window, F_lat)

                    # Extract window
                    lat_i = current[:, time_beg:time_end]
                    inp_i = inpaint_cf[:, time_beg:time_end]

                    # RoPE for this window size
                    win_len = time_end - time_beg
                    rope_i = pipe._prepare_rotary_embeddings(H_vid, W_vid, win_len)

                    # Single denoise step on this window
                    noise_pred_i = self.transformer(
                        hidden_states=lat_i,
                        encoder_hidden_states=prompt_embeds,
                        timestep=t_input,
                        inpaint_latents=inp_i,
                        image_rotary_emb=rope_i,
                    )
                    stepped_i = pipe.scheduler.step(noise_pred_i, t, lat_i)
                    mx.eval(stepped_i)

                    # Blending weights: linear ramp at overlap edges
                    w_parts = []
                    ramp_left = latent_stride if time_beg > 0 else 0
                    ramp_right = latent_stride if time_end < F_lat else 0
                    mid = win_len - ramp_left - ramp_right

                    if ramp_left > 0:
                        w_parts.append(mx.linspace(0, 1, ramp_left + 2)[1:-1].reshape(1, ramp_left, 1, 1, 1))
                    if mid > 0:
                        w_parts.append(mx.ones((1, mid, 1, 1, 1)))
                    if ramp_right > 0:
                        w_parts.append(mx.linspace(1, 0, ramp_right + 2)[1:-1].reshape(1, ramp_right, 1, 1, 1))
                    w_i = mx.concatenate(w_parts, axis=1) if len(w_parts) > 1 else w_parts[0]

                    canvas[:, time_beg:time_end] = canvas[:, time_beg:time_end] + stepped_i * w_i
                    weights[:, time_beg:time_end] = weights[:, time_beg:time_end] + w_i

                    # Slide window
                    if time_end >= F_lat:
                        break
                    time_beg = time_end - latent_stride

                current = canvas / mx.maximum(weights, 1e-8)
                mx.eval(current)

                if (i + 1) % 10 == 0 or i == 0:
                    print(f"    Step {i+1}/{num_inference_steps}")

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
        guidance_scale: float = 1.0,
        seed: int = 42,
        temporal_window: int = 85,
        temporal_stride: int = 16,
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
            temporal_window=temporal_window,
            temporal_stride=temporal_stride,
        )
