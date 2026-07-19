"""bf16 dtype-flow contracts for the VOID pipeline (dogfood 2026-07-20).

The reference runs weight_dtype = torch.bfloat16 end-to-end and casts
latents back to the model dtype after every scheduler step
(pipeline_cogvideox_fun_inpaint.py:1170). The port ran everything in fp32
(182 s of float32 gemms + 71k bf16->fp32 upcasts, smeltr session
void-dogfood-bf16). Pure unit tests: transformer/scheduler stubbed.
"""

import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from void_mlx.pipeline import WEIGHT_DTYPE, VOIDPipeline


def test_weight_dtype_mirrors_reference():
    assert WEIGHT_DTYPE == mx.bfloat16


def test_denoise_window_recasts_after_each_step():
    """scheduler.step promotes to fp32 (iso torch); every transformer input
    must be back in the latents dtype (reference :1170)."""
    seen_dtypes = []

    class StubTransformer:
        def __call__(self, hidden_states, **kwargs):
            seen_dtypes.append(hidden_states.dtype)
            return hidden_states

    class StubScheduler:
        def step(self, noise_pred, t, current):
            return current.astype(mx.float32)  # torch-like promotion

    pipe = VOIDPipeline.__new__(VOIDPipeline)
    pipe.transformer = StubTransformer()

    latents = mx.zeros((1, 2, 4, 4, 4), dtype=WEIGHT_DTYPE)
    out = pipe._denoise_window(
        latents, None, None, None, StubScheduler(), [999, 500, 0]
    )

    assert out.dtype == WEIGHT_DTYPE
    assert seen_dtypes == [WEIGHT_DTYPE] * 3, (
        "every transformer input must be bf16, got " + str(seen_dtypes)
    )
