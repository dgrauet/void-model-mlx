"""Two-pass memory lifecycle (VOID dogfood follow-up, 2026-07-20).

The reference runs the two passes as two separate PROCESSES (predict_v2v.py
then inference_with_pass1_warped_noise.py) — the merged single-process port
is port-only structure, so its memory behavior is ours to fix:
- T5-XXL bf16 (~9.5 GB) stayed resident for the whole run although it is
  only needed to encode ONE prompt (identical in both passes);
- run_pass2 ended by reloading pass1 weights "for next run" — 10 GB of
  load work inside the final decode window, for a run that usually ends;
- measured: plateau 21.6 GB, transition spike 35.3 GB (131% of budget).

Contracts: prompt encoded once then T5 freed (lazily reloaded for a new
prompt); checkpoint switches are lazy and deduplicated.
"""

import mlx.core as mx
import pytest

import void_mlx.pipeline as vp


class StubPipe:
    def __init__(self):
        self.encode_calls = 0
        self.text_encoder = object()
        self.tokenizer = object()

    def encode_prompt(self, prompt, negative, do_cfg):
        self.encode_calls += 1
        return mx.zeros((1, 4, 8), dtype=mx.float32)


@pytest.fixture
def pipe():
    p = vp.VOIDPipeline.__new__(vp.VOIDPipeline)
    p.base_model_path = "/nonexistent"
    p.pass1_checkpoint = "p1.safetensors"
    p.pass2_checkpoint = "p2.safetensors"
    p.t5 = object()
    p.tokenizer = object()
    p._prompt_cache = {}
    p._loaded_checkpoint = "p1.safetensors"
    p.low_ram = True
    return p


class TestPromptEncodeOnce:
    def test_second_encode_hits_cache_and_t5_freed(self, pipe):
        stub = StubPipe()
        e1 = pipe._encode_prompt(stub, "a prompt")
        assert stub.encode_calls == 1
        assert pipe.t5 is None, "T5 must be freed after encoding"
        assert pipe.tokenizer is None

        e2 = pipe._encode_prompt(stub, "a prompt")
        assert stub.encode_calls == 1, "same prompt must not re-encode"
        assert e2 is e1
        assert e1.dtype == vp.WEIGHT_DTYPE

    def test_new_prompt_reloads_t5_lazily(self, pipe, monkeypatch):
        stub = StubPipe()
        pipe._encode_prompt(stub, "first")

        reloaded = []
        monkeypatch.setattr(
            vp.T5Encoder, "from_pretrained", classmethod(lambda cls, p: reloaded.append(p) or object())
        )
        monkeypatch.setattr(vp, "T5Tokenizer", lambda p: object())

        pipe._encode_prompt(stub, "second")
        assert reloaded == ["/nonexistent"], "new prompt must reload T5 once"
        assert stub.encode_calls == 2
        assert pipe.t5 is None, "T5 freed again after the new encode"


class TestLazyCheckpointSwitch:
    def test_switch_loads_once_and_same_is_noop(self, pipe, monkeypatch):
        loads = []
        monkeypatch.setattr(vp, "load_void_weights", lambda tf, path: loads.append(path))
        pipe.transformer = object()

        pipe._ensure_checkpoint("p2.safetensors")
        pipe._ensure_checkpoint("p2.safetensors")
        assert loads == ["p2.safetensors"], "same checkpoint must not reload"

        pipe._ensure_checkpoint("p1.safetensors")
        assert loads == ["p2.safetensors", "p1.safetensors"]

    def test_run_pass2_has_no_trailing_pass1_reload(self):
        import inspect

        src = inspect.getsource(vp.VOIDPipeline.run_pass2)
        assert "Reload pass 1" not in src
        assert "load_void_weights(self.transformer, self.pass1_checkpoint)" not in src, (
            "pass1 must be reloaded lazily by run_pass1, not eagerly at the end "
            "of run_pass2 inside the decode memory window"
        )


class TestDefaultKeepsT5Resident:
    def test_without_low_ram_t5_stays_and_cache_still_works(self, pipe):
        """Default mode mirrors the reference residency ("model loaded
        ONCE"): T5 must survive encoding; the embed cache still dedupes."""
        pipe.low_ram = False
        t5 = pipe.t5
        stub = StubPipe()
        e1 = pipe._encode_prompt(stub, "a prompt")
        assert pipe.t5 is t5, "default mode must keep T5 resident"
        e2 = pipe._encode_prompt(stub, "a prompt")
        assert stub.encode_calls == 1 and e2 is e1
