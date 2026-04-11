"""Quadmask utilities for VOID.

VOID uses a 4-value quadmask encoding:
  0   = primary object to remove (black)
  63  = overlap region between primary and affected objects
  127 = affected/interaction region (objects that should react, e.g., fall)
  255 = background to preserve (white)
"""

import cv2
import numpy as np
from pathlib import Path


def adjust_frame_count(num_frames: int, temporal_compression: int = 4, patch_size_t: int = 2) -> int:
    """Adjust frame count to be compatible with the model architecture.

    The VAE compresses temporally by `temporal_compression` (4x), and
    the transformer patches temporally by `patch_size_t` (2x).
    The number of latent frames must be divisible by patch_size_t.

    Latent frames = (F - 1) // temporal_compression + 1
    This must be divisible by patch_size_t.

    Valid F values: 5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, ...

    Returns the largest valid F <= num_frames.
    """
    for f in range(num_frames, 0, -1):
        latent_f = (f - 1) // temporal_compression + 1
        if latent_f % patch_size_t == 0 and latent_f >= 2:
            return f
    return 5  # minimum valid


def load_quadmask_video(path: str, height: int, width: int, max_frames: int) -> np.ndarray:
    """Load a quadmask video and normalize to [0, 1] with 4 discrete values.

    Args:
        path: Path to quadmask video (mp4).
        height: Target height.
        width: Target width.
        max_frames: Maximum number of frames to load.

    Returns:
        (F, H, W, 1) float32 array with values in {0, 63/255, 127/255, 1.0}.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_NEAREST)
        frames.append(gray)
    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from {path}")

    # Pad or truncate to max_frames
    while len(frames) < max_frames:
        frames.append(frames[-1])
    frames = frames[:max_frames]

    mask = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return mask[..., None]  # (F, H, W, 1)


def load_video(path: str, height: int, width: int, max_frames: int) -> np.ndarray:
    """Load a video and resize to target dimensions.

    Args:
        path: Path to video file.
        height: Target height.
        width: Target width.
        max_frames: Maximum number of frames.

    Returns:
        (F, H, W, 3) float32 array in [0, 1].
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from {path}")

    while len(frames) < max_frames:
        frames.append(frames[-1])
    frames = frames[:max_frames]

    return np.stack(frames, axis=0).astype(np.float32) / 255.0


def load_sample(sample_dir: str, height: int = 384, width: int = 672, max_frames: int = 85):
    """Load a VOID sample (video + quadmask + prompt).

    Automatically adjusts max_frames to be compatible with the model
    (latent frame count must be divisible by patch_size_t=2).

    Args:
        sample_dir: Path to sample directory containing:
            - input_video.mp4
            - trimask_quadmask.mp4 (or quadmask_*.mp4)
            - prompt.json

    Returns:
        Tuple of (video, mask, prompt) where:
            video: (F, H, W, 3) float32 in [0, 1]
            mask: (F, H, W, 1) float32 quadmask
            prompt: str
    """
    import json
    sample_dir = Path(sample_dir)

    # Adjust frame count for model compatibility
    adjusted = adjust_frame_count(max_frames)
    if adjusted != max_frames:
        print(f"  Adjusted frames: {max_frames} -> {adjusted} (must produce even latent frames)")
    max_frames = adjusted

    # Load video
    video_path = sample_dir / "input_video.mp4"
    video = load_video(str(video_path), height, width, max_frames)

    # Load mask
    mask_path = sample_dir / "trimask_quadmask.mp4"
    if not mask_path.exists():
        # Try alternative naming
        mask_files = sorted(sample_dir.glob("quadmask_*.mp4"))
        if mask_files:
            mask_path = mask_files[0]
        else:
            mask_files = sorted(sample_dir.glob("mask_*.mp4"))
            if mask_files:
                mask_path = mask_files[0]
    mask = load_quadmask_video(str(mask_path), height, width, max_frames)

    # Load prompt
    prompt_file = sample_dir / "prompt.json"
    if prompt_file.exists():
        with open(prompt_file) as f:
            prompt_data = json.load(f)
        # VOID uses "bg" key for background description prompt
        prompt = prompt_data.get("prompt", prompt_data.get("bg", ""))
    else:
        prompt = ""

    return video, mask, prompt
