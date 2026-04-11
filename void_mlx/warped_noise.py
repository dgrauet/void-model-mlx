"""Warped noise generation for VOID pass 2.

Computes optical flow from the pass 1 output video, then warps
Gaussian noise using the flow to produce temporally coherent noise
for the refinement pass.

This is a simplified reimplementation of the Go-with-the-Flow approach
used in the original VOID model, using OpenCV instead of the `rp` library.
"""

import cv2
import numpy as np


def compute_optical_flow(video: np.ndarray) -> np.ndarray:
    """Compute dense optical flow between consecutive frames.

    Args:
        video: (F, H, W, 3) float32 in [0, 1].

    Returns:
        (F-1, H, W, 2) float32 optical flow (dx, dy per pixel).
    """
    F, H, W, _ = video.shape
    flows = []

    gray_prev = cv2.cvtColor((video[0] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    for i in range(1, F):
        gray_curr = cv2.cvtColor((video[i] * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow)
        gray_prev = gray_curr

    return np.stack(flows, axis=0)  # (F-1, H, W, 2)


def warp_image(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp an image using optical flow (backward warping).

    Args:
        image: (H, W, C) source image.
        flow: (H, W, 2) optical flow.

    Returns:
        (H, W, C) warped image.
    """
    H, W = flow.shape[:2]
    # Create coordinate grid
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    # Apply flow (backward warp: where did each pixel come from?)
    map_x = (grid_x + flow[:, :, 0]).astype(np.float32)
    map_y = (grid_y + flow[:, :, 1]).astype(np.float32)

    if image.ndim == 2:
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    else:
        channels = []
        for c in range(image.shape[2]):
            warped = cv2.remap(image[:, :, c], map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
            channels.append(warped)
        return np.stack(channels, axis=-1)


def generate_warped_noise(
    video: np.ndarray,
    num_latent_frames: int,
    latent_h: int,
    latent_w: int,
    latent_channels: int = 16,
    seed: int = 42,
) -> np.ndarray:
    """Generate temporally coherent warped noise from a video.

    The process:
    1. Compute optical flow between frames
    2. Start with random noise for frame 0
    3. For each subsequent frame, warp the previous noise using the flow
    4. Re-normalize to maintain Gaussian statistics
    5. Downsample to latent resolution

    Args:
        video: (F, H, W, 3) float32 in [0, 1] — the pass 1 output.
        num_latent_frames: Number of latent temporal frames.
        latent_h: Latent height.
        latent_w: Latent width.
        latent_channels: Number of latent channels (default 16).
        seed: Random seed.

    Returns:
        (num_latent_frames, latent_h, latent_w, latent_channels) float32 warped noise.
    """
    rng = np.random.RandomState(seed)
    F, H, W, _ = video.shape

    # Compute optical flow at video resolution
    print("  Computing optical flow...")
    flows = compute_optical_flow(video)

    # Scale flow to work at a higher resolution for better quality
    # then downsample the result
    noise_h = latent_h * 4  # work at 4x latent resolution
    noise_w = latent_w * 4

    # Resize flows to noise resolution
    scaled_flows = []
    for flow in flows:
        scaled = cv2.resize(flow, (noise_w, noise_h))
        # Scale flow values proportionally
        scaled[:, :, 0] *= noise_w / W
        scaled[:, :, 1] *= noise_h / H
        scaled_flows.append(scaled)

    # Temporal resampling: map F video frames to num_latent_frames
    frame_indices = np.linspace(0, F - 2, num_latent_frames).astype(int)

    # Generate warped noise
    print("  Warping noise...")
    noise_frames = []

    # Frame 0: random Gaussian noise
    noise = rng.randn(noise_h, noise_w, latent_channels).astype(np.float32)
    noise_frames.append(noise.copy())

    for i in range(1, num_latent_frames):
        # Get flow for this latent frame
        flow_idx = frame_indices[i]
        flow_idx = min(flow_idx, len(scaled_flows) - 1)
        flow = scaled_flows[flow_idx]

        # Warp previous noise using flow
        noise = warp_image(noise, flow)

        # Re-normalize to maintain Gaussian statistics
        # (warping + interpolation changes the distribution)
        noise = (noise - noise.mean()) / (noise.std() + 1e-8)

        noise_frames.append(noise.copy())

    warped = np.stack(noise_frames, axis=0)  # (T, noise_h, noise_w, C)

    # Downsample to latent resolution
    print("  Downsampling to latent resolution...")
    result = np.zeros((num_latent_frames, latent_h, latent_w, latent_channels),
                       dtype=np.float32)
    for t in range(num_latent_frames):
        for c in range(latent_channels):
            result[t, :, :, c] = cv2.resize(
                warped[t, :, :, c], (latent_w, latent_h),
                interpolation=cv2.INTER_AREA,  # area for downsampling
            )

    # Final normalization
    result = (result - result.mean()) / (result.std() + 1e-8)

    print(f"  Warped noise: shape={result.shape}, "
          f"mean={result.mean():.4f}, std={result.std():.4f}")

    return result
