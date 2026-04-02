#!/bin/bash
#
# Batch processing: Run inference on multiple videos using pass 1 warped noise
#

# Run from the void root directory regardless of where this script is called from
cd "$(dirname "$0")/.."

# ==================== CONFIGURATION ====================

# Your trained model checkpoint (update this to your checkpoint path)
MODEL_CHECKPOINT="<PATH_TO_YOUR_CHECKPOINT>/transformer/diffusion_pytorch_model.safetensors"

# Base model
MODEL_NAME="./CogVideoX-Fun-V1.5-5b-InP"

# Data directories
DATA_ROOTDIR="<PATH_TO_YOUR_DATA_ROOTDIR>"
PASS1_DIR="<PATH_TO_YOUR_PASS1_OUTPUT_DIR>"
OUTPUT_DIR="./pass2_output"
WARPED_NOISE_CACHE="./pass1_warped_noise_cache"
GWF_DIR="../Go-with-the-Flow"

# Video settings
TEMPORAL_WINDOW_SIZE=85
HEIGHT=384
WIDTH=672
SEED=42
GUIDANCE_SCALE=6.0
NUM_INFERENCE_STEPS=50

# ==================== VIDEO LIST ====================
# Add video names here (one per line)
VIDEO_NAMES=(
    # Add your video names here (must match folder names in DATA_ROOTDIR)
    # "my-video-1"
    # "my-video-2"
)

# ==================== PROCESS VIDEOS ====================

echo "=========================================="
echo "Batch Inference with Pass 1 Warped Noise"
echo "=========================================="
echo "Model: $MODEL_CHECKPOINT"
echo "Videos to process: ${#VIDEO_NAMES[@]}"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Process all videos in a single run (model loaded only once!)
python inference/cogvideox_fun/inference_with_pass1_warped_noise.py \
    --video_names "${VIDEO_NAMES[@]}" \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --model_name "$MODEL_NAME" \
    --data_rootdir "$DATA_ROOTDIR" \
    --pass1_dir "$PASS1_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --warped_noise_cache_dir "$WARPED_NOISE_CACHE" \
    --gwf_dir "$GWF_DIR" \
    --temporal_window_size $TEMPORAL_WINDOW_SIZE \
    --height $HEIGHT \
    --width $WIDTH \
    --seed $SEED \
    --guidance_scale $GUIDANCE_SCALE \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --use_quadmask

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "BATCH PROCESSING COMPLETE"
    echo "=========================================="
    echo "✓ Successfully processed ${#VIDEO_NAMES[@]} videos"
    echo "Output: $OUTPUT_DIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "BATCH PROCESSING FAILED"
    echo "=========================================="
    echo "✗ Check logs above for details"
    echo "=========================================="
    exit 1
fi
