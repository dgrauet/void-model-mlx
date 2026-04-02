from absl import app
from absl import flags
from ml_collections import config_flags
from loguru import logger
import json
import os
import sys
import glob
import cv2
import pprint
import numpy as np
import mediapy as media
import torch
from diffusers import (CogVideoXDDIMScheduler, DDIMScheduler,
                       DPMSolverMultistepScheduler,
                       EulerAncestralDiscreteScheduler, EulerDiscreteScheduler,
                       PNDMScheduler)
from PIL import Image

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from videox_fun.models import (AutoencoderKLCogVideoX,
                              CogVideoXTransformer3DModel, T5EncoderModel,
                              T5Tokenizer)
from videox_fun.pipeline import (CogVideoXFunPipeline,
                                CogVideoXFunInpaintPipeline)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora, create_network
from videox_fun.utils.fp8_optimization import convert_weight_dtype_wrapper
from videox_fun.utils.utils import get_video_mask_input, save_videos_grid, save_inout_row, get_video_mask_validation
from videox_fun.dist import set_multi_gpus_devices

if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config", "config/quadmask_cogvideox.py", "Path to the python config file"
    )

def load_pipeline(config):
    model_name = config.video_model.model_name
    weight_dtype = config.system.weight_dtype
    device = set_multi_gpus_devices(config.system.ulysses_degree, config.system.ring_degree)
 
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_name,
        subfolder="transformer",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float8_e4m3fn if config.system.gpu_memory_mode == "model_cpu_offload_and_qfloat8" else weight_dtype,
        use_vae_mask=config.video_model.use_vae_mask,
        stack_mask=config.video_model.stack_mask,
    ).to(weight_dtype)

    if config.video_model.transformer_path:
        logger.info(f"Load transformer from checkpoint: {config.video_model.transformer_path}")
        if config.video_model.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(config.video_model.transformer_path)
        else:
            state_dict = torch.load(config.video_model.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        param_name = 'patch_embed.proj.weight'

        if (
            (config.video_model.use_vae_mask or config.video_model.stack_mask) and
            state_dict[param_name].size(1) != transformer.state_dict()[param_name].size(1)
        ):
            logger.info('patch_embed.proj.weight size does not match the custom transformer ' +
                  f'{config.video_model.transformer_path}')
            latent_ch = 16
            feat_scale = 8
            feat_dim = int(latent_ch * feat_scale)
            old_total_dim = state_dict[param_name].size(1)
            new_total_dim = transformer.state_dict()[param_name].size(1)

            # Start with transformer's current pretrained weights (like training does)
            # Then overwrite certain channels with checkpoint weights
            new_weight = transformer.state_dict()[param_name].clone()
            # Overwrite first and last feat_dim channels with checkpoint weights
            new_weight[:, :feat_dim] = state_dict[param_name][:, :feat_dim]
            new_weight[:, -feat_dim:] = state_dict[param_name][:, -feat_dim:]
            # Middle channels keep the base pretrained weights
            state_dict[param_name] = new_weight
            logger.info(f'Adapted {param_name} from {old_total_dim} to {new_total_dim} channels (preserving base model middle channels)')

        m, u = transformer.load_state_dict(state_dict, strict=False)
        logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Vae
    vae = AutoencoderKLCogVideoX.from_pretrained(
        model_name,
        subfolder="vae"
    ).to(weight_dtype)

    if config.video_model.vae_path:
        logger.info(f"Load VAE from checkpoint: {config.video_model.vae_path}")
        if config.video_model.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(config.video_model.vae_path)
        else:
            state_dict = torch.load(config.video_model.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        logger.info(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get tokenizer and text_encoder
    tokenizer = T5Tokenizer.from_pretrained(
        model_name, subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        model_name, subfolder="text_encoder", torch_dtype=weight_dtype
    )
    Choosen_Scheduler = scheduler_dict = {
        "Euler": EulerDiscreteScheduler,
        "Euler A": EulerAncestralDiscreteScheduler,
        "DPM++": DPMSolverMultistepScheduler,
        "PNDM": PNDMScheduler,
        "DDIM_Cog": CogVideoXDDIMScheduler,
        "DDIM_Origin": DDIMScheduler,
    }[config.video_model.sampler_name]
    scheduler = Choosen_Scheduler.from_pretrained(
        model_name,
        subfolder="scheduler"
    )

    # load pipeline
    if transformer.config.in_channels != vae.config.latent_channels:
        pipeline = CogVideoXFunInpaintPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )
    else:
        pipeline = CogVideoXFunPipeline(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
        )

    if config.system.ulysses_degree > 1 or config.system.ring_degree > 1:
        transformer.enable_multi_gpus_inference()

    if config.system.gpu_memory_mode == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload(device=device)
    elif config.system.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif config.system.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    generator = torch.Generator(device=device).manual_seed(config.system.seed)

    if config.video_model.lora_path:
        print(f"DEBUG: About to load LoRA from: {config.video_model.lora_path}")
        print(f"DEBUG: LoRA weight multiplier: {config.video_model.lora_weight}")
        pipeline = merge_lora(pipeline, config.video_model.lora_path, config.video_model.lora_weight, device=device)
        print("LORA MERGED and loaded and set and ready to go")
    else:
        print("DEBUG: No LoRA path specified, running without LoRA")

    return pipeline, vae, generator


def run_inference(config, pipeline, vae, generator, input_video_name, keep_fg_ids=[-1]):
    save_video_name = f'{input_video_name}-fg=' + '_'.join([f'{i:02d}' for i in keep_fg_ids])
    if (config.experiment.skip_if_exists and
        sorted(list(glob.glob(os.path.join(config.experiment.save_path, f"{save_video_name}*.mp4"))))):
        logger.debug(f"Skipping {save_video_name} as it already exists")
        return

    video_length = config.data.max_video_length
    video_length = int((video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length != 1 else 1
    logger.debug(f'Video length: {video_length}')
    sample_size = config.data.sample_size
    sample_size = (int(sample_size.split('x')[0]), int(sample_size.split('x')[1]))
    if not config.experiment.validation:
        input_video, input_video_mask, prompt, _= get_video_mask_input(
            input_video_name,
            sample_size=sample_size,
            keep_fg_ids=keep_fg_ids,
            max_video_length=video_length,
            temporal_window_size=config.video_model.temporal_window_size,
            data_rootdir=config.data.data_rootdir,
            use_trimask=config.video_model.use_trimask,
            use_quadmask=config.video_model.use_quadmask,
            dilate_width=config.data.dilate_width,
        )
    else:
        input_video, input_video_mask, prompt = get_video_mask_validation(
            input_video_name,
            sample_size=sample_size,
            max_video_length=video_length,
            temporal_window_size=config.video_model.temporal_window_size,
            data_rootdir=config.data.data_rootdir,
            use_trimask=True,#config.video_model.use_trimask,
            dilate_width=config.data.dilate_width,
        )

    # vae experiment
    if config.experiment.skip_unet:
        if config.experiment.mask_to_vae:
            input_video = input_video_mask.repeat(1, 3, 1, 1, 1)

    with torch.no_grad():
        sample = pipeline(
            prompt,
            num_frames = config.video_model.temporal_window_size,
            negative_prompt = config.video_model.negative_prompt,
            height      = sample_size[0],
            width       = sample_size[1],
            generator   = generator,
            guidance_scale = config.video_model.guidance_scale,
            num_inference_steps = 30,
            video       = input_video,
            mask_video  = input_video_mask,
            strength    = config.video_model.denoise_strength,
            use_trimask = True, #config.video_model.use_trimask,
            zero_out_mask_region = config.video_model.zero_out_mask_region,
            skip_unet = config.experiment.skip_unet,
            use_vae_mask = config.video_model.use_vae_mask,
            stack_mask = config.video_model.stack_mask,
        ).videos

    if not os.path.exists(config.experiment.save_path):
        os.makedirs(config.experiment.save_path, exist_ok=True)

    index = len([path for path in os.listdir(config.experiment.save_path) if path.endswith('_tuple.mp4') and path.startswith(save_video_name)]) + 1
    prefix = save_video_name + f'-{index:04d}'

    if video_length == 1:
        save_sample_path = os.path.join(config.experiment.save_path, prefix + f".png")

        image = sample[0, :, 0]
        image = image.transpose(0, 1).transpose(1, 2)
        image = (image * 255).numpy().astype(np.uint8)
        image = Image.fromarray(image)
        image.save(save_sample_path)
    else:
        video_path = os.path.join(config.experiment.save_path, prefix + ".mp4")
        save_videos_grid(sample, video_path, fps=config.data.fps)
        save_inout_row(input_video, input_video_mask, sample, video_path[:-4] + "_tuple.mp4", fps=config.data.fps)


def main(_):
    config = FLAGS.config
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(config.to_dict())

    all_seqs_in_dir = sorted(os.listdir(config.data.data_rootdir))
    run_seqs = []
    if '/' in config.experiment.run_seqs:
        run_part, total_parts = config.experiment.run_seqs.split('/')
        run_part = int(run_part)
        total_parts = int(total_parts)
        n_per_part = len(all_seqs_in_dir) // total_parts
        part_start = (run_part - 1) * n_per_part
        part_end = min(run_part * n_per_part, len(all_seqs_in_dir))
        run_seqs = all_seqs_in_dir[part_start:part_end]
    else:
        run_seqs = config.experiment.run_seqs.split(',')
        run_seqs = [seq for seq in run_seqs if seq in all_seqs_in_dir]

    seq_fg_to_run = []
    if not config.experiment.validation:
        for seq in run_seqs:
            fg_ids = [-1]
            num_fgs = len(sorted(list(glob.glob(os.path.join(config.data.data_rootdir, seq, "quadmask_*.mp4")))))
            if num_fgs == 0:
                num_fgs = len(sorted(list(glob.glob(os.path.join(config.data.data_rootdir, seq, "mask_*.mp4")))))
            assert num_fgs > 0
            if config.experiment.matting_mode == "solo" and num_fgs > 1:
                fg_ids.extend(list(range(num_fgs)))
            for fg_id in fg_ids:
                seq_fg_to_run.append((seq, [fg_id]))
    else:
        # read training videos and random mask generation
        seq_fg_to_run = [(seq, [-1]) for seq in run_seqs]

    pipeline, vae, generator = load_pipeline(config)

    for seq_name, fg_id in seq_fg_to_run:
        logger.info(f'Sequence to run: {seq_name}, fgs to keep: {fg_id}')
        def _run_inference():
            run_inference(
                config=config,
                pipeline=pipeline,
                vae=vae,
                generator=generator,
                input_video_name=seq_name,
                keep_fg_ids=fg_id,
            )

        if config.system.allow_skipping_error:
            try:
                _run_inference()
            except Exception as e:
                logger.info(f'Error in {seq_name}, {fg_id}: {e}')
                continue
        else:
            _run_inference()


if __name__ == "__main__":
    app.run(main)