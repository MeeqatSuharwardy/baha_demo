import os
import random
import numpy as np
import PIL.Image
import spaces
import torch
from diffusers import AutoencoderKL, DiffusionPipeline

USE_TORCH_COMPILE = True
ENABLE_CPU_OFFLOAD = False
ENABLE_REFINER = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)
    if ENABLE_REFINER:
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
        if ENABLE_REFINER:
            refiner.enable_model_cpu_offload()
    else:
        pipe.to(device)
        if ENABLE_REFINER:
            refiner.to(device)

    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        if ENABLE_REFINER:
            refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
@spaces.GPU
def generate(
    prompt: str,
    negative_prompt: str = "Aspect Ratio 1:1",
    seed: int = 785245150,
    width: int = 1024,
    height: int = 1024,
    guidance_scale_base: float = 11.7,
    guidance_scale_refiner: float = 5.0,
    num_inference_steps_base: int = 25,
    num_inference_steps_refiner: int = 20,
    apply_refiner: bool = True,
) -> PIL.Image.Image:

    generator = torch.Generator().manual_seed(seed)
    if not apply_refiner:
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale_base,
            num_inference_steps=num_inference_steps_base,
            generator=generator,
            output_type="pil",
        ).images[0]
    else:
        latents = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale_base,
            num_inference_steps=num_inference_steps_base,
            generator=generator,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale_refiner,
            num_inference_steps=num_inference_steps_refiner,
            image=latents,
            generator=generator,
        ).images[0]
        return image