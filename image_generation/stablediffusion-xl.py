# https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
# https://huggingface.co/docs/diffusers/v0.24.0/en/api/pipelines/stable_diffusion/stable_diffusion_xl#stable-diffusion-xl

from diffusers import DiffusionPipeline
import torch
import os
import re

# Load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True
)
base.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", 
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
refiner.to("cuda")

# Set output directory
output_dir = "path/to/output_directory" # adjust
os.makedirs(output_dir, exist_ok=True)

# Load prompts from file
prompt_file = "path/to/prompt_list.txt" # adjust
with open(prompt_file, "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Denoising step division
n_steps = 50 # default=4.0
high_noise_frac = 0.8

# CFG Scale (default = 7.0)
guidance_scale = 5.0 

# Function to sanitize filenames
def safe_filename(text, maxlen=60):
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s]+', '_', text)
    return text.strip()

# Number of runs per prompt
num_runs = 4

# Generate and save each image, multiple runs
for i, prompt in enumerate(prompts, start=1):
    print(f"[{i}/{len(prompts)}] Generating images for: '{prompt}'")

    # Extend prompt for realistic images
    prompt_extended = f"{prompt}, realistic"

    negative_prompt = "text, letters, writing, captions, distorted, NSFW"

    for run in range(1, num_runs + 1):
        print(f"  Run {run} of {num_runs}")

        # Base stage
        latents = base(
            prompt=prompt_extended,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            guidance_scale=guidance_scale,
            output_type="latent",
            width=512, 
            height=512
        ).images

        # Refiner stage
        image = refiner(
            prompt=prompt_extended,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            guidance_scale=guidance_scale,
            image=latents
        ).images[0]

        # construct filename: 001_Prompt_Text_1.png, etc.
        cleaned = safe_filename(prompt)
        filename = f"{i:03d}_{cleaned}_SDXL_{run}.png"
        filepath = os.path.join(output_dir, filename)

        image.save(filepath)
        print(f"Saved image to: {filepath}")