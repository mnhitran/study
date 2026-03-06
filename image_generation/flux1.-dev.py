# https://huggingface.co/black-forest-labs/FLUX.1-dev

import torch
from diffusers import FluxPipeline
import warnings
from transformers import logging as transformers_logging
import re
import os
import random 

# Settings
device = "cuda"
num_runs = 4  # Number of runs per prompt
output_dir = "path/to/output_directory" # adjust
prompt_file = "path/to/prompt_list.txt" # adjust accordingly

# Suppress unnecessary warnings
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message="You set 'add_prefix_space'.")

# Sanitize filename
def sanitize_filename(prompt: str, maxlen=60) -> str:
    prompt = re.sub(r'[^\w\s-]', '', prompt)  # remove special characters
    prompt = re.sub(r'\s+', '_', prompt)      # replace spaces with underscores
    return prompt[:maxlen]

# Load the FLUX.1-dev pipeline
print("Loading FLUX.1‑dev pipeline…")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.to(device)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load prompts
with open(prompt_file, "r") as f:
    prompts = [
        line.strip() for line in f
        if line.strip() and not line.startswith("#")
    ]

# Generate images with multiple runs
for i, prompt in enumerate(prompts, start=1):
    print(f"[{i}/{len(prompts)}] Generating for prompt: '{prompt}'")
    cleaned = sanitize_filename(prompt)

    for run in range(1, num_runs + 1):
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device).manual_seed(seed)
        print(f"  Run {run} using seed: {seed}")

        image = pipe(
            prompt=prompt, 
            negative_prompt="text, letters, captions, signs, writings, speech bubbles",
            height=512, 
            width=512,
            guidance_scale=4.5, # default=3.5
            num_inference_steps=35, # default=50
            max_sequence_length=128,
            generator=generator
        ).images[0]

        filename = f"{i:03d}_{cleaned}_flux_{run}.png"
        save_path = os.path.join(output_dir, filename)
        image.save(save_path)

        print(f"Saved: {save_path}")

print("FLUX.1‑dev multiple run generation completed!")