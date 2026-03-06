# https://huggingface.co/kandinsky-community/kandinsky-3
# https://huggingface.co/docs/diffusers/v0.24.0/api/pipelines/kandinsky3

from diffusers import AutoPipelineForText2Image
import torch
import re
import os
import random

# Load the pipeline once
pipe = AutoPipelineForText2Image.from_pretrained(
    "kandinsky-community/kandinsky-3",
    variant="fp16",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Read prompts
prompt_file = "path/to/prompt_list.txt" # adjust accordingly
with open(prompt_file, 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Output base directory
base_output_dir = "path/to/output_directory" # adjust
os.makedirs(base_output_dir, exist_ok=True)

# Function to sanitize filenames
def clean_filename(text, maxlen=60):
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'\s+', '_', text)
    return text[:maxlen]

# negative prompt
negative_prompt = "text, letters, writing, captions, signs"

# === Loop over 4 runs ===
for run_number in range(1, 5):
    print(f"\n🎯 Starting run {run_number}/4...\n")

    for i, prompt in enumerate(prompts):
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25, # default=50
            guidance_scale=4.0, # default=3.0
            generator=generator
        ).images[0]

        cleaned = clean_filename(prompt)
        filename = f"{i+1:03d}_{cleaned}_kandinsky_{run_number}.png"
        save_path = os.path.join(base_output_dir, filename)

        image.save(save_path)
        print(f"Saved: {save_path}")