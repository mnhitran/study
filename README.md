# Image Generation and Elo Rating Scripts

This repository contains the Python scripts used for image generation and Elo rating calculations in the study.

## Repository Structure

- image_generation/
  - kandinsky-3.py
  - flux1.-dev.py
  - stablediffusion-xl.py
  - prompts_list.txt
- elo_rating/
  - elo_rating_model.py
  - elo_rating_statement_value.py
  - results_final.csv

## Image Generation

The scripts in image_generation/ generate images using three text-to-image models based on the prompts provided in prompts_list.txt.

- kandinsky-3.py
- flux.1-dev.py 
- stablediffusion-xl.py

The scripts are based on HuggingFace model implementations and include model-specific generation settings such as guidance scale, inference steps, image resolution, and negative prompts. 
Each script loads one model, generates images from the prompts, and saves the outputs to disk.

## Elo Rating Calculations

The scripts in elo_rating/ compute Elo ratings based on pairwise comparison data from the LimeSurvey study using the provided results-final.csv file.

- elo_rating_model.py computes one overall Elo ranking across all items
- elo_rating_statement_and_value.py computes Elo ratings per statement and per value

## Requirements

Python 3 with:

- torch
- diffusers
- transformers
- pandas
- matplotlib

Install with:

```bash
pip install torch diffusers transformers pandas matplotlib
```
## Data

The repository includes the following input files:

- image_generation/prompt_list.txt - prompts used for image generation
- elo_rating/results-survey591572.csv - LimeSurvey results used for the Elo rating calculations
- 
## Note

These scripts were developed for the experiments reported in the accompanying research study.
Further details on prompts, model settings, and the evaluation procedure are described there.
