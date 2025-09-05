# MaskToCorrect
## Installation

   ```bash
   git clone https://github.com/payelsantra/MaskToCorrect.git
   cd MaskToCorrect
```
## Environment Setup (using Conda)

We recommend creating a separate conda environment to avoid dependency conflicts.

```bash
# Create a new conda environment with Python 3.9
conda create --name masktocorrect_env python=3.9

# Activate the environment
conda activate masktocorrect_env

# Install the main dependencies
pip install -r requirements_main.txt

# Create and activate the scorer environment
conda create --name scorer python=3.9
conda activate scorer

# Install the scorer dependencies
pip install -r requirements_scorer.txt

```
## Running the Main Script

After completing the installation steps and activating the environment, you can run the main functionality of MaskToCorrect with:

```bash
conda activate masktocorrect_env

python main.py \
  --input_file <path_to_input_file> \
  --retriever <retriever_type> \
  --shots <number_of_shots> \
  --output_file <path_to_output_file> \
  --model <model_name_or_path> \
  --masker <masking_strategy>
