# MaskToCorrect
## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/payelsantra/MaskToCorrect.git
   cd MaskToCorrect

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
conda create --name scorer python=3.8
conda activate scorer

# Install the scorer dependencies
pip install -r requirements_scorer.txt

