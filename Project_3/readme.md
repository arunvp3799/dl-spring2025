# Adversarial Attack Testing Framework

This repository contains code to test various adversarial attacks against deep learning models for image classification. It supports ResNet34 and DenseNet121 architectures and includes multiple attack methods.

## Project Structure

```
.
├── predicted_samples/ 
├── data/                  # Data directory (you must add your dataset here)
├── src/                   # Source code directory
│   ├── data_utils.py      # Data loader utilities
│   ├── basic.py           # Basic model testing (no attack)
│   ├── fsgm.py            # Fast Sign Gradient Method attack
│   ├── pgd.py             # Projected Gradient Descent attack
│   ├── patched_pgd.py     # Patched PGD attack
│   └── random_patch_pgd.py # Random Patch PGD attack
├── requirements.txt       # Python dependencies
└── run.sh         # Bash script to run attacks
```

## Setup Instructions

### 1. Prepare the Dataset

Place your dataset in the `data/` directory. The data loader in `data_utils.py` expects the following structure:

```
data/
├── TestDataSet/
│   ├── Label1_Images/
│   ├── Label2_Images/
│   └── ...
```

### 2. Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running Attacks

You can run different attacks on the models using the provided bash script:

```bash
sh run_attacks.sh <attack> <model>
```

### Parameters:

- **attack**: The type of attack to run
  - `basic`: No attack, just evaluate model performance
  - `fsgm`: Fast Sign Gradient Method
  - `pgd`: Projected Gradient Descent
  - `patched_pgd`: Patched PGD attack
  - `random_patch_pgd`: Random Patch PGD attack

- **model**: The model architecture to use
  - `resnet34`: ResNet-34 architecture
  - `densenet121`: DenseNet-121 architecture

### Examples:

Test the basic performance of ResNet34 (no attack):
```bash
sh run_attacks.sh basic resnet34
```

Run FSGM attack on DenseNet121:
```bash
sh run_attacks.sh fsgm densenet121
```

Run PGD attack on ResNet34:
```bash
sh run_attacks.sh pgd resnet34
```

```
The Notebook is also present in src folder
```

## Understanding the Attacks

Each attack represents a different adversarial technique:

1. **Basic**: Not an attack, just evaluates the model's performance on clean images.

2. **FSGM (Fast Sign Gradient Method)**: A single-step attack that perturbs the input in the direction of the gradient of the loss with respect to the input.
   
   Think of this like taking a single step in the most harmful direction - similar to how a hiker might take one step in the steepest direction to quickly gain elevation.

3. **PGD (Projected Gradient Descent)**: An iterative version of FSGM that takes multiple small steps and projects the perturbation back onto a constraint set.
   
   This is like a hiker taking many careful steps, staying on a permitted path (the constraint set), but always moving in a harmful direction at each step.

4. **Patched PGD**: A variant of PGD that applies the perturbation only to specific patches of the image.
   
   Imagine placing sticky notes over parts of a stop sign - you're only modifying certain regions rather than the entire image.

5. **Random Patch PGD**: Similar to Patched PGD, but the patches are selected randomly.
   
   This is like randomly placing sticky notes on different parts of an image to find vulnerable spots.

## Troubleshooting

If you encounter issues:

1. Ensure your data is correctly organized in the data/ directory
2. Verify all dependencies are installed
3. Check that you're using the correct attack and model names
4. Make sure the bash script has execution permissions (`chmod +x run_attacks.sh`)

## Contributing

Feel free to add new attack methods by creating additional Python files in the src directory and updating the bash script accordingly.