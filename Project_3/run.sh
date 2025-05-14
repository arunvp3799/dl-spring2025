#!/bin/bash

# Check if the correct number of arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: sh run_attacks.sh <attack> <model>"
    echo "Available attacks: basic, fsgm, pgd, patched_pgd, random_patch_pgd"
    echo "Available models: resnet34, densenet121"
    exit 1
fi

# Store arguments
ATTACK=$1
MODEL=$2

# Validate model input
if [ "$MODEL" != "resnet34" ] && [ "$MODEL" != "densenet121" ]; then
    echo "Error: Invalid model. Choose either 'resnet34' or 'densenet121'"
    exit 1
fi

# Run appropriate script based on attack type
case $ATTACK in
    basic)
        echo "Running basic model testing with $MODEL..."
        python src/basic.py --model $MODEL
        ;;
    fsgm)
        echo "Running FSGM attack with $MODEL..."
        python src/fsgm.py --model $MODEL
        ;;
    pgd)
        echo "Running PGD attack with $MODEL..."
        python src/pgd.py --model $MODEL
        ;;
    patched_pgd)
        echo "Running Patched PGD attack with $MODEL..."
        python src/patched_pgd.py --model $MODEL
        ;;
    random_patch_pgd)
        echo "Running Random Patch PGD attack with $MODEL..."
        python src/random_patch_pgd.py --model $MODEL
        ;;
    *)
        echo "Error: Invalid attack type. Choose from: basic, fsgm, pgd, patched_pgd, random_patch_pgd"
        exit 1
        ;;
esac

echo "Attack completed!"