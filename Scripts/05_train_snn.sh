#! /bin/bash

#SBATCH --job-name=snn_calibration # Job name
#SBATCH --output=outputs/snn_train_output.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH -p cscc-gpu-p
#SBATCH --time=12:00:00
#SBATCH -q cscc-gpu-qos


# List of models to train
MODELS=("resnet18")

# Dataset
DATASETS=("cifar100")

# Latency
LATENCY=4

# Number of reference models in the experiment
REF_MODELS=(4)

# Loop through each model and run the training script
for REF in "${REF_MODELS[@]}"; do
	for DATASET in "${DATASETS[@]}"; do
		for MODEL in "${MODELS[@]}"; do
			echo "Training $MODEL on $DATASET..."
			echo "Training for T = $LATENCY"
			python3 main_training_parallel.py --t $LATENCY --epochs 100 --dataset "$DATASET" --reference_models $REF
			echo  "Training finished"
		done
		echo "Finished training all models on $DATASET"
			echo "================================="
	done
	echo "All models trained successfully"
done
