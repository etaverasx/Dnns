#!/bin/bash
# train2.sh - Run Task 2 experiments (20 total)
# Results -> reports/task2/task2-results.csv
# Plots   -> reports/task2/plots-task2
# Models  -> reports/task2/models-task2

# === Task 2 directories ===
TASK2_DIR="reports/task2"
PLOTS_DIR="$TASK2_DIR/plots-task2"
MODELS_DIR="$TASK2_DIR/models-task2"
CSV_PATH="$TASK2_DIR/task2-results.csv"

mkdir -p "$PLOTS_DIR"
mkdir -p "$MODELS_DIR"

# === Experiment configs ===
EPOCHS=20
LR1=0.001
LR2=0.01
BS1=32
BS2=128

# Models: only LeNet+MNIST and ResNet18+CIFAR10
declare -A MODELS
MODELS["lenet"]="MNIST"
MODELS["resnet18"]="CIFAR10"

echo "====================================="
echo ">>> Task 2 experiments starting"
echo "Results will be stored in: $CSV_PATH"
echo "Plots will be stored in:   $PLOTS_DIR/"
echo "Models will be stored in:  $MODELS_DIR/"
echo "====================================="

EXP_NUM=1

for model in "${!MODELS[@]}"; do
  dataset=${MODELS[$model]}

  echo "-------------------------------------"
  echo ">>> Running Task 2 for $model on $dataset"
  echo "-------------------------------------"

  # 1. Rotation (with vs without)
  for aug in rotation none; do
    python train_task2.py \
      --model $model --dataset $dataset \
      --epochs $EPOCHS --batch_size 64 --lr $LR1 \
      --optimizer adam --augment $aug \
      --exp_name "exp${EXP_NUM}_${model}_${dataset}_rotation_${aug}"
    ((EXP_NUM++))
  done

  # 2. Horizontal Flip (with vs without)
  for aug in flip none; do
    python train_task2.py \
      --model $model --dataset $dataset \
      --epochs $EPOCHS --batch_size 64 --lr $LR1 \
      --optimizer adam --augment $aug \
      --exp_name "exp${EXP_NUM}_${model}_${dataset}_flip_${aug}"
    ((EXP_NUM++))
  done

  # 3. Optimizer (Adam vs SGD)
  for opt in adam sgd; do
    python train_task2.py \
      --model $model --dataset $dataset \
      --epochs $EPOCHS --batch_size 64 --lr $LR1 \
      --optimizer $opt --augment none \
      --exp_name "exp${EXP_NUM}_${model}_${dataset}_${opt}"
    ((EXP_NUM++))
  done

  # 4. Batch size (small vs large)
  for bs in $BS1 $BS2; do
    python train_task2.py \
      --model $model --dataset $dataset \
      --epochs $EPOCHS --batch_size $bs --lr $LR1 \
      --optimizer adam --augment none \
      --exp_name "exp${EXP_NUM}_${model}_${dataset}_bs${bs}"
    ((EXP_NUM++))
  done

  # 5. Learning rate (low vs high)
  for lr in $LR1 $LR2; do
    python train_task2.py \
      --model $model --dataset $dataset \
      --epochs $EPOCHS --batch_size 64 --lr $lr \
      --optimizer adam --augment none \
      --exp_name "exp${EXP_NUM}_${model}_${dataset}_lr${lr}"
    ((EXP_NUM++))
  done

done

echo "====================================="
echo ">>> Task 2 experiments complete!"
echo ">>> Results saved in: $CSV_PATH"
echo ">>> Plots saved in:   $PLOTS_DIR/"
echo ">>> Models saved in:  $MODELS_DIR/"
echo "====================================="
