#!/bin/bash
# train.sh - Run Task 1 experiments (LeNet, ResNet18, ViT on MNIST and CIFAR10)
# Cleans old plots & checkpoints before each run
# Keeps results in a CSV file for easy analysis

MODELS=("lenet" "resnet18" "vit")   # Models to train
DATASETS=("MNIST" "CIFAR10")       # Datasets to test on
EPOCHS=50                          # Longer training for Task 1
BATCH_SIZE=64
LR=0.001

# === Task 1 directories ===
TASK1_DIR="reports/task1"
PLOTS_DIR="$TASK1_DIR/plots"
MODELS_DIR="$TASK1_DIR/models/checkpoints"
CSV_PATH="$TASK1_DIR/results.csv"

# === Ensure directories exist ===
mkdir -p "$PLOTS_DIR"
mkdir -p "$MODELS_DIR"

# === Clean old plots and checkpoints (but keep CSV history) ===
rm -rf ${PLOTS_DIR}/*
rm -rf ${MODELS_DIR}/*

# === Run ID handling (to track across multiple runs) ===
if [ -f "$CSV_PATH" ]; then
  # Get last run ID from the CSV (skip header, grab first column)
  LAST_RUN_ID=$(tail -n +2 "$CSV_PATH" | awk -F',' '{print $1}' | sort -n | tail -1)
  if [ -z "$LAST_RUN_ID" ]; then
    RUN_ID=1
  else
    RUN_ID=$((LAST_RUN_ID + 1))
  fi
else
  RUN_ID=1
fi

# === Count total experiments ===
EXP_NUM=1
TOTAL_EXPS=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))

# === Main experiment loop ===
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "====================================="
    echo ">>> Run ID: $RUN_ID | Experiment $EXP_NUM of $TOTAL_EXPS"
    echo ">>> Training $model on $dataset for $EPOCHS epochs"
    echo ">>> Started at: $(date)"
    echo "====================================="

    python train.py \
      --model $model \
      --dataset $dataset \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --lr $LR \
      --exp_num $EXP_NUM \
      --run_id $RUN_ID

    echo ">>> Finished Experiment $EXP_NUM of $TOTAL_EXPS"
    echo ""
    ((EXP_NUM++))
  done
done

echo "====================================="
echo ">>> All Task 1 experiments complete!"
echo ">>> Results saved in: $CSV_PATH"
echo ">>> Plots saved in:   $PLOTS_DIR/"
echo ">>> Models saved in:  $MODELS_DIR/"
echo "====================================="
