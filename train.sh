#!/bin/bash
# train.sh - Run training for all 6 experiments (Task 1)
# Cleans old plots & checkpoints each run, keeps CSV history

MODELS=("lenet" "resnet18" "vit")
DATASETS=("MNIST" "CIFAR10")
EPOCHS=50
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

# === Clean old plots and checkpoints ===
rm -rf ${PLOTS_DIR}/*
rm -rf ${MODELS_DIR}/*

# === Get current run ID from CSV ===
if [ -f "$CSV_PATH" ]; then
  LAST_RUN_ID=$(tail -n +2 "$CSV_PATH" | awk -F',' '{print $1}' | sort -n | tail -1)
  if [ -z "$LAST_RUN_ID" ]; then
    RUN_ID=1
  else
    RUN_ID=$((LAST_RUN_ID + 1))
  fi
else
  RUN_ID=1
fi

EXP_NUM=1
TOTAL_EXPS=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))

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
echo ">>> All experiments complete!"
echo ">>> Results saved in: $CSV_PATH"
echo ">>> Plots saved in:   $PLOTS_DIR/"
echo ">>> Models saved in:  $MODELS_DIR/"
echo "====================================="
