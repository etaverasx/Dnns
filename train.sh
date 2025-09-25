#!/bin/bash
# train.sh - Run training for all 6 experiments
# Cleans old plots & checkpoints each run, keeps CSV history

MODELS=("lenet" "resnet18" "vit")
DATASETS=("MNIST" "CIFAR10")
EPOCHS=10    # for testing, can increase later
BATCH_SIZE=64
LR=0.001

# === Clean old plots and checkpoints ===
rm -rf reports/task1_plots/*
rm -rf reports/task1_models/checkpoints/*

# === Get current run ID from CSV ===
CSV_PATH="reports/task1_results.csv"
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
echo ">>> Results saved in: reports/task1_results.csv"
echo ">>> Plots saved in:   reports/task1_plots/"
echo ">>> Models saved in:  reports/task1_models/checkpoints/"
echo "====================================="