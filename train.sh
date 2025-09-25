#!/bin/bash
# train.sh - Run training for all 6 experiments (10 epochs for testing speed)
# Each experiment produces 2 plots (Loss + Accuracy)

MODELS=("lenet" "resnet18" "vit")
DATASETS=("MNIST" "CIFAR10")
EPOCHS=10   # quick run for testing
BATCH_SIZE=64
LR=0.001

EXP_NUM=1
TOTAL_EXPS=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "====================================="
    echo ">>> Experiment $EXP_NUM of $TOTAL_EXPS"
    echo ">>> Training $model on $dataset for $EPOCHS epochs"
    echo ">>> Started at: $(date)"
    echo "====================================="

    python train.py \
      --model $model \
      --dataset $dataset \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --lr $LR

    echo ">>> Finished Experiment $EXP_NUM of $TOTAL_EXPS"
    echo ""
    ((EXP_NUM++))
  done
done
