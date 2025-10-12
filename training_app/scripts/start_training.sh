#!/bin/bash

# EXTREME STRETCHING for small hand-crafted dataset (300 vs 60K images)
# Using 50x multiplier with pre-augmentation = 15,000 effective hand-crafted images
# This balances the dataset better: 60K synthetic + 15K hand-crafted

cd /home/alec/git/pokemon/training_app && \
source venv/bin/activate && \
nohup python -u -m src.training.train_yolo_obb \
  --synthetic-data configs/yolo_obb_dataset.yaml \
  --handcrafted-data data/gold/gold_dataset.yaml \
  --eval-handcrafted-ratio 0.9 \
  --handcrafted-aug-multiplier 50 \
  --apply-pre-augmentation \
  --merged-dataset-dir data/merged_dataset_50x \
  --model yolo11m-obb.pt \
  --epochs 50 \
  --imgsz 1088 \
  --batch 8 \
  --fraction 0.05 \
  --project trading_cards_obb \
  --name yolo11m_obb_50x_stretched \
  --aug-intensity medium \
  --show-examples \
  --patience 5 \
  --save-period 10 \
  --validation-interval 5 \
  --no-mlflow \
  > training.log 2>&1 &


# tail -f /home/alec/git/pokemon/training_app/training.log
# tail -100 /home/alec/git/pokemon/training_app/training.log

# pkill -f train_yolo_obb