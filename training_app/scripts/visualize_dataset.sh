#!/bin/bash
# Quick dataset visualization script - shows dataset samples without training

cd /home/alec/git/pokemon/training_app && \
source venv/bin/activate && \
python -m src.training.train_yolo_obb \
  --synthetic-data configs/yolo_obb_dataset.yaml \
  --handcrafted-data data/gold/gold_dataset.yaml \
  --merged-dataset-dir data/merged_dataset \
  --project trading_cards_obb \
  --name dataset_visualization \
  --examples-only \
  --num-examples 16 \
  --show-examples

echo ""
echo "Visualizations saved to:"
echo "  - Source datasets: data/merged_dataset/source_datasets_visualization/"
echo "  - Merged dataset: data/merged_dataset/merged_dataset_visualization/"
echo "  - Augmented examples: trading_cards_obb/dataset_visualization/debug_visualizations/"

