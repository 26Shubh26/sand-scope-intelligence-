# Offroad Semantic Segmentation - Hackathon Submission

This repository contains the complete solution for the Offroad Segmentation Hackathon. Our goal was to develop a robust model capable of segmenting complex offroad terrains into 10 distinct categories.

## 🏆 Key Achievements
- **Mean IoU:** 0.8910260052906603 (Validated on 317 samples)
- **Model:** DeepLabV3+ with ResNet-50 backbone
- **Efficiency:** Optimized for real-time inference on edge devices (tested on MPS/CPU)

## 📁 Repository Structure
- `model.py`: Model architecture definition.
- `dataset.py`: Optimized data pipeline with remapping and augmentations.
- `train.py`: Training orchestration.
- `evaluate.py`: Performance evaluation script (Mean IoU).
- `predict.py`: Inference script for visualization.
- `analyze_failures.py`: Automated failure case analysis tool.
- `REPORT.md`: **Detailed Performance Evaluation & Analysis Report.**
- `requirements.txt`: Environment setup.

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Evaluate Model**:
   ```bash
   python evaluate.py
   ```

3. **Run Inference**:
   ```bash
   python predict.py
   ```

## 📊 Performance Visualization
- **Loss Graphs**: See `loss_graph.png`.
- **Inference Sample**: See `output.png`.
- **Failure Analysis**: Check the `failure_analysis/` directory for detailed visualizations of model successes and failures.

## 📝 Methodology
We used a DeepLabV3+ architecture to handle the multi-scale nature of offroad terrains. The data was remapped from raw pixel values to 10 logical classes and augmented to handle varied lighting and scale. For a deep dive into our findings, please refer to [REPORT.md](REPORT.md).
