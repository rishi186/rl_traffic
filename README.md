# Multi-Agent RL Traffic Signal Control

## Setup

1. **Install dependencies:**
pip install -r requirements.txt


2. **Set SUMO_HOME environment variable:**


## Training
python scripts/train.py --config config.yaml

## Evaluation
python scripts/evaluate.py --config config.yaml --model results/models/best_model.pth --gui


## Monitoring
View training progress with TensorBoard:

tensorboard --logdir results/tensorboard
