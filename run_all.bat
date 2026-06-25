@echo off
if not defined SUMO_HOME (
    set "SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo"
    echo Set SUMO_HOME to default: C:\Program Files ^(x86^)\Eclipse\Sumo
)

echo.
echo === 1. Running Fixed-Time Baseline ===
.venv\Scripts\python scripts\baseline_fixed.py --config config.yaml --episodes 5

echo.
echo === 2. Training DQN Agent ===
.venv\Scripts\python scripts\train.py --config config.yaml

echo.
echo === 3. Evaluating Trained Agent ===
.venv\Scripts\python scripts\evaluate.py --config config.yaml --model results\models\best_model.pth --episodes 5

echo.
echo === 4. Evaluating Generalization ===
.venv\Scripts\python scripts\evaluate_generalization.py --config config.yaml --model results\models\best_model.pth --mode runtime

echo.
echo === 5. Generating Performance Report ===
.venv\Scripts\python scripts\generate_report.py --results-dir results

echo.
echo === Pipeline Completed! Check results\report.md ===
