# Set SUMO_HOME if not already defined
if (-not $env:SUMO_HOME) {
    $env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
    Write-Host "Set SUMO_HOME to default: $env:SUMO_HOME" -ForegroundColor Yellow
}

# 1. Run Baseline
Write-Host "`n=== 1. Running Fixed-Time Baseline ===" -ForegroundColor Green
.venv\Scripts\python scripts/baseline_fixed.py --config config.yaml --episodes 5

# 2. Train Agent
Write-Host "`n=== 2. Training DQN Agent ===" -ForegroundColor Green
.venv\Scripts\python scripts/train.py --config config.yaml

# 3. Evaluate Agent
Write-Host "`n=== 3. Evaluating Trained Agent ===" -ForegroundColor Green
.venv\Scripts\python scripts/evaluate.py --config config.yaml --model results/models/best_model.pth --episodes 5

# 4. Evaluate Generalization
Write-Host "`n=== 4. Evaluating Generalization ===" -ForegroundColor Green
.venv\Scripts\python scripts/evaluate_generalization.py --config config.yaml --model results/models/best_model.pth --mode runtime

# 5. Generate Report
Write-Host "`n=== 5. Generating Performance Report ===" -ForegroundColor Green
.venv\Scripts\python scripts/generate_report.py --results-dir results

Write-Host "`n=== Pipeline Execution Completed! ===" -ForegroundColor Cyan
Write-Host "Check 'results/report.md' for the final results." -ForegroundColor Cyan
