param(
    [ValidateSet("smoke", "pilot", "publication")][string]$Profile = "smoke",
    [string]$OutputDir = "results/campaign-$(Get-Date -Format yyyyMMdd-HHmmss)"
)
$ErrorActionPreference = "Stop"
& .venv/Scripts/python.exe scripts/run_experiments.py --config config.yaml --benchmark-profile $Profile --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Benchmark failed with exit code $LASTEXITCODE" }
