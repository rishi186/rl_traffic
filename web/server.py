"""FastAPI backend for the RL Traffic web application.

Provides:
    - REST API for experiments, metrics, models, and evaluations
    - WebSocket for live training metric streaming
    - Serves the built React frontend
    - Endpoints to launch/monitor training jobs

Run:
    python web/server.py
    # or
    uvicorn web.server:app --reload --port 8000
"""

import sys
import json
import asyncio
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RL Traffic Control",
    description="Web dashboard for RL-based traffic signal optimization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Live training state shared with WebSocket clients
_live_state: Dict[str, Any] = {
    "is_training": False,
    "current_episode": 0,
    "total_episodes": 0,
    "metrics_history": [],
    "latest_metrics": {},
    "training_log": [],
}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    config_path: str = "config.yaml"
    overrides: Optional[Dict[str, Any]] = None
    algorithm: str = "dqn"


class ExperimentInfo(BaseModel):
    name: str
    path: str
    created: str
    has_models: bool
    has_metrics: bool
    has_tensorboard: bool


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _get_experiments() -> List[dict]:
    """Scan results directory for experiment runs."""
    experiments = []
    if not RESULTS_DIR.exists():
        return experiments

    for entry in sorted(RESULTS_DIR.iterdir(), key=lambda e: e.stat().st_mtime if e.exists() else 0, reverse=True):
        if not entry.is_dir():
            continue
        if entry.name in ("__pycache__",):
            continue

        info = {
            "name": entry.name,
            "path": str(entry),
            "created": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(),
            "has_models": (entry / "models").exists(),
            "has_metrics": (entry / "training_metrics.json").exists(),
            "has_tensorboard": (entry / "tensorboard").exists(),
        }

        # Load metrics summary if available
        metrics_path = entry / "training_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                if metrics:
                    info["num_episodes"] = len(metrics)
                    info["best_reward"] = max(m.get("reward", 0) for m in metrics)
                    info["last_reward"] = metrics[-1].get("reward", 0)
                    info["avg_reward"] = sum(m.get("reward", 0) for m in metrics) / len(metrics)
            except Exception:
                info["num_episodes"] = 0

        # List models
        models_dir = entry / "models"
        if models_dir.exists():
            info["models"] = [m.name for m in models_dir.iterdir() if m.is_file()]

        experiments.append(info)

    return experiments


def _safe_path(base: Path, user_input: str) -> Path:
    """Join user input to base path, preventing path traversal."""
    resolved = (base / user_input).resolve()
    if not str(resolved).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved


def _get_experiment_metrics(exp_name: str) -> dict:
    """Load training metrics for a specific experiment."""
    exp_dir = _safe_path(RESULTS_DIR, exp_name)
    metrics_path = exp_dir / "training_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"No metrics found for experiment '{exp_name}'")

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(status_code=422, detail=f"Corrupted metrics file for '{exp_name}'")

    waiting_times = [m.get("avg_waiting_time") for m in metrics if m.get("avg_waiting_time") is not None]
    queue_lengths = [m.get("avg_queue") for m in metrics if m.get("avg_queue") is not None]

    return {
        "experiment": exp_name,
        "num_episodes": len(metrics),
        "metrics": metrics,
        "summary": {
            "best_reward": max(m.get("reward", 0) for m in metrics) if metrics else 0,
            "avg_reward": sum(m.get("reward", 0) for m in metrics) / len(metrics) if metrics else 0,
            "last_reward": metrics[-1].get("reward", 0) if metrics else 0,
            "best_waiting": min(waiting_times) if waiting_times else 0,
            "best_queue": min(queue_lengths) if queue_lengths else 0,
        },
    }


def _get_project_info() -> dict:
    """Get project metadata for the showcase page."""
    try:
        config = _load_config()
    except Exception:
        config = {}
    training = config.get("training", {}) if isinstance(config, dict) else {}
    return {
        "name": "Deep Q-Network Traffic Signal Optimization",
        "description": "A production-grade DQN agent that autonomously optimises traffic signal phases across a multi-intersection urban grid using PyTorch, SUMO simulation, and the TraCI API.",
        "algorithm": training.get("algorithm", "dqn"),
        "total_episodes": training.get("total_episodes", 100),
        "features": [
            "Dueling Double DQN with Prioritized Experience Replay",
            "Soft target updates (Polyak averaging)",
            "Cosine annealing LR scheduler",
            "Gradient clipping",
            "Attention-based Q-Network",
            "Multi-agent parameter sharing",
            "Improved reward shaping (throughput, switch, congestion)",
            "Curriculum learning",
            "Early stopping",
            "WandB integration",
            "Hyperparameter sweep runner",
            "Simulation video recording",
        ],
        "tech_stack": ["PyTorch", "SUMO", "TraCI", "Gymnasium", "FastAPI", "React"],
        "config": config,
    }


# ---------------------------------------------------------------------------
# REST API routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/project")
async def project_info():
    """Get project metadata for showcase page."""
    return _get_project_info()


@app.get("/api/experiments")
async def list_experiments():
    """List all experiment runs."""
    return _get_experiments()


@app.get("/api/experiments/{exp_name}")
async def get_experiment(exp_name: str):
    """Get details and metrics for a specific experiment."""
    # Prevent path traversal
    if "/" in exp_name or "\\" in exp_name or ".." in exp_name:
        raise HTTPException(status_code=400, detail="Invalid experiment name")
    return _get_experiment_metrics(exp_name)


@app.get("/api/experiments/{exp_name}/compare/{other_name}")
async def compare_experiments(exp_name: str, other_name: str):
    """Compare two experiments side by side."""
    for name in (exp_name, other_name):
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=400, detail="Invalid experiment name")
    exp1 = _get_experiment_metrics(exp_name)
    exp2 = _get_experiment_metrics(other_name)
    return {"experiment_a": exp1, "experiment_b": exp2}


@app.get("/api/config")
async def get_config():
    """Get the current config.yaml."""
    return _load_config()


@app.put("/api/config")
async def update_config(config: dict):
    """Update config.yaml with validation."""
    # Validate config using the project's validator
    try:
        from src.utils.config_validator import ConfigValidator
        validator = ConfigValidator()
        validator.validate(config)
    except ImportError:
        pass  # Validator not available, skip validation
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Config validation failed: {str(e)}")

    config_path = CONFIG_PATH
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return {"status": "updated", "config": config}


@app.get("/api/models")
async def list_models():
    """List all saved models across experiments."""
    models = []
    for exp in _get_experiments():
        if "models" in exp:
            for model_name in exp["models"]:
                models.append({
                    "name": model_name,
                    "experiment": exp["name"],
                    "path": f"{exp['path']}/models/{model_name}",
                })
    return models


@app.get("/api/live-state")
async def get_live_state():
    """Get current live training state."""
    return _live_state


@app.post("/api/train")
async def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Start a training run in the background."""
    if _live_state["is_training"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    _live_state["is_training"] = True
    _live_state["current_episode"] = 0
    _live_state["metrics_history"] = []
    _live_state["latest_metrics"] = {}
    _live_state["training_log"] = []

    background_tasks.add_task(_run_training, req)
    return {"status": "started", "message": "Training started in background"}


@app.post("/api/train/stop")
async def stop_training():
    """Request training to stop."""
    _live_state["is_training"] = False
    return {"status": "stopping"}


# ---------------------------------------------------------------------------
# Training background task
# ---------------------------------------------------------------------------

def _run_training(req: TrainRequest):
    """Run training in a background thread, updating _live_state."""
    import subprocess

    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "train.py"),
           "--config", req.config_path]

    _live_state["training_log"].append(f"Starting training: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        for line in process.stdout:
            line = line.strip()
            if line:
                _live_state["training_log"].append(line)
                # Cap log at 500 lines to prevent unbounded memory growth
                if len(_live_state["training_log"]) > 500:
                    _live_state["training_log"] = _live_state["training_log"][-300:]
                # Try to parse episode info from log lines
                if "Episode" in line and "Reward=" in line:
                    try:
                        parts = line.split("|")
                        ep_part = parts[0].split()
                        ep_num = int(ep_part[1].split("/")[0])
                        _live_state["current_episode"] = ep_num

                        reward = float(parts[1].split("=")[1])
                        _live_state["latest_metrics"] = {
                            "episode": ep_num,
                            "reward": reward,
                            "timestamp": datetime.now().isoformat(),
                        }
                        _live_state["metrics_history"].append(_live_state["latest_metrics"])
                        # Cap metrics history at 1000 entries
                        if len(_live_state["metrics_history"]) > 1000:
                            _live_state["metrics_history"] = _live_state["metrics_history"][-500:]
                    except (IndexError, ValueError):
                        pass

            # Check if stop was requested
            if not _live_state["is_training"]:
                process.terminate()
                _live_state["training_log"].append("Training stopped by user")
                break

        process.wait()
        _live_state["training_log"].append(f"Training finished with exit code {process.returncode}")

    except Exception as e:
        _live_state["training_log"].append(f"Training error: {str(e)}")
    finally:
        _live_state["is_training"] = False


# ---------------------------------------------------------------------------
# WebSocket for live updates
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        # Send current state on connect
        await ws.send_json({"type": "state", "data": _live_state})

        while True:
            # Poll and broadcast live state every 2 seconds
            await asyncio.sleep(2)
            await ws.send_json({"type": "state", "data": _live_state})
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ---------------------------------------------------------------------------
# Serve React frontend (production)
# ---------------------------------------------------------------------------

FRONTEND_BUILD = PROJECT_ROOT / "web" / "frontend" / "dist"

if FRONTEND_BUILD.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_BUILD / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        # Prevent path traversal
        file_path = (FRONTEND_BUILD / full_path).resolve()
        if not str(file_path).startswith(str(FRONTEND_BUILD.resolve())):
            raise HTTPException(status_code=400, detail="Invalid path")
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(FRONTEND_BUILD / "index.html"))
else:
    @app.get("/")
    async def dev_notice():
        return {
            "message": "Frontend not built. Run `cd web/frontend && npm install && npm run dev`",
            "api_docs": "/docs",
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.server:app", host="0.0.0.0", port=8000, reload=True)
