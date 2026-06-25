# RL Traffic Web Application

A full-stack web app for the RL Traffic Signal Optimization project.

## Features

- **Landing Page** — Project showcase with architecture, features, and tech stack
- **Training Dashboard** — Live training metrics via WebSocket, reward curves, training logs
- **Experiment Tracker** — Browse past runs, view metrics, compare experiments
- **Interactive Demo** — Configure and run simulations with trained models
- **Config Editor** — Edit `config.yaml` from the browser with live toggles and sliders

## Architecture

```
web/
├── server.py              # FastAPI backend (REST + WebSocket)
└── frontend/
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    └── src/
        ├── App.jsx         # Main app with sidebar navigation
        ├── main.jsx        # React entry point
        ├── lib/
        │   ├── api.js      # API client + WebSocket helper
        │   └── utils.js    # Tailwind class merge utility
        └── pages/
            ├── Landing.jsx     # Project showcase
            ├── Dashboard.jsx   # Live training dashboard
            ├── Experiments.jsx # Experiment browser
            ├── Demo.jsx        # Interactive simulation
            └── Config.jsx      # Config editor
```

## Quick Start

### 1. Install Backend Dependencies

```bash
pip install fastapi uvicorn websockets
```

### 2. Install Frontend Dependencies

```bash
cd web/frontend
npm install
```

### 3. Run the Backend

```bash
# From project root
python web/server.py
# or
uvicorn web.server:app --reload --port 8000
```

### 4. Run the Frontend (Dev Mode)

```bash
cd web/frontend
npm run dev
```

Open http://localhost:5173 in your browser.

### 5. Build for Production

```bash
cd web/frontend
npm run build
```

The built frontend will be served by FastAPI at http://localhost:8000.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/project` | Project metadata |
| GET | `/api/experiments` | List all experiment runs |
| GET | `/api/experiments/{name}` | Get experiment details & metrics |
| GET | `/api/experiments/{a}/compare/{b}` | Compare two experiments |
| GET | `/api/config` | Get current config.yaml |
| PUT | `/api/config` | Update config.yaml |
| GET | `/api/models` | List all saved models |
| GET | `/api/live-state` | Get live training state |
| POST | `/api/train` | Start training in background |
| POST | `/api/train/stop` | Request training to stop |
| WS | `/ws` | WebSocket for live training updates |
