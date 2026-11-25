# Tirupati traffic signal benchmark

Compare **fixed time, SUMO actuated, PyTorch DQN and PyTorch PPO** on a modeled
road network in central Tirupati, India. The FastAPI and React demo replays
600 seconds of a matched fixed-time and DQN simulation.

[Run the web demo](#quick-start-windows) · [Open the map in SUMO](#open-the-tirupati-map) ·
[Run all four policies](#reproduce-smoke-pilot-or-publication-candidate) ·
[Read the Bengaluru pilot results](#measured-results-and-limits)

## Quick start (Windows)

Install Python, Node.js and Eclipse SUMO. Add SUMO's `bin` directory to PATH.
Run these commands in PowerShell from the repository root:

```powershell
python -m venv .venv  # skip if .venv already exists
.venv/Scripts/python.exe -m pip install -r requirements.txt
npm.cmd --prefix web/frontend ci
npm.cmd --prefix web/frontend run build
.venv/Scripts/python.exe -m uvicorn web.server:app --host 127.0.0.1 --port 8000
```

Open [Demo](http://127.0.0.1:8000/demo) and press **Play**. The bundled replay
uses synthetic traffic and SUMO-generated signals. Results describe the simulation.
The reference Windows runs used SUMO/TraCI 1.26.0, Python 3.14.5, PyTorch 2.12.0,
and Node 24.15.0. [Full Python environment](docs/benchmark/environment-windows.txt)

## Open the Tirupati map

New runs use central Tirupati by default. The scenario covers the railway station
and APSRTC bus station area with OpenStreetMap roads, ten SUMO-generated signal
controllers and synthetic passenger trips. See [scenario details and map attribution](maps/tirupati/README.md).

```powershell
sumo-gui -c maps/tirupati/tirupati.sumocfg
```

Press **Play** to start the built-in fixed-time programs. The Bengaluru scenario
is still available at `maps/sumo.sumocfg`. Its trained models use Bengaluru's
signal and lane IDs. Train fresh checkpoints for Tirupati.

## Verify before running experiments

```powershell
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m flake8 src scripts tests web/server.py --max-line-length=120 --ignore=E402,W503,E501
.venv/Scripts/python.exe -m pytest tests -m integration -v
```

The tiny one-junction fixture is checked in with netconvert input files. Integration
tests start a real simulator and verify shutdown. Tests also cover action timing,
network accounting, PPO behavior alignment and GAE, singleton updates, checkpoint
schemas, replay boundaries, missing matches and exact episode-boundary resume.
CI installs SUMO, executes integration checks and builds React with `npm ci`.
GitHub CI itself has not been run by this local upgrade.

## Reproduce smoke, pilot or publication candidate

Every output directory must be new. Existing results/checkpoints are never reused
or overwritten implicitly. These commands use Tirupati. To reproduce the
Bengaluru protocol, pass a copy of `config.yaml` whose `sumo.cfg_file` is
`maps/sumo.sumocfg`. Run from the repository root:

```powershell
# Tiny runtime budget on the existing road network: 1 training episode per learner,
# one normal demand/seed pair, four policies, 30 measured seconds.
.venv/Scripts/python.exe scripts/run_experiments.py --config config.yaml --benchmark-profile smoke --output-dir results/smoke-reproduction

# Actual pilot protocol: 2 training episodes per learner at 120 measured seconds,
# validation light demand/seed 7001; smoke first, then 3 demands x 5 seeds x 4 policies.
# Evaluation: 300 measured seconds + 5 warm-up seconds, one training seed (42).
.venv/Scripts/python.exe scripts/run_experiments.py --config config.yaml --benchmark-profile pilot --output-dir results/pilot-reproduction

# Exact publication-candidate command (not executed in this upgrade):
# DQN/PPO: 100 episodes x training seeds 42,43,44; 5000 measured seconds;
# evaluation: light/normal/heavy/held-out rush_hour x seeds 101..105.
.venv/Scripts/python.exe scripts/run_experiments.py --config config.yaml --benchmark-profile publication --output-dir results/publication-v1
```

`run_all.ps1` and `run_all.bat` call the same campaign entry point and default to
smoke. The larger campaign trains three seeds per learner and reports them
separately. Its bootstrap intervals describe variation across evaluation seeds.

To run the smoke campaign in a container, use `docker compose run --build --rm smoke`.
It writes to `results/docker-smoke`. Pass `--output-dir results/docker-smoke-2`
after `smoke` for a fresh run. Use `docker compose run --rm evaluate --help`
or `docker compose run --rm generalization --help` to see checkpoint and
manifest arguments.

Campaign directories contain explicit demand files and their manifest, isolated
DQN/PPO training runs, a smoke matrix, and (for pilot/publication) an `evaluation/`
bundle: `raw.csv`, `manifest.json`, `summary.json`, `paired-comparison.png`,
`report.md`, full TripInfo and per-second replay logs. Configs, seeds, Git revision,
actual versions, source/scenario/model hashes and exact dirty-source archives
(`source.zip`) are saved. Outputs carry the actual source digest even before commit.

The demand generator preserves vehicle types and complete named/inline routes,
sorts explicit departure schedules, and rejects unexpanded flows/distributions.
Light/normal/heavy use 0.5/1/1.5 times the selected base vehicles. Rush-hour shifts
normal demand into a central peak and is excluded from training/validation.

## Timing, control and metric contract

- `sumo.num_seconds` is the **measured window in simulated seconds** after
  `warmup_seconds`; all policies use step-length 1 second, the same warm-up,
  demand, seed and final partial decision. Legacy `max_steps_per_episode` does
  not stop the simulator. Fixed-window evaluation keeps sampling after demand
  clears. Natural completion and artificial time-limit truncation are distinct.
- Signals and incoming lanes are sorted. The checkpoint sidecar records lane and
  signal IDs, green-state order, feature normalization, padding, dimensions and
  action schema. Binary requests mean hold / next cyclic green; minimum green,
  maximum green and complete yellow/all-red sequences govern what is applied.
  Requested and applied switches are logged separately. All compared policies
  receive the same derived clearance plan. Fixed green durations use the source
  program clipped to min/max; actuated uses SUMO gap actuation on that plan.
- **Primary:** `network_stopped_vehicle_seconds`, the 1 Hz count of all network
  vehicles at speed <= 0.1 m/s integrated over the measured window. Warm-up is
  excluded. This is not a mean trip wait.
- `controlled_lane_queue_mean` averages the deduplicated incoming controlled-lane
  halting count at 1 Hz. It excludes other roads/internal lanes and is not a
  whole-network queue. Legacy lane-wait snapshots are vehicle-second sums across
  controlled lanes, not average vehicle waiting times.
- TripInfo supplies completed-trip waiting/duration means, waiting p95/max,
  completion fraction, unfinished and uninserted demand. Whole-trip statistics
  include warm-up exposure. Unfinished waiting is a censored lower bound.
  Scheduled demand counts departures before the episode end; future departures
  are outside this episode. Empty completed samples are null, never zero.
- Arrivals/departures are accumulated once per network tick. Reward components
  are **proxies**: lane queue, lane waiting snapshots, emergency current wait,
  incoming-count decrease, switch and density penalties. The legacy `pressure`
  reward is incoming vehicle count, not movement pressure. No max-pressure
  baseline is claimed.
- A comparison is keyed by scenario/input hashes, seed, timing, safety parameters,
  actual window length and observation schema. Improvements are null for missing
  matches or a zero baseline. Reports/plots regenerate from the same raw CSV;
  percentile bootstrap uses matched evaluation-seed differences (10,000 draws,
  RNG seed 42). Negative policy-minus-fixed differences favor the policy.

## Individual entry points and checkpoints

```powershell
.venv/Scripts/python.exe scripts/generate_density_configs.py --config config.yaml --output-dir results/demand-new
.venv/Scripts/python.exe scripts/train.py --config config.yaml --algorithm dqn --run-dir results/train-new
.venv/Scripts/python.exe scripts/baseline_fixed.py --config config.yaml --episodes 5 --output-dir results/fixed-new
.venv/Scripts/python.exe scripts/evaluate.py --config config.yaml --model results/train-new/models/best_model.pth --episodes 5 --output-dir results/eval-new
.venv/Scripts/python.exe scripts/generate_report.py --results-dir results/pilot-reproduction/evaluation
.venv/Scripts/python.exe src/visualization/plot_comparison.py --results-dir results/pilot-reproduction/evaluation
```

The campaign is the recommended paired entry point: individual aggregates must not
be combined across densities/horizons. The former `--mode runtime` generalization
path was replaced with explicit `--manifest`, `--dqn-model`, `--ppo-model`,
`--seeds`, `--scenarios`, and `--horizon` arguments; see `--help`.

Training selects `best_model.pth` on validation stopped vehicle-seconds. Pass
`--validation-scenarios` and `--validation-seeds` for explicit scenarios; standalone
training defaults to its training scenario with held-out validation seed 7001.
Use `--warm-start model.pth` for a **weights-only restart**. Legacy checkpoints
without a verifiable observation schema require `--allow-unverified`; this never
asserts compatibility. Different schemas are rejected even if tensor shapes match.

`--resume path/to/resume.pt` is an **exact episode-boundary resume** and requires
matching config, source/dependencies, Python/SUMO, training/validation input hashes,
and schema. It restores replay, optimizer/scheduler, RNG, counters and validation
selection state into a new run directory. Only the episode stopping budget and
output directory may change. Load only trusted local resume bundles. The explicit
DQN cosine schedule length stays fixed when the episode budget is extended.
Benchmark training uses one-step DQN; the unsafe legacy interleaved n-step collector
and ID-embedding configurations are rejected. The existing agents/network
architectures remain available for compatible weights; legacy optional training
orchestration features are not benchmark modes.

## Genuine replay

```powershell
.venv/Scripts/python.exe -m uvicorn web.server:app --host 127.0.0.1 --port 8000
```

The default replay in `docs/benchmark/tirupati/` compares fixed time and DQN
for one matched 600-second run with synthetic demand. Set
`$env:RL_TRAFFIC_BENCHMARK_DIR` to `docs/benchmark/pilot` to view the recorded
Bengaluru pilot, or to a new campaign's `evaluation` directory. Set it before
starting the server. Both policy panels, signal-state display, network stopped
counts, selected-signal queues and chart cursor share one per-second clock.
Final completion and TripInfo statistics are explicitly separated from the current
frame. Provenance and limitations are visible. Missing or mismatched artifacts
produce an error, never placeholder metrics. The API checks hashes, match IDs,
time alignment and timeline totals against raw data. This is artifact playback,
not an interactive live SUMO execution or a video.

## Measured results and limits

Completed Bengaluru pilot: 60 real SUMO evaluations (three demands, five matched seeds,
four policies), with 300 measured seconds plus five warm-up seconds per run.
Both learners trained for two 120-second episodes with one training seed.

Mean **network stopped vehicle-seconds** (lower is better):

| Demand | Fixed time | SUMO actuated | DQN | PPO |
|---|---:|---:|---:|---:|
| light | 3,038.8 | 2,430.4 | 2,833.0 | 2,696.6 |
| normal | 5,281.8 | 4,538.2 | 5,086.4 | 4,877.6 |
| heavy | 7,884.0 | 7,324.2 | 7,837.0 | 7,581.4 |

Actuated has the lowest mean in all three demands. Learner-versus-fixed paired
95% bootstrap intervals cross zero in normal and heavy demand, so those pilot
differences do not establish a reliable benefit. Group mean completion is only
21.8–27.6%; completed-trip waiting statistics alone are insufficient evidence.

Actual evaluation runtime: **16.57 seconds mean**, 15.41 seconds median,
13.03–30.75 seconds per run; 994.19 seconds summed evaluation runtime. Startup
and telemetry are included in per-run times. Full publication training was not
executed: its 600 training episodes plus validation represent a much larger
budget than this pilot. The exact publication command is above.

See [the generated report](docs/benchmark/pilot/report.md) for all outcomes,
waiting tails, queues and paired intervals, and [the verification ledger](docs/benchmark/verification.md)
for commands, source versions, tests and limitations. Full TripInfo/replays and
checkpoint/source archives remain under the isolated local campaign directories;
the compact report/raw-data/replay bundle is retained in `docs/benchmark/pilot/`.

The original network is uncalibrated. SUMO reports a missing-yellow source program
and one never-green source movement. Emergency-braking and red-light emergency-stop
warnings were also observed in heavy fixed-time evaluation. Derived common plans
supply legal clearance sequences but do not calibrate the road network or certify
road safety. No real-world effectiveness or generalized superiority is established.
Pilot uncertainty uses only five evaluation seeds and one training seed.

The replay API/data checks and frontend build passed. Browser origin approval
blocked visual interaction QA; no browser screenshot or visual verification is claimed.

## Semantics references

Implementation decisions follow the [SUMO traffic-light control API](https://sumo.dlr.de/docs/TraCI/Change_Traffic_Lights_State.html),
[SUMO TripInfo definitions](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html),
and [Gymnasium terminal/truncation bootstrap guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
The repository's executable tests and raw simulator artifacts supply verification.

Project code is distributed under the [MIT License](LICENSE). The Tirupati road
data uses the [Open Database License](maps/tirupati/README.md).
