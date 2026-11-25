# Upgrade evidence ledger

All work is local on `upgrade/reproducible-benchmark`. No push or publication.

## M1 — base and simulator

Fetched `origin/master`: cached `267f647` advanced to `0424647`. Original clean
`master` at `eca43db` is preserved. Branch starts at fetched `0424647`.
Ignored `results/` (including old checkpoints), `.venv/`, `graphify-out/`,
`.devin/`, and frontend `node_modules/` and `dist/` were present at audit.
Downloads checkout was not accessed. No existing artifacts were removed.

Read config, environment, DQN/PPO, train, fixed baseline, evaluation,
generalization, density generation, report, four plot scripts, server, Demo,
tests and CI. Existing tests use many formula/stub tests rather than TraCI.

Commands (from repository root, PowerShell):

```
git status --short --branch
git remote -v
git status --short --ignored
git fetch origin
git switch -c upgrade/reproducible-benchmark origin/master
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m flake8 src scripts tests --max-line-length=120 --ignore=E402,W503,E501
netconvert --node-files tests/fixtures/tiny/nodes.nod.xml --edge-files tests/fixtures/tiny/edges.edg.xml --output-file tests/fixtures/tiny/tiny.net.xml --no-turnarounds true
.venv/Scripts/python.exe -m pytest tests -q
```

Original `eca43db`: 71 passed (16.88 s); lint failed with existing warnings.
Fetched `0424647`: 71 passed (6.88 s); lint clean.
Added fixture: 72 passed (8.94 s), lint clean. Real SUMO started, advanced
10 seconds and closed; TraCI confirmed unloaded. Network generated successfully.
Changed: tiny fixture XML, integration test, pytest marker, engineering contract.

Actual tools: Windows Python 3.14.5, torch 2.12.0, numpy 2.4.6, traci and SUMO
1.26.0, gymnasium 1.3.0, pytest 9.0.3, flake8 7.3.0, Node 24.15.0,
npm 11.12.1, Git 2.53.0.windows.2. Full Python snapshot in
`environment-windows.txt`. FastAPI/uvicorn were missing and installed for API verification.

VERDICT: PASS

## Decisions before M2–M5

- Replace the environment lifecycle/step and action state machine, retaining the
  observation features and agent metric helpers. Evidence: baseline uses ticks,
  RL uses decisions; old action code never completes its own yellow transition.
- Keep legacy reward options but label incoming-count and waiting snapshots as
  proxies. Use network per-tick counts and SUMO TripInfo for evaluation.
- Share one episode runner across normal entry points, train validation and
  benchmark. Reject runtime vehicle deletion/cloning for paired experiments.
- Fix PPO behavior data and separate bootstrap/trace masks. Preserve compatible
  network architecture, require schema checks for benchmark checkpoints.
- Use SUMO actuated as rule baseline: original incoming count is not movement
  pressure; no unsupported max-pressure claim.

Risk: medium (simulation and learning semantics); within declared task scope.

## M2 — simulation semantics

Changed `config.yaml`, environment lifecycle, `timing.py`, `episode_metrics.py`,
shared `src/evaluation.py`, fixed/evaluate entry points, provenance helpers and
`test_episode_semantics.py`. All policies use one-second SUMO ticks, the same
warm-up and explicit `sumo.num_seconds` measured window. Legacy training
decision count no longer caps the simulator. Final partial decisions are kept.
Sorted signals/lanes and versioned observation schema are recorded. RL holds
greens, enforces min/max and traverses yellow/all-red; all policies receive the
same derived clearance plan. Requested and applied requests are logged.
Runtime density deletion/cloning is rejected.

Network arrivals are accumulated once each tick, including separate warm-up
and measured-window totals. Queue is a deduplicated controlled-lane count;
network stopped vehicle-seconds use all vehicle speed subscriptions at 1 Hz.
Completed trip waiting/duration and tails come from TripInfo, with censored
unfinished trips separate. Empty completed samples remain null, never zero.

```
.venv/Scripts/python.exe -m pytest tests/test_episode_semantics.py -q
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m flake8 src scripts tests --max-line-length=120 --ignore=E402,W503,E501
```

6 new semantic tests passed, including fixed/actuated/RL real SUMO, 103-second
partial final interval, accounting identity, stable ordering, deterministic
replays and lawful holds/clearances. Full suite: 78 passed (13.38 s); lint clean
after formatting fixes. Original-network 15-second measured window plus
5-second warm-up ran in 3.73 s (83 signals, 20 scheduled/inserted, zero completed).
This was a startup measurement, not a performance result.

Source SUMO programs warned about missing yellow and one never-green movement.
Derived benchmark plans now yellow every released green before all-red. The
source-network never-green movement remains a documented scenario limitation;
this task does not claim to calibrate or repair the road network.

VERDICT: PASS

## M3 — learning and checkpoint semantics

Changed `scripts/train.py`, DQN/PPO agents, provenance, shared runner and
`test_training_semantics.py`. PPO now stores same-step behavior data and true
next observations. GAE bootstraps truncations, cuts traces at both boundaries,
and masks natural terminal bootstrap. Singleton normalization is finite.
DQN replay stores natural terminal masks, preserving truncation bootstrap.
Only verified one-step replay is enabled in benchmark training; legacy
multi-agent n-step and ID-embedding configurations fail explicitly.

Exact resume includes optimizer/scheduler, replay, RNG, episode, metrics and
validation-best state; it checks source, Python/packages, SUMO and scenario
hashes. Weights-only restart is distinct. Checkpoint sidecars stamp observation
schema and hashes; legacy schema-less loading requires explicit acknowledgement.
Runs isolate outputs and save config, provenance and exact dirty-source archive.
Selection uses validation scenarios/seeds and network stopped vehicle-seconds,
not training reward. Training proxy returns are labeled.

```
.venv/Scripts/python.exe -m pytest tests/test_training_semantics.py -q
.venv/Scripts/python.exe -m flake8 src scripts tests --max-line-length=120 --ignore=E402,W503,E501
```

8 regression tests passed (22.32 s), including real SUMO exact-resume equality
for DQN and PPO. Lint clean at this stage. The first DQN resume test failed:
cosine T_max depended on episode stopping budget. Fixed with explicit
`lr_scheduler_total_steps`; the same reproducer then passed. A concurrent full
suite was invalidated by source changes (resume guard correctly rejected a
changed source digest); final full suite will be run without concurrent edits.

VERDICT: PASS


## Campaign correction before longer evaluation

`test_clearance_advance_is_not_an_additional_applied_switch` reproduced a bug:
the yellow-to-all-red advance was recorded/penalized as an additional applied
switch (expected 0, got 1). `_advance_phase` now marks a switch only when leaving
green. Same real-SUMO reproducer passed (1 passed, 1.44 s). Full suite then passed
90 tests (40.22 s). Targeted semantics/paired-data/API checks after normalization
schema updates: 11 passed (11.33 s). One upstream Starlette/httpx deprecation
warning is recorded; it is not a failed request or simulator check.

The first local campaign `results/benchmark-pilot-v1/` was interrupted before
longer evaluation and retained as diagnostic data. It is excluded from the final
pilot. Corrected campaign uses `results/benchmark-pilot-v2/`; no prior outputs
were deleted or overwritten. The corrected DQN validation scores observed so
far are 526 and 482 stopped vehicle-seconds (not evaluation claims).

VERDICT: PASS

## Telemetry verification and final evaluation protocol

The first two full-telemetry pilot runs took 33.55 and 36.74 s. Per-tick lane
and signal queries were replaced with SUMO subscriptions without changing policy
or reward queries. A new real-SUMO regression compares every subscribed queue,
network stopped count and signal state against direct queries. All eight semantic
tests at that point passed (5.47 s). A 300-second original-network reference
reproduced 3,062 stopped vehicle-seconds and 37/148 completed trips exactly,
reducing runtime to 14.18 s. The partial slow evaluation in
`benchmark-pilot-v2/evaluation/` remains preserved and excluded.

The corrected two-episode training runs remain in `benchmark-pilot-v2/dqn-42`
and `ppo-42`: 48 DQN gradient-update steps and 2 PPO rollout updates; every recorded
loss/update statistic was checked finite. Validation scores were DQN [526, 482],
PPO [437, 429], and both selected episode 2. Validation uses light demand and
seed 7001; training uses normal demand/seeds 42,43. Evaluation uses 101–105.
The subscription optimization changes metric retrieval, not actions/rewards;
checkpoint source archives preserve their exact training implementation.

Final smoke command:

```powershell
.venv/Scripts/python.exe scripts/evaluate_generalization.py --config results/benchmark-pilot-v2/campaign-config.yaml --manifest results/benchmark-pilot-v2/demand/manifest.json --dqn-model results/benchmark-pilot-v2/dqn-42/models/best_model.pth --ppo-model results/benchmark-pilot-v2/ppo-42/models/best_model.pth --seeds 101 --scenarios normal --horizon 30 --output-dir results/benchmark-final-smoke --label 'Smoke only - subscription telemetry'
.venv/Scripts/python.exe scripts/verify_benchmark.py --results-dir results/benchmark-final-smoke
```

Four real runs validated against TripInfo, per-tick timeline, schema and matched
summary. In-process replay API returned HTTP 200 and 30 synchronized frames.
Smoke has zero completed trips; it supports pipeline startup, not effectiveness.

Final pilot command (completed successfully):

```powershell
.venv/Scripts/python.exe scripts/evaluate_generalization.py --config results/benchmark-pilot-v2/campaign-config.yaml --manifest results/benchmark-pilot-v2/demand/manifest.json --dqn-model results/benchmark-pilot-v2/dqn-42/models/best_model.pth --ppo-model results/benchmark-pilot-v2/ppo-42/models/best_model.pth --seeds 101 102 103 104 105 --scenarios light normal heavy --horizon 300 --output-dir results/benchmark-final-pilot --label 'Pilot: 2 training episodes, 1 training seed'
```

Demand files: light 148, normal 296, heavy 444 routed vehicles before 305 s,
plus a held-out rush-hour file (296) reserved for a future publication campaign.
No runtime demand deletion/cloning. Source warnings remain visible in logs.

## M4 — paired benchmark completed

Changed: `src/benchmark.py`, `scripts/generate_density_configs.py`,
`evaluate_generalization.py`, `generate_report.py`, `run_experiments.py`,
`verify_benchmark.py`, comparison/generalization/training/signal plot scripts,
Windows run wrappers, paired-data regression tests, and compact artifacts in
`docs/benchmark/pilot/`. Complete source/input/model provenance and all per-run
TripInfo/replays remain in the isolated local results directories.

Final smoke: 4 real policy runs at 30 seconds, all matched; validator passed.
Final pilot: **60/60 runs**, light/normal/heavy, seeds 101–105, fixed/actuated/DQN/PPO,
300 measured seconds plus 5 warm-up seconds. Each group has all 5 matched seeds;
there are no missing comparisons. All raw rows independently cross-checked with
TripInfo, per-second stopped/queue totals, schema, input hashes and derived summary.

```
.venv/Scripts/python.exe scripts/generate_report.py --results-dir results/benchmark-final-pilot
.venv/Scripts/python.exe scripts/verify_benchmark.py --results-dir results/benchmark-final-pilot
```

Validator: `Validated 60 real runs against TripInfo, timeline, schema and matched summary`.
Raw CSV, manifest, machine summary, two plots, generated report, compact paired
replay and derivation hashes were copied unchanged to `docs/benchmark/pilot/`.
Plots were opened and visually inspected as local image artifacts.

Mean stopped vehicle-seconds:

| Demand | Fixed | Actuated | DQN | PPO |
|---|---:|---:|---:|---:|
| Light | 3038.8 | 2430.4 | 2833.0 | 2696.6 |
| Normal | 5281.8 | 4538.2 | 5086.4 | 4877.6 |
| Heavy | 7884.0 | 7324.2 | 7837.0 | 7581.4 |

Actuated has the lowest observed mean in all three demands. DQN/PPO versus fixed
95% paired intervals cross zero in normal and heavy demand. Heavy DQN difference:
-47 veh-s, interval [-1179.2, 1096.2]; no reliable benefit established. Mean group
completion ranges from 21.8% to 27.6%, making completed-only waits insufficient
for effectiveness claims. Every outcome and tail metric is retained.

Measured wall time per final pilot evaluation (startup/telemetry included):
mean **16.5699 s**, median **15.4111 s**, minimum **13.0291 s**, maximum **30.7513 s**;
sum **994.1913 s** across 60 runs. The full publication protocol was not executed.
It would require 600 training episodes plus validation and a larger evaluation
matrix; convergence and training-population uncertainty remain unverified.

Exact publication-candidate command:

```powershell
.venv/Scripts/python.exe scripts/run_experiments.py --config config.yaml --benchmark-profile publication --output-dir results/publication-v1
```

This includes independent training seeds 42/43/44, 100 episodes each for DQN/PPO,
5000 measured seconds, explicit light/normal/heavy/held-out rush-hour demand and
five matched evaluation seeds. It creates new outputs and does not publish.

SUMO diagnostics observed in heavy fixed-time evaluation included emergency
braking and a red-light emergency stop for `v000059` on `362206700#11_0` at 272 s.
These are recorded as limitations, not hidden or interpreted as a safety result.
The source never-green movement and uncalibrated network remain unresolved
scenario limitations. No field or road-safety claim is supported.

VERDICT: PASS

## M5 — genuine replay and final verification

Changed: `web/server.py`, Demo, Dashboard, Config, Experiments and Landing pages,
README, requirements and CI. The existing FastAPI/React stack is retained.
Demo loads actual paired artifacts, uses one shared per-second index for both
panels, signals, queues and chart cursor, and displays final completion/TripInfo
metrics and provenance separately. Missing/mismatched data never generates
placeholder metrics. API checks raw/manifest hashes, matching identities,
synchronized clocks, per-tick totals and raw metrics. Tests exercise real saved
pilot data and reject corruption. Legacy wait/queue labels are explicit proxies.

CI retains the upstream frontend build job and now installs SUMO and runs real
integration checks with an executable preflight to prevent a missing-SUMO skip.
GitHub Actions itself was not executed (no push requested).

Final commands after code changes:

```
.venv/Scripts/python.exe -m pytest tests -q
.venv/Scripts/python.exe -m flake8 src scripts tests web/server.py --max-line-length=120 --ignore=E402,W503,E501
git diff --check
cd web/frontend
npm.cmd run build
```

Results: **95 passed, zero failures/skips, 40.42 s**. Lint clean; diff check passed.
One installed Starlette/httpx deprecation warning remains. Vite build succeeded
in **10.06 s**, with a 630.12 kB JS chunk warning; no build error. `npm ls --depth=0`
passed and reported React 18.3.1 / Vite 5.4.21. No UI screenshot was fabricated.

Browser visual QA is **UNVERIFIED**: browser approval rejected opening
`http://127.0.0.1:8010/demo` because origin/network permission was declined.
No alternate browser or workaround was attempted. The preview server was stopped.
In-process API verification and frontend build do not establish visual UI quality.

All changes remain local and uncommitted on `upgrade/reproducible-benchmark`,
based on fetched `0424647`. Original `master` remains at `eca43db`. Old checkpoint
files retain their pre-task timestamps (for example, `results/models/best_model.pth`
is dated 2026-06-24); no old result/checkpoint path was overwritten. No Downloads
checkout access and no push/publication were performed.

VERDICT: PASS

## GitHub review and simulation performance

The public default branch `master` was at `0424647` when checked on
2026-11-25. GitHub showed a successful CI run for that commit and its README
still presented the old 14% waiting-time claim. The local branch contains
the Tirupati default and a measured replay. The README quick start now
runs from a checkout on any Windows user account. CI selects unit and
integration tests separately to run each test once per Python version.

The environment now reads lane counts, speeds and waits plus vehicle class
and wait from TraCI subscriptions. The subscriptions deliver these values
with each simulator step. Commands that change traffic signals are unchanged.

Measured on Windows with SUMO 1.26.0 and seed 101. Three runs per map used
alternating hold and switch requests with the timeline enabled. The table
reports median wall time after simulator reset, excluding startup. The
before and after traces matched by SHA-256 across all six repetitions.
The trace included observations, rewards, termination flags, actions,
signal and queue timelines and TripInfo-derived summaries.

| Map | Signals | Simulated seconds | Before | After |
|---|---:|---:|---:|---:|
| Tirupati | 10 | 300 | 6.08 s | 2.46 s |
| Bengaluru | 83 | 120 | 10.00 s | 3.43 s |

Including startup, median wall time went from 9.00 s to 6.02 s for Tirupati
and from 12.13 s to 6.10 s for Bengaluru. Fixed and actuated Tirupati traces
also matched in one run per policy. A separate 600-second evaluation reused
the four Tirupati Demo policies, demand and seed. All four raw metric rows
and per-second replays matched the original Demo recording except for
measured runtime. The saved recording remained in place.

```powershell
.venv/Scripts/python.exe -m pytest tests -m 'not integration' -q
.venv/Scripts/python.exe -m pytest tests -m integration -q
.venv/Scripts/python.exe -m flake8 src scripts tests web/server.py --max-line-length=120 --ignore=E402,W503,E501
npm.cmd --prefix web/frontend run build
.venv/Scripts/python.exe scripts/verify_benchmark.py --results-dir results/tirupati-optimized-check/evaluation
git diff --check
```

The Docker daemon was unavailable. Docker execution and the updated GitHub
Actions workflow were outside the local verification.
