# CPU Thermal-Event Prediction on Apple Silicon

This repository contains our team's take on the final project for our Statistical Machine Learning course.

A supervised learning pipeline that predicts, from telemetry alone, whether a Mac's CPU will reach a distress temperature within the next two minutes — before it happens, not after.

**Status:** 22 sessions collected, 45,328 labelled samples.

## Motivation

Consumer operating systems already expose rich CPU telemetry — die temperature, package power, core utilization, frequency — through APIs like `powermetrics` on macOS. Thermal throttling and shutdown protection typically react to temperature *after* it crosses a threshold. We asked a narrower, more useful question: **can the OS anticipate a thermal event early enough to act pre-emptively** (e.g. pausing a batch job, warning a user, or shifting load) using only signals it already has, with no additional sensors and no hardware access?

This is a binary forecasting problem, not a regression problem: we do not attempt to predict the exact future temperature curve, only whether a hot excursion is imminent. Prior work in this space has largely targeted HPC clusters. We adapt the same underlying idea to a laptop-class consumer chip and a binary early-warning framing.

## Task Definition

Telemetry is sampled every 5 seconds. A sample is labelled **1** if CPU die temperature reaches **75 °C** at any point in the following **two minutes** (24 steps), and **0** otherwise. The model is never asked "is it hot now?" — only "is it about to be?"

Two properties of the data drove every methodological choice:

- **Temperature is strongly autocorrelated.** A trivial rule that assumes "the next state looks like the current state" scores deceptively well at short horizons — a model can *appear* accurate while having learned nothing beyond inertia. To make this visible rather than hidden, every result is reported against a **persistence baseline**, and the prediction horizon was deliberately raised from 1 to 2 minutes after finding that at 1 minute the task was almost entirely solvable by inertia alone.
- **Positives arrive in bursts.** Thermal events cluster tightly around workload episodes, not individual samples. A naive row-level random train/test split would place adjacent samples from the same thermal episode on both sides of the split, leaking near-identical information across the boundary — a well-documented failure mode in time-series ML. Splits here are **blocked by session and chronological**: entire sessions are assigned wholesale to train, validation, or test.

## Why Apple Silicon Only (and Not Linux)

The pipeline currently targets macOS on Apple Silicon exclusively, via `powermetrics`. A Linux port was attempted and abandoned during the project: the sensor schema (`hwmon`/`lm-sensors`) matched closely enough to reuse the same feature code, but the *semantics* did not — die-level temperature readouts, effective thermal ceilings, and throttling behavior differ substantially by platform and were not comparable to the Apple Silicon threshold. 

## Data-Generating Process

Idle logging alone produces almost no positive labels — real machines rarely self-heat to 75 °C without sustained load. The dataset therefore had to be *manufactured* deliberately, not just observed.

**Telemetry logger** (`m_series_telemetry_logger_v3_sessionized.py`): records structured samples at a fixed interval via `powermetrics` (root required), writes one CSV per session, and updates a session registry.

**Stress generator** (`intermittent_stress_runner.py`): the logger only records — it does not produce heat. This companion script drives *repeated, bounded* approaches to the thermal ceiling rather than one long plateau, because the label depends on the *approach*, not the sustained state:

- **Burst** — synthetic CPU/memory load (120–180 s default), via `stress-ng` or a `yes`-process fallback.
- **Cool-down** — load released (120–180 s default), letting the machine shed heat.
- **Block** — 2–4 burst/cool-down cycles run back to back.
- **Gap** — 30–60 minutes between blocks, so each block is thermally independent of the last.

Logger and stress runner are run concurrently in separate terminals; the runner is always stopped *before* the logger, so the cool-down tail of the final burst is captured rather than truncated. Seeds and load mode (`cpu`, `cpu-vm`, `matrix`) were varied deliberately between sessions to avoid correlated, low-information repeats — identical schedules add far less information than their row count implies.

## Results

22 sessions, 45,328 labelled samples, 16.0% positive class. Session-blocked, chronological split: 15 / 3 / 4 sessions (train / validation / test).

| Test set (positive class) | Persistence baseline | Random forest |
|---|---|---|
| Precision | 0.948 | 0.976 |
| Recall | 0.411 | 0.607 |
| F1 | 0.573 | 0.750 |
| ROC-AUC | n/a | 0.910 |
| PR-AUC | n/a | 0.835 |

### Model comparison

All four model families converge to the same validation F1, within noise: logistic regression 0.787, random forest 0.791, gradient boosting 0.789, neural net 0.790 (spread: 0.005). This is an informative negative result — when a linear model matches a neural network on this task, the ceiling is set by the **features and the label definition**, not by model capacity. Further gains, if they exist, are more likely to come from better features or event-level framing than from a more expressive model.

## Method Summary

- **Labels.** Forward-window maximum of `cpu_die_temp_c` against 75 °C over 24 steps (2 minutes). Samples whose look-ahead window is truncated by session end are dropped, never assumed negative.
- **Imputation.** Causal only, never backward-looking. 10.9% of die temperature (the label source) is imputed; a missingness flag is retained as a feature rather than discarded.
- **Features.** Five base telemetry signals expanded into lags, rolling statistics (30 s–15 min windows), differences, accelerations, and two interaction terms, all computed strictly within session boundaries (no cross-session leakage). 24 features retained after importance filtering. The **current-time** temperature reading is deliberately excluded — only its lag is admitted — since the label is itself a function of future temperature, and including the live reading would leak the answer.
- **Preprocessing.** Mode-aware scaling for multimodal columns, power transforms for skewed ones, fitted on training sessions only.
- **Models.** Logistic regression, random forest, gradient boosting, and a PyTorch feed-forward MLP — spanning linear, tree-ensemble, and neural approaches to stress-test whether model family mattered (it didn't).
- **Evaluation.** Precision, recall, F1, ROC-AUC, and PR-AUC on the positive class, all benchmarked against a persistence baseline rather than in isolation, per standard practice for autocorrelated time-series classification.

## Known Limitations

- Two machines are pooled with no per-machine evaluation; a fixed 75 °C threshold may not carry the same thermal meaning across both.
- Validation data was reused for both feature selection and model selection, so reported validation scores are optimistic by construction.


## Repository Structure

\```text
thermal-prediction/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── code/
│   ├── m_series_telemetry_logger_v3_sessionized.py
│   ├── intermittent_stress_runner.py
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── data_transform.py
│   ├── data_labels.py
│   ├── data_eda.py
│   ├── data_features.py
│   ├── data_prep.py
│   ├── nn_architecture.py
│   └── playground.ipynb
├── data/
│   ├── raw/
│   └── interim/
├── artifacts/    (gitignored — regenerated by playground.ipynb)
\```

### What each part does

**`code/m_series_telemetry_logger_v3_sessionized.py`** — Runs the telemetry logger. Records structured samples at a chosen interval, writes each run to a raw session CSV, and updates the session registry. Apple Silicon macOS only, via `powermetrics` (see "Collecting Data").

**`code/intermittent_stress_runner.py`** — Drives intermittent load bursts separated by cool-downs, so sessions contain repeated approaches to the threshold instead of flat idle traces (see "Generating Stress").

**`code/data_loader.py`** — Loads session CSVs and merges them into a single dataframe for downstream processing.

**`code/data_quality.py`** — Time-order checks, trimming of session edges, and temporal gap management.

**`code/data_transform.py`** — Missing-value treatment, scaling, power transforms, and target-distribution checks. Imputation is causal only.

**`code/data_labels.py`** — Constructs the prediction target: the forward-window maximum of CPU die temperature against the threshold, over the look-ahead horizon.

**`code/data_eda.py`** — Exploratory plots, descriptive statistics, relationship inspection, and the feature-importance diagnostics (impurity + permutation importance) used to decide which engineered features to keep.

**`code/data_features.py`** — Builds ML features: lags, rolling summaries, differences, accelerations, and engineered proxies. Produces columns only; importance analysis lives in `data_eda.py`.

**`code/data_prep.py`** — Splitting and preprocessing helpers: session-blocked chronological splits, preprocessors fitted on train only.

**`code/nn_architecture.py`** — PyTorch feed-forward classifier and training loop, used as one of the four model families.

**`code/playground.ipynb`** — Main working notebook: runs the full pipeline from raw sessions through model evaluation. Imports `code/` modules directly, so its working directory must be set to `code/`.

**`data/raw/`** — Raw per-session CSV files produced by the logger.

**`data/interim/`** — Intermediate outputs, chiefly the session registry.

**`artifacts/`** — Gitignored. Holds the fitted preprocessing bundle (`preprocessors.joblib`) and other generated outputs, regenerated by rerunning the notebook. Nothing here is meant to be committed.

## Setup

\```bash
git clone https://github.com/saifali03/thermal-prediction.git
cd thermal-prediction
conda create -n thermal-prediction python=3.11 -y
conda activate thermal-prediction
pip install -r requirements.txt
jupyter notebook   # then open code/playground.ipynb
\```

## Collecting Data

macOS on Apple Silicon only. `powermetrics` requires root, so run the logger under `sudo`.

\```bash
sudo python3 code/m_series_telemetry_logger_v3_sessionized.py \
  --interval 5 \
  --machine-id mac_m5 \
  --base-dir data/raw \
  --registry-path data/interim/session_registry.csv \
  --notes "workload description"
\```

One CSV per session, at `data/raw/<machine_id>/<YYYY-MM-DD>/<machine_id>_<yyyymmdd>_<run>.csv`, plus a row in the registry. Session IDs look like `mac_m5_20260627_001`.

## Generating Stress

The model learns from the *approach* to a thermal event, so the dataset needs machines repeatedly climbing toward the threshold and cooling back down — idle logging alone produces almost no positive labels.

### How it works

- **Burst** — synthetic load period (120–180 s default).
- **Cool-down** — follows each burst (120–180 s default), letting the machine shed heat.
- **Block** — 2–4 burst/cool-down cycles back to back.
- **Gap** — 30–60 minutes between blocks, so each block is an independent thermal episode.

Runs for `--hours` total, producing many separate heat-and-cool episodes rather than one long plateau. Uses `stress-ng` if installed (recommended: `brew install stress-ng`), otherwise falls back to spawning `yes` processes (works, but gives less control over load level).

### Running it

The stress runner **only generates heat** — it does not record anything. Run the logger alongside it, in a separate terminal:

1. Start the logger (terminal A).
2. Start the stress runner (terminal B).
3. When the runner finishes, stop the logger with Ctrl-C.

Order matters at the end: stop the **runner first**, let the logger capture the cool-down tail, then stop the logger. Otherwise the session ends mid-burst and the final approach window is truncated.

\```bash
python3 code/intermittent_stress_runner.py --hours 3 --seed 1
\```

### Options

| Flag | Default | What it does |
|---|---|---|
| `--hours` | 4.0 | Total runtime |
| `--seed` | none | Random seed; **vary this between sessions** |
| `--mode` | `cpu` | Load style: `cpu`, `cpu-vm` (adds memory pressure), `matrix` (FPU-heavy) |
| `--cpu-load` | 65 | Target CPU load percent |
| `--cpu-workers` | cores/3 | Parallel workers per burst |
| `--burst-min-sec` / `--burst-max-sec` | 120 / 180 | Burst length range |
| `--cool-min-sec` / `--cool-max-sec` | 120 / 180 | Cool-down length range |
| `--gap-min-sec` / `--gap-max-sec` | 1800 / 3600 | Time between blocks |
| `--cycles-min` / `--cycles-max` | 2 / 4 | Bursts per block |
| `--vm-workers` / `--vm-bytes` | 1 / 256M | Memory pressure when `--mode cpu-vm` |


\```bash
python3 code/intermittent_stress_runner.py \
  --hours 3 --mode cpu-vm --cpu-load 100 \
  --burst-min-sec 240 --burst-max-sec 360 \
  --cool-min-sec 60 --cool-max-sec 120 \
  --seed 7
\```

Note in the logger's `--notes` field whether the run was deliberately stressed, what was running, and whether the charger was connected.

## License

MIT.
