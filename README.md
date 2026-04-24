# 🏎️ F1 Lap Time Predictor

A machine learning tool that predicts Formula 1 lap times using real-world telemetry and weather data. Built with [FastF1](https://docs.fastf1.dev/) and scikit-learn.

## Overview

This project fetches historical F1 race data — including tire compounds, weather conditions, and lap telemetry — and trains a Random Forest model to predict lap times based on configurable parameters.

### Features

- **Real F1 data** sourced via the FastF1 API (official F1 timing data)
- **Random Forest regression** model with automatic training and persistence
- **Interactive CLI** for inputting custom race conditions and getting predictions
- **Configurable parameters**: track, tire compound, tyre life, air/track temperature, humidity, and lap number

## Setup

### Prerequisites

- Python 3.9+

### Installation

1. **Clone the repository** (or navigate to the project directory):

   ```bash
   cd "F1 Lap Predictor"
   ```

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train the Model

On first run, or to retrain with fresh data:

```bash
python main.py --train
```

This will:
- Download race data for the first 5 rounds of the 2024 season
- Engineer features from lap and weather data
- Train a Random Forest model
- Save the trained model to `predictor_model.joblib`

> **Note:** The first training run will take several minutes as FastF1 downloads and caches race data.

### Predict Lap Times

Once a model is trained, run without the `--train` flag:

```bash
python main.py
```

You'll be prompted to enter race parameters interactively:

```
=== F1 Lap Time Predictor ===
Enter parameters to predict lap time (press Enter to use defaults):

Track (default: Bahrain Grand Prix):
Tire Compound [SOFT/MEDIUM/HARD] (default: SOFT):
Tyre Life in laps (default: 5.0):
Air Temperature °C (default: 25.0):
Track Temperature °C (default: 32.0):
Humidity % (default: 45.0):
Lap Number / Fuel Load proxy (default: 15):
```

Press **Enter** on any field to accept the default value.

### Example Output

```
--- Model Prediction Input ---
  Track      : Bahrain Grand Prix
  Compound   : SOFT
  Tyre Life  : 5.0 laps
  Air Temp   : 25.0°C
  Track Temp : 32.0°C
  Humidity   : 45.0%
  Lap Number : 15 (Fuel impact)

🏎️  => Predicted Lap Time: 1:33.456 (93.456 seconds)
```

## Project Structure

```
.
├── main.py                  # Core predictor class and CLI entry point
├── test_fastf1.py           # Quick smoke test for FastF1 data loading
├── requirements.txt         # Python dependencies
├── predictor_model.joblib   # Trained model (generated after training)
├── cache/                   # FastF1 data cache (auto-created)
└── README.md
```

## Parameters Guide

| Parameter       | Description                                      | Default              |
|-----------------|--------------------------------------------------|----------------------|
| Track           | Grand Prix name (must match training data)       | Bahrain Grand Prix   |
| Compound        | Tire compound: `SOFT`, `MEDIUM`, or `HARD`       | SOFT                 |
| Tyre Life       | Number of laps on current tire set               | 5.0                  |
| Air Temp        | Ambient air temperature in °C                    | 25.0                 |
| Track Temp      | Track surface temperature in °C                  | 32.0                 |
| Humidity        | Relative humidity percentage                     | 45.0                 |
| Lap Number      | Current lap number (proxy for fuel load)         | 15                   |

## Dependencies

- [FastF1](https://docs.fastf1.dev/) — F1 telemetry and timing data
- [pandas](https://pandas.pydata.org/) — Data manipulation
- [NumPy](https://numpy.org/) — Numerical computing
- [scikit-learn](https://scikit-learn.org/) — Machine learning (Random Forest, preprocessing)
- [joblib](https://joblib.readthedocs.io/) — Model serialization (included with scikit-learn)

## License

This project is for educational and personal use. F1 timing data is provided by the FastF1 library under its own terms.
