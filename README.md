# BloodFlow AI

A forecasting system for hospital blood banks. It predicts daily consumption for each of the 8 blood types, warns before a shortage, estimates how many units will expire, and suggests compatible substitutions. It accepts new data in any reasonable format (English or Arabic column names, wide or long tables, CSV or Excel).

Graduation project — AI track.

## Requirements

- Python 3.10 or newer
- Windows, Linux or macOS

## Installation

```
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

## Data files

Put the data files in the project folder (or in any folder you pass as an argument). The file name only needs to contain one of these words:

| Data | File name contains | Required columns (any common name) |
| --- | --- | --- |
| Consumption | `consumption`, `usage`, `transfusion`, `استهلاك` | date, blood type, units |
| Donations | `donation`, `تبرع` | date, blood type, units |
| Stock batches | `batch`, `inventory`, `دفعات` | blood type, collection date, expiry date, units |

`load_data.py` cleans the data automatically: it unifies blood type spelling, parses date formats, removes negative values and duplicates, and fills missing days.

## Running the scripts

Run them in this order:

```
python compare_models.py        # compares Baseline, Prophet, XGBoost  -> model_comparison.csv
python shortage_all.py          # shortage forecast and alerts          -> shortage_results.csv
python expiry_forecast.py       # units expected to expire (FIFO)       -> expiry_results.csv
python compatibility.py         # substitutions between blood types     -> substitutions.csv
```

`compatibility.py` reads `shortage_results.csv`, so run it after `shortage_all.py`.

Faster versions that reuse saved forecasts:

```
python shortage_all_v2.py
python expiry_forecast_v2.py
```

To run on data in another folder, pass the folder name:

```
python shortage_all.py new_data
```

## Running the API

```
python -m uvicorn api:app --port 8000
```

Wait for `Application startup complete` (training takes about 1–2 minutes), then open:

- `http://127.0.0.1:8000/docs` — interactive test page
- `http://127.0.0.1:8000/health` — status check

To let another machine (the .NET backend) reach the API:

```
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Every endpoint except `/health` needs the header `x-api-key`. The default key is `bloodflow-secret`. Change it with an environment variable before starting the server:

```
set BLOODFLOW_API_KEY=your-key        # Windows (cmd)
$env:BLOODFLOW_API_KEY="your-key"     # Windows (PowerShell)
export BLOODFLOW_API_KEY=your-key     # Linux / macOS
```

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Server status (no key) |
| GET | `/stock` | Current stock per blood type |
| GET | `/forecast/all` | 14-day consumption forecast |
| POST | `/predict/shortage` | Shortage alerts (RED / YELLOW / GREEN) |
| POST | `/predict/expiry` | Units expected to expire |
| POST | `/substitutions` | Compatible substitutions |
| POST | `/predict/demand` | Forecast from a history you send |
| POST | `/retrain` | Send new data and retrain |

Full request and response details are in the API contract document shared with the backend team.

## Saved results

The `results/` folder holds the outputs of the last run on the main data, so they can be reviewed without running anything. Running a script writes a fresh copy next to the scripts.

`new_data/` is a deliberately messy copy of the data (Arabic column names, negative values, a missing day) used to test that the scripts accept any format.

## Project files

| File | Role |
| --- | --- |
| `load_data.py` | Reads and cleans any data format |
| `compare_models.py` | Model comparison (rolling backtest, MAE and WMAPE) |
| `shortage_all.py` | Shortage forecast and alerts |
| `expiry_forecast.py` | Expiry forecast |
| `compatibility.py` | Blood type substitutions |
| `forecast_cache.py` | Saves forecasts and retrains only when data changes |
| `shortage_all_v2.py`, `expiry_forecast_v2.py` | Fast versions using the cache |
| `api.py` | FastAPI service for the backend |

## Limitations

- The data used in development is synthetic, not from a real hospital.
- Forecasts cover 14 days ahead.
- In the API, a blood type needs at least 60 days of history for Prophet; shorter histories use a simpler fallback.
