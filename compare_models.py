import logging
import numpy as np
import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

from load_data import load_consumption          # جديد
df = load_consumption()                          # بدل pd.read_csv
HORIZON = 14
FOLDS = 5   # 5 فترات اختبار مختلفة (آخر 5 مرات × 14 يوم)

def make_features(s):
    d = s.copy()
    d["dow"] = d["date"].dt.dayofweek
    d["month"] = d["date"].dt.month
    for lag in [14, 21, 28]:          # lag >= 14 عشان مفيش تسريب بيانات
        d[f"lag_{lag}"] = d["y"].shift(lag)
    d["roll7"] = d["y"].shift(14).rolling(7).mean()
    d["roll30"] = d["y"].shift(14).rolling(30).mean()
    return d.dropna()

def wmape(actual, pred):
    return np.abs(actual - pred).sum() / max(actual.sum(), 1)

rows = []
for bt in sorted(df["blood_type"].unique()):
    s = df[df["blood_type"] == bt][["date", "units_consumed"]]
    s.columns = ["date", "y"]
    s = s.reset_index(drop=True)
    if len(s) < FOLDS * HORIZON + 60:               # جديد: الداتا قليلة على 5 فترات اختبار
        print("اتخطيت", bt, ": داتا قليلة")
        continue
    feats = make_features(s)
    X_cols = [c for c in feats.columns if c not in ["date", "y"]]

    err = {"Baseline": [], "Prophet": [], "XGBoost": []}
    act = []
    for k in range(FOLDS, 0, -1):
        end = len(s) - (k - 1) * HORIZON
        train = s.iloc[: end - HORIZON]
        test = s.iloc[end - HORIZON : end]
        actual = test["y"].values
        act.append(actual)

        # Baseline
        err["Baseline"].append(np.abs(train["y"].tail(7).mean() - actual))

        # Prophet
        m = Prophet(weekly_seasonality=True, yearly_seasonality=len(train) >= 600)  # جديد
        m.fit(train.rename(columns={"date": "ds"}))
        fc = m.predict(m.make_future_dataframe(periods=HORIZON)).tail(HORIZON)
        err["Prophet"].append(np.abs(np.maximum(fc["yhat"].values, 0) - actual))

        # XGBoost
        cutoff = train["date"].max()
        tr = feats[feats["date"] <= cutoff]
        te = feats[(feats["date"] > cutoff) & (feats["date"] <= test["date"].max())]
        xgb = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05)
        xgb.fit(tr[X_cols], tr["y"])
        err["XGBoost"].append(np.abs(np.maximum(xgb.predict(te[X_cols]), 0) - te["y"].values))

    total_actual = np.concatenate(act).sum()
    row = {"blood_type": bt}
    for name, e in err.items():
        row[f"{name}_MAE"] = np.concatenate(e).mean()
        row[f"{name}_WMAPE%"] = np.concatenate(e).sum() / max(total_actual, 1) * 100
    rows.append(row)
    print("خلصت", bt)

out = pd.DataFrame(rows).round(2)
print("\n", out.to_string(index=False))
print("\nمتوسط كل موديل:")
print(out.drop(columns="blood_type").mean().round(2))
out.to_csv("model_comparison.csv", index=False)
