# -*- coding: utf-8 -*-
"""
BloodFlow AI: خدمة API للباك (.NET)

التشغيل:
    python -m uvicorn api:app --port 8000
وبعدين افتحي http://127.0.0.1:8000/docs

- بيستخدم load_data.py نفسه، فبيقبل أي شكل داتا زي باقي المشروع
- نفس منطق shortage_all.py و expiry_forecast.py و compatibility.py بالظبط
- مفتاح الـ API من متغير البيئة BLOODFLOW_API_KEY (الباك يبعته في Header اسمه x-api-key)
"""
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from prophet import Prophet

import load_data as ld

logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").setLevel(logging.ERROR)

# ====== الإعدادات ======
API_KEY = os.environ.get("BLOODFLOW_API_KEY", "bloodflow-secret")
HORIZON = 14
TYPES = ld.TYPES
DAYS_COVER = {"O+": 3, "A+": 3, "B+": 3, "AB+": 3,
              "O-": 5, "A-": 5, "B-": 5, "AB-": 5}
MIN_THRESHOLD = 3
YEARLY_MIN_DAYS = 600        # نفس القاعدة اللي في باقي الأكواد
LIVE_DIR = "live_data"       # آخر داتا اتبعتت عن طريق /retrain بتتحفظ هنا

DONORS = {
    "O-": ["O-"],
    "O+": ["O+", "O-"],
    "A-": ["A-", "O-"],
    "A+": ["A+", "A-", "O+", "O-"],
    "B-": ["B-", "O-"],
    "B+": ["B+", "B-", "O+", "O-"],
    "AB-": ["AB-", "A-", "B-", "O-"],
    "AB+": ["AB+", "AB-", "A+", "A-", "B+", "B-", "O+", "O-"],
}
RECIPIENTS = {d: [r for r in DONORS if d in DONORS[r]] for d in DONORS}

# ====== حالة السيرفر ======
STATE = {"cons": None, "don": None, "batches": None, "today": None, "source": None}
FORECAST, METHOD, INFO = {}, {}, {}
AVAIL = pd.DataFrame()


# ---------------------------------------------------------------
# التوقع: Prophet لو الداتا كفاية، وطريقة أبسط لو قليلة
# ---------------------------------------------------------------
def fit_forecast(s, horizon):
    """s: جدول فيه ds و y. بيرجّع (توقع, اسم الطريقة)"""
    s = s.sort_values("ds").reset_index(drop=True)
    n = len(s)
    if n >= 60 and s["y"].sum() > 0:
        m = Prophet(weekly_seasonality=True,
                    yearly_seasonality=n >= YEARLY_MIN_DAYS,
                    interval_width=0.8)
        m.fit(s)
        fc = m.predict(m.make_future_dataframe(periods=horizon)).tail(horizon)
        fc = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)
        for c in ["yhat", "yhat_lower", "yhat_upper"]:
            fc[c] = fc[c].clip(lower=0)
        return fc, ("prophet" if n >= YEARLY_MIN_DAYS else "prophet-weekly")

    # داتا قليلة: متوسط نفس يوم الأسبوع، أو المتوسط العام
    last = s["ds"].max() if n else pd.Timestamp.today().normalize()
    future = pd.date_range(last + pd.Timedelta(days=1), periods=horizon)
    mean = float(s["y"].mean()) if n else 0.0
    std = float(np.nan_to_num(s["y"].std())) if n > 1 else mean ** 0.5
    by_dow = s.groupby(s["ds"].dt.dayofweek)["y"].mean().to_dict() if n >= 14 else {}
    yhat = np.array([by_dow.get(d.dayofweek, mean) for d in future], dtype=float)
    fc = pd.DataFrame({"ds": future, "yhat": yhat,
                       "yhat_lower": np.maximum(yhat - 1.28 * std, 0),
                       "yhat_upper": yhat + 1.28 * std})
    return fc, ("weekday-average" if n >= 14 else "simple-mean")


# ---------------------------------------------------------------
# تحميل الداتا وتدريب الموديلات
# ---------------------------------------------------------------
def load_from(folder):
    """بيقرا الفولدر بـ load_data.py (نفس التنضيف بتاع باقي المشروع)"""
    ld.DATA_DIR = folder
    cons = ld.load_consumption()
    don = ld.load_donations()
    today = cons["date"].max()      # «النهاردة» = آخر يوم في الداتا
    batches = ld.load_batches(today)
    return cons, don, batches, today


def train_all():
    global AVAIL
    cons, don, b, today = (STATE["cons"], STATE["don"], STATE["batches"],
                           STATE["today"])
    if len(b):
        AVAIL = b[b["status"] == "available"].copy()
        AVAIL["days_left"] = (AVAIL["expiry_date"] - today).dt.days
    else:
        AVAIL = pd.DataFrame(columns=["blood_type", "units_remaining", "days_left"])

    FORECAST.clear(), METHOD.clear(), INFO.clear()
    for bt in TYPES:
        s = cons[cons.blood_type == bt][["date", "units_consumed"]]
        s.columns = ["ds", "y"]
        FORECAST[bt], METHOD[bt] = fit_forecast(s, HORIZON)
        avg_cons = float(s["y"].tail(30).mean()) if len(s) else 0.0
        d = don[don.blood_type == bt].sort_values("date")["units_donated"].tail(30)
        avg_don = float(d.mean()) if len(d) else 0.0
        INFO[bt] = {
            "avg_cons": avg_cons,
            "avg_don": avg_don,
            "threshold": max(MIN_THRESHOLD, round(DAYS_COVER.get(bt, 5) * avg_cons)),
            "has_data": len(s) > 0,
        }
        print("جاهز:", bt, "|", METHOD[bt])


def real_stock():
    s = AVAIL.groupby("blood_type")["units_remaining"].sum().to_dict() if len(AVAIL) else {}
    return {t: int(s.get(t, 0)) for t in TYPES}


@asynccontextmanager
async def lifespan(app: FastAPI):
    folder = LIVE_DIR if os.path.exists(os.path.join(LIVE_DIR, "consumption.csv")) else "."
    STATE["source"] = folder
    STATE["cons"], STATE["don"], STATE["batches"], STATE["today"] = load_from(folder)
    train_all()
    if API_KEY == "bloodflow-secret":
        print("⚠️ بتستخدمي المفتاح الافتراضي. في الاستخدام الحقيقي حطي BLOODFLOW_API_KEY")
    yield


app = FastAPI(title="BloodFlow AI Service", version="2.0", lifespan=lifespan)


def check_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------
# شكل الطلبات
# ---------------------------------------------------------------
class StockRequest(BaseModel):
    # مثال: {"stocks": {"O+": 240, "B-": 3}}. الفصيلة اللي مش مكتوبة بتاخد المخزون الحقيقي
    stocks: Optional[Dict[str, float]] = None


class DemandRequest(BaseModel):
    # مثال: {"blood_type": "O+", "days_ahead": 7,
    #        "history": [{"date": "2026-01-01", "units_consumed": 70}, ...]}
    blood_type: str
    history: List[Dict[str, Any]]
    days_ahead: int = 7


class RetrainRequest(BaseModel):
    # كل جدول = قايمة صفوف JSON بأي أسماء أعمدة يفهمها load_data.py
    consumption: List[Dict[str, Any]]
    donations: Optional[List[Dict[str, Any]]] = None
    batches: Optional[List[Dict[str, Any]]] = None
    mode: str = "replace"      # replace = استبدال | append = إضافة على القديم


def get_stocks(req: Optional[StockRequest]):
    stocks = real_stock()
    if req is None or not req.stocks:
        return stocks
    for t, v in req.stocks.items():
        key = ld.norm_type(t)
        if key is None:
            raise HTTPException(400, f"Unknown blood type: {t}")
        if v < 0:
            raise HTTPException(400, f"Stock for {t} cannot be negative")
        stocks[key] = v
    return stocks


# ---------------------------------------------------------------
# المنطق: نفس الأكواد الأساسية
# ---------------------------------------------------------------
def simulate(stock0, daily_don, demand, threshold):
    stock, first_day, path = stock0, None, []
    for i, d in enumerate(demand, 1):
        stock = stock + daily_don - max(float(d), 0)
        path.append(round(float(stock), 1))
        if stock < threshold and first_day is None:
            first_day = i
    return first_day, path


def shortage_for(bt, stock0):
    info, fc = INFO[bt], FORECAST[bt]
    day_exp, path = simulate(stock0, info["avg_don"], fc["yhat"].values, info["threshold"])
    day_worst, _ = simulate(stock0, info["avg_don"], fc["yhat_upper"].values,
                            info["threshold"])
    deficit = max(0.0, info["threshold"] - min(path + [stock0]))
    if day_exp is not None and day_exp <= 3:
        alert = "RED"
    elif day_exp is not None or day_worst is not None:
        alert = "YELLOW"
    else:
        alert = "GREEN"
    return {
        "blood_type": bt,
        "current_stock": stock0,
        "safety_threshold": info["threshold"],
        "days_of_supply": round(stock0 / max(info["avg_cons"], 0.1), 1),
        "predicted_consumption_14d": round(float(fc["yhat"].sum())),
        "shortage_in_days": day_exp,
        "worst_case_shortage_in_days": day_worst,
        "expected_deficit_units": round(deficit, 1),
        "alert_level": alert,
        "forecast_method": METHOD[bt],
        "daily": [{"date": str(fc["ds"][i].date()),
                   "predicted_consumption": round(float(fc["yhat"][i]), 1),
                   "predicted_stock": path[i]} for i in range(len(fc))],
    }


def fifo_expiry(batch_list, demand):
    bl = sorted([list(x) for x in batch_list], key=lambda x: x[1])
    expired = 0.0
    for day, need in enumerate(demand, 1):
        kept = []
        for units, left in bl:
            if left <= day:
                expired += units
            else:
                kept.append([units, left])
        bl = kept
        for x in bl:
            take = min(x[0], need)
            x[0] -= take
            need -= take
        bl = [x for x in bl if x[0] > 0]
    return expired


def expiry_for(bt):
    sb = AVAIL[AVAIL.blood_type == bt] if len(AVAIL) else AVAIL
    fc = FORECAST[bt]
    bl = list(zip(sb["units_remaining"], sb["days_left"])) if len(sb) else []
    available = float(sb["units_remaining"].sum()) if len(sb) else 0.0
    expected = float(fifo_expiry(bl, fc["yhat"].values))
    low = float(fifo_expiry(bl, fc["yhat_lower"].values))
    risk = low / max(available, 1) * 100
    return {
        "blood_type": bt,
        "available_units": int(available),
        "expiring_in_3_days": int(sb[sb.days_left <= 3]["units_remaining"].sum()) if len(sb) else 0,
        "expiring_in_7_days": int(sb[sb.days_left <= 7]["units_remaining"].sum()) if len(sb) else 0,
        "expected_waste_units": round(expected, 1),
        "waste_if_low_demand_units": round(low, 1),
        "waste_risk_percent": round(risk, 1),
        "risk_level": "HIGH" if risk > 15 else "MEDIUM" if risk >= 5 else "LOW",
    }


def substitutions_for(stocks):
    thr = {t: INFO[t]["threshold"] for t in TYPES}
    need = {t: max(0, round(thr[t] - stocks[t])) for t in TYPES}
    surplus = {t: max(0, round(stocks[t] - thr[t])) for t in TYPES}
    soon = {}
    for t in TYPES:
        sb = AVAIL[(AVAIL.blood_type == t) & (AVAIL.days_left <= 7)] if len(AVAIL) else []
        soon[t] = min(int(sb["units_remaining"].sum()) if len(sb) else 0, int(stocks[t]))

    out = []
    for target in TYPES:
        remaining = need[target]
        if remaining <= 0:
            continue
        options = [d for d in DONORS[target] if d != target]
        options.sort(key=lambda d: (d == "O-", -surplus[d]))    # O- آخر خيار
        moves = []
        for d in options:
            if remaining <= 0:
                break
            give = min(remaining, surplus[d])
            if give <= 0:
                continue
            moves.append({"from": d, "units": give, "near_expiry_units": min(give, soon[d])})
            surplus[d] -= give
            remaining -= give
        out.append({"recipient": target, "units_needed": need[target],
                    "suggestions": moves, "still_uncovered": remaining,
                    "action": ("donation campaign or transfer from another hospital"
                               if remaining > 0 else "covered by substitutions")})

    near_expiry = []
    for t in TYPES:
        if soon[t] >= 1:
            needy = [r for r in RECIPIENTS[t] if need[r] > 0]
            near_expiry.append({"blood_type": t, "units_expiring_7d": soon[t],
                                "use_for": needy,
                                "action": "use internally" if needy else
                                          "transfer to a nearby hospital"})
    return {"substitutions": out, "near_expiry": near_expiry}


# ---------------------------------------------------------------
# الـ Endpoints
# ---------------------------------------------------------------
@app.get("/health")
def health():
    c = STATE["cons"]
    return {
        "status": "ok",
        "today": str(STATE["today"].date()),
        "data_source": STATE["source"],
        "consumption_rows": int(len(c)),
        "date_range": [str(c["date"].min().date()), str(c["date"].max().date())],
        "methods": METHOD,
    }


@app.get("/forecast/all", dependencies=[Depends(check_key)])
def forecast_all():
    return {bt: [{"date": str(fc["ds"][i].date()),
                  "predicted": round(float(fc["yhat"][i]), 1),
                  "lower": round(float(fc["yhat_lower"][i]), 1),
                  "upper": round(float(fc["yhat_upper"][i]), 1)}
                 for i in range(len(fc))]
            for bt, fc in FORECAST.items()}


@app.get("/stock", dependencies=[Depends(check_key)])
def stock():
    """المخزون الحالي المحسوب من الدفعات المتاحة"""
    return {"today": str(STATE["today"].date()), "stock": real_stock()}


@app.post("/predict/shortage", dependencies=[Depends(check_key)])
def predict_shortage(req: Optional[StockRequest] = None):
    stocks = get_stocks(req)
    return [shortage_for(bt, stocks[bt]) for bt in TYPES if INFO[bt]["has_data"]]


@app.post("/predict/expiry", dependencies=[Depends(check_key)])
def predict_expiry():
    return [expiry_for(bt) for bt in TYPES if INFO[bt]["has_data"]]


@app.post("/substitutions", dependencies=[Depends(check_key)])
def substitutions(req: Optional[StockRequest] = None):
    return substitutions_for(get_stocks(req))


@app.post("/predict/demand", dependencies=[Depends(check_key)])
def predict_demand(req: DemandRequest):
    """توقع من تاريخ بيتبعت في الطلب نفسه (مابيغيرش الموديلات المحفوظة)"""
    bt = ld.norm_type(req.blood_type)
    if bt is None:
        raise HTTPException(400, f"Unknown blood type: {req.blood_type}")
    if not 1 <= req.days_ahead <= 30:
        raise HTTPException(400, "days_ahead must be between 1 and 30")
    df = pd.DataFrame(req.history)
    dcol, vcol = ld.find_col(df, ld.DATE), ld.find_col(df, ld.VALUE)
    if dcol is None or vcol is None:
        raise HTTPException(400, f"history needs a date and a units column, got {list(df.columns)}")
    s = pd.DataFrame({"ds": ld.to_date(df[dcol]), "y": ld.to_num(df[vcol]).clip(lower=0)})
    s = s.dropna().groupby("ds", as_index=False)["y"].sum()
    if len(s) < 7:
        raise HTTPException(400, "Need at least 7 days of history")
    fc, method = fit_forecast(s, req.days_ahead)
    return {"blood_type": bt, "history_days_used": int(len(s)), "method": method,
            "forecast": [{"date": str(fc["ds"][i].date()),
                          "predicted": round(float(fc["yhat"][i]), 1),
                          "lower": round(float(fc["yhat_lower"][i]), 1),
                          "upper": round(float(fc["yhat_upper"][i]), 1)}
                         for i in range(len(fc))]}


@app.post("/retrain", dependencies=[Depends(check_key)])
def retrain(req: RetrainRequest):
    """بيستقبل داتا جديدة بأي شكل، ينضّفها بـ load_data.py، ويعيد التدريب"""
    if req.mode not in ("replace", "append"):
        raise HTTPException(400, "mode must be 'replace' or 'append'")

    # 1) نكتب اللي اتبعت في فولدر مؤقت، ونخلّي load_data.py يقراه وينضفه
    tmp = tempfile.mkdtemp()
    try:
        pd.DataFrame(req.consumption).to_csv(os.path.join(tmp, "consumption.csv"), index=False)
        if req.donations:
            pd.DataFrame(req.donations).to_csv(os.path.join(tmp, "donations.csv"), index=False)
        if req.batches:
            pd.DataFrame(req.batches).to_csv(os.path.join(tmp, "inventory_batches.csv"), index=False)
        try:
            cons, don, bat, _ = load_from(tmp)
        except (ValueError, KeyError, SystemExit) as e:
            raise HTTPException(400, "Could not read the data: "
                                + str(e).replace(os.path.join(tmp, ""), ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 2) إضافة على القديم لو mode = append
    if req.mode == "append":
        cons = pd.concat([STATE["cons"], cons]).drop_duplicates(
            ["date", "blood_type"], keep="last")
        if req.donations:
            don = pd.concat([STATE["don"], don]).drop_duplicates(
                ["date", "blood_type"], keep="last")
        else:
            don = STATE["don"]
        if req.batches:
            bat = pd.concat([STATE["batches"], bat], ignore_index=True)
            if "batch_id" in bat.columns:
                bat = bat.drop_duplicates("batch_id", keep="last")
        else:
            bat = STATE["batches"]

    if cons["date"].nunique() < 7:
        raise HTTPException(400, "Need at least 7 days of consumption data")

    # 3) نحفظ الداتا النضيفة ونقراها تاني (عشان الأيام الناقصة والحالة يتحسبوا صح)
    os.makedirs(LIVE_DIR, exist_ok=True)
    cons.to_csv(os.path.join(LIVE_DIR, "consumption.csv"), index=False)
    for name, df in [("donations.csv", don), ("inventory_batches.csv", bat)]:
        path = os.path.join(LIVE_DIR, name)
        if df is not None and len(df):
            df.drop(columns=["status"], errors="ignore").to_csv(path, index=False)
        elif os.path.exists(path):
            os.remove(path)          # replace من غير الجدول ده = مفيش داتا ليه
    STATE["source"] = LIVE_DIR
    STATE["cons"], STATE["don"], STATE["batches"], STATE["today"] = load_from(LIVE_DIR)
    train_all()

    warnings = [f"{bt}: no consumption data" for bt in TYPES if not INFO[bt]["has_data"]]
    warnings += [f"{bt}: short history, using {METHOD[bt]}" for bt in TYPES
                 if INFO[bt]["has_data"] and METHOD[bt] not in ("prophet", "prophet-weekly")]
    if not req.donations and req.mode == "replace":
        warnings.append("no donations sent: shortage assumes zero donations")
    c = STATE["cons"]
    return {"status": "retrained", "mode": req.mode, "today": str(STATE["today"].date()),
            "consumption_rows": int(len(c)),
            "date_range": [str(c["date"].min().date()), str(c["date"].max().date())],
            "methods": METHOD, "warnings": warnings}
