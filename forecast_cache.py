import logging
import os
import pickle

HORIZON = 14
CACHE_FILE = "forecast_cache.pkl"


def _fingerprint(cons):
    """بصمة للداتا: لو اتغيّرت (صفوف جديدة أو أرقام اتعدّلت) يبقى لازم نعيد التدريب"""
    return (len(cons), str(cons["date"].max()), int(cons["units_consumed"].sum()))


def get_forecasts(cons, force=False):
    """
    بيرجّع توقع 14 يوم لكل فصيلة.
    - لو الداتا ما اتغيّرتش: بيحمّل التوقع المحفوظ (ثواني).
    - لو داتا جديدة أو أول مرة: بيدرّب الموديلات من جديد ويحفظ النتيجة.
    """
    fp = _fingerprint(cons)
    if not force and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            if cache.get("fingerprint") == fp:
                print("الداتا ما اتغيّرتش: بحمّل التوقعات المحفوظة (سريع)")
                return cache["forecast"]
        except Exception:
            pass   # لو الملف تالف نعيد التدريب عادي

    print("داتا جديدة أو أول مرة: بدرّب الموديلات (دقيقة أو اتنين)...")
    from prophet import Prophet    # هنا بس عشان التشغيل السريع مايستناش تحميله
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    result = {}
    for bt in sorted(cons["blood_type"].unique()):
        s = cons[cons.blood_type == bt][["date", "units_consumed"]]
        s.columns = ["ds", "y"]
        m = Prophet(weekly_seasonality=True,
                    yearly_seasonality=len(s) >= 600,   # جديد: الموسمية السنوية محتاجة سنتين
                    interval_width=0.8)
        m.fit(s)
        fc = m.predict(m.make_future_dataframe(periods=HORIZON)).tail(HORIZON)
        result[bt] = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)
        print("خلصت", bt)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"fingerprint": fp, "forecast": result}, f)
    return result
