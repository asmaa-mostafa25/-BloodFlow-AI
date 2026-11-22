import pandas as pd
from forecast_cache import get_forecasts
from load_data import load_consumption, load_donations, load_batches, real_stock  # جديد

cons = load_consumption()                        # بدل pd.read_csv
don = load_donations()                           # بدل pd.read_csv

# المخزون الحقيقي محسوب من ملف الدفعات (بدل الأرقام المكتوبة)
CURRENT_STOCK = real_stock(load_batches(cons["date"].max()))

# ====== للتجربة في العرض: أي رقم هنا بيغطي على المخزون الحقيقي ======
# مثال: TEST_STOCK = {"O+": 240, "A+": 300, "B+": 120, "AB+": 40,
#                     "O-": 8, "A-": 12, "B-": 3, "AB-": 6}
TEST_STOCK = {}
CURRENT_STOCK.update(TEST_STOCK)

DAYS_COVER = {"O+": 3, "A+": 3, "B+": 3, "AB+": 3,
              "O-": 5, "A-": 5, "B-": 5, "AB-": 5}
MIN_THRESHOLD = 3
# ==========================================


def simulate(start_stock, daily_don, demand, threshold):
    stock = start_stock
    first_day, lowest = None, start_stock
    for i, d in enumerate(demand, 1):
        stock = stock + daily_don - max(d, 0)
        lowest = min(lowest, stock)
        if stock < threshold and first_day is None:
            first_day = i
    return first_day, lowest


# التوقعات: بتتحمّل من الملف المحفوظ، أو بتتدرّب من جديد لو الداتا اتغيّرت
forecasts = get_forecasts(cons)

rows = []
for bt in CURRENT_STOCK:
    s = cons[cons.blood_type == bt][["date", "units_consumed"]]
    s.columns = ["ds", "y"]
    if bt not in forecasts:                         # جديد: فصيلة مش موجودة في الداتا
        continue
    fc = forecasts[bt]

    avg_cons = s["y"].tail(30).mean()
    avg_don = (don[don.blood_type == bt].sort_values("date")
               ["units_donated"].tail(30).mean())
    if pd.isna(avg_don):                            # جديد: مفيش تبرعات للفصيلة دي
        avg_don = 0
    threshold = max(MIN_THRESHOLD, round(DAYS_COVER.get(bt, 5) * avg_cons))
    stock0 = CURRENT_STOCK[bt]

    day_exp, low_exp = simulate(stock0, avg_don, fc["yhat"].values, threshold)
    day_worst, _ = simulate(stock0, avg_don, fc["yhat_upper"].values, threshold)

    deficit = max(0, threshold - low_exp)

    if day_exp is not None and day_exp <= 3:
        alert = "RED"
    elif day_exp is not None or day_worst is not None:
        alert = "YELLOW"
    else:
        alert = "GREEN"

    rows.append({
        "type": bt,
        "stock": stock0,
        "threshold": threshold,
        "days_of_supply": round(stock0 / max(avg_cons, 0.1), 1),
        "pred_cons_14d": round(fc["yhat"].clip(lower=0).sum()),
        "shortage_day": day_exp,
        "worst_case_day": day_worst,
        "deficit": round(deficit, 1),
        "alert": alert,
    })

out = pd.DataFrame(rows)
print()
print(out.to_string(index=False))
out.to_csv("shortage_results.csv", index=False)
