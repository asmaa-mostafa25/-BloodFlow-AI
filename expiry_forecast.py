import logging
import pandas as pd
from prophet import Prophet
from load_data import load_consumption, load_batches   # جديد

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

HORIZON = 14

cons = load_consumption()                        # بدل pd.read_csv

# "النهاردة" = آخر يوم في الداتا
TODAY = cons["date"].max()
batches = load_batches(TODAY, required=True)                  # بدل pd.read_csv (بعد TODAY)

# الوحدات الموجودة فعلًا في المخزن دلوقتي
stock = batches[batches["status"] == "available"].copy()
stock["days_left"] = (stock["expiry_date"] - TODAY).dt.days


def fifo_expiry(batch_list, demand):
    """
    batch_list: قايمة [وحدات, أيام_متبقية_للصلاحية]
    demand: الاستهلاك المتوقع لكل يوم
    بيسحب من الأقدم الأول (FIFO) وبيرجّع كام وحدة هتنتهي صلاحيتها من غير استخدام
    """
    bl = sorted([list(b) for b in batch_list], key=lambda b: b[1])
    expired = 0.0
    for day, need in enumerate(demand, 1):
        kept = []
        for units, left in bl:
            if left <= day:
                expired += units          # الصلاحية خلصت
            else:
                kept.append([units, left])
        bl = kept
        for b in bl:                       # الاستهلاك بيبدأ بالأقدم
            take = min(b[0], need)
            b[0] -= take
            need -= take
        bl = [b for b in bl if b[0] > 0]
    return expired


rows = []
for bt in sorted(stock["blood_type"].unique()):
    s = cons[cons.blood_type == bt][["date", "units_consumed"]]
    s.columns = ["ds", "y"]
    if len(s) < 2:                                  # جديد: مفيش استهلاك للفصيلة دي
        continue
    m = Prophet(weekly_seasonality=True,
                yearly_seasonality=len(s) >= 600,   # جديد: الموسمية السنوية محتاجة سنتين
                interval_width=0.8)
    m.fit(s)
    fc = m.predict(m.make_future_dataframe(periods=HORIZON)).tail(HORIZON)

    sb = stock[stock.blood_type == bt]
    batch_list = list(zip(sb["units_remaining"], sb["days_left"]))
    available = sb["units_remaining"].sum()

    exp_3 = sb[sb.days_left <= 3]["units_remaining"].sum()
    exp_7 = sb[sb.days_left <= 7]["units_remaining"].sum()

    # السيناريو المتوقع، وسيناريو الاستهلاك المنخفض (الأخطر على الصلاحية)
    expected = fifo_expiry(batch_list, fc["yhat"].clip(lower=0).values)
    low_demand = fifo_expiry(batch_list, fc["yhat_lower"].clip(lower=0).values)

    risk = low_demand / max(available, 1) * 100
    level = "HIGH" if risk > 15 else "MEDIUM" if risk >= 5 else "LOW"

    if low_demand >= 1:
        action = f"استخدمي أو انقلي حوالي {round(low_demand)} وحدة قبل ما تنتهي"
    else:
        action = "مفيش إجراء مطلوب"

    rows.append({
        "type": bt,
        "available": int(available),
        "exp_in_3d": int(exp_3),
        "exp_in_7d": int(exp_7),
        "expected_waste": round(expected, 1),
        "waste_if_low_demand": round(low_demand, 1),
        "risk_%": round(risk, 1),
        "risk_level": level,
        "action": action,
    })
    print("خلصت", bt)

out = pd.DataFrame(rows)
print()
print(out.drop(columns="action").to_string(index=False))
print()
for r in rows:
    if r["waste_if_low_demand"] >= 1:
        print(f"{r['type']}: {r['action']}")
out.to_csv("expiry_results.csv", index=False)
