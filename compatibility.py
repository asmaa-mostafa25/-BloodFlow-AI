import pandas as pd
from load_data import load_consumption, load_batches   # جديد

shortage = pd.read_csv("shortage_results.csv")
# "النهاردة" = آخر يوم في داتا الاستهلاك
TODAY = load_consumption()["date"].max()
batches = load_batches(TODAY, required=True)                  # بدل pd.read_csv

# ====== سيناريو تجربة: نفترض إن فصيلة معينة مخزونها قل ======
# اتركيها فاضية {} لو عاوزة المخزون الحقيقي بس
SIMULATE_STOCK = {"B-": 2}
# =========================================================

# مين يقدر يدّي مين (كرات الدم الحمراء): المستقبِل ← المتبرعين المناسبين
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
# العكس: كل متبرع يقدر يدّي مين
RECIPIENTS = {d: [r for r in DONORS if d in DONORS[r]] for d in DONORS}

# المخزون الحقيقي من الدفعات المتاحة
avail = batches[batches["status"] == "available"].copy()
avail["days_left"] = (avail["expiry_date"] - TODAY).dt.days
stock = avail.groupby("blood_type")["units_remaining"].sum().to_dict()
soon = (avail[avail["days_left"] <= 7]
        .groupby("blood_type")["units_remaining"].sum().to_dict())

for bt, units in SIMULATE_STOCK.items():
    stock[bt] = units          # سيناريو التجربة
    soon[bt] = min(soon.get(bt, 0), units)

threshold = {t: 3 for t in DONORS}               # جديد: فصيلة مالهاش نتيجة ياخد حد أمان 3
threshold.update(dict(zip(shortage["type"], shortage["threshold"])))

# العجز = كام وحدة ناقصة عشان نوصل لحد الأمان، والفايض = اللي فوق حد الأمان
need = {t: max(0, threshold[t] - stock.get(t, 0)) for t in DONORS}
surplus = {t: max(0, stock.get(t, 0) - threshold[t]) for t in DONORS}

print("المخزون الحالي | حد الأمان | عجز | فايض | هتنتهي خلال 7 أيام")
for t in DONORS:
    print(f"{t:>4}: {stock.get(t, 0):>5} | {threshold[t]:>4} | "
          f"{need[t]:>3} | {surplus[t]:>5} | {soon.get(t, 0):>3}")

rows = []
print("\n===== اقتراحات الاستبدال =====")
for target in DONORS:
    remaining = need[target]
    if remaining <= 0:
        continue
    # المتبرعين المناسبين: O- في الآخر لأنه المتبرع العام ومحمي للطوارئ
    options = [d for d in DONORS[target] if d != target]
    options.sort(key=lambda d: (d == "O-", -surplus[d]))
    print(f"\n{target} ناقصة {need[target]} وحدة:")
    for d in options:
        if remaining <= 0:
            break
        give = min(remaining, surplus[d])
        if give <= 0:
            continue
        near_expiry = min(give, soon.get(d, 0))
        note = f" (منها {near_expiry} قربت تنتهي)" if near_expiry else ""
        print(f"   تحويل {give} وحدة من {d}{note}")
        rows.append({"to": target, "from": d, "units": give,
                     "near_expiry": near_expiry})
        surplus[d] -= give
        remaining -= give
    if remaining > 0:
        print(f"   لسه ناقص {remaining} وحدة {target}: المطلوب حملة تبرع أو نقل من مستشفى تانية")

print("\n===== الوحدات القريبة من الانتهاء =====")
for t, units in soon.items():
    if units < 1:
        continue
    needy = [r for r in RECIPIENTS[t] if need[r] > 0]
    if needy:
        print(f"{t}: {units} وحدة هتنتهي خلال 7 أيام، استخدامها لـ {', '.join(needy)}")
    else:
        print(f"{t}: {units} وحدة هتنتهي خلال 7 أيام ومفيش احتياج داخلي، نقلها لمستشفى قريبة")

pd.DataFrame(rows, columns=["to", "from", "units", "near_expiry"]).to_csv(
    "substitutions.csv", index=False)
