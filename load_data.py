# -*- coding: utf-8 -*-
# load_data.py: بيقرا وينضّف أي داتا، ويرجّعها بنفس أسماء الأعمدة القديمة
# تشغيل على داتا تانية:  python shortage_all.py اسم_الفولدر
import os
import re
import sys

import pandas as pd

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
TYPES = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
SHELF_LIFE = 35

# الأسماء المقبولة لكل عمود (إنجليزي وعربي)
DATE = ["date", "day", "ds", "transaction_date", "transfusion_date",
        "التاريخ", "تاريخ", "اليوم"]
TYPE = ["blood_type", "blood_group", "bloodtype", "type", "group",
        "patient_blood_group", "الفصيلة", "فصيلة", "فصيلة_الدم"]
VALUE = ["units_consumed", "units_donated", "consumed", "donated", "units",
         "quantity", "qty", "count", "الكمية", "عدد_الوحدات", "الوحدات", "عدد"]
BATCH = {
    "batch_id": ["batch_id", "batch", "id", "bag_id", "رقم_الدفعة"],
    "units": ["units", "quantity", "qty", "الكمية", "عدد_الوحدات"],
    "units_remaining": ["units_remaining", "remaining", "المتبقي"],
    "collection_date": ["collection_date", "collected", "date", "تاريخ_التبرع"],
    "expiry_date": ["expiry_date", "expiry", "expiration_date", "تاريخ_الانتهاء"],
    "status": ["status", "state", "الحالة"],
}
FILES = {
    "consumption": ["consumption", "usage", "transfusion", "blood_data", "استهلاك"],
    "donations": ["donation", "تبرع"],
    "batches": ["batch", "inventory", "دفعات"],
    "stock": ["stock", "مخزون"],
}
AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


# ---------- أدوات صغيرة ----------
def clean_name(c):
    """ "Blood Group " ← "blood_group" """
    return re.sub(r"[\s\-\.]+", "_", str(c).replace("\ufeff", "").strip().lower())


def find_col(df, names):
    """بيرجع اسم العمود الموجود فعلًا في الملف من قايمة الأسماء المقبولة"""
    cols = {clean_name(c): c for c in df.columns}
    for n in names:
        if clean_name(n) in cols:
            return cols[clean_name(n)]
    return None


def norm_type(x):
    """ O pos / O positive / 0+ / O موجب / أ سالب ← O+ / A- """
    s = str(x).strip()
    for w in ["موجب", "positive", "pos"]:
        s = re.sub(w, "+", s, flags=re.I)
    for w in ["سالب", "negative", "neg"]:
        s = re.sub(w, "-", s, flags=re.I)
    s = s.replace("أب", "AB").replace("أ", "A").replace("ب", "B").replace("−", "-")
    s = re.sub(r"[\s_]|RH", "", s.upper()).replace("0", "O")
    return s if s in TYPES else None


def to_date(col):
    """بيجرب شهر/يوم ويوم/شهر ويختار اللي فهم تواريخ أكتر"""
    txt = col.astype(str).str.translate(AR_DIGITS)
    a = pd.to_datetime(txt, errors="coerce", format="mixed")
    b = pd.to_datetime(txt, errors="coerce", format="mixed", dayfirst=True)
    return (b if b.notna().sum() > a.notna().sum() else a).dt.normalize()


def to_num(col):
    return pd.to_numeric(col.astype(str).str.translate(AR_DIGITS)
                         .str.replace(",", ""), errors="coerce")


def read_file(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "cp1256"):          # cp1256 = عربي ويندوز
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            pass
    raise ValueError(f"مش قادرة أقرا {path}")


def find_file(kind):
    files = [f for f in sorted(os.listdir(DATA_DIR))
             if f.lower().endswith((".csv", ".xlsx", ".xls"))]
    for k in FILES[kind]:            # الكلمة الأولى في القايمة ليها الأولوية
        for f in files:
            if k in f.lower():
                return os.path.join(DATA_DIR, f)
    return None


# ---------- الاستهلاك والتبرعات ----------
def load_daily(kind, value_name):
    path = find_file(kind)
    if path is None:
        return None
    df = read_file(path).dropna(how="all")
    comp = find_col(df, ["component", "product", "المكون"])
    if comp is not None:                         # كرات حمراء بس (مش بلازما وصفائح)
        df = df[df[comp].astype(str).str.lower().str.contains("rbc|red|packed|حمراء")]
    date_col = find_col(df, DATE)
    if date_col is None:
        raise ValueError(f"{path}: مش لاقية عمود التاريخ. الأعمدة: {list(df.columns)}")

    type_cols = [c for c in df.columns if norm_type(c)]
    if len(type_cols) >= 2:                      # شكل عريض: عمود لكل فصيلة
        df = df.melt(id_vars=[date_col], value_vars=type_cols,
                     var_name="blood_type", value_name="v")
    else:
        type_col, val_col = find_col(df, TYPE), find_col(df, VALUE)
        if type_col is None:
            raise ValueError(f"{path}: مش لاقية عمود الفصيلة")
        df = df.rename(columns={type_col: "blood_type"})
        df["v"] = df[val_col] if val_col else 1   # مفيش كمية: كل صف = وحدة
    df = df.rename(columns={date_col: "date"})[["date", "blood_type", "v"]]

    df["date"] = to_date(df["date"])
    df["blood_type"] = df["blood_type"].map(norm_type)
    df["v"] = to_num(df["v"]).clip(lower=0)      # السالب ← صفر
    bad = df.isna().any(axis=1).sum()
    df = df.dropna()
    if bad:
        print(f"   {kind}: شلت {bad} صف خربان")

    # التكرار: لو أغلب الأيام صف واحد يبقى التكرار غلط، غير كده سجل عمليات فبنجمع
    if df.duplicated(["date", "blood_type"]).mean() < 0.10:
        df = df.drop_duplicates(["date", "blood_type"], keep="last")
    df = df.groupby(["date", "blood_type"], as_index=False)["v"].sum()

    # الأيام الناقصة: يوم مفيهوش أي تسجيل نكمّله من اللي قبله وبعده، والباقي صفر
    days = pd.date_range(df["date"].min(), df["date"].max())
    recorded = set(df["date"])
    out = []
    for bt, g in df.groupby("blood_type"):
        v = g.set_index("date")["v"].reindex(days)
        v[v.isna() & v.index.isin(list(recorded))] = 0
        v = v.interpolate(limit_direction="both").round()
        out.append(pd.DataFrame({"date": days, "blood_type": bt, value_name: v.values}))
    out = pd.concat(out, ignore_index=True)
    print(f"   {os.path.basename(path)}: {len(out)} صف | "
          f"{out['date'].min().date()} → {out['date'].max().date()} | "
          f"{out['blood_type'].nunique()} فصائل")
    return out


def load_consumption():
    df = load_daily("consumption", "units_consumed")
    if df is None:
        sys.exit(f"❌ مفيش ملف استهلاك في {DATA_DIR}")
    return df


def load_donations():
    df = load_daily("donations", "units_donated")
    if df is None:
        print("   ⚠️ مفيش ملف تبرعات، هعتبرها صفر")
        df = pd.DataFrame({"date": pd.to_datetime([]), "blood_type": [],
                           "units_donated": []})
    return df


# ---------- الدفعات والمخزون ----------
def load_batches(today, required=False):
    path = find_file("batches")
    if path is None:
        if required:
            sys.exit("❌ الكود ده محتاج ملف دفعات (inventory_batches)")
        return pd.DataFrame({"blood_type": [], "units_remaining": [], "status": [],
                             "collection_date": pd.to_datetime([]),
                             "expiry_date": pd.to_datetime([])})
    raw = read_file(path).dropna(how="all")
    b = pd.DataFrame({"blood_type": raw[find_col(raw, TYPE)].map(norm_type)})
    for name, options in BATCH.items():
        col = find_col(raw, options)
        if col is not None:
            b[name] = raw[col]
    for c in ["collection_date", "expiry_date"]:
        if c in b:
            b[c] = to_date(b[c])
    if "expiry_date" not in b:                   # مفيش تاريخ انتهاء: التبرع + 35
        b["expiry_date"] = b["collection_date"] + pd.Timedelta(days=SHELF_LIFE)
    if "collection_date" not in b:
        b["collection_date"] = b["expiry_date"] - pd.Timedelta(days=SHELF_LIFE)
    b["units"] = to_num(b["units"]) if "units" in b else 1
    if "units_remaining" in b:
        b["units_remaining"] = to_num(b["units_remaining"])
    else:                                        # من عمود الحالة لو موجود
        b["units_remaining"] = b["units"]
        if "status" in b:
            used = b["status"].astype(str).str.lower().str.contains(
                "consum|used|expired|مستخدم|منتهي|اتصرف")
            b.loc[used, "units_remaining"] = 0
    b = b.dropna(subset=["blood_type", "units", "expiry_date", "collection_date"])

    # الحالة بتتحسب من جديد حسب «النهاردة»
    b["status"] = "consumed"
    left = b["units_remaining"] > 0
    b.loc[left & (b["expiry_date"] <= today), "status"] = "expired"
    b.loc[left & (b["expiry_date"] > today)
          & (b["collection_date"] <= today), "status"] = "available"
    print(f"   {os.path.basename(path)}: {len(b)} دفعة، "
          f"{int(b.loc[b.status == 'available', 'units_remaining'].sum())} وحدة متاحة")
    return b.reset_index(drop=True)


def real_stock(batches):
    """المخزون الحالي: من ملف المخزون لو موجود، غير كده من الدفعات المتاحة"""
    path = find_file("stock")
    if path and path != find_file("batches"):
        s = read_file(path)
        type_cols = [c for c in s.columns if norm_type(c)]
        if len(type_cols) >= 2:                  # صف واحد: عمود لكل فصيلة
            stock = {norm_type(c): to_num(s[c]).iloc[-1] for c in type_cols}
        else:
            stock = dict(zip(s[find_col(s, TYPE)].map(norm_type),
                             to_num(s[find_col(s, VALUE + ["stock"])])))
    else:
        stock = (batches[batches["status"] == "available"]
                 .groupby("blood_type")["units_remaining"].sum().to_dict())
    if not stock:
        print("   ⚠️ مفيش دفعات ولا ملف مخزون: المخزون صفر. استخدمي TEST_STOCK")
    return {t: int(stock.get(t, 0) or 0) for t in TYPES}
