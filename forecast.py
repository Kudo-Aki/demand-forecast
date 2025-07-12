#!/usr/bin/env python
# =============================================================
# 需要予測フル版 2025-07 (no-error)
# =============================================================
import os, json, base64, re, unicodedata, logging
from datetime import date, timedelta, datetime
import numpy as np, pandas as pd, requests, gspread
from google.oauth2.service_account import Credentials
from catboost import CatBoostRegressor

# ---------- env ----------
SID         = os.getenv("GSHEET_ID")
SA_JSON     = os.getenv("GSPREAD_SA_JSON")
DB_SHEET    = os.getenv("DB_SHEET", "データベース")
FC_SHEET    = os.getenv("FORECAST_SHEET", "需要予測")
FORECAST_D  = int(os.getenv("FORECAST_DAYS", 7))
LABEL_ROWS  = int(os.getenv("LABEL_ROWS", 10))
LAT, LON    = 36.3740, 140.5662

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- helpers ----------
def sa_creds():
    raw = SA_JSON or ""
    data = json.loads(raw) if raw.lstrip().startswith("{") \
           else json.loads(base64.b64decode(raw))
    return Credentials.from_service_account_info(
        data, scopes=["https://www.googleapis.com/auth/spreadsheets"])

def num_clean(s: pd.Series):
    return (pd.to_numeric(
        s.astype(str)
         .str.replace(r"[^\d.\-]", "", regex=True)
         .str.replace("．", ".", regex=False)
         .replace("", np.nan),
        errors="coerce").fillna(0))

def norm(t):
    return re.sub(r"[ 　【】\[\]\(\)]", "",
                  unicodedata.normalize("NFKC", str(t))).lower()

def fuzzy_row(df, key):
    nk = norm(key)
    for r in df.index:
        if nk in norm(r):
            return r
    return None

def ensure_series(obj):
    """DataFrame の場合は先頭行を返す"""
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[0]
    return obj

def weather(lat, lon, days):
    url = ("https://api.open-meteo.com/v1/forecast?"
           f"latitude={lat}&longitude={lon}"
           "&daily=weathercode,temperature_2m_max,temperature_2m_min"
           "&timezone=Asia%2FTokyo")
    d = requests.get(url, timeout=10).json()["daily"]
    return (pd.DataFrame({
        "dt":   pd.to_datetime(d["time"]),
        "code": d["weathercode"],
        "tmax": d["temperature_2m_max"],
        "tmin": d["temperature_2m_min"],
    }).set_index("dt").iloc[:days])

ROKUYO = ["先勝","友引","先負","仏滅","大安","赤口"]
def rokuyo(start, days):
    base = datetime(1900,1,1)
    return [ROKUYO[((start+timedelta(i)) - base.date()).days % 6]
            for i in range(days)]

def cat_pred(y, horizon, extra=None):
    y = ensure_series(y).astype(float)
    idx = y.dropna().index
    if idx.empty or float(y.sum()) == 0:
        return np.zeros(horizon)
    X = pd.DataFrame({"dow": idx.weekday, "mon": idx.month}, index=idx)
    if extra:
        for k, s in extra.items():
            X[k] = s.reindex(idx).fillna(s.mean())
    m = CatBoostRegressor(depth=6, learning_rate=0.15,
                          loss_function="RMSE", random_state=42, verbose=False)
    m.fit(X, y)
    fut = pd.date_range(idx.max()+timedelta(1), periods=horizon)
    Xf = pd.DataFrame({"dow": fut.weekday, "mon": fut.month}, index=fut)
    if extra:
        for k,s in extra.items():
            Xf[k] = s.tail(30).mean()
    return np.clip(m.predict(Xf), 0, None)

# ---------- main ----------
def main():
    # ── read sheet
    gc = gspread.authorize(sa_creds())
    ws = gc.open_by_key(SID).worksheet(DB_SHEET)
    df = pd.DataFrame(ws.get_all_values())
    df.columns, df = df.iloc[0], df.drop(0)

    dates = pd.to_datetime(df.columns[1:], errors="coerce")
    mask  = ~dates.isna()
    wide  = df.set_index(df.columns[0]).iloc[:, mask.values]
    wide.columns = dates[mask]
    wide = wide.apply(num_clean, axis=1)

    # ── rows
    r_sales = fuzzy_row(wide, "売上")
    r_cust  = fuzzy_row(wide, "客数")
    r_unit  = fuzzy_row(wide, "客単価")
    r_tmax  = fuzzy_row(wide, "最高気温")
    r_tmin  = fuzzy_row(wide, "最低気温")

    start = date.today()+timedelta(1)
    fut_d = pd.date_range(start, periods=FORECAST_D)
    wdf   = weather(LAT, LON, FORECAST_D).reindex(fut_d, method="nearest")

    extra = {
        "tmax": ensure_series(wide.loc[r_tmax]) if r_tmax else pd.Series(index=wide.columns),
        "tmin": ensure_series(wide.loc[r_tmin]) if r_tmin else pd.Series(index=wide.columns)
    }

    agg_rows = [r_sales, r_cust, r_unit]
    agg_lbls = ["売上", "客数", "客単価"]
    item_rows = list(wide.index.drop([r for r in agg_rows if r], errors="ignore"))
    lbls = agg_lbls + item_rows
    rows = agg_rows + item_rows

    pred = {l: cat_pred(wide.loc[r], FORECAST_D, extra) for l, r in zip(lbls, rows)}

    # ── write sheet
    sh = gc.open_by_key(SID)
    ws_o = sh.worksheet(FC_SHEET) if FC_SHEET in [w.title for w in sh.worksheets()] \
           else sh.add_worksheet(FC_SHEET, rows=2000, cols=400)
    ws_o.resize(rows=LABEL_ROWS+len(lbls), cols=1+FORECAST_D)

    ws_o.update("A1", [["日付"]+[d.strftime("%Y/%m/%d") for d in fut_d]])
    meta = [
        ["曜日"] + ["月火水木金土日"[d.weekday()] for d in fut_d],
        ["六曜"] + rokuyo(start, FORECAST_D),
        ["年中行事"] + [""]*FORECAST_D,
        ["天気"] + wdf["code"].astype(int).map({
            0:"快晴",1:"晴",2:"薄曇",3:"曇",45:"霧",51:"霧雨",61:"小雨",
            63:"雨",65:"大雨",71:"雪",80:"にわか雨",95:"雷雨"}).fillna("－").tolist(),
        ["最高気温"] + wdf["tmax"].round(1).tolist(),
        ["最低気温"] + wdf["tmin"].round(1).tolist()
    ]
    ws_o.update("A2", meta)

    for i, lbl in enumerate(lbls, start=8):
        vals = pred[lbl]
        ws_o.update_cell(i,1,lbl)
        ws_o.update(f"B{i}", [vals.round(1).tolist() if i<=10 else
                              vals.round().astype(int).tolist()])

    logging.info("✅ Sheet updated without errors")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("🚨 Fatal")
