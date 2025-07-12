#!/usr/bin/env python
# =============================================================
# 需要予測フル版 2025-07  – batch write / no duplicates
# =============================================================
import os, json, base64, re, unicodedata, logging, warnings
from datetime import date, timedelta, datetime
import numpy as np, pandas as pd, requests, gspread
from google.oauth2.service_account import Credentials
from catboost import CatBoostRegressor

# ---------- 設定 ----------
SID         = os.getenv("GSHEET_ID")
SA_JSON     = os.getenv("GSPREAD_SA_JSON")
DB_SHEET    = os.getenv("DB_SHEET", "データベース")
FC_SHEET    = os.getenv("FORECAST_SHEET", "需要予測")
FORECAST_D  = int(os.getenv("FORECAST_DAYS", 7))
LABEL_ROWS  = int(os.getenv("LABEL_ROWS", 10))
LAT, LON    = 36.3740, 140.5662

META_ROWS = ["曜日","六曜","年中行事","天気","最高気温","最低気温"]  # ← 重複除外用

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- ユーティリティ ----------
def creds():
    raw = SA_JSON or ""
    data = json.loads(raw) if raw.lstrip().startswith("{") \
           else json.loads(base64.b64decode(raw))
    return Credentials.from_service_account_info(
        data, scopes=["https://www.googleapis.com/auth/spreadsheets"])

def num_clean(s):
    return (pd.to_numeric(
            s.astype(str).str.replace(r"[^\d.\-]", "", regex=True)
             .str.replace("．", ".", regex=False)
             .replace("", np.nan),
            errors="coerce").fillna(0))

def norm(t):  # 全半角・カッコ無視
    return re.sub(r"[ 　【】\[\]\(\)]", "",
                  unicodedata.normalize("NFKC", str(t))).lower()

def fuzzy_row(df, key):
    nk = norm(key)
    for r in df.index:
        if nk in norm(r):
            return r
    return None

def ensure_series(x):
    return x.iloc[0] if isinstance(x, pd.DataFrame) else x

def weather(days):
    url = (f"https://api.open-meteo.com/v1/forecast?"
           f"latitude={LAT}&longitude={LON}"
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
    return [ROKUYO[((start+timedelta(i))-base.date()).days % 6] for i in range(days)]

def cat_pred(y, horizon, extra=None):
    y = ensure_series(y).astype(float)
    idx = y.dropna().index
    if idx.empty or float(y.sum()) == 0:
        return np.zeros(horizon)

    X = pd.DataFrame({
        "dow": idx.weekday,
        "mon": idx.month,
        "doy": idx.dayofyear   # ← 通日を追加
    }, index=idx)
    if extra:
        for k,s in extra.items():
            X[k] = s.reindex(idx).fillna(s.mean())

    model = CatBoostRegressor(depth=6, learning_rate=0.15,
                              loss_function="RMSE", random_state=42,
                              verbose=False)
    model.fit(X, y)

    fut = pd.date_range(idx.max()+timedelta(1), periods=horizon)
    Xf = pd.DataFrame({
        "dow": fut.weekday,
        "mon": fut.month,
        "doy": fut.dayofyear
    }, index=fut)
    if extra:
        for k,s in extra.items():
            Xf[k] = s.tail(30).mean()

    return np.clip(model.predict(Xf), 0, None)

# ---------- main ----------
def main():
    gc   = gspread.authorize(creds())
    wsdb = gc.open_by_key(SID).worksheet(DB_SHEET)
    df   = pd.DataFrame(wsdb.get_all_values())
    df.columns, df = df.iloc[0], df.drop(0)

    dates = pd.to_datetime(df.columns[1:], errors="coerce")
    valid = ~dates.isna()
    wide  = df.set_index(df.columns[0]).iloc[:, valid]
    wide.columns = dates[valid]
    wide = wide.apply(num_clean, axis=1)

    # 対象行抽出
    rows_remove = META_ROWS + ["売上","客数","客単価"]
    r_sales = fuzzy_row(wide, "売上")
    r_cust  = fuzzy_row(wide, "客数")
    r_unit  = fuzzy_row(wide, "客単価")
    r_tmax  = fuzzy_row(wide, "最高気温")
    r_tmin  = fuzzy_row(wide, "最低気温")

    start   = date.today()+timedelta(1)
    fut_idx = pd.date_range(start, periods=FORECAST_D)
    wdf     = weather(FORECAST_D).reindex(fut_idx, method="nearest")

    extra = {
        "tmax": ensure_series(wide.loc[r_tmax]) if r_tmax else pd.Series(index=wide.columns),
        "tmin": ensure_series(wide.loc[r_tmin]) if r_tmin else pd.Series(index=wide.columns)
    }

    agg_lbls = ["売上","客数","客単価"]
    agg_rows = [r_sales, r_cust, r_unit]
    item_rows = [r for r in wide.index
                 if r not in rows_remove and r not in META_ROWS]
    labels = agg_lbls + item_rows
    rows   = agg_rows + item_rows

    pred = {l: cat_pred(wide.loc[r], FORECAST_D, extra) for l,r in zip(labels,rows)}

    # ---------- Sheets 出力（バッチ書込） ----------
    sh  = gc.open_by_key(SID)
    wso = sh.worksheet(FC_SHEET) if FC_SHEET in [w.title for w in sh.worksheets()] \
          else sh.add_worksheet(FC_SHEET, rows=2000, cols=400)
    wso.resize(rows=LABEL_ROWS+len(labels), cols=1+FORECAST_D)

    # A1〜: ヘッダとメタ
    header = [["日付"]+[d.strftime("%Y/%m/%d") for d in fut_idx]]
    meta   = [
        ["曜日"] + ["月火水木金土日"[d.weekday()] for d in fut_idx],
        ["六曜"] + rokuyo(start, FORECAST_D),
        ["年中行事"] + [""]*FORECAST_D,
        ["天気"] + wdf["code"].astype(int).map({
            0:"快晴",1:"晴",2:"薄曇",3:"曇",45:"霧",51:"霧雨",61:"小雨",
            63:"雨",65:"大雨",71:"雪",80:"にわか雨",95:"雷雨"}).fillna("－").tolist(),
        ["最高気温"] + wdf["tmax"].round(1).tolist(),
        ["最低気温"] + wdf["tmin"].round(1).tolist()
    ]
    wso.update(range_name="A1", values=header+meta)

    # LABEL_ROWS 以降をまとめて書込
    body = []
    for lbl in labels:
        body.append([lbl] + (pred[lbl].round(1) if lbl in agg_lbls
                             else pred[lbl].round().astype(int)).tolist())
    wso.update(range_name=f"A{LABEL_ROWS+1}", values=body)

    logging.info("✅ 完了 — quota を超える書込なし")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("🚨 Fatal")
