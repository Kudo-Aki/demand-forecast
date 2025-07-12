#!/usr/bin/env python
# ==================================================================
# 需要予測フル版 2025-07 – recency-weighted & accuracy log (dup-index fix)
# ==================================================================
import os, json, base64, re, unicodedata, logging, warnings
from datetime import date, timedelta, datetime
import numpy as np, pandas as pd, requests, gspread
from google.oauth2.service_account import Credentials
from catboost import CatBoostRegressor

# ---------- 設定 ----------
SID          = os.getenv("GSHEET_ID")
SA_JSON      = os.getenv("GSPREAD_SA_JSON")
DB_SHEET     = os.getenv("DB_SHEET",      "データベース")
FC_SHEET     = os.getenv("FORECAST_SHEET","需要予測")
HIST_SHEET   = os.getenv("HISTORY_SHEET", "予測履歴")
METRIC_SHEET = os.getenv("METRIC_SHEET",  "予測精度")
FORECAST_D   = int(os.getenv("FORECAST_DAYS", 7))
LABEL_ROWS   = int(os.getenv("LABEL_ROWS", 10))
LAT, LON     = 36.3740, 140.5662

META_ROWS = ["曜日","六曜","年中行事","天気","最高気温","最低気温"]

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- util ----------
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

def norm(t):
    return re.sub(r"[ 　【】\[\]\(\)]", "",
                  unicodedata.normalize("NFKC", str(t))).lower()

def fuzzy_row(df, key):
    nk = norm(key)
    for r in df.index:
        if nk in norm(r):
            return r
    return None

def ensure_series(x):          # DataFrame → 先頭行
    return x.iloc[0] if isinstance(x, pd.DataFrame) else x

W2C = {"快晴":0,"晴":1,"薄曇":2,"曇":3,"霧":45,"霧雨":51,"小雨":61,"雨":63,
       "大雨":65,"小雪":71,"雪":73,"大雪":75,"にわか雨":80,"雷雨":95,
       "—":np.nan,"ｰ":np.nan,"":np.nan}

def weather_forecast(days):
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

# ---------- モデル ----------
def cat_predict(y: pd.Series,
                X_extra: pd.DataFrame,
                Xf_extra: pd.DataFrame) -> np.ndarray:
    y = ensure_series(y).astype(float)
    idx = y.dropna().index
    if idx.empty or float(y.sum()) == 0:
        return np.zeros(len(Xf_extra))

    X  = pd.DataFrame({
        "dow": idx.weekday.astype(int),
        "mon": idx.month.astype(int),
        "doy": idx.dayofyear.astype(int)
    }, index=idx).join(X_extra.reindex(idx))

    Xf = pd.DataFrame({
        "dow": Xf_extra.index.weekday.astype(int),
        "mon": Xf_extra.index.month.astype(int),
        "doy": Xf_extra.index.dayofyear.astype(int)
    }, index=Xf_extra.index).join(Xf_extra)

    w = np.linspace(1.0, 2.0, len(idx))         # ndarray OK

    model = CatBoostRegressor(depth=8, learning_rate=0.1,
                              loss_function="RMSE",
                              random_state=42, verbose=False)
    model.fit(X, y.loc[idx], sample_weight=w, cat_features=["dow","mon"])
    return np.clip(model.predict(Xf), 0, None)

# ---------- main ----------
def main():
    gc  = gspread.authorize(creds())
    sh  = gc.open_by_key(SID)

    # --- データベース読み込み
    df0 = pd.DataFrame(sh.worksheet(DB_SHEET).get_all_values())
    df0.columns; df = df0.drop(0)
    df.columns = df0.iloc[0]

    dates = pd.to_datetime(df.columns[1:], errors="coerce")
    wide  = df.set_index(df.columns[0]).iloc[:, ~dates.isna()]
    wide.columns = dates[~dates.isna()]
    wide = wide.apply(num_clean, axis=1)

    # --- 行を特定
    r_sales = fuzzy_row(wide, "売上");  r_cust  = fuzzy_row(wide, "客数")
    r_unit  = fuzzy_row(wide, "客単価"); r_tmax  = fuzzy_row(wide, "最高気温")
    r_tmin  = fuzzy_row(wide, "最低気温"); r_wtxt = fuzzy_row(wide, "天気")
    wcode_hist = wide.loc[r_wtxt].replace(W2C).astype(float) if r_wtxt else pd.Series(index=wide.columns, dtype=float)

    # --- 未来外生
    start   = date.today()+timedelta(1)
    fut_idx = pd.date_range(start, periods=FORECAST_D)
    wdf     = weather_forecast(FORECAST_D).reindex(fut_idx, method="nearest")

    Xf_extra = pd.DataFrame({
        "code": wdf["code"].astype(float),
        "tmax": wdf["tmax"].astype(float),
        "tmin": wdf["tmin"].astype(float)
    }, index=fut_idx)

    X_extra = pd.DataFrame({
        "code": wcode_hist,
        "tmax": ensure_series(wide.loc[r_tmax]) if r_tmax else np.nan,
        "tmin": ensure_series(wide.loc[r_tmin]) if r_tmin else np.nan
    })

    # --- 予測
    agg_lbls = ["売上","客数","客単価"];    agg_rows = [r_sales, r_cust, r_unit]
    item_rows = [r for r in wide.index
                 if r not in META_ROWS+agg_lbls and not norm(r).startswith("天気")]
    labels = agg_lbls + item_rows;       rows = agg_rows + item_rows
    preds  = {l: cat_predict(wide.loc[r], X_extra, Xf_extra) for l,r in zip(labels,rows)}

    # ---------- 1) 需要予測シート ----------
    ws = sh.worksheet(FC_SHEET) if FC_SHEET in [w.title for w in sh.worksheets()] \
         else sh.add_worksheet(FC_SHEET, rows=2000, cols=400)
    ws.resize(rows=LABEL_ROWS+len(labels), cols=1+FORECAST_D)

    header = [["日付"]+[d.strftime("%Y/%m/%d") for d in fut_idx]]
    meta   = [
        ["曜日"] + ["月火水木金土日"[d.weekday()] for d in fut_idx],
        ["六曜"] + rokuyo(start, FORECAST_D),
        ["年中行事"] + [""]*FORECAST_D,
        ["天気"] + wdf["code"].map({
            0:"快晴",1:"晴",2:"薄曇",3:"曇",45:"霧",51:"霧雨",61:"小雨",
            63:"雨",65:"大雨",71:"雪",75:"大雪",80:"にわか雨",95:"雷雨"}).fillna("－").tolist(),
        ["最高気温"] + wdf["tmax"].round(1).tolist(),
        ["最低気温"] + wdf["tmin"].round(1).tolist()
    ]
    ws.update(values=header+meta, range_name="A1")

    body=[[lbl]+(preds[lbl].round(1) if lbl in agg_lbls
                 else preds[lbl].round().astype(int)).tolist()
          for lbl in labels]
    ws.update(values=body, range_name=f"A{LABEL_ROWS+1}")

    # ---------- 2) 予測履歴 ----------
    hist_ws = sh.worksheet(HIST_SHEET) if HIST_SHEET in [w.title for w in sh.worksheets()] \
              else sh.add_worksheet(HIST_SHEET, rows=1, cols=5)
    if hist_ws.row_count == 0 or hist_ws.cell(1,1).value != "run_date":
        hist_ws.update(values=[["run_date","target_date","label","pred"]],
                       range_name="A1")

    run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hist_rows = [[run_date, td.strftime("%Y-%m-%d"), lbl, float(val)]
                 for lbl in labels for td, val in zip(fut_idx, preds[lbl])]
    hist_ws.append_rows(hist_rows, value_input_option="USER_ENTERED")

    # ---------- 3) 精度レポート ----------
    hist_df = pd.DataFrame(hist_ws.get_all_values()[1:],
                           columns=["run","target","label","pred"])
    hist_df["target"] = pd.to_datetime(hist_df["target"])
    hist_df["pred"]   = pd.to_numeric(hist_df["pred"], errors="coerce")

    # ★ duplicate-index safe dict
    actual_map = {
        lab: ensure_series(wide.loc[lab]) if lab in wide.index
             else pd.Series(index=wide.columns, dtype=float)
        for lab in wide.index.unique()
    }

    rec=[]; cutoff=date.today()
    for _, r in hist_df.iterrows():
        if r["target"].date() >= cutoff: continue
        lab=r["label"]; d=r["target"]
        if lab in actual_map and d in actual_map[lab]:
            act=actual_map[lab][d]
            if pd.notna(act):
                err=abs(act-r["pred"]); ape=err/act*100 if act else np.nan
                rec.append([lab, err, ape])
    if rec:
        rep=(pd.DataFrame(rec, columns=["label","ae","ape"])
               .groupby("label").agg(MAE=("ae","mean"),MAPE=("ape","mean"))
               .reset_index().round(2))
        met_ws = sh.worksheet(METRIC_SHEET) if METRIC_SHEET in [w.title for w in sh.worksheets()] \
                 else sh.add_worksheet(METRIC_SHEET, rows=2000, cols=10)
        met_ws.clear()
        met_ws.update(values=[rep.columns.tolist()]+rep.values.tolist(),
                      range_name="A1")

    logging.info("✅ 完了 — duplicate-index fix & all sheets updated")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("🚨 Fatal")
