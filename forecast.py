#!/usr/bin/env python
# =============================================================
# 需要予測フル版 2025-07  (Python 3.9+)
# =============================================================
import os, json, base64, re, unicodedata, logging
from datetime import date, timedelta, datetime
import numpy as np, pandas as pd, requests, gspread
from google.oauth2.service_account import Credentials
from catboost import CatBoostRegressor

# ---------- 0) 環境変数 ----------
SID             = os.getenv("GSHEET_ID")                # ★必須
SA_JSON_RAW     = os.getenv("GSPREAD_SA_JSON")          # ★必須
DB_SHEET        = os.getenv("DB_SHEET",        "データベース")
FC_SHEET        = os.getenv("FORECAST_SHEET",  "需要予測")
FORECAST_DAYS   = int(os.getenv("FORECAST_DAYS", 7))
LABEL_ROWS      = int(os.getenv("LABEL_ROWS", 10))
LAT, LON        = 36.3740, 140.5662                     # 天気座標

# ---------- logger ----------
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- util ----------
def load_sa():
    raw = SA_JSON_RAW or ""
    j = json.loads(raw) if raw.lstrip().startswith("{") else json.loads(base64.b64decode(raw))
    return Credentials.from_service_account_info(j,
        scopes=["https://www.googleapis.com/auth/spreadsheets"])

def num_clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
         .str.replace(r'[^\d.\-]', '', regex=True)
         .str.replace('．', '.', regex=False)
         .replace('', np.nan),
        errors='coerce').fillna(0)

def norm(txt):
    return re.sub(r'[ 　【】\[\]\(\)]', '',
                  unicodedata.normalize('NFKC', str(txt))).lower()

def weather_forecast(lat, lon, days):
    url = ("https://api.open-meteo.com/v1/forecast?"
           f"latitude={lat}&longitude={lon}"
           "&daily=weathercode,temperature_2m_max,temperature_2m_min"
           "&timezone=Asia%2FTokyo")
    j = requests.get(url, timeout=10).json().get("daily", {})
    return (pd.DataFrame({"dt": pd.to_datetime(j["time"]),
                          "code": j["weathercode"],
                          "tmax": j["temperature_2m_max"],
                          "tmin": j["temperature_2m_min"]})
              .set_index("dt").iloc[:days])

ROKUYO = ["先勝","友引","先負","仏滅","大安","赤口"]
def get_rokuyo(start: date, days: int):
    base = datetime(1900,1,1)  # この日が大安
    return [ROKUYO[( (start + timedelta(i)) - base.date()).days % 6]
            for i in range(days)]

# ---------- ML ----------
def catboost_pred(y: pd.Series, horizon: int, feats_extra=None) -> np.ndarray:
    idx = y.dropna().index
    if idx.empty or y.sum() == 0:
        return np.zeros(horizon)
    df = pd.DataFrame({"y": y.reindex(idx)}).copy()
    df["dow"], df["month"] = idx.weekday, idx.month
    if feats_extra:
        for k, s in feats_extra.items():
            df[k] = s.reindex(idx).fillna(s.mean())
    mdl = CatBoostRegressor(depth=6, learning_rate=0.15,
                            loss_function="RMSE", verbose=False, random_state=42)
    mdl.fit(df.drop(columns="y"), df["y"], cat_features=["dow"])
    fut_idx = pd.date_range(y.index.max() + timedelta(1), periods=horizon)
    fut = pd.DataFrame({"dow": fut_idx.weekday, "month": fut_idx.month})
    if feats_extra:
        for k, s in feats_extra.items():
            fut[k] = s.tail(30).mean()
    return np.clip(mdl.predict(fut), 0, None)

# ---------- main ----------
def main():
    gc = gspread.authorize(load_sa())
    ws_db = gc.open_by_key(SID).worksheet(DB_SHEET)
    raw   = ws_db.get_all_values()
    df_db = pd.DataFrame(raw); df_db.columns = df_db.iloc[0]; df_db = df_db.drop(0)

    # 日付列
    date_cols = pd.to_datetime(df_db.columns[1:], errors='coerce')
    mask_date = ~date_cols.isna()
    dates     = date_cols[mask_date]

    # 実績 wide
    wide = df_db.set_index(df_db.columns[0]).drop(df_db.columns[0], axis=1)
    wide = wide.iloc[:, mask_date.values]              # 日付列のみ
    wide.columns = dates                               # ★ ← 修正：列数一致
    wide = wide.apply(num_clean, axis=1)

    # 予報
    start = date.today() + timedelta(1)
    fut_dates = pd.date_range(start, periods=FORECAST_DAYS)
    wdf = weather_forecast(LAT, LON, FORECAST_DAYS).reindex(fut_dates, method="nearest")
    rokuyo_seq = get_rokuyo(start, FORECAST_DAYS)

    # 追加特徴
    extra = {
        "tmax": wide.loc.get("最高気温", pd.Series(index=dates)),
        "tmin": wide.loc.get("最低気温", pd.Series(index=dates)),
    }

    target_rows = ["売上","客数","客単価"] + list(wide.index.drop(
                     ["売上","客数","客単価"], errors='ignore'))

    pred = {name: catboost_pred(wide.loc[name], FORECAST_DAYS, extra)
            for name in target_rows}

    # ---------- 需要予測シート ----------
    sh = gc.open_by_key(SID)
    ws_fc = (sh.worksheet(FC_SHEET) if FC_SHEET in [w.title for w in sh.worksheets()]
             else sh.add_worksheet(FC_SHEET, rows=2000, cols=400))
    ws_fc.resize(rows=LABEL_ROWS + len(target_rows), cols=1 + FORECAST_DAYS)

    # ヘッダ + メタ
    ws_fc.update("A1", [["日付"] + [d.strftime("%Y/%m/%d") for d in fut_dates]])
    meta = [
        ["曜日"]      + [ "月火水木金土日"[d.weekday()] for d in fut_dates],
        ["六曜"]      + rokuyo_seq,
        ["年中行事"]  + [""]*FORECAST_DAYS,
        ["天気"]      + wdf["code"].map({
            0:"快晴",1:"晴",2:"薄曇",3:"曇",45:"霧",51:"霧雨",61:"小雨",
            63:"雨",65:"大雨",71:"雪",80:"にわか雨",95:"雷雨"}).fillna("－").tolist(),
        ["最高気温"]  + wdf["tmax"].round(1).tolist(),
        ["最低気温"]  + wdf["tmin"].round(1).tolist(),
    ]
    ws_fc.update("A2", meta)

    # 集計予測
    for i, row in enumerate(["売上","客数","客単価"], start=8):
        ws_fc.update_cell(i, 1, row)
        ws_fc.update(f"B{i}", [pred[row].round(1).tolist()])

    # 授与品
    start_row = LABEL_ROWS + 1
    ws_fc.update(f"A{start_row}", [[r] for r in target_rows[3:]])
    ws_fc.update(f"B{start_row}",
                 [pred[r].round().astype(int).tolist() for r in target_rows[3:]],
                 value_input_option="USER_ENTERED")

    logging.info("✅ 需要予測シート更新完了")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"🚨 Fatal: {e}")
