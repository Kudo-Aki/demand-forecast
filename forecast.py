#!/usr/bin/env python
# =============================================================
# 需要予測フル版 2025-07  (Python 3.9+)
# - 曜日, 六曜, 天気(予報), 気温 も書き込み
# - 売上 / 客数 / 客単価 を集計予測
# - 授与品別も従来通り予測
# =============================================================
import os, json, base64, re, unicodedata, logging, textwrap
from datetime import date, timedelta, datetime
import numpy as np
import pandas as pd
import requests, gspread
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
TZ              = os.getenv("TIMEZONE", "Asia/Tokyo")

# ---------- logger ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------- util ----------
def load_sa():
    raw = SA_JSON_RAW or ""
    j = json.loads(raw) if raw.lstrip().startswith("{") else json.loads(base64.b64decode(raw))
    return Credentials.from_service_account_info(j, scopes=[
        "https://www.googleapis.com/auth/spreadsheets"
    ])

def num_clean(s: pd.Series) -> pd.Series:
    """全角→半角, 数字以外除去, 空→0"""
    return pd.to_numeric(
        s.astype(str)
         .str.replace(r'[^\d.\-]', '', regex=True)
         .str.replace('．', '.', regex=False)
         .replace('', np.nan),
        errors='coerce'
    ).fillna(0)

def norm(txt):  # 列名マッチ用
    return re.sub(r'[ 　【】\[\]\(\)]', '',
                  unicodedata.normalize('NFKC', str(txt))).lower()

def weather_forecast(lat, lon, days):
    """Open-Meteo 7日予報 (daily)"""
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&daily=weathercode,temperature_2m_max,temperature_2m_min"
        "&timezone=Asia%2FTokyo"
    )
    j = requests.get(url, timeout=10).json().get("daily", {})
    return pd.DataFrame({
        "dt": pd.to_datetime(j["time"]),
        "code": j["weathercode"],
        "tmax": j["temperature_2m_max"],
        "tmin": j["temperature_2m_min"],
    }).set_index("dt").iloc[:days]

ROKUYO = ["先勝","友引","先負","仏滅","大安","赤口"]
def get_rokuyo(start: date, days: int):
    """六曜カレンダーを簡易生成（固定周期ではなく本来は API 必要）。
       精度が要る場合は CAL ics をパースした関数に置換してください。
    """
    base = datetime(1900,1,1)  # この日が大安
    seq  = []
    for d in (start + timedelta(n) for n in range(days)):
        idx = (d - base.date()).days % 6
        seq.append(ROKUYO[idx])
    return seq

# ---------- CatBoost Helper ----------
def catboost_pred(y: pd.Series, horizon: int, feats_extra=None) -> np.ndarray:
    idx = y.dropna().index
    if idx.empty or y.sum() == 0:
        return np.zeros(horizon)
    df = pd.DataFrame({"y": y.reindex(idx)}).copy()
    df["dow"] = idx.weekday
    df["month"] = idx.month
    if feats_extra is not None:
        for k, s in feats_extra.items():
            df[k] = s.reindex(idx).fillna(s.mean())
    model = CatBoostRegressor(depth=6, learning_rate=0.15,
                              loss_function="RMSE", verbose=False, random_state=42)
    model.fit(df.drop(columns="y"), df["y"], cat_features=["dow"])
    fut_idx = pd.date_range(y.index.max() + timedelta(1), periods=horizon)
    fut = pd.DataFrame({"dow": fut_idx.weekday, "month": fut_idx.month})
    if feats_extra is not None:
        for k, s in feats_extra.items():
            fut[k] = s.tail(30).mean()
    return np.clip(model.predict(fut), 0, None)

# ---------- main ----------
def main():
    ## 1) Sheets 読込
    gc = gspread.authorize(load_sa())
    ws_db = gc.open_by_key(SID).worksheet(DB_SHEET)
    raw   = ws_db.get_all_values()
    df_db = pd.DataFrame(raw); df_db.columns = df_db.iloc[0]; df_db = df_db.drop(0)
    # 日付列
    date_cols = pd.to_datetime(df_db.columns[1:], errors='coerce')
    mask_date = ~date_cols.isna()
    dates     = date_cols[mask_date]
    # 実績行 Series 化
    wide = df_db.set_index(df_db.columns[0])
    wide.columns = list(df_db.columns[:1]) + list(dates)  # datetime index
    wide = wide.drop(df_db.columns[0], axis=1)
    wide = wide.apply(num_clean, axis=1)

    ## 2) 予報データ
    START = date.today() + timedelta(1)
    fut_dates = pd.date_range(START, periods=FORECAST_DAYS)
    wdf = weather_forecast(LAT, LON, FORECAST_DAYS).reindex(fut_dates, method="nearest")
    rokuyo_seq = get_rokuyo(START, FORECAST_DAYS)

    ## 3) 予測ループ
    pred_dict = {}
    extra_feats = {
        "tmax": pd.to_numeric(wide.loc.get("最高気温", pd.Series(index=dates)), errors='coerce'),
        "tmin": pd.to_numeric(wide.loc.get("最低気温", pd.Series(index=dates)), errors='coerce'),
    }
    target_rows = ["売上", "客数", "客単価"] + list(wide.index.drop(["売上","客数","客単価"], errors='ignore'))

    for name in target_rows:
        y = wide.loc.get(name)
        if y is None:
            logging.warning(f"行 '{name}' が見つかりません。スキップ")
            continue
        pred_dict[name] = catboost_pred(y, FORECAST_DAYS, extra_feats)

    ## 4) 需要予測シート書込
    ws_fc = gc.open_by_key(SID).worksheet(FC_SHEET) if FC_SHEET in [w.title for w in gc.open_by_key(SID).worksheets()] \
            else gc.open_by_key(SID).add_worksheet(FC_SHEET, rows=2000, cols=400)

    # サイズ調整
    ws_fc.resize(rows=LABEL_ROWS + len(target_rows), cols=1 + FORECAST_DAYS)

    # 1 行目: 日付
    ws_fc.update("A1", [["日付"] + [d.strftime("%Y/%m/%d") for d in fut_dates]])

    # 2～7 行目: メタ
    meta_rows = [
        ["曜日"]   + list("月火水木金土日"[d.weekday()] for d in fut_dates),
        ["六曜"]   + rokuyo_seq,
        ["年中行事"] + [""]*FORECAST_DAYS,          # 空けておく
        ["天気"]   + wdf["code"].map({
            0:"快晴",1:"晴",2:"薄曇",3:"曇",45:"霧",51:"霧雨",61:"小雨",
            63:"雨",65:"大雨",71:"雪",80:"にわか雨",95:"雷雨"
        }).fillna("－").tolist(),
        ["最高気温"] + wdf["tmax"].round(1).tolist(),
        ["最低気温"] + wdf["tmin"].round(1).tolist(),
    ]
    ws_fc.update(f"A2", meta_rows)

    # 8～10 行: 集計予測
    for i, row_name in enumerate(["売上","客数","客単価"], start=8):
        ws_fc.update_cell(i, 1, row_name)
        ws_fc.update(f"B{i}", [pred_dict.get(row_name, np.zeros(FORECAST_DAYS)).round(1).tolist()])

    # 11 行目以降: 授与品
    start_row = LABEL_ROWS + 1
    ws_fc.update(f"A{start_row}", [[r] for r in target_rows[3:]])  # アイテム名
    value_block = [pred_dict.get(n, np.zeros(FORECAST_DAYS)).round().astype(int).tolist()
                   for n in target_rows[3:]]
    ws_fc.update(f"B{start_row}", value_block, value_input_option="USER_ENTERED")

    logging.info("✅ 需要予測シート更新完了")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"🚨 Fatal: {e}")
