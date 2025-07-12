#!/usr/bin/env python
# =============================================================
# 需要予測フル版 2025-07  (error-free)
# =============================================================
import os, json, base64, re, unicodedata, logging
from datetime import date, timedelta, datetime
import numpy as np, pandas as pd, requests, gspread
from google.oauth2.service_account import Credentials
from catboost import CatBoostRegressor

# ---------- 環境 ----------
SID             = os.getenv("GSHEET_ID")
SA_JSON_RAW     = os.getenv("GSPREAD_SA_JSON")
DB_SHEET        = os.getenv("DB_SHEET",        "データベース")
FC_SHEET        = os.getenv("FORECAST_SHEET",  "需要予測")
FORECAST_DAYS   = int(os.getenv("FORECAST_DAYS", 7))
LABEL_ROWS      = int(os.getenv("LABEL_ROWS", 10))
LAT, LON        = 36.3740, 140.5662            # 天気座標

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- util ----------
def load_sa():
    raw = SA_JSON_RAW or ""
    info = json.loads(raw) if raw.lstrip().startswith("{") \
           else json.loads(base64.b64decode(raw))
    return Credentials.from_service_account_info(
        info, scopes=["https://www.googleapis.com/auth/spreadsheets"])

def num_clean(s: pd.Series) -> pd.Series:
    return (pd.to_numeric(
            s.astype(str)
             .str.replace(r"[^\d.\-]", "", regex=True)
             .str.replace("．", ".", regex=False)
             .replace("", np.nan),
            errors="coerce")
            .fillna(0))

def norm(text: str) -> str:
    """全角・半角・括弧類を無視した比較用文字列"""
    return re.sub(r"[ 　【】\[\]\(\)]", "",
                  unicodedata.normalize("NFKC", str(text))).lower()

def find_row(df: pd.DataFrame, key: str):
    """`key` を含む行ラベルをあいまい検索して返す"""
    nk = norm(key)
    for r in df.index:
        if nk in norm(r):
            return r
    return None

def weather_forecast(lat, lon, days):
    url = ("https://api.open-meteo.com/v1/forecast?"
           f"latitude={lat}&longitude={lon}"
           "&daily=weathercode,temperature_2m_max,temperature_2m_min"
           "&timezone=Asia%2FTokyo")
    daily = requests.get(url, timeout=10).json()["daily"]
    return (pd.DataFrame({
        "dt":   pd.to_datetime(daily["time"]),
        "code": daily["weathercode"],
        "tmax": daily["temperature_2m_max"],
        "tmin": daily["temperature_2m_min"],
    }).set_index("dt").iloc[:days])

ROKUYO = ["先勝","友引","先負","仏滅","大安","赤口"]
def get_rokuyo(start: date, days: int):
    base = datetime(1900, 1, 1)
    return [ROKUYO[((start + timedelta(i)) - base.date()).days % 6]
            for i in range(days)]

def catboost_pred(y: pd.Series, horizon: int, extra=None):
    """CatBoost で horizon 日先まで予測"""
    idx = y.dropna().index
    if idx.empty or y.sum() == 0:
        return np.zeros(horizon)
    X = pd.DataFrame({
        "dow":   idx.weekday,
        "month": idx.month,
    }, index=idx)
    if extra:
        for k, s in extra.items():
            X[k] = s.reindex(idx).fillna(s.mean())
    model = CatBoostRegressor(depth=6, learning_rate=0.15,
                              loss_function="RMSE", random_state=42, verbose=False)
    model.fit(X, y.reindex(idx), cat_features=["dow"])

    fut_idx = pd.date_range(idx.max() + timedelta(1), periods=horizon)
    Xfut = pd.DataFrame({
        "dow":   fut_idx.weekday,
        "month": fut_idx.month,
    }, index=fut_idx)
    if extra:
        for k, s in extra.items():
            Xfut[k] = s.tail(30).mean()
    return np.clip(model.predict(Xfut), 0, None)

# ---------- main ----------
def main():
    # 1) シート読込み --------------------------------------------------------
    gc = gspread.authorize(load_sa())
    ws_db = gc.open_by_key(SID).worksheet(DB_SHEET)
    df_db = pd.DataFrame(ws_db.get_all_values())
    df_db.columns = df_db.iloc[0]
    df_db = df_db.drop(0)

    # 2) 日付列判定 ----------------------------------------------------------
    date_cols = pd.to_datetime(df_db.columns[1:], errors="coerce")
    mask = ~date_cols.isna()                 # NumPy bool 配列
    dates = pd.Index(date_cols[mask])

    # 3) ワイド表整形 --------------------------------------------------------
    wide = df_db.set_index(df_db.columns[0])
    wide = wide.iloc[:, mask]                # ← ブールマスクは iloc で
    wide.columns = dates
    wide = wide.apply(num_clean, axis=1)

    # 4) 行ラベルあいまい取得 ----------------------------------------------
    row_sales  = find_row(wide, "売上")
    row_cust   = find_row(wide, "客数")
    row_unit   = find_row(wide, "客単価")
    row_tmax   = find_row(wide, "最高気温")
    row_tmin   = find_row(wide, "最低気温")

    # 5) 予報データ ----------------------------------------------------------
    start = date.today() + timedelta(1)
    fut_dates = pd.date_range(start, periods=FORECAST_DAYS)
    wdf = weather_forecast(LAT, LON, FORECAST_DAYS).reindex(
        fut_dates, method="nearest")
    rokuyo_seq = get_rokuyo(start, FORECAST_DAYS)

    extra = {
        "tmax": wide.loc[row_tmax] if row_tmax else pd.Series(index=dates),
        "tmin": wide.loc[row_tmin] if row_tmin else pd.Series(index=dates),
    }

    # 6) 予測 ---------------------------------------------------------------
    agg_rows   = [row_sales, row_cust, row_unit]
    agg_labels = ["売上", "客数", "客単価"]
    item_rows  = list(wide.index.drop([r for r in agg_rows if r], errors="ignore"))
    targets = agg_rows + item_rows
    labels  = agg_labels + item_rows           # アイテムは行名をそのまま表示

    pred = {lbl: catboost_pred(wide.loc[row], FORECAST_DAYS, extra)
            for lbl, row in zip(labels, targets)}

    # 7) 出力シート ----------------------------------------------------------
    sh = gc.open_by_key(SID)
    ws_fc = (sh.worksheet(FC_SHEET) if FC_SHEET in [w.title for w in sh.worksheets()]
             else sh.add_worksheet(FC_SHEET, rows=2000, cols=400))
    ws_fc.resize(rows=LABEL_ROWS + len(labels), cols=1 + FORECAST_DAYS)

    # 7-1) 1 行目: 日付
    ws_fc.update("A1", [["日付"] + [d.strftime("%Y/%m/%d") for d in fut_dates]])

    # 7-2) 2〜7 行目: メタ情報
    meta = [
        ["曜日"]      + ["月火水木金土日"[d.weekday()] for d in fut_dates],
        ["六曜"]      + rokuyo_seq,
        ["年中行事"]  + [""] * FORECAST_DAYS,
        ["天気"]      + wdf["code"].round().astype(int).map({
            0:"快晴",1:"晴",2:"薄曇",3:"曇",45:"霧",51:"霧雨",61:"小雨",
            63:"雨",65:"大雨",71:"雪",80:"にわか雨",95:"雷雨"}).fillna("－").tolist(),
        ["最高気温"]  + wdf["tmax"].round(1).tolist(),
        ["最低気温"]  + wdf["tmin"].round(1).tolist(),
    ]
    ws_fc.update("A2", meta)

    # 7-3) 8 行目以降: 予測値
    for i, lbl in enumerate(labels, start=8):
        ws_fc.update_cell(i, 1, lbl)
        vals = pred[lbl]
        ws_fc.update(f"B{i}", [vals.round(1).tolist() if i <= 10
                               else vals.round().astype(int).tolist()])

    logging.info("✅ 需要予測シート更新完了")

# ---------- 実行 ----------
if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("🚨 Fatal")
