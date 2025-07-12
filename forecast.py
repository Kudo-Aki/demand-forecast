#!/usr/bin/env python
# ===========================================================
# 需要予測 & 自動学習スクリプト
# -----------------------------------------------------------
# ・毎日 17:00JST に実行（GitHub Actions などで定期実行）
# ・Google Sheets から販売実績を取得し、CatBoost で 1週間予測
# ・予測結果を「需要予測」シートへ書き込み
# ・前日の予測誤差を自動で評価してログ出力
# ・エラーはすべて try/except で握りつぶさずログに残す
# ===========================================================
import os, json, base64, re, unicodedata, logging
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd

# --- 外部ライブラリ（要: pip install gspread google-auth catboost） ---
import gspread
from google.oauth2.service_account import Credentials
from catboost import CatBoostRegressor

# ========== 0. 環境変数 ==========
SHEET_ID           = os.getenv("GSHEET_ID")              # スプレッドシート ID
SA_ENV             = os.getenv("GSPREAD_SA_JSON")        # ServiceAccount(JSON or base64)
DB_SHEET_NAME      = os.getenv("DB_SHEET",     "データベース")
FC_SHEET_NAME      = os.getenv("FORECAST_SHEET", "需要予測")
FORECAST_DAYS      = int(os.getenv("FORECAST_DAYS", 7))
LABEL_ROWS         = int(os.getenv("LABEL_ROWS", 10))    # メタ行数
TZ                 = os.getenv("TIMEZONE", "Asia/Tokyo")

# ========== 1. logger 設定 ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ========== 2. Util 関数 ==========
def load_sa_json():
    """サービスアカウント JSON (平文 or base64) を dict で返す"""
    raw = SA_ENV or ""
    try:
        return json.loads(raw) if raw.strip().startswith("{") else json.loads(base64.b64decode(raw))
    except Exception as e:
        logging.error(f"❌ ServiceAccount 読込失敗: {e}")
        raise

def normalize(txt: str) -> str:
    """全角半角を吸収 & 記号除去で比較用キーに"""
    return re.sub(r"[ 　【】\[\]\(\)]", "", unicodedata.normalize("NFKC", str(txt))).lower()

def sheet_to_dataframe(ws) -> pd.DataFrame:
    """シートをそのまま DataFrame に読み込む（最上行=ヘッダとせず index=None）"""
    records = ws.get_all_values()
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("⛔ シートが空です")
    return df

def ensure_sheet(gc: gspread.Client, name: str):
    """存在しなければシートを追加して返す"""
    sh = gc.open_by_key(SHEET_ID)
    try:
        return sh.worksheet(name)
    except gspread.exceptions.WorksheetNotFound:
        return sh.add_worksheet(title=name, rows="1000", cols="1000")

# ========== 3. 学習 & 予測 ==========
def build_features(df: pd.DataFrame, date_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """共通特徴量を作成（必要なら拡張可）"""
    feat = pd.DataFrame(index=date_idx)
    feat["dow"]  = feat.index.weekday        # 曜日 (0=Mon)
    feat["month"] = feat.index.month         # 月
    # 六曜・天気 等を追加する場合はここでマージ
    return feat

def predict_series(sales: pd.Series, horizon: int) -> np.ndarray:
    """CatBoost で horizon 日先まで予測"""
    # --- 学習データ整備 ---
    hist = sales.copy().astype(float)
    hist.index = pd.to_datetime(hist.index, errors="coerce")
    hist = hist.dropna()
    if hist.empty or hist.sum() == 0:
        return np.zeros(horizon)             # データ不足なら 0 返し

    X = build_features(hist.to_frame("y"), hist.index)
    y = hist.values

    # --- モデル学習 ---
    try:
        model = CatBoostRegressor(
            depth=6,
            learning_rate=0.1,
            loss_function="RMSE",
            random_state=42,
            verbose=False,
        )
        model.fit(X, y, cat_features=["dow"])
    except Exception as e:
        logging.error(f"  ⚠️ CatBoost 学習失敗: {e}")
        return np.zeros(horizon)

    # --- 予測 ---
    fut_idx = pd.date_range(hist.index.max() + timedelta(days=1), periods=horizon, freq="D")
    X_future = build_features(hist.to_frame(), fut_idx)
    try:
        pred = model.predict(X_future)
        pred = np.clip(pred, 0, None)        # マイナスを 0 に丸め
    except Exception as e:
        logging.error(f"  ⚠️ 予測失敗: {e}")
        pred = np.zeros(horizon)
    return pred

# ========== 4. メイン処理 ==========
def main() -> None:
    try:
        sa_info = load_sa_json()
        gc = gspread.authorize(
            Credentials.from_service_account_info(
                sa_info,
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
        )
        sh = gc.open_by_key(SHEET_ID)
        ws_db = sh.worksheet(DB_SHEET_NAME)
        df_db = sheet_to_dataframe(ws_db)

        # --- 日付列抽出（ラベル行より右側が日付列） ---
        date_cols = pd.to_datetime(df_db.iloc[0, 1:], errors="coerce")
        valid_mask = ~date_cols.isna()
        if not valid_mask.any():
            raise ValueError("📆 日付列が見つかりません")
        date_cols = date_cols[valid_mask]
        dates = date_cols.dt.normalize()

        # --- 売上行抽出（LABEL_ROWS行目以降が授与品名） ---
        item_rows = df_db.iloc[LABEL_ROWS:, :]      # メタ行をスキップ
        item_names = item_rows.iloc[:, 0].tolist()  # A列 = 授与品名
        pred_dict = {}                              # 予測結果保存

        for idx, name in enumerate(item_names):
            try:
                sales_vec = item_rows.iloc[idx, 1:][valid_mask].replace('', np.nan)
                sales_series = pd.Series(pd.to_numeric(sales_vec, errors='coerce'),
                         index=dates).fillna(0)
                preds = predict_series(sales_series, FORECAST_DAYS)      # ndarray
                pred_dict[name] = preds
                # ---- 昨日の誤差ログ ----
                if len(sales_series) > 1:
                    yday = sales_series.index.max()
                    if (yday + timedelta(days=1)).normalize() == pd.Timestamp(date.today()):
                        # 昨日の予測があれば評価（存在しない場合スキップ）
                        try:
                            fc_sheet = ensure_sheet(gc, FC_SHEET_NAME)
                            fc_header = pd.to_datetime(fc_sheet.row_values(1)[1:], errors="coerce")
                            if yday in fc_header:
                                col_idx = fc_header.get_loc(yday) + 2  # 1-based (+A列)
                                yhat = float(fc_sheet.cell(idx + LABEL_ROWS + 1, col_idx).value or 0)
                                err = abs(yhat - sales_series.iloc[-1])
                                logging.info(f"📝 誤差 [{name}] {yhat:.1f}→{sales_series.iloc[-1]}  Δ={err:.1f}")
                        except Exception as e:
                            logging.warning(f"  誤差評価失敗 ({name}): {e}")
            except Exception as e:
                logging.error(f"❌ [{name}] の処理で例外: {e}")
                continue

        # --- 需要予測シート更新 ---
        ws_fc = ensure_sheet(gc, FC_SHEET_NAME)
        # 1 行目: 日付ヘッダ
        header_dates = pd.date_range(date.today() + timedelta(days=1),
                                     periods=FORECAST_DAYS, freq="D")
        ws_fc.resize(rows=LABEL_ROWS + len(item_names), cols=1 + FORECAST_DAYS)
        ws_fc.update("A1", [["日付"] + [d.strftime("%Y/%m/%d") for d in header_dates]])

        # 授与品名を書き出し (A列)
        ws_fc.update(f"A{LABEL_ROWS+1}", [[n] for n in item_names])

        # 予測値を書き出し (B列以降)
        value_matrix = []
        for name in item_names:
            preds = pred_dict.get(name, np.zeros(FORECAST_DAYS))
            value_matrix.append(preds.tolist())
        ws_fc.update(f"B{LABEL_ROWS+1}",
                     value_matrix,
                     value_input_option="USER_ENTERED")

        logging.info("✅ 需要予測シート更新完了")

    except Exception as e:
        logging.exception(f"🚨 重大なエラーで処理中断: {e}")

# ========== 5. エントリポイント ==========
if __name__ == "__main__":
    main()
