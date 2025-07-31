#!/usr/bin/env python
# ==================================================================
# 需要予測フル版 2025-07  – CatBoost / Vertex / Hybrid
#   * Vertex 情報が未設定でもエラーにならない安全設計版 *
# ==================================================================
import os, json, base64, re, unicodedata, logging, warnings, time, math
from datetime import date, timedelta, datetime
import numpy as np, pandas as pd, requests, gspread

from google.oauth2.service_account import Credentials
from catboost import CatBoostRegressor

# Vertex（遅延 import）
try:        from google.cloud import aiplatform
except Exception: aiplatform = None

# ---------- 基本設定 ----------
SID          = os.getenv("GSHEET_ID")
SA_JSON      = os.getenv("GSPREAD_SA_JSON")
DB_SHEET     = os.getenv("DB_SHEET",       "データベース")

FC_SHEET_CB  = os.getenv("FORECAST_SHEET_CB", "需要予測_CB")
FC_SHEET_VX  = os.getenv("FORECAST_SHEET_VX", "需要予測_VX")
FC_SHEET_HY  = os.getenv("FORECAST_SHEET_HY", "需要予測_HY")

HIST_SHEET   = os.getenv("HISTORY_SHEET",  "予測履歴")
METRIC_SHEET = os.getenv("METRIC_SHEET",   "予測精度")

FORECAST_D   = int(os.getenv("FORECAST_DAYS", 7))    # 明日から n 日
LABEL_ROWS   = int(os.getenv("LABEL_ROWS", 10))

ERR_WEIGHT_SCALE = float(os.getenv("ERR_WEIGHT_SCALE", 30000))
ERR_WEIGHT_CAP   = float(os.getenv("ERR_WEIGHT_CAP", 1.0))

# Vertex 環境
USE_VERTEX_RAW      = os.getenv("USE_VERTEX", "1") == "1"
VERTEX_PROJECT      = os.getenv("VERTEX_PROJECT")
VERTEX_LOCATION     = os.getenv("VERTEX_LOCATION", "asia-northeast1")
VERTEX_ENDPOINT_ID  = os.getenv("VERTEX_ENDPOINT_ID")
VERTEX_TIMEOUT      = int(os.getenv("VERTEX_TIMEOUT", 30))

# 地理 (天気)
LAT, LON = 36.3740, 140.5662

META_ROWS = ["曜日","六曜","年中行事","天気","最高気温","最低気温"]

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- util ----------
def creds():
    raw = SA_JSON or ""
    data = json.loads(raw) if raw.lstrip().startswith("{") else json.loads(base64.b64decode(raw))
    return Credentials.from_service_account_info(
        data, scopes=["https://www.googleapis.com/auth/spreadsheets"])

def num_clean(s):
    return (pd.to_numeric(
        s.astype(str)
         .str.replace(r"[^\d.\-]", "", regex=True)
         .str.replace("．", ".", regex=False)
         .replace("", np.nan),
        errors="coerce").fillna(0))

def norm(t): return re.sub(r"[ 　【】\[\]\(\)]", "", unicodedata.normalize("NFKC", str(t))).lower()

def fuzzy_row(df, key):
    nk = norm(key)
    for r in df.index:
        if nk in norm(r): return r
    return None

def ensure_series(x): return x.iloc[0] if isinstance(x, pd.DataFrame) else x

W2C = {"快晴":0,"晴":1,"薄曇":2,"曇":3,"霧":45,"霧雨":51,"小雨":61,"雨":63,
       "大雨":65,"小雪":71,"雪":73,"大雪":75,"にわか雨":80,"雷雨":95,
       "—":np.nan,"ｰ":np.nan,"":np.nan,"－":np.nan}

def weather_forecast(days):
    url = (f"https://api.open-meteo.com/v1/forecast?"
           f"latitude={LAT}&longitude={LON}"
           "&daily=weathercode,temperature_2m_max,temperature_2m_min"
           "&timezone=Asia%2FTokyo")
    try:
        d = requests.get(url, timeout=15).json()["daily"]
        return (pd.DataFrame({
            "dt":   pd.to_datetime(d["time"]),
            "code": d["weathercode"],
            "tmax": d["temperature_2m_max"],
            "tmin": d["temperature_2m_min"],
        }).set_index("dt").iloc[:days])
    except Exception as e:
        logging.warning(f"天気API失敗 fallback: {e}")
        idx = pd.date_range(date.today()+timedelta(1), periods=days)
        return pd.DataFrame({"code": np.nan, "tmax": np.nan, "tmin": np.nan}, index=idx)

ROKUYO = ["先勝","友引","先負","仏滅","大安","赤口"]
def rokuyo(start, days):
    base = datetime(1900,1,1)
    return [ROKUYO[((start+timedelta(i))-base.date()).days % 6] for i in range(days)]

def prepare_cat_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "code" in out.columns:
        out["code_cat"] = out["code"].fillna(-999).round().astype(int).astype(str)
    return out

def sanitize_sheet_values(df: pd.DataFrame):
    safe = df.copy()
    for c in safe.columns:
        safe[c] = safe[c].apply(lambda v:
            "" if (isinstance(v, (float,int)) and (pd.isna(v) or np.isinf(v)))
            else v)
    return [safe.columns.tolist()] + safe.astype(str).values.tolist()

# ---------- CatBoost ----------
def cat_predict(label_name: str, y: pd.Series,
                X_extra: pd.DataFrame, Xf_extra: pd.DataFrame,
                history_df: pd.DataFrame) -> np.ndarray:
    y = ensure_series(y).astype(float)
    idx = y.dropna().index
    if idx.empty or float(y.sum()) == 0:
        return np.zeros(len(Xf_extra))

    X = pd.DataFrame({
        "dow": idx.weekday.astype(int),
        "mon": idx.month.astype(int),
        "doy": idx.dayofyear.astype(int)
    }, index=idx).join(X_extra.reindex(idx))

    Xf = pd.DataFrame({
        "dow": Xf_extra.index.weekday.astype(int),
        "mon": Xf_extra.index.month.astype(int),
        "doy": Xf_extra.index.dayofyear.astype(int)
    }, index=Xf_extra.index).join(Xf_extra)

    X, Xf = prepare_cat_columns(X), prepare_cat_columns(Xf)

    n = len(idx)
    recency_w = np.linspace(1.0, 2.0, n)
    error_factor = np.ones(n)

    if not history_df.empty:
        hc = (history_df[(history_df["label"] == label_name) &
                         (history_df["model"].isin(["catboost", ""]))]  # 旧レコード互換
              .sort_values("run").drop_duplicates(subset="target", keep="last")
              .set_index("target")["pred"])
        for i, d in enumerate(idx):
            if d in hc.index:
                err = abs(y.loc[d] - hc.loc[d])
                error_factor[i] = 1.0 + min(err / ERR_WEIGHT_SCALE, ERR_WEIGHT_CAP)

    w = recency_w * error_factor
    cat_feats = [c for c in ["dow","mon","code_cat"] if c in X.columns]

    model = CatBoostRegressor(
        depth=8, learning_rate=0.1, loss_function="RMSE",
        random_state=42, verbose=False
    )
    model.fit(X, y.loc[idx], sample_weight=w, cat_features=cat_feats)

    preds = model.predict(Xf)
    preds = np.clip(preds, 0, None)
    return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

# ---------- Vertex ----------
def init_vertex() -> bool:
    """True: 初期化成功 / False: Vertex 無効化"""
    if not USE_VERTEX_RAW:
        return False
    if not (VERTEX_PROJECT and VERTEX_ENDPOINT_ID):
        logging.warning("Vertex 環境変数未設定 -> Vertex 無効化")
        return False
    if aiplatform is None:
        logging.warning("google-cloud-aiplatform が未インストール -> Vertex 無効化")
        return False
    try:
        aiplatform.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
        return True
    except Exception as e:
        logging.warning(f"Vertex init error: {e}")
        return False

def vertex_predict_matrix(label_name: str,
                          Xf_extra: pd.DataFrame,
                          fut_idx: pd.DatetimeIndex,
                          endpoint=None) -> np.ndarray:
    if endpoint is None:
        return np.zeros(len(fut_idx))

    recs=[]
    for d in fut_idx:
        recs.append({
            "dow": int(d.weekday()), "mon": int(d.month), "doy": int(d.dayofyear),
            "code": float(Xf_extra.loc[d,"code"]) if "code" in Xf_extra else 0.0,
            "tmax": float(Xf_extra.loc[d,"tmax"]) if "tmax" in Xf_extra else 0.0,
            "tmin": float(Xf_extra.loc[d,"tmin"]) if "tmin" in Xf_extra else 0.0,
            "label": label_name
        })
    try:
        prediction = endpoint.predict(instances=recs, timeout=VERTEX_TIMEOUT)
        raw = prediction.predictions
        vals=[]
        for r in raw:
            if isinstance(r, dict):
                if "value" in r: vals.append(float(r["value"]))
                elif "predictions" in r and r["predictions"]: vals.append(float(r["predictions"][0]))
                else: vals.append(0.0)
            else:
                try: vals.append(float(r))
                except: vals.append(0.0)
        if len(vals) < len(fut_idx): vals += [0.0]*(len(fut_idx)-len(vals))
        arr=np.array(vals[:len(fut_idx)])
        return np.nan_to_num(np.clip(arr,0,None), nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logging.warning(f"Vertex 予測失敗(label={label_name}): {e}")
        return np.zeros(len(fut_idx))

# ---------- Hybrid ----------
def hybrid_blend(cb_preds, vx_preds, model_mae) -> np.ndarray:
    if cb_preds is None and vx_preds is None: return np.zeros(0)
    if vx_preds is None or len(vx_preds)==0:  return cb_preds
    if cb_preds is None or len(cb_preds)==0:  return vx_preds
    mae_cb = model_mae.get("catboost"); mae_vx = model_mae.get("vertex")
    if not (mae_cb and mae_vx) or mae_cb<=0 or mae_vx<=0:
        w_cb=w_vx=0.5
    else:
        w_cb, w_vx = 1.0/(mae_cb+1e-6), 1.0/(mae_vx+1e-6)
        s=w_cb+w_vx; w_cb/=s; w_vx/=s
    return w_cb*cb_preds + w_vx*vx_preds

# ---------- Main ----------
def main():
    logging.info("=== 需要予測開始 ===")

    # Vertex 初期化（成功なら endpoint インスタンスを返す）
    endpoint = None
    global USE_VERTEX  # ← 先頭で宣言しておけば SyntaxError 回避
    USE_VERTEX = init_vertex()
    if USE_VERTEX:
        try:
            endpoint = aiplatform.Endpoint(VERTEX_ENDPOINT_ID)
            logging.info("Vertex 使用モード")
        except Exception as e:
            logging.warning(f"Endpoint 取得失敗 -> Vertex 無効化: {e}")
            USE_VERTEX = False

    gc = gspread.authorize(creds())
    sh = gc.open_by_key(SID)

    # --- データベース読み込み ---
    db_ws = sh.worksheet(DB_SHEET)
    raw = db_ws.get_all_values()
    if not raw or len(raw) < 2:
        raise RuntimeError("データベースシートにデータがありません。")

    df0 = pd.DataFrame(raw); df = df0.drop(0); df.columns = df0.iloc[0]
    date_cols = pd.to_datetime(df.columns[1:], errors="coerce")
    wide = df.set_index(df.columns[0]).iloc[:, ~date_cols.isna()]
    wide.columns = date_cols[~date_cols.isna()]
    wide = wide.apply(num_clean, axis=1)

    # 行特定
    r_sales = fuzzy_row(wide,"売上"); r_cust=fuzzy_row(wide,"客数"); r_unit=fuzzy_row(wide,"客単価")
    r_tmax=fuzzy_row(wide,"最高気温"); r_tmin=fuzzy_row(wide,"最低気温"); r_wtxt=fuzzy_row(wide,"天気")
    wcode_hist = (wide.loc[r_wtxt].replace(W2C).astype(float) if r_wtxt else pd.Series(index=wide.columns,dtype=float))

    start = date.today()+timedelta(1)
    fut_idx = pd.date_range(start, periods=FORECAST_D)
    wdf = weather_forecast(FORECAST_D).reindex(fut_idx, method="nearest")

    Xf_extra = pd.DataFrame({"code":wdf["code"],"tmax":wdf["tmax"],"tmin":wdf["tmin"]}, index=fut_idx)
    X_extra  = pd.DataFrame({
        "code": wcode_hist,
        "tmax": ensure_series(wide.loc[r_tmax]) if r_tmax else np.nan,
        "tmin": ensure_series(wide.loc[r_tmin]) if r_tmin else np.nan
    })

    agg_lbls=["売上","客数","客単価"]; agg_rows=[r_sales,r_cust,r_unit]
    item_rows=[r for r in wide.index if r not in META_ROWS+agg_lbls and r and not norm(r).startswith("天気")]
    labels, rows = agg_lbls+item_rows, agg_rows+item_rows

    # --- 履歴読込 ---
    hist_ws = sh.worksheet(HIST_SHEET) if HIST_SHEET in [w.title for w in sh.worksheets()] \
              else sh.add_worksheet(HIST_SHEET, rows=1, cols=6)
    hist_vals = hist_ws.get_all_values()
    if not hist_vals:
        hist_ws.update(values=[["run_date","target_date","model","label","pred"]], range_name="A1")
        hist_vals = hist_ws.get_all_values()
    if hist_vals[0]==["run_date","target_date","label","pred"]:
        body=hist_vals[1:]; new_body=[[r[0],r[1],"catboost",r[2],r[3]] for r in body if len(r)>=4]
        hist_ws.clear(); hist_ws.update([["run_date","target_date","model","label","pred"]]+new_body,"A1")
        hist_vals = hist_ws.get_all_values()
    hist_df = (pd.DataFrame(hist_vals[1:], columns=hist_vals[0])
               .assign(run=lambda d:pd.to_datetime(d["run_date"],errors="coerce"),
                       target=lambda d:pd.to_datetime(d["target_date"],errors="coerce"),
                       pred=lambda d:pd.to_numeric(d["pred"],errors="coerce"),
                       model=lambda d:d["model"].fillna("catboost"))) if len(hist_vals)>1 \
              else pd.DataFrame(columns=["run","target","model","label","pred"])

    # --- 予測 ---
    preds_cb, preds_vx, preds_hy, model_mae = {}, {}, {}, {}
    for lbl, r in zip(labels, rows):
        series = wide.loc[r] if r is not None else pd.Series(dtype=float)
        # CatBoost
        try: preds_cb[lbl] = cat_predict(lbl, series, X_extra, Xf_extra, hist_df)
        except Exception as e:
            logging.exception(f"CatBoost予測失敗 label={lbl}: {e}")
            preds_cb[lbl]=np.zeros(len(fut_idx))
        # Vertex
        preds_vx[lbl] = vertex_predict_matrix(lbl, Xf_extra, fut_idx, endpoint) if USE_VERTEX \
                        else np.zeros(len(fut_idx))

    # MAE 計算（ハイブリッド重み）
    if not hist_df.empty:
        tmp=hist_df[(hist_df["target"].notna())&(hist_df["target"].dt.date<date.today())]
        actual_map={lab:ensure_series(wide.loc[lab]) if lab in wide.index
                    else pd.Series(index=wide.columns,dtype=float) for lab in wide.index.unique()}
        rows_mae=[]
        for _,row in tmp.iterrows():
            lab=row["label"]; d=row["target"]; 
            if lab in actual_map and d in actual_map[lab]:
                act=actual_map[lab].get(d,np.nan)
                if pd.notna(act): rows_mae.append([row["model"],abs(act-row["pred"])])
        if rows_mae:
            dfm=pd.DataFrame(rows_mae,columns=["model","ae"])
            model_mae=dfm.groupby("model")["ae"].mean().to_dict()

    # Hybrid
    for lbl in labels:
        preds_hy[lbl]=hybrid_blend(preds_cb[lbl], preds_vx[lbl], model_mae)

    # ---------- シート更新 ----------
    def upsert(sheet_name, preds_dict):
        ws = sh.worksheet(sheet_name) if sheet_name in [w.title for w in sh.worksheets()] \
             else sh.add_worksheet(sheet_name, rows=2000, cols=400)
        ws.resize(rows=LABEL_ROWS+len(labels), cols=1+FORECAST_D)
        header=[["日付"]+[d.strftime("%Y/%m/%d") for d in fut_idx]]
        meta=[
            ["曜日"]+["月火水木金土日"[d.weekday()] for d in fut_idx],
            ["六曜"]+rokuyo(start,FORECAST_D),
            ["年中行事"]+[""]*FORECAST_D,
            ["天気"]+wdf["code"].map({
                0:"快晴",1:"晴",2:"薄曇",3:"曇",45:"霧",51:"霧雨",61:"小雨",
                63:"雨",65:"大雨",71:"雪",75:"大雪",80:"にわか雨",95:"雷雨"}).fillna("－").tolist(),
            ["最高気温"]+wdf["tmax"].round(1).tolist(),
            ["最低気温"]+wdf["tmin"].round(1).tolist()
        ]
        ws.update(header+meta,"A1")
        body=[[lbl]+(preds_dict[lbl].round(1) if lbl in ["売上","客数","客単価"]
                     else preds_dict[lbl].round().astype(int)).tolist() for lbl in labels]
        ws.update(body, f"A{LABEL_ROWS+1}")
        logging.info(f"{sheet_name} 更新完了")
    upsert(FC_SHEET_CB,preds_cb); upsert(FC_SHEET_VX,preds_vx); upsert(FC_SHEET_HY,preds_hy)

    # ---------- 履歴追記 ----------
    run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def hist_append(model_name, pred_map):
        rows_add=[[run_date,d.strftime("%Y-%m-%d"),model_name,lbl,float(v)]
                  for lbl in labels for d,v in zip(fut_idx,pred_map[lbl])]
        for _ in range(3):
            try: hist_ws.append_rows(rows_add,value_input_option="USER_ENTERED"); break
            except Exception as e: logging.warning(f"履歴 append 再試行: {e}"); time.sleep(2)
    hist_append("catboost",preds_cb)
    if USE_VERTEX: hist_append("vertex",preds_vx)
    hist_append("hybrid",preds_hy)
    logging.info("履歴シート更新完了")

    # ---------- 精度レポート ----------
    hist_vals2=hist_ws.get_all_values()
    if len(hist_vals2)>1:
        hdf=pd.DataFrame(hist_vals2[1:],columns=hist_vals2[0])
        hdf["target"]=pd.to_datetime(hdf["target_date"],errors="coerce")
        hdf["pred"]=pd.to_numeric(hdf["pred"],errors="coerce"); hdf["model"]=hdf["model"].fillna("catboost")
        actual_map={lab:ensure_series(wide.loc[lab]) if lab in wide.index
                    else pd.Series(index=wide.columns,dtype=float) for lab in wide.index.unique()}
        rec=[]; cutoff=date.today()
        for _,r in hdf.iterrows():
            if pd.isna(r["target"]) or r["target"].date()>=cutoff: continue
            lab=r["label"]; d=r["target"]
            if lab in actual_map and d in actual_map[lab]:
                act=actual_map[lab].get(d,np.nan)
                if pd.notna(act):
                    err=abs(act-r["pred"]); ape=err/act*100 if act else np.nan
                    rec.append([r["model"],lab,err,ape])
        if rec:
            rep=(pd.DataFrame(rec,columns=["model","label","ae","ape"])
                 .groupby(["model","label"]).agg(MAE=("ae","mean"),MAPE=("ape","mean"))
                 .reset_index().round(2))
            met_ws=sh.worksheet(METRIC_SHEET) if METRIC_SHEET in [w.title for w in sh.worksheets()] \
                   else sh.add_worksheet(METRIC_SHEET, rows=2000, cols=20)
            met_ws.clear(); met_ws.update(sanitize_sheet_values(rep),"A1")
            logging.info("精度レポート更新完了")
    logging.info("✅ 完了 — 3予測/履歴/精度 更新")

if __name__ == "__main__":
    try: main()
    except Exception: logging.exception("🚨 Fatal"); raise
