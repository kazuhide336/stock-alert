"""
天井サイン検出スクリプト v2.1 ─ GitHub Actions対応版
Discord Webhook URLは環境変数 DISCORD_WEBHOOK_URL から読み込む
"""

import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import time
from datetime import datetime

# =====================================================================
# 銘柄リスト
# =====================================================================
WATCHLIST = [
    "MU","NVDA","AVGO","ARM","MRVL","ANET","ASML",
    "COHR","AXTI","AAOI","TSEM","SNDK",
    "ETN","DELL","CLS","FN","MOD","LITE",
    "CRWD","PANW","NOW","CEG",
    "KTOS","AVAV","RCAT","UMAC",
    "QUBT","RGTI","IONQ",
    "ASPI","ONDS","AAPL","TSLA",
    "QQQ","GLDM","GDX","COPX",
    "000660.KS",  # SK Hynix
    "005930.KS",  # Samsung Electronics
    "6503.T","6504.T","6622.T","6762.T","6976.T","6779.T",
    "5802.T","5801.T","5803.T","5706.T","285A.T","3131.T","7974.T",
]

# ★ 環境変数から取得（GitHub Actions Secrets）
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# =====================================================================
# パラメータ
# =====================================================================
EMA_DEV_MIN   = 20.0
EMA_DEV_MAX   = 60.0
RSI_SCORE_MIN = 75
RSI_SCORE_MAX = 95
BB_PERIOD     = 20
BB_SIGMA      = 2.5

DIV_PARAMS = {
    "1wk": {"lookback": 24, "window": 2},
    "1d":  {"lookback": 60, "window": 4},
    "4h":  {"lookback": 60, "window": 6},
    "1h":  {"lookback": 48, "window": 4},
}
EMA_LONG = {"1wk": 52, "1d": 200, "4h": 240, "1h": 200}

ALERT_SCORE   = 55
CAUTION_SCORE = 35

SIGNAL_DEFS = [
    ("rsi_div",  "1wk", 3.0, "週足  RSIダイバージェンス",    "★★★"),
    ("rsi_div",  "1d",  2.5, "日足  RSIダイバージェンス",    "★★★"),
    ("vol_div",  "1d",  2.5, "日足  出来高ダイバージェンス", "★★★"),
    ("rsi_hot",  "1wk", 2.0, "週足  RSI買われすぎ",          "★★☆"),
    ("ema_dev",  "1d",  2.0, "日足  EMA乖離率",              "★★☆"),
    ("macd_div", "1d",  2.0, "日足  MACDダイバージェンス",   "★★☆"),
    ("bb_break", "1d",  2.0, "日足  BB+2.5σ逸脱",            "★★☆"),
    ("rsi_hot",  "1d",  1.5, "日足  RSI買われすぎ",          "★★☆"),
    ("rsi_div",  "4h",  1.5, "4時間 RSIダイバージェンス",    "★★☆"),
    ("macd_div", "4h",  1.5, "4時間 MACDダイバージェンス",   "★★☆"),
    ("vol_div",  "4h",  1.0, "4時間 出来高ダイバージェンス", "★☆☆"),
    ("ema_dev",  "4h",  1.0, "4時間 EMA乖離率",              "★☆☆"),
    ("bb_break", "4h",  1.0, "4時間 BB+2.5σ逸脱",            "★☆☆"),
    ("rsi_hot",  "4h",  1.0, "4時間 RSI買われすぎ",          "★☆☆"),
    ("rsi_div",  "1h",  0.5, "1時間 RSIダイバージェンス",    "★☆☆"),
    ("macd_div", "1h",  0.5, "1時間 MACDダイバージェンス",   "★☆☆"),
    ("rsi_hot",  "1h",  0.5, "1時間 RSI買われすぎ",          "★☆☆"),
]
# =====================================================================

FETCH_CONFIG = {
    "1wk": {"period": "5y",   "interval": "1wk"},
    "1d":  {"period": "400d", "interval": "1d"},
    "4h":  {"period": "200d", "interval": "4h"},
    "1h":  {"period": "60d",  "interval": "1h"},
}
TF_LABEL = {"1wk":"週", "1d":"日", "4h":"4h", "1h":"1h"}


def fetch_timeframes(ticker):
    data, status = {}, {}
    for tf, cfg in FETCH_CONFIG.items():
        try:
            df = yf.download(ticker, period=cfg["period"], interval=cfg["interval"],
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(how="all")
            min_bars = EMA_LONG[tf] + 30
            if len(df) >= min_bars:
                data[tf], status[tf] = df, "ok"
            elif len(df) > 0:
                data[tf], status[tf] = None, f"少データ({len(df)}本)"
            else:
                data[tf], status[tf] = None, "取得失敗"
        except Exception:
            data[tf], status[tf] = None, "エラー"
    return data, status


def calc(df, tf):
    df = df.copy()
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    df["ema_long"]  = ta.ema(close, length=EMA_LONG[tf])
    df["rsi"]       = ta.rsi(close, length=14)
    m               = ta.macd(close, fast=12, slow=26, signal=9)
    df["macd_hist"] = m["MACDh_12_26_9"]
    b               = ta.bbands(close, length=BB_PERIOD, std=BB_SIGMA)
    df["bb_upper"]  = b[f"BBU_{BB_PERIOD}_{BB_SIGMA}"]
    df["vol_ma20"]  = volume.rolling(20).mean()
    return df.dropna()


def two_peaks(df, col, tf):
    p = DIV_PARAMS[tf]
    lb, win = p["lookback"], p["window"]
    r  = df.tail(lb)
    ap = r["Close"].values
    ac = r[col].values
    n  = len(ap)
    pks = [i for i in range(win, n-win)
           if ap[i] >= max(ap[max(0,i-win):i+win+1])]
    if len(pks) < 2: return None
    p1, p2 = pks[-2], pks[-1]
    return ap[p1], ac[p1], ap[p2], ac[p2]


def check_rsi_hot(df, tf):
    rsi = float(df["rsi"].iloc[-1])
    s   = max(0.0, min(1.0,(rsi-RSI_SCORE_MIN)/(RSI_SCORE_MAX-RSI_SCORE_MIN)))
    lv  = "🔥天井圏" if rsi>=85 else "過熱" if rsi>=80 else "警戒" if rsi>=75 else "正常"
    return s, f"RSI {rsi:.0f} {lv}"

def check_rsi_div(df, tf):
    r = two_peaks(df,"rsi",tf)
    if r is None: return 0.0,"ピーク不足"
    pr1,rsi1,pr2,rsi2 = r
    if pr2<=pr1: return 0.0,"価格未更新"
    d = rsi1-rsi2
    if d<=0: return 0.0,f"RSIも更新"
    return min(1.0,d/10.0), f"価格↑ RSI {rsi1:.1f}→{rsi2:.1f} (差{d:.1f}pt)"

def check_vol_div(df, tf):
    r = two_peaks(df,"Volume",tf)
    if r is None: return 0.0,"ピーク不足"
    pr1,v1,pr2,v2 = r
    if pr2<=pr1: return 0.0,"価格未更新"
    ratio = v2/v1 if v1>0 else 1.0
    if ratio>=0.85: return 0.0,f"出来高正常({ratio*100:.0f}%)"
    return min(1.0,(0.85-ratio)/0.35), f"価格↑ 出来高↓(前回比{ratio*100:.0f}%)"

def check_ema_dev(df, tf):
    price = float(df["Close"].iloc[-1])
    ema   = float(df["ema_long"].iloc[-1])
    dev   = (price-ema)/ema*100
    s     = max(0.0,min(1.0,(dev-EMA_DEV_MIN)/(EMA_DEV_MAX-EMA_DEV_MIN)))
    return s, f"EMA{EMA_LONG[tf]} 乖離{dev:+.1f}%"

def check_macd_div(df, tf):
    r = two_peaks(df,"macd_hist",tf)
    if r is None: return 0.0,"ピーク不足"
    pr1,h1,pr2,h2 = r
    if pr2<=pr1 or h1<=0: return 0.0,"条件未成立"
    ratio = h2/h1
    if ratio>=0.75: return 0.0,f"縮小なし({ratio*100:.0f}%)"
    return min(1.0,(0.75-ratio)/0.75), f"MACDヒスト縮小({h1:.2f}→{h2:.2f})"

def check_bb_break(df, tf):
    price = float(df["Close"].iloc[-1])
    bb_u  = float(df["bb_upper"].iloc[-1])
    if price<=bb_u: return 0.0,f"上限内"
    pct = (price-bb_u)/bb_u*100
    return min(1.0,0.5+pct/4.0), f"上限を{pct:.1f}%超過"

CHECK_FN = {
    "rsi_hot":check_rsi_hot,"rsi_div":check_rsi_div,
    "vol_div":check_vol_div,"ema_dev":check_ema_dev,
    "macd_div":check_macd_div,"bb_break":check_bb_break,
}


def evaluate(ticker, tf_data, tf_status):
    calc_data = {}
    for tf, df in tf_data.items():
        try:    calc_data[tf] = calc(df, tf) if df is not None else None
        except: calc_data[tf] = None

    signals = []
    for key,tf,weight,label,star in SIGNAL_DEFS:
        df_c = calc_data.get(tf)
        if df_c is None or len(df_c)<10:
            signals.append({"label":label,"star":star,"tf":tf,"weight":weight,
                            "score":0.0,"detail":tf_status.get(tf,"不明"),"na":True})
            continue
        try:    score,detail = CHECK_FN[key](df_c,tf)
        except: score,detail = 0.0,"計算エラー"
        signals.append({"label":label,"star":star,"tf":tf,"weight":weight,
                        "score":score,"detail":detail,"na":False})

    valid   = [s for s in signals if not s["na"]]
    total_w = sum(s["weight"] for s in valid) or 1
    pct     = sum(s["weight"]*s["score"] for s in valid)/total_w*100

    latest = next((calc_data[tf] for tf in ["1d","1wk","4h","1h"]
                   if calc_data.get(tf) is not None), None)
    price  = float(latest["Close"].iloc[-1]) if latest is not None else 0

    return {"ticker":ticker,"price":price,"score":pct,
            "signals":signals,"status":tf_status}


def score_bar(pct, w=10):
    f   = round(pct/100*w)
    bar = "█"*f + "░"*(w-f)
    col = "🔴" if pct>=70 else "🟠" if pct>=55 else "🟡" if pct>=35 else "🟢" if pct>=15 else "⚪"
    return f"{col} {bar} {pct:.0f}%"

def tf_coverage(status):
    return " ".join(
        f"{TF_LABEL[tf]}{'✅' if status.get(tf)=='ok' else '❌'}"
        for tf in ["1wk","1d","4h","1h"]
    )

def tf_signals(signals, r):
    out = []
    for tf,lbl in TF_LABEL.items():
        if r["status"].get(tf) != "ok":
            out.append(f"{lbl}➖"); continue
        grp   = [s for s in signals if s["tf"]==tf and not s["na"]]
        fired = sum(1 for s in grp if s["score"]>=0.5)
        total = len(grp)
        out.append(f"{lbl}{'🔴' if fired==total else '🟡' if fired>0 else '⚪'}")
    return " ".join(out)

def sig_dot(score, na):
    if na: return "➖"
    return "🔴" if score>=0.7 else "🟡" if score>=0.35 else "⚪"


def build_summary(results):
    danger  = [r for r in results if r["score"]>=ALERT_SCORE]
    caution = [r for r in results if CAUTION_SCORE<=r["score"]<ALERT_SCORE]
    lines   = []
    if danger:
        lines.append(f"**🚨 天井危険圏 {ALERT_SCORE}%以上 — {len(danger)}銘柄**")
        for r in sorted(danger,key=lambda x:-x["score"]):
            lines.append(
                f"`{r['ticker']:<12}` {score_bar(r['score'])}\n"
                f"{'':14}{tf_signals(r['signals'],r)}  {tf_coverage(r['status'])}"
            )
    if caution:
        lines.append(f"\n**⚠️  注意圏 — {len(caution)}銘柄**")
        for r in sorted(caution,key=lambda x:-x["score"]):
            lines.append(f"`{r['ticker']:<12}` {score_bar(r['score'])}  {tf_signals(r['signals'],r)}")
    if not danger and not caution:
        lines.append("天井シグナルに達している銘柄はありません ✅")
    partial = [r for r in results if any(s!="ok" for s in r["status"].values())]
    if partial:
        names = ", ".join(r["ticker"] for r in partial[:8])
        lines.append(f"\n⚠️ 一部足が取得不能: {names}{'...' if len(partial)>8 else ''}")
    color = 0xFF2222 if danger else (0xFF8800 if caution else 0x00CC66)
    return {"embeds":[{"title":f"🔍 天井チェック {len(results)}銘柄 — {datetime.now():%m/%d %H:%M}",
                       "description":"\n".join(lines)[:4000],"color":color}]}


def build_detail(r):
    lines = [f"**スコア {score_bar(r['score'])}  ${r['price']:.2f}**",
             f"データ: {tf_coverage(r['status'])}\n"]
    for star_label,star_key in [("★★★ 最重要","★★★"),("★★☆ 重要","★★☆"),("★☆☆ 補助","★☆☆")]:
        grp = [s for s in r["signals"] if s["star"]==star_key]
        if not grp: continue
        lines.append(f"**{star_label}**")
        for s in grp:
            dot = sig_dot(s["score"],s["na"])
            bar = "▓"*round(s["score"]*5)+"░"*(5-round(s["score"]*5))
            na_note = f" *({s['detail']})*" if s["na"] else ""
            lines.append(f"{dot} [{bar}] `{s['label']}`{na_note}")
            if not s["na"]: lines.append(f"   └ {s['detail']}")
        lines.append("")
    return {"embeds":[{"title":f"🚨 {r['ticker']} 天井詳細",
                       "description":"\n".join(lines)[:4000],"color":0xFF2222,
                       "footer":{"text":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}}]}


def send(payload):
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        time.sleep(0.6)
        return resp.status_code == 204
    except Exception as e:
        print(f"  送信エラー: {e}")
        return False


def main():
    print(f"\n{'='*60}")
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]  天井サイン検出 v2.1")
    print(f"銘柄: {len(WATCHLIST)}  シグナル/銘柄: {len(SIGNAL_DEFS)}")
    print(f"{'='*60}")

    if not DISCORD_WEBHOOK_URL:
        print("⚠️  環境変数 DISCORD_WEBHOOK_URL が未設定です")
        print("   GitHub: Settings → Secrets → DISCORD_WEBHOOK_URL を追加")
        return

    results = []
    for i, ticker in enumerate(WATCHLIST, 1):
        print(f"[{i:>2}/{len(WATCHLIST)}] {ticker:<14}", end=" ", flush=True)
        try:
            tf_data, tf_status = fetch_timeframes(ticker)
            r = evaluate(ticker, tf_data, tf_status)
            results.append(r)
            cov = " ".join(
                f"{TF_LABEL[tf]}{'✅' if tf_status[tf]=='ok' else '❌'}"
                for tf in ["1wk","1d","4h","1h"]
            )
            print(f"{score_bar(r['score'])}  {cov}")
        except Exception as e:
            print(f"エラー: {e}")

    print()
    ok = send(build_summary(results))
    print(f"サマリー通知: {'✅' if ok else '❌'}")

    for r in [x for x in results if x["score"]>=ALERT_SCORE]:
        ok = send(build_detail(r))
        print(f"詳細 {r['ticker']}: {'✅' if ok else '❌'}")

    if results:
        top = max(results, key=lambda x: x["score"])
        print(f"\n最高スコア: {top['ticker']} {top['score']:.1f}%")


if __name__ == "__main__":
    main()
