"""
天井サイン検出スクリプト v2.2 ─ GitHub Actions対応・ランキング表示版
"""

import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import time
from datetime import datetime

WATCHLIST = [
    "MU","NVDA","AVGO","ARM","MRVL","ANET","ASML",
    "COHR","AXTI","AAOI","TSEM","SNDK",
    "ETN","DELL","CLS","FN","MOD","LITE",
    "CRWD","PANW","NOW","CEG",
    "KTOS","AVAV","RCAT","UMAC",
    "QUBT","RGTI","IONQ",
    "ASPI","ONDS","AAPL","TSLA",
    "QQQ","GLDM","GDX","COPX",
    "000660.KS","005930.KS",
    "6503.T","6504.T","6622.T","6762.T","6976.T","6779.T",
    "5802.T","5801.T","5803.T","5706.T","285A.T","3131.T","7974.T",
]

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

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
    ("rsi_div",  "1wk", 3.0, "週足  RSIダイバージェンス",    "AAA"),
    ("rsi_div",  "1d",  2.5, "日足  RSIダイバージェンス",    "AAA"),
    ("vol_div",  "1d",  2.5, "日足  出来高ダイバージェンス", "AAA"),
    ("rsi_hot",  "1wk", 2.0, "週足  RSI買われすぎ",          "AA"),
    ("ema_dev",  "1d",  2.0, "日足  EMA乖離率",              "AA"),
    ("macd_div", "1d",  2.0, "日足  MACDダイバージェンス",   "AA"),
    ("bb_break", "1d",  2.0, "日足  BB+2.5sigma逸脱",        "AA"),
    ("rsi_hot",  "1d",  1.5, "日足  RSI買われすぎ",          "AA"),
    ("rsi_div",  "4h",  1.5, "4時間 RSIダイバージェンス",    "AA"),
    ("macd_div", "4h",  1.5, "4時間 MACDダイバージェンス",   "AA"),
    ("vol_div",  "4h",  1.0, "4時間 出来高ダイバージェンス", "A"),
    ("ema_dev",  "4h",  1.0, "4時間 EMA乖離率",              "A"),
    ("bb_break", "4h",  1.0, "4時間 BB+2.5sigma逸脱",        "A"),
    ("rsi_hot",  "4h",  1.0, "4時間 RSI買われすぎ",          "A"),
    ("rsi_div",  "1h",  0.5, "1時間 RSIダイバージェンス",    "A"),
    ("macd_div", "1h",  0.5, "1時間 MACDダイバージェンス",   "A"),
    ("rsi_hot",  "1h",  0.5, "1時間 RSI買われすぎ",          "A"),
]

FETCH_CONFIG = {
    "1wk": {"period": "5y",   "interval": "1wk"},
    "1d":  {"period": "400d", "interval": "1d"},
    "4h":  {"period": "200d", "interval": "4h"},
    "1h":  {"period": "60d",  "interval": "1h"},
}
TF_LABEL = {"1wk": "週", "1d": "日", "4h": "4h", "1h": "1h"}


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
                data[tf], status[tf] = None, "少データ"
            else:
                data[tf], status[tf] = None, "取得失敗"
        except Exception:
            data[tf], status[tf] = None, "エラー"
    return data, status

def calc(df, tf):
    df = df.copy()
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    df["ema_long"] = ta.ema(close, length=EMA_LONG[tf])
    df["rsi"]      = ta.rsi(close, length=14)

    m = ta.macd(close, fast=12, slow=26, signal=9)
    if m is None:
        raise ValueError("MACD計算失敗")
    macd_h_col = next((c for c in m.columns if "MACDh" in c), None)
    if macd_h_col is None:
        raise ValueError(f"MACDh列なし: {list(m.columns)}")
    df["macd_hist"] = m[macd_h_col]

    b = ta.bbands(close, length=BB_PERIOD, std=BB_SIGMA)
    if b is None:
        raise ValueError("BB計算失敗")
    bb_u_col = next((c for c in b.columns if c.startswith("BBU_")), None)
    if bb_u_col is None:
        raise ValueError(f"BBU列なし: {list(b.columns)}")
    df["bb_upper"] = b[bb_u_col]

    df["vol_ma20"] = volume.rolling(20).mean()
    return df.dropna()


def two_peaks(df, col, tf):
    p   = DIV_PARAMS[tf]
    lb, win = p["lookback"], p["window"]
    r   = df.tail(lb)
    ap  = r["Close"].values
    ac  = r[col].values
    n   = len(ap)
    pks = [i for i in range(win, n - win)
           if ap[i] >= max(ap[max(0, i - win): i + win + 1])]
    if len(pks) < 2:
        return None
    p1, p2 = pks[-2], pks[-1]
    return ap[p1], ac[p1], ap[p2], ac[p2]


def check_rsi_hot(df, tf):
    rsi = float(df["rsi"].iloc[-1])
    s   = max(0.0, min(1.0, (rsi - RSI_SCORE_MIN) / (RSI_SCORE_MAX - RSI_SCORE_MIN)))
    lv  = "天井圏" if rsi >= 85 else "過熱" if rsi >= 80 else "警戒" if rsi >= 75 else "正常"
    return s, "RSI " + str(round(rsi)) + " " + lv


def check_rsi_div(df, tf):
    r = two_peaks(df, "rsi", tf)
    if r is None:
        return 0.0, "ピーク不足"
    pr1, rsi1, pr2, rsi2 = r
    if pr2 <= pr1:
        return 0.0, "価格未更新"
    d = rsi1 - rsi2
    if d <= 0:
        return 0.0, "RSIも更新"
    return min(1.0, d / 10.0), "価格↑ RSI " + str(round(rsi1, 1)) + "→" + str(round(rsi2, 1)) + " 差" + str(round(d, 1)) + "pt"


def check_vol_div(df, tf):
    r = two_peaks(df, "Volume", tf)
    if r is None:
        return 0.0, "ピーク不足"
    pr1, v1, pr2, v2 = r
    if pr2 <= pr1:
        return 0.0, "価格未更新"
    ratio = v2 / v1 if v1 > 0 else 1.0
    if ratio >= 0.85:
        return 0.0, "出来高正常 " + str(round(ratio * 100)) + "%"
    return min(1.0, (0.85 - ratio) / 0.35), "価格↑ 出来高↓ 前回比" + str(round(ratio * 100)) + "%"


def check_ema_dev(df, tf):
    price = float(df["Close"].iloc[-1])
    ema   = float(df["ema_long"].iloc[-1])
    dev   = (price - ema) / ema * 100
    s     = max(0.0, min(1.0, (dev - EMA_DEV_MIN) / (EMA_DEV_MAX - EMA_DEV_MIN)))
    return s, "EMA乖離 " + ("+" if dev >= 0 else "") + str(round(dev, 1)) + "%"


def check_macd_div(df, tf):
    r = two_peaks(df, "macd_hist", tf)
    if r is None:
        return 0.0, "ピーク不足"
    pr1, h1, pr2, h2 = r
    if pr2 <= pr1 or h1 <= 0:
        return 0.0, "条件未成立"
    ratio = h2 / h1
    if ratio >= 0.75:
        return 0.0, "縮小なし " + str(round(ratio * 100)) + "%"
    return min(1.0, (0.75 - ratio) / 0.75), "MACDヒスト縮小 " + str(round(h1, 2)) + "→" + str(round(h2, 2))


def check_bb_break(df, tf):
    price = float(df["Close"].iloc[-1])
    bb_u  = float(df["bb_upper"].iloc[-1])
    if price <= bb_u:
        return 0.0, "上限内"
    pct = (price - bb_u) / bb_u * 100
    return min(1.0, 0.5 + pct / 4.0), "上限を " + str(round(pct, 1)) + "% 超過"


CHECK_FN = {
    "rsi_hot":  check_rsi_hot,
    "rsi_div":  check_rsi_div,
    "vol_div":  check_vol_div,
    "ema_dev":  check_ema_dev,
    "macd_div": check_macd_div,
    "bb_break": check_bb_break,
}


def evaluate(ticker, tf_data, tf_status):
    calc_data = {}
    for tf, df in tf_data.items():
        try:
            calc_data[tf] = calc(df, tf) if df is not None else None
        except Exception as e:
            print(f"  calc失敗 {tf}: {e}")
            calc_data[tf] = None

    signals = []
    for key, tf, weight, label, star in SIGNAL_DEFS:
        df_c = calc_data.get(tf)
        if df_c is None or len(df_c) < 10:
            signals.append({"label": label, "star": star, "tf": tf, "weight": weight,
                            "score": 0.0, "detail": tf_status.get(tf, "不明"), "na": True})
            continue
        try:
            score, detail = CHECK_FN[key](df_c, tf)
        except Exception:
            score, detail = 0.0, "計算エラー"
        signals.append({"label": label, "star": star, "tf": tf, "weight": weight,
                        "score": score, "detail": detail, "na": False})

    valid   = [s for s in signals if not s["na"]]
    total_w = sum(s["weight"] for s in valid) or 1
    pct     = sum(s["weight"] * s["score"] for s in valid) / total_w * 100

    latest = next((calc_data[tf] for tf in ["1d", "1wk", "4h", "1h"]
                   if calc_data.get(tf) is not None), None)
    price  = float(latest["Close"].iloc[-1]) if latest is not None else 0

    return {"ticker": ticker, "price": price, "score": pct,
            "signals": signals, "status": tf_status}


def score_bar(pct, w=10):
    f   = round(pct / 100 * w)
    bar = "█" * f + "░" * (w - f)
    if   pct >= 70: col = "🔴"
    elif pct >= 55: col = "🟠"
    elif pct >= 35: col = "🟡"
    elif pct >= 15: col = "🟢"
    else:           col = "⚪"
    return col + " " + bar + " " + str(round(pct)) + "%"


def tf_coverage(status):
    parts = []
    for tf, lbl in TF_LABEL.items():
        st = status.get(tf, "不明")
        parts.append(lbl + ("✅" if st == "ok" else "❌"))
    return " ".join(parts)


def tf_signals(signals, r):
    out = []
    for tf, lbl in TF_LABEL.items():
        if r["status"].get(tf) != "ok":
            out.append(lbl + "➖")
            continue
        grp   = [s for s in signals if s["tf"] == tf and not s["na"]]
        fired = sum(1 for s in grp if s["score"] >= 0.5)
        total = len(grp)
        if   fired == total: out.append(lbl + "🔴")
        elif fired > 0:      out.append(lbl + "🟡")
        else:                out.append(lbl + "⚪")
    return " ".join(out)


def sig_dot(score, na):
    if na:            return "➖"
    if score >= 0.7:  return "🔴"
    if score >= 0.35: return "🟡"
    return "⚪"


def build_summary(results):
    danger  = [r for r in results if r["score"] >= ALERT_SCORE]
    caution = [r for r in results if CAUTION_SCORE <= r["score"] < ALERT_SCORE]
    watch   = [r for r in results if 10 <= r["score"] < CAUTION_SCORE]
    lines   = []

    if not danger and not caution and not watch:
        lines.append("天井シグナルなし ✅")

    if danger:
        lines.append("**🚨 危険圏 " + str(ALERT_SCORE) + "%以上**")
        for r in sorted(danger, key=lambda x: -x["score"]):
            lines.append("`" + r["ticker"].ljust(12) + "` " + score_bar(r["score"]))
            lines.append("              " + tf_signals(r["signals"], r) + "  " + tf_coverage(r["status"]))

    if caution:
        lines.append("\n**⚠️ 注意圏 " + str(CAUTION_SCORE) + "〜" + str(ALERT_SCORE) + "%**")
        for r in sorted(caution, key=lambda x: -x["score"]):
            lines.append("`" + r["ticker"].ljust(12) + "` " + score_bar(r["score"]) + "  " + tf_signals(r["signals"], r))

    if watch:
        lines.append("\n**👀 監視圏 10〜" + str(CAUTION_SCORE) + "%**")
        for r in sorted(watch, key=lambda x: -x["score"]):
            lines.append("`" + r["ticker"].ljust(12) + "` " + score_bar(r["score"]))

    lines.append("\n**📊 全銘柄ランキング（上位15）**")
    top15 = sorted(results, key=lambda x: -x["score"])[:15]
    for r in top15:
        price_str = str(round(r["price"], 2))
        lines.append("`" + r["ticker"].ljust(12) + "` " + score_bar(r["score"]) + "  $" + price_str)

    partial = [r for r in results if any(s != "ok" for s in r["status"].values())]
    if partial:
        names = ", ".join(r["ticker"] for r in partial[:8])
        lines.append("\n⚠️ 一部足取得不能: " + names)

    color = 0xFF2222 if danger else (0xFF8800 if caution else 0x3399FF)
    return {
        "embeds": [{
            "title": "🔍 天井チェック " + str(len(results)) + "銘柄 — " + datetime.now().strftime("%m/%d %H:%M"),
            "description": "\n".join(lines)[:4000],
            "color": color,
        }]
    }


def build_detail(r):
    lines = [
        "**スコア " + score_bar(r["score"]) + "  $" + str(round(r["price"], 2)) + "**",
        "データ: " + tf_coverage(r["status"]) + "\n",
    ]
    for star_label, star_key in [("★★★ 最重要", "AAA"), ("★★☆ 重要", "AA"), ("★☆☆ 補助", "A")]:
        grp = [s for s in r["signals"] if s["star"] == star_key]
        if not grp:
            continue
        lines.append("**" + star_label + "**")
        for s in grp:
            dot = sig_dot(s["score"], s["na"])
            filled = round(s["score"] * 5)
            bar = "▓" * filled + "░" * (5 - filled)
            na_note = " (" + s["detail"] + ")" if s["na"] else ""
            lines.append(dot + " [" + bar + "] `" + s["label"] + "`" + na_note)
            if not s["na"]:
                lines.append("   └ " + s["detail"])
        lines.append("")
    return {
        "embeds": [{
            "title": "🚨 " + r["ticker"] + " 天井詳細",
            "description": "\n".join(lines)[:4000],
            "color": 0xFF2222,
            "footer": {"text": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        }]
    }


def send(payload):
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        time.sleep(0.6)
        return resp.status_code == 204
    except Exception as e:
        print("送信エラー: " + str(e))
        return False


def main():
    print("=" * 60)
    print("[" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "]  天井サイン検出 v2.2")
    print("銘柄: " + str(len(WATCHLIST)) + "  シグナル/銘柄: " + str(len(SIGNAL_DEFS)))
    print("=" * 60)

    if not DISCORD_WEBHOOK_URL:
        print("環境変数 DISCORD_WEBHOOK_URL が未設定です")
        return

    results = []
    for i, ticker in enumerate(WATCHLIST, 1):
        print("[" + str(i).rjust(2) + "/" + str(len(WATCHLIST)) + "] " + ticker.ljust(14), end=" ", flush=True)
        try:
            tf_data, tf_status = fetch_timeframes(ticker)
            r = evaluate(ticker, tf_data, tf_status)
            results.append(r)
            cov = " ".join(TF_LABEL[tf] + ("✅" if tf_status[tf] == "ok" else "❌")
                           for tf in ["1wk", "1d", "4h", "1h"])
            print(score_bar(r["score"]) + "  " + cov)
        except Exception as e:
            print("エラー: " + str(e))

    print()
    ok = send(build_summary(results))
    print("サマリー通知: " + ("✅" if ok else "❌"))

    for r in [x for x in results if x["score"] >= ALERT_SCORE]:
        ok = send(build_detail(r))
        print("詳細 " + r["ticker"] + ": " + ("✅" if ok else "❌"))

    if results:
        top = max(results, key=lambda x: x["score"])
        print("最高スコア: " + top["ticker"] + " " + str(round(top["score"], 1)) + "%")


if __name__ == "__main__":
    main()
