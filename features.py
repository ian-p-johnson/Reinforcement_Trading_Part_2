from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx_local = out.index
    idx_utc = idx_local.tz_convert("UTC") if idx_local.tz is not None else idx_local.tz_localize("UTC")

    minute_of_day = idx_local.hour * 60 + idx_local.minute
    day_of_week = idx_local.day_of_week

    out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    out["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    h_utc = idx_utc.hour
    out["session_asia"] = ((h_utc >= 0) & (h_utc < 7)).astype(int)
    out["session_london"] = ((h_utc >= 7) & (h_utc < 16)).astype(int)
    out["session_newyork"] = ((h_utc >= 13) & (h_utc < 22)).astype(int)
    out["session_london_ny_overlap"] = ((h_utc >= 13) & (h_utc < 16)).astype(int)
    return out


def add_stationary_features(df: pd.DataFrame, atr_period: int = 14, rsi_period: int = 14) -> Tuple[pd.DataFrame, List[str]]:
    """Add causal technical indicators and stationary/relative features.

    Returns: (feature dataframe, list of columns intended for ML/RL observation)
    """
    out = df.copy()

    # Raw causal indicators.
    out["atr"] = atr(out, atr_period)
    out["ema20"] = out["Close"].ewm(span=20, adjust=False, min_periods=20).mean()
    out["ema50"] = out["Close"].ewm(span=50, adjust=False, min_periods=50).mean()
    out["ema200"] = out["Close"].ewm(span=200, adjust=False, min_periods=200).mean()
    out["rsi14"] = rsi(out["Close"], rsi_period)

    ema12 = out["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    bb_mid = out["Close"].rolling(20, min_periods=20).mean()
    bb_std = out["Close"].rolling(20, min_periods=20).std()
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_width"] = out["bb_upper"] - out["bb_lower"]

    # Avoid division by zero.
    eps = 1e-12
    a = out["atr"].replace(0, np.nan)

    # Trend / mean distance.
    out["close_ema20_atr"] = (out["Close"] - out["ema20"]) / (a + eps)
    out["close_ema50_atr"] = (out["Close"] - out["ema50"]) / (a + eps)
    out["close_ema200_atr"] = (out["Close"] - out["ema200"]) / (a + eps)
    out["ema20_ema50_atr"] = (out["ema20"] - out["ema50"]) / (a + eps)
    # DROPPED (collinearity audit 2026-06-02): |r|=0.96 with close_ema50_atr and
    # 0.93 with ema20_ema50_atr — the EMA-distance features already encode trend
    # slope, so this added redundant capacity that helped the agent overfit.
    # out["ema50_slope5_atr"] = (out["ema50"] - out["ema50"].shift(5)) / (a + eps)

    # Momentum.
    # DROPPED: |r|=0.98 with close_ema20_atr (0.95 with close_ema50_atr) — RSI vs
    # the mid-distance carried essentially the same information on this data.
    # out["rsi_centered"] = out["rsi14"] - 50.0
    out["macd_hist_atr"] = out["macd_hist"] / (a + eps)
    # DROPPED: |r|=1.00 with ret5_atr — roc5_atr is the identical formula
    # ((Close - Close.shift(5)) / atr), so it was a pure duplicate of ret5_atr.
    # out["roc5_atr"] = (out["Close"] - out["Close"].shift(5)) / (a + eps)
    out["roc20_atr"] = (out["Close"] - out["Close"].shift(20)) / (a + eps)

    # Volatility regime.
    out["atr_close"] = out["atr"] / out["Close"]
    atr_fast = out["atr"].ewm(span=5, adjust=False, min_periods=5).mean()
    atr_slow = out["atr"].ewm(span=30, adjust=False, min_periods=30).mean()
    out["atr_fast_slow"] = atr_fast / atr_slow.replace(0, np.nan)
    out["bb_width_close"] = out["bb_width"] / out["Close"]

    # Candle shape.
    # DROPPED: |r|=1.00 with ret1_atr — on a gapless intraday series Open[t] ≈
    # Close[t-1], so body (Close-Open) and the 1-bar return were duplicates.
    # out["body_atr"] = (out["Close"] - out["Open"]) / (a + eps)
    out["range_atr"] = (out["High"] - out["Low"]) / (a + eps)
    candle_range = (out["High"] - out["Low"]).replace(0, np.nan)
    out["upper_wick_ratio"] = (out["High"] - out[["Open", "Close"]].max(axis=1)) / candle_range
    out["lower_wick_ratio"] = (out[["Open", "Close"]].min(axis=1) - out["Low"]) / candle_range

    # Microstructure / recent moves.
    for k in range(1, 6):
        out[f"ret{k}_atr"] = (out["Close"] - out["Close"].shift(k)) / (a + eps)

    out = add_time_features(out)

    # Observation columns for ML/RL.  Four highly-collinear features were
    # dropped in the 2026-06-02 audit (see DROPPED notes above): ema50_slope5_atr,
    # rsi_centered, roc5_atr, body_atr.  25 features remain.
    feature_cols = [
        "close_ema20_atr", "close_ema50_atr", "close_ema200_atr",
        "ema20_ema50_atr",                       # ema50_slope5_atr dropped (|r|=0.96)
        "macd_hist_atr", "roc20_atr",            # rsi_centered (|r|=0.98), roc5_atr (|r|=1.00) dropped
        "atr_close", "atr_fast_slow", "bb_width_close",
        "range_atr", "upper_wick_ratio", "lower_wick_ratio",   # body_atr (|r|=1.00) dropped
        "ret1_atr", "ret2_atr", "ret3_atr", "ret4_atr", "ret5_atr",
        "tod_sin", "tod_cos", "dow_sin", "dow_cos",
        "session_asia", "session_london", "session_newyork", "session_london_ny_overlap",
    ]

    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    return out, feature_cols


def prepare_feature_frame(df: pd.DataFrame, warmup_bars: int = 250, atr_period: int = 14, rsi_period: int = 14):
    feat, feature_cols = add_stationary_features(df, atr_period=atr_period, rsi_period=rsi_period)
    feat = feat.iloc[warmup_bars:].copy()
    feat = feat.dropna(subset=feature_cols + ["atr", "ema20", "ema50", "ema200"])
    return feat, feature_cols
