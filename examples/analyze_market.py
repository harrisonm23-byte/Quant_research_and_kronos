"""
analyze_market.py - End-to-end market analysis with Kronos.

Three modes, one tool:

  forecast   Probabilistic forecast of the next `pred_len` bars for a single
             asset. Draws several sample paths and reports a mean forecast plus
             quantile uncertainty bands. Saves a plot and a CSV.

  backtest   Walk-forward evaluation of the *model* (not just a trading rule).
             Repeatedly forecasts `pred_len` bars ahead from rolling origins and
             scores the forecasts against what actually happened: directional
             accuracy, MAE / RMSE / MAPE on close, plus a simple forecast-driven
             long/flat strategy compared against buy-and-hold.

  signal     Decision-support report. Combines the latest forecast with simple
             trend and support/resistance reads into a single JSON report with a
             confidence derived from the dispersion of the sample paths.

The model weights are pulled from the Hugging Face Hub by default, but every
path accepts a local directory via --tokenizer-path / --model-path so the tool
runs fully offline once the weights have been downloaded once.

Examples
--------
  # Forecast the next 30 bars from a CSV, averaging 30 sample paths
  python analyze_market.py forecast --csv data/btc_1h.csv --pred-len 30 --samples 30

  # Walk-forward backtest, re-forecasting every 24 bars
  python analyze_market.py backtest --csv data/btc_1h.csv --pred-len 24 --step 24

  # Signal report written to JSON
  python analyze_market.py signal --csv data/btc_1h.csv --pred-len 24 --out report.json
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Kronos, KronosTokenizer, KronosPredictor

# Default Hugging Face repos for each model size. Override with --model-path /
# --tokenizer-path to load from a local directory (offline).
MODEL_REPOS = {
    "mini":  {"tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",   "model": "NeoQuasar/Kronos-mini",  "max_context": 2048},
    "small": {"tokenizer": "NeoQuasar/Kronos-Tokenizer-base", "model": "NeoQuasar/Kronos-small", "max_context": 512},
    "base":  {"tokenizer": "NeoQuasar/Kronos-Tokenizer-base", "model": "NeoQuasar/Kronos-base",  "max_context": 512},
}

OHLCV = ["open", "high", "low", "close", "volume", "amount"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_csv(path, time_col="timestamps", rename=None):
    """Load an OHLCV CSV into a standardized DataFrame.

    Returns a DataFrame indexed 0..N-1 with a datetime `timestamps` column and
    at least open/high/low/close columns. volume/amount are filled with zeros if
    absent (Kronos treats them as optional).
    """
    df = pd.read_csv(path)
    if rename:
        df = df.rename(columns=rename)
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found. Available: {list(df.columns)}")
    df = df.rename(columns={time_col: "timestamps"})
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    missing = [c for c in ["open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"Required price columns missing: {missing}. Available: {list(df.columns)}")
    for col in ["volume", "amount"]:
        if col not in df.columns:
            df[col] = 0.0
    return df


def infer_future_timestamps(hist_ts, pred_len):
    """Extend a timestamp series by `pred_len` steps using the median spacing."""
    hist_ts = pd.to_datetime(pd.Series(hist_ts).reset_index(drop=True))
    if len(hist_ts) < 2:
        raise ValueError("Need at least 2 historical timestamps to infer cadence.")
    step = hist_ts.diff().dropna().median()
    last = hist_ts.iloc[-1]
    return pd.Series([last + step * (i + 1) for i in range(pred_len)])


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_predictor(args):
    repo = MODEL_REPOS[args.model_size]
    tok_src = args.tokenizer_path or repo["tokenizer"]
    mdl_src = args.model_path or repo["model"]
    max_context = args.max_context or repo["max_context"]

    print(f"Loading tokenizer from {tok_src}")
    tokenizer = KronosTokenizer.from_pretrained(tok_src)
    print(f"Loading model from {mdl_src}")
    model = Kronos.from_pretrained(mdl_src)

    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=max_context)
    print(f"Predictor ready on device: {predictor.device}")
    return predictor


def sample_paths(predictor, x_df, x_ts, y_ts, pred_len, n_samples, T, top_p, top_k):
    """Draw `n_samples` independent forecast paths.

    Returns an array of shape (n_samples, pred_len, len(OHLCV)). Each call to
    predict() uses sample_count=1 so the paths reflect the model's own sampling
    dispersion rather than being pre-averaged.
    """
    paths = []
    for i in range(n_samples):
        pred = predictor.predict(
            df=x_df[OHLCV], x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=T, top_p=top_p, top_k=top_k,
            sample_count=1, verbose=False,
        )
        paths.append(pred[OHLCV].values)
    return np.stack(paths, axis=0)


# --------------------------------------------------------------------------- #
# Mode: forecast
# --------------------------------------------------------------------------- #
def summarize_paths(paths, y_ts):
    """Collapse sample paths into mean + quantile bands for the close price."""
    close = paths[:, :, OHLCV.index("close")]  # (n_samples, pred_len)
    summary = pd.DataFrame(index=pd.DatetimeIndex(y_ts))
    summary["close_mean"] = close.mean(axis=0)
    summary["close_p10"] = np.percentile(close, 10, axis=0)
    summary["close_p50"] = np.percentile(close, 50, axis=0)
    summary["close_p90"] = np.percentile(close, 90, axis=0)
    summary["close_std"] = close.std(axis=0)
    # mean OHLCV for completeness
    for j, col in enumerate(OHLCV):
        summary[f"{col}_mean"] = paths[:, :, j].mean(axis=0)
    return summary


def run_forecast(args, predictor):
    df = load_csv(args.csv, args.time_col, parse_rename(args.rename))
    lookback = args.lookback or min(len(df) - 1, predictor.max_context)
    if lookback >= len(df):
        raise ValueError(f"lookback ({lookback}) must be < rows ({len(df)})")

    x_df = df.iloc[-lookback:].reset_index(drop=True)
    x_ts = x_df["timestamps"]
    y_ts = infer_future_timestamps(x_ts, args.pred_len)

    print(f"Forecasting {args.pred_len} bars from {lookback} bars of history "
          f"with {args.samples} sample path(s)...")
    paths = sample_paths(predictor, x_df, x_ts, y_ts, args.pred_len,
                         args.samples, args.T, args.top_p, args.top_k)
    summary = summarize_paths(paths, y_ts)

    print("\nForecast (close):")
    print(summary[["close_mean", "close_p10", "close_p90", "close_std"]].head(10).to_string())

    if args.out:
        summary.to_csv(args.out)
        print(f"\nSaved forecast to {args.out}")
    if args.plot:
        plot_forecast(x_df, summary, args.plot)
        print(f"Saved plot to {args.plot}")
    return summary


def plot_forecast(x_df, summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_df["timestamps"], x_df["close"], color="#1f77b4", label="History", linewidth=1.3)
    ax.plot(summary.index, summary["close_mean"], color="#d62728", label="Forecast (mean)", linewidth=1.5)
    ax.fill_between(summary.index, summary["close_p10"], summary["close_p90"],
                    color="#d62728", alpha=0.2, label="P10-P90 band")
    ax.set_ylabel("Close")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Mode: signal
# --------------------------------------------------------------------------- #
def support_resistance(close, window=20):
    """Most recent swing low / high over the trailing window."""
    recent = close[-window:] if len(close) >= window else close
    return float(np.min(recent)), float(np.max(recent))


def run_signal(args, predictor):
    df = load_csv(args.csv, args.time_col, parse_rename(args.rename))
    lookback = args.lookback or min(len(df) - 1, predictor.max_context)
    x_df = df.iloc[-lookback:].reset_index(drop=True)
    x_ts = x_df["timestamps"]
    y_ts = infer_future_timestamps(x_ts, args.pred_len)

    paths = sample_paths(predictor, x_df, x_ts, y_ts, args.pred_len,
                         max(args.samples, 10), args.T, args.top_p, args.top_k)
    summary = summarize_paths(paths, y_ts)

    last_close = float(x_df["close"].iloc[-1])
    fc_close = float(summary["close_mean"].iloc[-1])
    implied_return = (fc_close - last_close) / last_close

    # trend from recent realized history (linear slope over lookback, normalized)
    n = min(lookback, 60)
    hist = x_df["close"].iloc[-n:].values
    slope = np.polyfit(np.arange(n), hist, 1)[0] / (hist.mean() + 1e-9)

    # confidence: how tightly the sample paths agree at the horizon, relative to
    # the size of the move. High dispersion => low confidence.
    terminal = paths[:, -1, OHLCV.index("close")]
    disp = terminal.std() / (last_close + 1e-9)
    agreement = float(np.mean(np.sign(terminal - last_close) == np.sign(implied_return)))
    confidence = round(float(agreement * np.exp(-abs(disp) * 5)), 4)

    sup, res = support_resistance(x_df["close"].values)
    direction = "up" if implied_return > 0 else ("down" if implied_return < 0 else "flat")

    report = {
        "asset_csv": os.path.basename(args.csv),
        "as_of": str(x_ts.iloc[-1]),
        "model": MODEL_REPOS[args.model_size]["model"] if not args.model_path else args.model_path,
        "lookback_bars": lookback,
        "horizon_bars": args.pred_len,
        "last_close": round(last_close, 6),
        "forecast_close": round(fc_close, 6),
        "implied_return": round(implied_return, 6),
        "direction": direction,
        "path_agreement": round(agreement, 4),
        "terminal_dispersion": round(float(disp), 6),
        "confidence": confidence,
        "recent_trend_slope": round(float(slope), 8),
        "trend": "up" if slope > 0 else ("down" if slope < 0 else "flat"),
        "support_level": round(sup, 6),
        "resistance_level": round(res, 6),
    }

    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved signal report to {args.out}")
    return report


# --------------------------------------------------------------------------- #
# Mode: backtest (walk-forward model evaluation)
# --------------------------------------------------------------------------- #
def run_backtest(args, predictor):
    df = load_csv(args.csv, args.time_col, parse_rename(args.rename))
    lookback = args.lookback or min(predictor.max_context, 256)
    pred_len = args.pred_len
    step = args.step or pred_len

    origins = list(range(lookback, len(df) - pred_len, step))
    if not origins:
        raise ValueError("Not enough data for a single window. Reduce --lookback or --pred-len.")
    print(f"Walk-forward backtest: {len(origins)} windows "
          f"(lookback={lookback}, pred_len={pred_len}, step={step})")

    rows = []
    for k, o in enumerate(origins):
        x_df = df.iloc[o - lookback:o].reset_index(drop=True)
        x_ts = x_df["timestamps"]
        y_ts = df["timestamps"].iloc[o:o + pred_len].reset_index(drop=True)
        actual = df["close"].iloc[o:o + pred_len].reset_index(drop=True).values

        paths = sample_paths(predictor, x_df, x_ts, y_ts, pred_len,
                             args.samples, args.T, args.top_p, args.top_k)
        pred_close = paths[:, :, OHLCV.index("close")].mean(axis=0)

        last_close = float(x_df["close"].iloc[-1])
        pred_dir = np.sign(pred_close[-1] - last_close)
        actual_dir = np.sign(actual[-1] - last_close)

        # per-step directional agreement across the whole horizon (not just the
        # terminal bar): of the pred_len steps, how often does the predicted
        # move-from-origin share the actual move-from-origin's sign.
        step_dir_hit = float(np.mean(np.sign(pred_close - last_close)
                                     == np.sign(actual - last_close)))

        rows.append({
            "origin": o,
            "t": str(df["timestamps"].iloc[o]),
            "last_close": last_close,
            "pred_terminal": float(pred_close[-1]),
            "actual_terminal": float(actual[-1]),
            "pred_ret": (pred_close[-1] - last_close) / last_close,
            "actual_ret": (actual[-1] - last_close) / last_close,
            "dir_hit": float(pred_dir == actual_dir),
            "step_dir_hit": step_dir_hit,
            "mae": float(np.mean(np.abs(pred_close - actual))),
            "rmse": float(np.sqrt(np.mean((pred_close - actual) ** 2))),
            "mape": float(np.mean(np.abs((pred_close - actual) / (actual + 1e-9)))),
            "abs_pct_err_terminal": float(abs(pred_close[-1] - actual[-1]) / (actual[-1] + 1e-9)),
            "signed_pct_err_terminal": float((pred_close[-1] - actual[-1]) / (actual[-1] + 1e-9)),
        })
        print(f"  [{k+1}/{len(origins)}] {rows[-1]['t']}  "
              f"dir={'HIT' if rows[-1]['dir_hit'] else 'miss'}  "
              f"MAPE={rows[-1]['mape']:.3%}")

    res = pd.DataFrame(rows)
    metrics = compute_backtest_stats(res, args.threshold,
                                     meta={"symbol_csv": os.path.basename(args.csv),
                                           "lookback": lookback, "pred_len": pred_len,
                                           "step": step, "samples": args.samples})

    print("\n=== Backtest statistics ===")
    print(json.dumps(metrics, indent=2))
    if args.out:
        res.to_csv(args.out, index=False)
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved per-window results to {args.out} and summary JSON alongside it.")
    return metrics, res


def compute_backtest_stats(res, threshold, meta=None):
    """Comprehensive evaluation stats for a walk-forward backtest.

    Grouped into:
      dataset      - what was tested
      directional  - can the model call the direction of the move?
      accuracy     - how far off are the price forecasts (margin of inaccuracy)?
      bias         - does the model systematically over/under-shoot?
      strategy     - economics of trading the signal vs buy-and-hold
      baseline     - naive yardsticks to judge the above against
    """
    n = len(res)
    pred_ret = res["pred_ret"].values
    actual_ret = res["actual_ret"].values
    pred_dir = np.sign(pred_ret)
    actual_dir = np.sign(actual_ret)

    # ---- directional ----
    up = pred_dir > 0
    down = pred_dir < 0
    n_up, n_down = int(up.sum()), int(down.sum())
    base_rate_up = float((actual_dir > 0).mean())          # how often it really rose
    directional = {
        "terminal_accuracy": round(float(res["dir_hit"].mean()), 4),
        "per_step_accuracy": round(float(res["step_dir_hit"].mean()), 4),
        "precision_when_predicting_up": round(float((actual_dir[up] > 0).mean()), 4) if n_up else None,
        "precision_when_predicting_down": round(float((actual_dir[down] < 0).mean()), 4) if n_down else None,
        "pct_calls_up": round(float(up.mean()), 4),
        "pct_calls_down": round(float(down.mean()), 4),
        "actual_up_rate": round(base_rate_up, 4),
        "return_correlation": (round(float(np.corrcoef(pred_ret, actual_ret)[0, 1]), 4)
                               if n > 1 and pred_ret.std() > 1e-12 and actual_ret.std() > 1e-12 else None),
    }

    # ---- accuracy (margin of inaccuracy on the terminal close) ----
    abs_pct = res["abs_pct_err_terminal"].values
    accuracy = {
        "terminal_mape": round(float(abs_pct.mean()), 6),
        "terminal_mape_median": round(float(np.median(abs_pct)), 6),
        "terminal_mape_p90": round(float(np.percentile(abs_pct, 90)), 6),
        "terminal_mape_worst": round(float(abs_pct.max()), 6),
        "path_mape_mean": round(float(res["mape"].mean()), 6),
        "mae_mean": round(float(res["mae"].mean()), 6),
        "rmse_mean": round(float(res["rmse"].mean()), 6),
    }

    # ---- bias (signed: + = model overshoots / too bullish) ----
    signed_pct = res["signed_pct_err_terminal"].values
    bias = {
        "mean_signed_pct_err": round(float(signed_pct.mean()), 6),
        "mean_predicted_return": round(float(pred_ret.mean()), 6),
        "mean_actual_return": round(float(actual_ret.mean()), 6),
        "return_bias": round(float((pred_ret - actual_ret).mean()), 6),
    }

    # ---- strategy: long the horizon when predicted return > threshold ----
    traded = (pred_ret > threshold)
    strat_ret = np.where(traded, actual_ret, 0.0)
    tr = actual_ret[traded]
    wins, losses = tr[tr > 0], tr[tr < 0]
    eq = np.cumprod(1 + strat_ret)
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / np.where(peak == 0, 1, peak)).min()) if n else 0.0
    strategy = {
        "trades_taken": int(traded.sum()),
        "win_rate": round(float((tr > 0).mean()), 4) if traded.any() else 0.0,
        "avg_win": round(float(wins.mean()), 6) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 6) if len(losses) else 0.0,
        "profit_factor": (round(float(wins.sum() / abs(losses.sum())), 4)
                          if losses.sum() != 0 else None),
        "avg_return_per_window": round(float(strat_ret.mean()), 6),
        "total_return": round(float(np.prod(1 + strat_ret) - 1), 6),
        "return_per_window_std": round(float(strat_ret.std()), 6),
        "sharpe_per_window": (round(float(strat_ret.mean() / strat_ret.std()), 4)
                              if strat_ret.std() > 1e-12 else None),
        "max_drawdown": round(max_dd, 6),
    }

    # ---- naive baselines to judge against ----
    bh = actual_ret
    baseline = {
        "buyhold_avg_return_per_window": round(float(bh.mean()), 6),
        "buyhold_total_return": round(float(np.prod(1 + bh) - 1), 6),
        "always_up_dir_accuracy": round(base_rate_up, 4),
        "majority_class_dir_accuracy": round(max(base_rate_up, 1 - base_rate_up), 4),
    }

    out = {"dataset": {**(meta or {}), "windows": n}}
    out.update({"directional": directional, "accuracy": accuracy,
                "bias": bias, "strategy": strategy, "baseline": baseline})
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_rename(spec):
    """Parse 'csvcol=open,csvcol2=close' into a rename dict."""
    if not spec:
        return None
    out = {}
    for pair in spec.split(","):
        k, v = pair.split("=")
        out[k.strip()] = v.strip()
    return out


def build_arg_parser():
    p = argparse.ArgumentParser(description="Market analysis with Kronos (forecast / backtest / signal).")
    sub = p.add_subparsers(dest="mode", required=True)

    def common(sp):
        sp.add_argument("--csv", required=True, help="Path to OHLCV CSV.")
        sp.add_argument("--time-col", default="timestamps", help="Name of the timestamp column.")
        sp.add_argument("--rename", default=None,
                        help="Comma list mapping CSV columns to standard names, e.g. 'Close=close,Vol=volume'.")
        sp.add_argument("--model-size", default="small", choices=list(MODEL_REPOS), help="Model capacity.")
        sp.add_argument("--tokenizer-path", default=None, help="Local tokenizer dir (offline).")
        sp.add_argument("--model-path", default=None, help="Local model dir (offline).")
        sp.add_argument("--max-context", type=int, default=None, help="Override max context length.")
        sp.add_argument("--device", default=None, help="cpu / cuda:0 / mps (auto if omitted).")
        sp.add_argument("--lookback", type=int, default=None, help="History bars fed to the model.")
        sp.add_argument("--pred-len", type=int, default=24, help="Bars to forecast ahead.")
        sp.add_argument("--samples", type=int, default=1, help="Number of sample paths to draw.")
        sp.add_argument("--T", type=float, default=1.0, help="Sampling temperature.")
        sp.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling probability.")
        sp.add_argument("--top-k", type=int, default=0, help="Top-k filter (0 = off).")
        sp.add_argument("--out", default=None, help="Output file (CSV or JSON depending on mode).")

    f = sub.add_parser("forecast", help="Probabilistic forecast for one asset.")
    common(f)
    f.add_argument("--plot", default=None, help="Path to save a forecast PNG.")

    s = sub.add_parser("signal", help="Decision-support signal report (JSON).")
    common(s)

    b = sub.add_parser("backtest", help="Walk-forward model evaluation.")
    common(b)
    b.add_argument("--step", type=int, default=None, help="Bars between forecast origins (default = pred_len).")
    b.add_argument("--threshold", type=float, default=0.0,
                   help="Min predicted return to take a long position in the toy strategy.")
    return p


def main():
    args = build_arg_parser().parse_args()
    predictor = build_predictor(args)
    if args.mode == "forecast":
        run_forecast(args, predictor)
    elif args.mode == "signal":
        run_signal(args, predictor)
    elif args.mode == "backtest":
        run_backtest(args, predictor)


if __name__ == "__main__":
    main()
