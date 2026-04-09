#!/usr/bin/env python3
"""
diagnostics.py — SARIMA model diagnostics for Lake Sevan
Loads cached CSVs and runs diagnostic plots without touching the frontend.

Usage:
    python diagnostics.py
"""

import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

import sevanSA

BASE_DIR = pathlib.Path(__file__).parent


def load_data():
    cache_csv   = BASE_DIR / sevanSA.CONFIG["cache_csv"]
    landsat_csv = BASE_DIR / sevanSA.CONFIG["landsat_cache_csv"]

    if not cache_csv.exists():
        raise FileNotFoundError(f"{cache_csv} not found — run sevanSA.py first to build the cache.")

    jrc_df     = pd.read_csv(cache_csv, parse_dates=["date"])
    landsat_df = pd.read_csv(landsat_csv, parse_dates=["date"]) \
                 if landsat_csv.exists() else pd.DataFrame()

    raw_df = sevanSA.combine_jrc_landsat(jrc_df, landsat_df)
    df     = sevanSA.build_complete_timeseries(raw_df)
    return df


def fit_model(df):
    clim, clim_std, anomaly = sevanSA.compute_climatology(df)

    ts = df.set_index("date")["area_km2"].copy()
    ts.index = pd.DatetimeIndex(ts.index)
    train = ts[ts.index.year >= sevanSA.CONFIG["sarima_start_year"]].asfreq("MS")

    # Same imputation as sevanSA.fit_sarima
    filled = train.copy()
    filled[filled.isna()] = filled.index[filled.isna()].map(
        lambda d: clim.get(d.month, np.nan)
    )
    filled = filled.interpolate(method="linear").bfill().ffill()

    model = SARIMAX(
        filled,
        order=sevanSA.CONFIG["sarima_order"],
        seasonal_order=sevanSA.CONFIG["sarima_seasonal"],
        enforce_stationarity=False,
        enforce_invertibility=False,
        freq="MS",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp=False, method="lbfgs", maxiter=200)

    return result, filled, clim, anomaly


def print_summary(result):
    print("\n" + "=" * 60)
    print("SARIMA MODEL SUMMARY")
    print("=" * 60)
    print(result.summary())

    resid = result.resid.dropna()
    dw    = durbin_watson(resid)
    lb    = acorr_ljungbox(resid, lags=[10, 20], return_df=True)

    print("\n--- Residual Diagnostics ---")
    print(f"  Durbin-Watson statistic : {dw:.4f}  (2.0 = no autocorrelation)")
    print(f"  Mean residual           : {resid.mean():.4f}  (should be ~0)")
    print(f"  Std  residual           : {resid.std():.4f}")
    print(f"\n  Ljung-Box test (no autocorrelation in residuals):")
    print(lb.to_string())

    print("\n--- Interpretation ---")
    if dw < 1.5:
        print("  [!] Durbin-Watson < 1.5 — positive autocorrelation in residuals (underfitting signal)")
    elif dw > 2.5:
        print("  [!] Durbin-Watson > 2.5 — negative autocorrelation in residuals")
    else:
        print("  [ok] Durbin-Watson looks healthy")

    if (lb["lb_pvalue"] < 0.05).any():
        print("  [!] Ljung-Box p < 0.05 — residuals still contain autocorrelation (model leaves signal on table)")
    else:
        print("  [ok] Ljung-Box: residuals appear white noise")

    aic, bic = result.aic, result.bic
    print(f"\n  AIC = {aic:.1f}   BIC = {bic:.1f}")
    print("  (Lower is better; useful for comparing alternative model orders)")


def plot_diagnostics(result, filled, clim, anomaly):
    resid = result.resid.dropna()

    # ── 1. Built-in statsmodels diagnostic panel ────────────────────────────
    fig1 = result.plot_diagnostics(figsize=(14, 8))
    fig1.suptitle("SARIMA Residual Diagnostics (statsmodels)", fontsize=13, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(BASE_DIR / "diag_residuals.png", dpi=150, bbox_inches="tight")
    print("\nSaved: diag_residuals.png")

    # ── 2. In-sample fit vs actuals ─────────────────────────────────────────
    fitted_vals = result.fittedvalues

    fig2, ax = plt.subplots(figsize=(14, 4))
    ax.plot(filled.index, filled.values, color="#2166ac", linewidth=0.8, label="Actual (imputed)")
    ax.plot(fitted_vals.index, fitted_vals.values, color="crimson", linewidth=0.8,
            linestyle="--", label="SARIMA in-sample fit")
    ax.set_title("In-Sample Fit vs Actuals", fontsize=12, fontweight="bold")
    ax.set_ylabel("Surface Area (km²)")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig2.tight_layout()
    fig2.savefig(BASE_DIR / "diag_fit.png", dpi=150, bbox_inches="tight")
    print("Saved: diag_fit.png")

    # ── 3. Residuals over time ───────────────────────────────────────────────
    fig3, ax = plt.subplots(figsize=(14, 3))
    ax.bar(resid.index, resid.values, width=25, color=np.where(resid.values >= 0, "#d6604d", "#4393c3"), alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Residuals Over Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Residual (km²)")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig3.tight_layout()
    fig3.savefig(BASE_DIR / "diag_residuals_time.png", dpi=150, bbox_inches="tight")
    print("Saved: diag_residuals_time.png")

    plt.close("all")


def auto_select_model(filled):
    """
    Grid search over candidate SARIMA orders and return the best by AIC.
    Candidates are chosen based on what the diagnostics suggest:
      - AR term was insignificant → try p=0
      - MA(1)≈-1 suggests over-differencing → try d=0
      - Ljung-Box failed → try higher p/q
    """
    candidates = [
        # (p,d,q)(P,D,Q,s)  — label
        ((0, 1, 1), (0, 1, 1, 12), "drop AR"),
        ((1, 1, 2), (0, 1, 1, 12), "extra MA"),
        ((2, 1, 1), (0, 1, 1, 12), "extra AR"),
        ((2, 1, 2), (0, 1, 1, 12), "AR+MA expanded"),
        ((1, 1, 1), (1, 1, 1, 12), "add seasonal AR"),
        ((0, 1, 2), (0, 1, 1, 12), "drop AR + extra MA"),
        ((1, 0, 1), (0, 1, 1, 12), "remove non-seasonal diff"),
        ((0, 1, 1), (0, 0, 1, 12), "remove seasonal diff"),
        ((1, 1, 1), (0, 1, 1, 12), "current (baseline)"),
    ]

    print("\n" + "=" * 70)
    print("AUTO MODEL SELECTION — Grid Search by AIC")
    print("=" * 70)
    print(f"  {'Model':<35} {'AIC':>10}  {'BIC':>10}  {'LjungBox(10)':>14}  Notes")
    print("  " + "-" * 80)

    results = []
    for order, seasonal_order, label in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = SARIMAX(filled, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False,
                            freq="MS")
                r = m.fit(disp=False, method="lbfgs", maxiter=300)

            lb_p = acorr_ljungbox(r.resid.dropna(), lags=[10], return_df=True)["lb_pvalue"].iloc[0]
            lb_ok = "ok" if lb_p >= 0.05 else f"FAIL p={lb_p:.3f}"
            model_str = f"SARIMA{order}x{seasonal_order}"
            print(f"  {model_str:<35} {r.aic:>10.1f}  {r.bic:>10.1f}  {lb_ok:>14}  {label}")
            results.append((r.aic, order, seasonal_order, label, r))
        except Exception as e:
            model_str = f"SARIMA{order}x{seasonal_order}"
            print(f"  {model_str:<35} {'FAILED':>10}  {'':>10}  {'':>14}  {label} ({e})")

    results.sort(key=lambda x: x[0])
    best_aic, best_order, best_seasonal, best_label, best_result = results[0]

    print(f"\n  Best model: SARIMA{best_order}x{best_seasonal}  ({best_label})  AIC={best_aic:.1f}")
    print(f"\n  To use this in sevanSA.py, update CONFIG:")
    print(f'    "sarima_order":    {best_order},')
    print(f'    "sarima_seasonal": {best_seasonal},')

    return best_result, best_order, best_seasonal


def plot_structural_break(df):
    """
    Plot the full area time series and mark the structural break visually.
    Uses a rolling 24-month mean to make the regime shift clear.
    Also prints a before/after comparison of mean and std.
    """
    ts = df.set_index("date")["area_km2"].dropna()
    ts.index = pd.DatetimeIndex(ts.index)

    BREAK_YEAR = 2007
    pre  = ts[ts.index.year <  BREAK_YEAR]
    post = ts[ts.index.year >= BREAK_YEAR]

    print("\n" + "=" * 60)
    print("STRUCTURAL BREAK ANALYSIS  (split year: 2007)")
    print("=" * 60)
    print(f"  Pre-2007  : n={len(pre):3d}  mean={pre.mean():.1f} km²  std={pre.std():.1f} km²")
    print(f"  Post-2007 : n={len(post):3d}  mean={post.mean():.1f} km²  std={post.std():.1f} km²")
    print(f"  Mean shift: {post.mean() - pre.mean():+.1f} km²")
    print(f"\n  Training SARIMA only on post-2007 data removes the declining")
    print(f"  regime from the fit, giving the model a homogeneous signal.")

    rolling = ts.rolling(window=24, min_periods=12).mean()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.scatter(ts.index, ts.values, s=5, color="#2166ac", alpha=0.5, label="Monthly obs (good)")
    ax.plot(rolling.index, rolling.values, color="black", linewidth=1.5, label="24-month rolling mean")
    ax.axvline(pd.Timestamp(f"{BREAK_YEAR}-01-01"), color="crimson", linewidth=1.5,
               linestyle="--", label=f"Structural break ({BREAK_YEAR})")
    ax.axvspan(ts.index.min(), pd.Timestamp(f"{BREAK_YEAR}-01-01"),
               color="#f4a58220", alpha=0.3, label="Declining regime (excluded from SARIMA)")
    ax.set_title("Lake Sevan — Structural Break Detection", fontsize=12, fontweight="bold")
    ax.set_ylabel("Surface Area (km²)")
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "diag_structural_break.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("\nSaved: diag_structural_break.png")


def backtest_comparison(df):
    """
    Hold out the last 24 good-quality months, fit both the old and new model
    on everything before that window, forecast 24 months ahead, and compare
    RMSE / MAE / MAPE against the actual observations.
    """
    clim, _, _ = sevanSA.compute_climatology(df)

    # Good observations only — these are the ground truth
    good = df[df["quality_flag"] == "good"].set_index("date")["area_km2"].copy()
    good.index = pd.DatetimeIndex(good.index)

    HOLDOUT = 24
    cutoff = good.index[-HOLDOUT]
    actual = good[good.index >= cutoff]

    def impute_and_fit(ts_train, order, seasonal_order):
        train = ts_train.asfreq("MS")
        filled = train.copy()
        filled[filled.isna()] = filled.index[filled.isna()].map(
            lambda d: clim.get(d.month, np.nan)
        )
        filled = filled.interpolate(method="linear").bfill().ffill()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SARIMAX(filled, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False, freq="MS")
            r = m.fit(disp=False, method="lbfgs", maxiter=300)
        fcast = r.get_forecast(steps=HOLDOUT)
        return fcast.predicted_mean, fcast.conf_int(alpha=0.05)

    def score(label, order, seasonal_order, start_year):
        ts_full = df.set_index("date")["area_km2"].copy()
        ts_full.index = pd.DatetimeIndex(ts_full.index)
        ts_train = ts_full[(ts_full.index.year >= start_year) & (ts_full.index < cutoff)]

        pred, ci = impute_and_fit(ts_train, order, seasonal_order)

        # Align predictions to actual good-obs dates only
        pred = pred.reindex(actual.index, method="nearest", tolerance=pd.Timedelta("15D")).dropna()
        obs  = actual.reindex(pred.index).dropna()
        pred = pred.reindex(obs.index)

        errors = obs - pred
        rmse   = np.sqrt((errors ** 2).mean())
        mae    = errors.abs().mean()
        mape   = (errors.abs() / obs.abs()).mean() * 100
        score_pct = max(0.0, 100 - mape)

        return {"label": label, "rmse": rmse, "mae": mae, "mape": mape,
                "score": score_pct, "pred": pred, "obs": obs, "ci": ci}

    print("\n" + "=" * 60)
    print(f"BACKTEST — Hold-out last {HOLDOUT} months  (cutoff: {cutoff.strftime('%Y-%m')})")
    print("=" * 60)

    old     = score("OLD  SARIMA(1,1,1)(0,1,1,12) from 1998", (1,1,1), (0,1,1,12), 1998)
    new     = score("NEW  SARIMA(0,1,2)(0,1,1,12) from 2007", (0,1,2), (0,1,1,12), 2007)
    combo   = score("MIX  SARIMA(0,1,2)(0,1,1,12) from 1998", (0,1,2), (0,1,1,12), 1998)

    for r in [old, new, combo]:
        print(f"\n  {r['label']}")
        print(f"    RMSE : {r['rmse']:.1f} km²   (avg forecast error magnitude)")
        print(f"    MAE  : {r['mae']:.1f} km²")
        print(f"    MAPE : {r['mape']:.2f}%        (% off actual area)")
        print(f"    Score: {r['score']:.1f} / 100")

    best = min([old, new, combo], key=lambda x: x["mape"])
    print(f"\n  Winner: {best['label'].strip()}  (MAPE={best['mape']:.2f}%)")

    # Plot backtest
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(old["obs"].index, old["obs"].values, color="black", linewidth=1.5,
            label="Actual (held-out)", zorder=5)
    ax.plot(old["pred"].index, old["pred"].values, color="#e74c3c", linewidth=1.2,
            linestyle="--", label=f"OLD  MAPE={old['mape']:.1f}%  score={old['score']:.0f}/100")
    ax.plot(new["pred"].index, new["pred"].values, color="#2ecc71", linewidth=1.2,
            linestyle="--", label=f"NEW  MAPE={new['mape']:.1f}%  score={new['score']:.0f}/100")
    ax.plot(combo["pred"].index, combo["pred"].values, color="#9b59b6", linewidth=1.2,
            linestyle="--", label=f"MIX  MAPE={combo['mape']:.1f}%  score={combo['score']:.0f}/100")
    ax.fill_between(combo["ci"].index, combo["ci"].iloc[:, 0], combo["ci"].iloc[:, 1],
                    color="#9b59b6", alpha=0.15, label="Mix model 95% CI")
    ax.set_title(f"Backtest: Last {HOLDOUT} months held out — 3-way comparison", fontsize=12, fontweight="bold")
    ax.set_ylabel("Surface Area (km²)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(BASE_DIR / "diag_backtest.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    print("\nSaved: diag_backtest.png")


if __name__ == "__main__":
    print("Loading cached data...")
    df = load_data()

    print("Fitting SARIMA model...")
    result, filled, clim, anomaly = fit_model(df)

    print_summary(result)
    plot_structural_break(df)
    best_result, best_order, best_seasonal = auto_select_model(filled)
    backtest_comparison(df)
    plot_diagnostics(result, filled, clim, anomaly)
