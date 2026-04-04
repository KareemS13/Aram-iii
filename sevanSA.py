#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sevanSA.py — Lake Sevan (Armenia) Surface-Area Time Series & Forecast
======================================================================
Data source : JRC Global Surface Water MonthlyHistory v1.4
              Pekel et al. (2016), Nature 540, 418-422
              © European Commission / Copernicus Programme
              Asset: ee://JRC/GSW1_4/MonthlyHistory  (1984-01 to 2021-12)

Access method: Google Earth Engine Python API (earthengine-api)
Resolution   : 30 m Landsat-based, geographic CRS (EPSG:4326)
AOI          : Lake Sevan, Armenia — lon 44.85-45.75E, lat 39.95-40.65N

JRC pixel encoding: 0 = no observation, 1 = not water, 2 = water

Usage:
    python sevanSA.py --project YOUR_GEE_PROJECT_ID
    python sevanSA.py --project YOUR_GEE_PROJECT_ID --no-cache
    python sevanSA.py --forecast-only          # reload CSV, re-run forecast & plot
"""

# ---------------------------------------------------------------------------
# Section 0 — Configuration
# ---------------------------------------------------------------------------

CONFIG = {
    "aoi_bounds":         [44.85, 39.95, 45.75, 40.65],  # [lon_min, lat_min, lon_max, lat_max]
    "collection":         "JRC/GSW1_4/MonthlyHistory",
    "scale_m":            30,
    "min_coverage":       0.50,    # fraction of AOI observed; below -> NaN
    "sarima_order":       (1, 1, 1),
    "sarima_seasonal":    (0, 1, 1, 12),
    "sarima_start_year":  1998,    # skip 1988-1997 Caucasus data gap
    "clim_ref_start":     "1991-01",
    "clim_ref_end":       "2020-12",
    "cache_csv":          "sevan_monthly_area.csv",
    "landsat_cache_csv":  "sevan_landsat_2022.csv",
    "metadata_json":      "sevan_metadata.json",
    "output_png":         "sevan_surface_area.png",
    "expected_range_km2": (1200, 1450),  # post-2000 summer validation range
    "n_forecast":         12,            # months ahead to forecast
}

# ---------------------------------------------------------------------------
# Section 1 — Imports
# ---------------------------------------------------------------------------

import argparse
import json
import logging
import pathlib
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:
    raise ImportError(
        "statsmodels is required: pip install statsmodels"
    ) from exc

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section 2 — GEE Initialisation
# ---------------------------------------------------------------------------

def initialize_gee(project=None):
    """Authenticate and initialise GEE; return (ee module, AOI geometry)."""
    import ee  # deferred so --forecast-only skips the GEE import entirely
    import os, json, tempfile

    sa_json_str = os.environ.get("GEE_SERVICE_ACCOUNT_JSON")
    if sa_json_str:
        sa_info = json.loads(sa_json_str)
        sa_email = sa_info["client_email"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sa_info, f)
            key_file = f.name
        credentials = ee.ServiceAccountCredentials(sa_email, key_file)
        ee.Initialize(credentials, project=project)
        log.info("GEE initialised via service account (project=%s)", project or "legacy")
    else:
        try:
            ee.Initialize(project=project)
            log.info("GEE initialised (project=%s)", project or "legacy")
        except Exception:
            log.info("GEE token missing — launching browser authentication...")
            ee.Authenticate()
            ee.Initialize(project=project)

    bounds = CONFIG["aoi_bounds"]
    aoi = ee.Geometry.Rectangle(bounds)
    return ee, aoi


def get_aoi_total_pixels(ee, aoi):
    """Count total 30 m pixels inside the AOI (denominator for coverage fraction)."""
    result = (
        ee.Image.constant(1)
        .rename("ones")
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=CONFIG["scale_m"],
            maxPixels=int(1e9),
        )
        .getInfo()
    )
    return int(result["ones"])


# ---------------------------------------------------------------------------
# Section 3 — GEE Extraction
# ---------------------------------------------------------------------------

def _make_mapper(ee, aoi):
    """Return a closure that maps an ee.Image to an ee.Feature with area stats."""
    def mapper(img):
        water_area_m2 = (
            img.eq(2)
               .multiply(ee.Image.pixelArea())
               .rename("water_m2")
        )
        no_obs = img.eq(0).rename("no_obs")
        stats = (
            water_area_m2.addBands(no_obs)
                         .reduceRegion(
                             reducer=ee.Reducer.sum(),
                             geometry=aoi,
                             scale=CONFIG["scale_m"],
                             maxPixels=int(1e9),
                         )
        )
        return ee.Feature(None, {
            "year":      img.get("year"),
            "month":     img.get("month"),
            "area_km2":  ee.Number(stats.get("water_m2")).divide(1e6),
            "no_obs_px": stats.get("no_obs"),
        })
    return mapper


def _parse_gee_result(features, total_pixels):
    """Convert raw getInfo() feature list to a tidy DataFrame."""
    rows = []
    for f in features:
        p = f["properties"]
        no_obs = int(p.get("no_obs_px") or 0)
        area   = float(p.get("area_km2") or 0.0)
        rows.append({
            "year":          int(p["year"]),
            "month":         int(p["month"]),
            "area_km2":      area,
            "no_obs_px":     no_obs,
            "coverage_frac": 1.0 - no_obs / max(total_pixels, 1),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(
        {"year": df["year"], "month": df["month"], "day": 1}
    )
    return df.sort_values("date").reset_index(drop=True)


def extract_monthly_areas(
    ee,
    aoi,
    total_pixels,
    cache_path=None,
    metadata_path=None,
    force_refresh=False,
):
    """
    Download or load from cache the per-month water area (km2) for Lake Sevan.
    Full extraction takes ~60 s. Results are cached to CSV.
    Cache is auto-invalidated when the GEE collection grows.
    """
    col = ee.ImageCollection(CONFIG["collection"])
    live_size = col.size().getInfo()

    if cache_path and cache_path.exists() and not force_refresh:
        stored_size = None
        if metadata_path and metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text())
                stored_size = meta.get("collection_size")
            except Exception:
                pass
        if stored_size == live_size:
            log.info("Cache current (%d images). Loading %s", live_size, cache_path)
            return pd.read_csv(cache_path, parse_dates=["date"])
        log.info("Collection grew %s -> %d. Re-fetching...", stored_size, live_size)

    log.info("Querying GEE (collection=%d images). This takes ~60 s...", live_size)
    mapper = _make_mapper(ee, aoi)
    fc     = col.map(mapper)
    data   = fc.getInfo()

    df = _parse_gee_result(data["features"], total_pixels)

    if cache_path:
        df.to_csv(cache_path, index=False)
        log.info("Saved %d rows to %s", len(df), cache_path)

    if metadata_path:
        meta = {"total_pixels": total_pixels, "collection_size": live_size}
        metadata_path.write_text(json.dumps(meta, indent=2))
        log.info("Metadata saved to %s", metadata_path)

    return df


# ---------------------------------------------------------------------------
# Section 3b — Landsat 8/9 Extension (2022-present)
# ---------------------------------------------------------------------------

def extract_landsat_monthly_areas(ee, aoi, total_pixels, start_date="2022-01-01",
                                   cache_path=None, force_refresh=False):
    """
    Compute monthly water area (km2) for Lake Sevan using Landsat 8/9
    Collection 2 Level-2 SR imagery from start_date to last complete month.

    Uses NDWI = (Green - NIR) / (Green + NIR) > 0 as the water mask.
    Cloud/shadow pixels are masked via QA_PIXEL before compositing.
    Each month is processed as a median composite, one GEE call per month.
    """
    if cache_path and cache_path.exists() and not force_refresh:
        log.info("Loading Landsat cache from %s", cache_path)
        return pd.read_csv(cache_path, parse_dates=["date"])

    # Build lake mask from JRC permanent water occurrence (>50% = permanent lake body).
    # This constrains detection to within the known lake boundary, avoiding
    # snow-covered surrounding mountains being misclassified as water.
    occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    lake_mask  = occurrence.gte(50)

    def mask_clouds(img):
        qa     = img.select("QA_PIXEL")
        cloud  = qa.bitwiseAnd(1 << 3).neq(0)
        shadow = qa.bitwiseAnd(1 << 4).neq(0)
        return img.updateMask(cloud.Or(shadow).Not())

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(aoi).map(mask_clouds)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(aoi).map(mask_clouds)
    landsat = l8.merge(l9)

    today = datetime.now()
    start = datetime.strptime(start_date, "%Y-%m-%d")

    # If cache exists, only fetch months after the last cached month
    if cache_path and cache_path.exists() and not force_refresh:
        existing = pd.read_csv(cache_path, parse_dates=["date"])
        if not existing.empty:
            last = pd.to_datetime(existing["date"]).max()
            start = (last + pd.DateOffset(months=1)).to_pydatetime()
            log.info("Incremental fetch: starting from %s", start.strftime("%Y-%m"))

    # Build list of complete months up to last complete month
    months = []
    y, m = start.year, start.month
    while (y < today.year) or (y == today.year and m < today.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1

    if not months:
        log.info("Landsat cache is already up to date.")
        return pd.read_csv(cache_path, parse_dates=["date"]) if cache_path and cache_path.exists() else pd.DataFrame()

    log.info("Fetching Landsat monthly composites: %d months (%s to %d-%02d)...",
             len(months), start_date, months[-1][0], months[-1][1])

    rows = []
    for year, month in months:
        s = f"{year}-{month:02d}-01"
        e = f"{year+1}-01-01" if month == 12 else f"{year}-{month+1:02d}-01"

        monthly = landsat.filterDate(s, e)

        # MNDWI = (Green - SWIR1) / (Green + SWIR1) — better than NDWI for water vs snow/ice
        mndwi_col   = monthly.map(lambda img: img.normalizedDifference(["SR_B3", "SR_B6"]).rename("MNDWI"))
        mndwi_count = mndwi_col.count().rename("count")
        mndwi_med   = mndwi_col.median()

        # Restrict detection to known lake pixels (JRC occurrence >= 50%)
        water    = mndwi_med.gt(0.0).updateMask(mndwi_count.gt(0)).updateMask(lake_mask).rename("water")
        no_obs   = mndwi_count.eq(0).rename("no_obs")

        stats = (
            water.multiply(ee.Image.pixelArea()).rename("water_m2")
                 .addBands(no_obs)
                 .reduceRegion(
                     reducer=ee.Reducer.sum(),
                     geometry=aoi,
                     scale=CONFIG["scale_m"],
                     maxPixels=int(1e9),
                 )
                 .getInfo()
        )

        area_km2      = float(stats.get("water_m2") or 0.0) / 1e6
        no_obs_px     = int(stats.get("no_obs") or 0)
        coverage_frac = 1.0 - no_obs_px / max(total_pixels, 1)

        # Ice contamination check: if measured area < 900 km2, the lake is
        # likely partially frozen — MNDWI cannot detect ice as water.
        # Force coverage_frac to 0 so build_complete_timeseries masks it as NaN.
        ice_flag = area_km2 < 900.0
        if ice_flag:
            coverage_frac = 0.0
            log.info("  %d-%02d: %.1f km2 — ice-contaminated, flagged NaN",
                     year, month, area_km2)
        else:
            log.info("  %d-%02d: %.1f km2 (coverage=%.0f%%)",
                     year, month, area_km2, coverage_frac * 100)

        rows.append({
            "year": year, "month": month,
            "area_km2": area_km2,
            "no_obs_px": no_obs_px,
            "coverage_frac": coverage_frac,
            "source": "landsat",
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["date"] = pd.to_datetime({"year": df["year"], "month": df["month"], "day": 1})
        df = df.sort_values("date").reset_index(drop=True)

    if cache_path and len(df) > 0:
        if cache_path.exists():
            existing = pd.read_csv(cache_path, parse_dates=["date"])
            df = pd.concat([existing, df], ignore_index=True)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
        df.to_csv(cache_path, index=False)
        log.info("Landsat data saved to %s", cache_path)

    return df


def combine_jrc_landsat(jrc_df, landsat_df):
    """
    Merge JRC (1984-2021) and Landsat (2022-present) DataFrames.
    Landsat rows are appended; JRC rows take precedence for any overlap.
    """
    if landsat_df.empty:
        return jrc_df

    jrc_df     = jrc_df.copy()
    landsat_df = landsat_df.copy()

    if "source" not in jrc_df.columns:
        jrc_df["source"] = "jrc"

    # Keep only Landsat months that are not already in JRC
    existing_dates = set(pd.to_datetime(jrc_df["date"]).dt.to_period("M"))
    new_rows = landsat_df[
        ~pd.to_datetime(landsat_df["date"]).dt.to_period("M").isin(existing_dates)
    ]

    combined = pd.concat([jrc_df, new_rows], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.sort_values("date").reset_index(drop=True)

    log.info("Combined series: %d JRC + %d Landsat = %d total rows",
             len(jrc_df), len(new_rows), len(combined))
    return combined


# ---------------------------------------------------------------------------
# Section 4 — Data Cleaning & Validation
# ---------------------------------------------------------------------------

def build_complete_timeseries(raw_df, min_coverage=0.50):
    """
    Align raw extraction to a complete monthly DatetimeIndex.
    Months with coverage_frac < min_coverage are masked as NaN.
    Adds quality_flag: 'good', 'low_coverage', or 'missing'.
    """
    full_idx = pd.date_range(
        start=raw_df["date"].min().replace(day=1),
        end=raw_df["date"].max().replace(day=1),
        freq="MS",
    )
    df = raw_df.set_index("date").reindex(full_idx)
    df.index.name = "date"

    df["quality_flag"] = "good"
    df.loc[df["coverage_frac"] < min_coverage, "quality_flag"] = "low_coverage"
    df.loc[df["area_km2"].isna(), "quality_flag"] = "missing"

    df.loc[df["quality_flag"] != "good", "area_km2"] = np.nan

    lo, hi = CONFIG["expected_range_km2"]
    summer_good = df[
        (df.index.month.isin([6, 7, 8, 9])) &
        (df.index.year >= 2000) &
        (df["quality_flag"] == "good")
    ]["area_km2"]
    out_of_range = summer_good[(summer_good < lo) | (summer_good > hi)]
    if not out_of_range.empty:
        log.warning(
            "%d post-2000 summer months outside expected [%d, %d] km2:\n%s",
            len(out_of_range), lo, hi, out_of_range,
        )

    n_good = (df["quality_flag"] == "good").sum()
    log.info("Quality: %d good / %d total months", n_good, len(df))
    return df.reset_index()


# ---------------------------------------------------------------------------
# Section 5 — Climatology & Anomaly
# ---------------------------------------------------------------------------

def compute_climatology(df, reference_start="1991-01", reference_end="2020-12"):
    """
    Compute WMO 1991-2020 monthly climatological mean and std,
    and the anomaly series (observed - clim[month]).

    Returns (clim, clim_std, anomaly) — all pd.Series.
    """
    ts = df.set_index("date")["area_km2"]

    ref_mask = (ts.index >= reference_start) & (ts.index <= reference_end)
    ref_ts   = ts[ref_mask]

    clim     = ref_ts.groupby(ref_ts.index.month).mean()
    clim_std = ref_ts.groupby(ref_ts.index.month).std()

    # Fill months with no reference data (e.g. winter ice/cloud) by interpolation
    clim     = clim.reindex(range(1, 13)).interpolate(method="linear").bfill().ffill()
    clim_std = clim_std.reindex(range(1, 13)).interpolate(method="linear").bfill().ffill()
    clim.index.name     = "month"
    clim_std.index.name = "month"

    anomaly = ts - ts.index.map(lambda d: clim.get(d.month, np.nan))
    anomaly.name = "anomaly_km2"

    log.info("Climatology Jun-Sep mean = %.1f km2", clim.loc[[6, 7, 8, 9]].mean())
    return clim, clim_std, anomaly


# ---------------------------------------------------------------------------
# Section 6 — SARIMA Forecast
# ---------------------------------------------------------------------------

def fit_sarima(
    df,
    clim,
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 12),
    train_start_year=1998,
    n_forecast=1,
):
    """
    Fit SARIMAX on the monthly area series.
    Missing months are imputed with the climatological mean for that month
    before fitting — the 53% NaN rate causes Kalman-filter variance explosion
    if passed raw.

    Returns (fitted_result, forecast_df).
    """
    ts = df.set_index("date")["area_km2"].copy()
    ts.index = pd.DatetimeIndex(ts.index)

    train = ts[ts.index.year >= train_start_year].asfreq("MS")
    n_valid = train.notna().sum()
    log.info(
        "SARIMA training: %d valid / %d total months from %d",
        n_valid, len(train), train_start_year,
    )

    # Impute NaN months with climatological mean for that calendar month,
    # then smooth residual gaps with linear interpolation
    clim_filled = train.copy()
    clim_filled[clim_filled.isna()] = clim_filled.index[clim_filled.isna()].map(
        lambda d: clim.get(d.month, np.nan)
    )
    clim_filled = clim_filled.interpolate(method="linear").bfill().ffill()
    train = clim_filled

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        freq="MS",
    )

    result = None
    for method in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(disp=False, method=method, maxiter=200)
            log.info(
                "SARIMA fit OK (method=%s) AIC=%.1f BIC=%.1f",
                method, result.aic, result.bic,
            )
            break
        except Exception as exc:
            log.warning("SARIMA fit failed (method=%s): %s", method, exc)

    if result is None:
        raise RuntimeError("SARIMA fitting failed with both lbfgs and powell.")

    fcast    = result.get_forecast(steps=n_forecast)
    fcast_mu = fcast.predicted_mean
    fcast_ci = fcast.conf_int(alpha=0.05)

    rows = []
    for i in range(n_forecast):
        fdate  = pd.Timestamp(fcast_mu.index[i])
        area   = float(fcast_mu.iloc[i])
        lo     = float(fcast_ci.iloc[i, 0])
        hi     = float(fcast_ci.iloc[i, 1])
        anom   = area - float(clim.get(fdate.month, np.nan))
        rows.append({
            "date":        fdate,
            "area_km2":    area,
            "ci_lower_95": lo,
            "ci_upper_95": hi,
            "anomaly_km2": anom,
        })

    forecast_df = pd.DataFrame(rows)

    print(f"\n=== SARIMA {n_forecast}-Month Forecast ===")
    print(f"{'Month':<10} {'Area (km2)':>10} {'Anomaly':>10}  {'95% CI'}")
    print("-" * 52)
    for _, r in forecast_df.iterrows():
        print(f"{r['date'].strftime('%Y-%m'):<10} {r['area_km2']:>10.1f} "
              f"{r['anomaly_km2']:>+10.1f}  [{r['ci_lower_95']:.1f}, {r['ci_upper_95']:.1f}]")

    return result, forecast_df


# ---------------------------------------------------------------------------
# Section 7 — Visualisation
# ---------------------------------------------------------------------------

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_timeseries(df, clim, clim_std, anomaly, forecast_df, output_path, show=True):
    """
    Publication-quality 3-panel figure:
      Panel 1 (50%): Full area time series + forecast
      Panel 2 (25%): Monthly climatology bar chart (1991-2020)
      Panel 3 (25%): Anomaly time series + forecast anomaly
    """
    fig = plt.figure(figsize=(14, 10))
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1], hspace=0.50)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    ts_date = pd.to_datetime(df["date"])
    area    = df["area_km2"].values
    quality = df["quality_flag"].values
    good    = quality == "good"

    # ---- Panel 1: Full time series ----------------------------------------
    ax1.set_title("Lake Sevan Surface Area — Monthly Time Series",
                  fontsize=13, fontweight="bold")

    ax1.axvspan(
        pd.Timestamp("1988-01-01"), pd.Timestamp("1997-12-31"),
        color="#d0d0d0", alpha=0.5, label="Limited data coverage (1988-1997)",
    )

    ax1.plot(ts_date[good], area[good], color="#2166ac", linewidth=0.9, alpha=0.7, zorder=2)
    ax1.scatter(ts_date[good], area[good], s=6, color="#2166ac", zorder=3, label="Good quality")

    lc = quality == "low_coverage"
    if lc.any():
        ax1.scatter(ts_date[lc], df["area_km2"].values[lc],
                    s=6, color="orange", zorder=3, label="Low coverage (excluded)")

    # 12-month forecast ribbon
    f_dates = pd.to_datetime(forecast_df["date"])
    ax1.fill_between(f_dates, forecast_df["ci_lower_95"], forecast_df["ci_upper_95"],
                     color="crimson", alpha=0.15, label="Forecast 95% CI")
    ax1.plot(f_dates, forecast_df["area_km2"], color="crimson", linewidth=1.5,
             marker="o", markersize=4, zorder=5,
             label=f"SARIMA forecast ({f_dates.iloc[0].strftime('%Y-%m')} – {f_dates.iloc[-1].strftime('%Y-%m')})")

    ax1.set_ylabel("Surface Area (km2)", fontsize=10)
    ax1.set_ylim(1050, 1750)
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # ---- Panel 2: Monthly climatology -------------------------------------
    ax2.set_title("1991-2020 Monthly Climatology", fontsize=11, fontweight="bold")

    months     = np.arange(1, 13)
    clim_vals  = np.array([clim.get(m, np.nan) for m in months])
    clim_errs  = np.array([clim_std.get(m, 0.0) for m in months])

    valid_vals = clim_vals[~np.isnan(clim_vals)]
    norm       = Normalize(vmin=valid_vals.min(), vmax=valid_vals.max())
    cmap       = plt.cm.Blues_r
    colors     = [cmap(norm(v)) if not np.isnan(v) else "#cccccc" for v in clim_vals]

    ax2.barh(months, clim_vals, color=colors, edgecolor="white", height=0.7)
    ax2.errorbar(
        clim_vals, months,
        xerr=clim_errs, fmt="none",
        color="black", capsize=3, linewidth=0.8,
    )
    ax2.set_yticks(months)
    ax2.set_yticklabels(MONTH_NAMES, fontsize=8)
    ax2.set_xlabel("Surface Area (km2)", fontsize=9)
    ax2.invert_yaxis()
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    # ---- Panel 3: Anomaly -------------------------------------------------
    ax3.set_title(
        "Surface Area Anomaly (relative to 1991-2020 climatology)",
        fontsize=11, fontweight="bold",
    )

    anom_dates = anomaly.index
    anom_vals  = anomaly.values
    pos        = anom_vals >= 0

    ax3.bar(anom_dates[pos],  anom_vals[pos],  width=25, color="#d6604d", alpha=0.8, label="Above normal")
    ax3.bar(anom_dates[~pos], anom_vals[~pos], width=25, color="#4393c3", alpha=0.8, label="Below normal")
    ax3.axhline(0, color="black", linewidth=0.8)

    # 12-month forecast anomalies
    f_dates = pd.to_datetime(forecast_df["date"])
    f_anom  = forecast_df["anomaly_km2"].values
    ax3.plot(f_dates, f_anom, color="crimson", linewidth=1.5,
             marker="D", markersize=5, zorder=5, label="Forecast anomaly")

    ax3.set_ylabel("Anomaly (km2)", fontsize=10)
    ax3.xaxis.set_major_locator(mdates.YearLocator(5))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax3.tick_params(axis="x", rotation=45)
    ax3.legend(fontsize=8, loc="lower right")
    ax3.grid(axis="y", linestyle="--", alpha=0.4)

    # ---- Attribution footer -----------------------------------------------
    fig.text(
        0.5, 0.002,
        "Data: JRC Global Surface Water v1.4 (Pekel et al. 2016, Nature 540, 418-422) "
        "European Commission / Copernicus Programme  |  Accessed via Google Earth Engine",
        ha="center", va="bottom", fontsize=7, color="#555555", style="italic",
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info("Figure saved to %s", output_path)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Section 8 — Main Entry Point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Lake Sevan surface-area time series & SARIMA forecast"
    )
    parser.add_argument("--project",       type=str,          default="ee-kareemsaffarini9",
                        help="GEE Cloud project ID")
    parser.add_argument("--no-cache",      action="store_true",
                        help="Force re-download from GEE even if CSV exists")
    parser.add_argument("--forecast-only", action="store_true",
                        help="Skip GEE; load cached CSV and re-run forecast + plot")
    parser.add_argument("--min-coverage",  type=float,        default=CONFIG["min_coverage"],
                        help="Minimum AOI coverage fraction (default 0.5)")
    parser.add_argument("--output-dir",    type=pathlib.Path, default=pathlib.Path("."),
                        help="Directory for output CSV and PNG")
    parser.add_argument("--no-plot-show",  action="store_true",
                        help="Save figure without opening interactive window")
    args = parser.parse_args(argv)

    out_dir           = args.output_dir
    cache_csv         = out_dir / CONFIG["cache_csv"]
    landsat_cache_csv = out_dir / CONFIG["landsat_cache_csv"]
    metadata_json     = out_dir / CONFIG["metadata_json"]
    out_png           = out_dir / CONFIG["output_png"]

    # Step 1: Data acquisition
    if args.forecast_only:
        if not cache_csv.exists():
            sys.exit(f"Error: {cache_csv} not found. Run without --forecast-only first.")
        log.info("--forecast-only: loading cached CSVs")
        jrc_df     = pd.read_csv(cache_csv, parse_dates=["date"])
        landsat_df = pd.read_csv(landsat_cache_csv, parse_dates=["date"]) \
                     if landsat_cache_csv.exists() else pd.DataFrame()
    else:
        ee, aoi = initialize_gee(project=args.project)

        total_pixels = None
        if metadata_json.exists():
            try:
                meta         = json.loads(metadata_json.read_text())
                total_pixels = meta.get("total_pixels")
            except Exception:
                pass
        if total_pixels is None:
            log.info("Computing AOI pixel count...")
            total_pixels = get_aoi_total_pixels(ee, aoi)
            log.info("AOI total pixels: %d", total_pixels)

        # JRC 1984-2021
        jrc_df = extract_monthly_areas(
            ee, aoi, total_pixels,
            cache_path=cache_csv,
            metadata_path=metadata_json,
            force_refresh=args.no_cache,
        )

        # Landsat 8/9 2022-present
        landsat_df = extract_landsat_monthly_areas(
            ee, aoi, total_pixels,
            start_date="2022-01-01",
            cache_path=landsat_cache_csv,
            force_refresh=args.no_cache,
        )

    # Combine JRC + Landsat
    raw_df = combine_jrc_landsat(jrc_df, landsat_df)

    # Step 2: Build clean time series
    df = build_complete_timeseries(raw_df, min_coverage=args.min_coverage)

    # Step 3: Climatology + anomaly
    clim, clim_std, anomaly = compute_climatology(
        df,
        reference_start=CONFIG["clim_ref_start"],
        reference_end=CONFIG["clim_ref_end"],
    )

    # Step 4: SARIMA forecast (12 months ahead)
    _model_result, forecast_df = fit_sarima(
        df, clim,
        order=CONFIG["sarima_order"],
        seasonal_order=CONFIG["sarima_seasonal"],
        train_start_year=CONFIG["sarima_start_year"],
        n_forecast=CONFIG["n_forecast"],
    )

    # Step 5: Summary
    n_good = (df["quality_flag"] == "good").sum()
    print("\n=== Lake Sevan Dataset Summary ===")
    print(f"  Period:       {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Valid months: {n_good} / {len(df)}")
    print(f"  Clim Jun-Sep: {clim.loc[[6,7,8,9]].mean():.1f} km2")

    # Step 6: Plot
    plot_timeseries(
        df, clim, clim_std, anomaly, forecast_df,
        output_path=out_png,
        show=not args.no_plot_show,
    )


if __name__ == "__main__":
    main()
