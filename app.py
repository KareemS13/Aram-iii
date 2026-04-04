#!/usr/bin/env python3
"""
app.py — Minimal Flask frontend for Lake Sevan Surface Area Monitor
Run: python app.py --project ee-kareemsaffarini9
"""

import argparse
import io
import json
import logging
import pathlib
import threading
import base64

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Flask

from flask import Flask, jsonify, render_template, request

import sevanSA

log = logging.getLogger(__name__)
app = Flask(__name__)

BASE_DIR   = pathlib.Path(__file__).parent
GEE_PROJECT = "ee-kareemsaffarini9"  # default GEE Cloud project ID

# Shared state for background update job
_update_status = {"running": False, "message": "Idle", "error": None}
_update_lock   = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_pipeline(force_refresh=False):
    """Run the full sevanSA pipeline and return (df, clim, clim_std, anomaly, forecast_df)."""
    ee, aoi = sevanSA.initialize_gee(project=GEE_PROJECT)

    meta_path = BASE_DIR / sevanSA.CONFIG["metadata_json"]
    total_pixels = None
    if meta_path.exists():
        try:
            total_pixels = json.loads(meta_path.read_text()).get("total_pixels")
        except Exception:
            pass
    if total_pixels is None:
        total_pixels = sevanSA.get_aoi_total_pixels(ee, aoi)

    jrc_df = sevanSA.extract_monthly_areas(
        ee, aoi, total_pixels,
        cache_path=BASE_DIR / sevanSA.CONFIG["cache_csv"],
        metadata_path=meta_path,
        force_refresh=force_refresh,
    )
    landsat_df = sevanSA.extract_landsat_monthly_areas(
        ee, aoi, total_pixels,
        start_date="2022-01-01",
        cache_path=BASE_DIR / sevanSA.CONFIG["landsat_cache_csv"],
        force_refresh=force_refresh,
    )

    raw_df     = sevanSA.combine_jrc_landsat(jrc_df, landsat_df)
    df         = sevanSA.build_complete_timeseries(raw_df)
    clim, clim_std, anomaly = sevanSA.compute_climatology(df)
    _, forecast_df = sevanSA.fit_sarima(
        df, clim,
        order=sevanSA.CONFIG["sarima_order"],
        seasonal_order=sevanSA.CONFIG["sarima_seasonal"],
        train_start_year=sevanSA.CONFIG["sarima_start_year"],
        n_forecast=sevanSA.CONFIG["n_forecast"],
    )
    return df, clim, clim_std, anomaly, forecast_df


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with _update_lock:
        return jsonify(_update_status.copy())


@app.route("/api/update", methods=["POST"])
def api_update():
    """Trigger an incremental data update + re-run forecast in background."""
    with _update_lock:
        if _update_status["running"]:
            return jsonify({"error": "Update already running"}), 409
        _update_status["running"] = True
        _update_status["message"] = "Starting update..."
        _update_status["error"]   = None

    def worker():
        try:
            with _update_lock:
                _update_status["message"] = "Fetching new data from GEE..."
            _run_pipeline(force_refresh=False)
            with _update_lock:
                _update_status["message"] = "Done"
        except Exception as exc:
            with _update_lock:
                _update_status["error"]   = str(exc)
                _update_status["message"] = "Failed"
        finally:
            with _update_lock:
                _update_status["running"] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"message": "Update started"})


@app.route("/api/data")
def api_data():
    """Return time series + forecast as JSON for the frontend chart."""
    try:
        cache_csv     = BASE_DIR / sevanSA.CONFIG["cache_csv"]
        landsat_csv   = BASE_DIR / sevanSA.CONFIG["landsat_cache_csv"]

        if not cache_csv.exists():
            return jsonify({"error": "No data cached yet. Run an update first."}), 404

        jrc_df     = pd.read_csv(cache_csv,   parse_dates=["date"])
        landsat_df = pd.read_csv(landsat_csv, parse_dates=["date"]) \
                     if landsat_csv.exists() else pd.DataFrame()

        raw_df     = sevanSA.combine_jrc_landsat(jrc_df, landsat_df)
        df         = sevanSA.build_complete_timeseries(raw_df)
        clim, clim_std, anomaly = sevanSA.compute_climatology(df)
        _, forecast_df = sevanSA.fit_sarima(
            df, clim,
            order=sevanSA.CONFIG["sarima_order"],
            seasonal_order=sevanSA.CONFIG["sarima_seasonal"],
            train_start_year=sevanSA.CONFIG["sarima_start_year"],
            n_forecast=sevanSA.CONFIG["n_forecast"],
        )

        # Time series (good months only)
        good = df[df["quality_flag"] == "good"][["date", "area_km2"]].copy()
        good["date"] = good["date"].dt.strftime("%Y-%m-%d")

        # Forecast table
        fcast = forecast_df.copy()
        fcast["date"] = pd.to_datetime(fcast["date"]).dt.strftime("%Y-%m")

        # Summary stats
        last_good = df[df["quality_flag"] == "good"].iloc[-1]

        return jsonify({
            "timeseries": good.to_dict(orient="records"),
            "forecast":   fcast.to_dict(orient="records"),
            "summary": {
                "period_start": str(df["date"].min().date()),
                "period_end":   str(df["date"].max().date()),
                "valid_months": int((df["quality_flag"] == "good").sum()),
                "total_months": len(df),
                "last_obs_date": last_good["date"].strftime("%Y-%m"),
                "last_obs_area": round(float(last_good["area_km2"]), 1),
                "clim_summer_mean": round(float(clim.loc[[6,7,8,9]].mean()), 1),
            },
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global GEE_PROJECT
    parser = argparse.ArgumentParser(description="Lake Sevan Flask dashboard")
    parser.add_argument("--project", type=str, default="ee-kareemsaffarini9",
                        help="GEE Cloud project ID")
    parser.add_argument("--port",    type=int, default=5000)
    parser.add_argument("--debug",   action="store_true")
    args = parser.parse_args()

    GEE_PROJECT = args.project
    port = int(os.environ.get("PORT", args.port))
    app.run(host="0.0.0.0", port=port, debug=args.debug)


if __name__ == "__main__":
    main()
