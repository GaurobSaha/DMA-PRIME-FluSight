"""Shared preparation, chronological calibration, and checks for the Flu notebooks."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

QUANTILES = np.array([.01, .025, .05, .10, .15, .20, .25, .30, .35, .40,
                     .45, .50, .55, .60, .65, .70, .75, .80, .85, .90,
                     .95, .975, .99])
COLUMNS = ["reference_date", "target", "horizon", "target_end_date",
           "location", "output_type", "output_type_id", "value"]
TARGET = "wk inc flu hosp"
HUB_URL = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main"


def load_settings(root):
    settings = json.loads((Path(root) / "flusight_config.json").read_text(encoding="utf-8"))
    return settings


def now_eastern():
    return datetime.now(ZoneInfo("America/New_York"))


def select_reference_date(mode, last_observed, requested=None):
    if mode not in ["dataset", "live", "retrospective"]:
        raise ValueError("Mode must be dataset, live, or retrospective.")
    dataset_reference = pd.Timestamp(last_observed).normalize() + pd.Timedelta(weeks=1)
    if pd.isna(dataset_reference) or dataset_reference.dayofweek != 5:
        raise ValueError("The last observed week must be a valid Saturday.")
    if requested:
        reference = pd.Timestamp(requested).normalize()
    elif mode in ["dataset", "retrospective"]:
        reference = dataset_reference
    else:
        today = pd.Timestamp(now_eastern().date())
        reference = today + pd.Timedelta(days=(5 - today.weekday()) % 7)
    if reference.dayofweek != 5:
        raise ValueError("ReferenceDate must be a Saturday in YYYY-MM-DD format.")
    if mode == "dataset" and reference != dataset_reference:
        raise ValueError(f"Dataset reference date must be one week after the last common observation: {dataset_reference.date()}.")
    if mode == "live":
        today = pd.Timestamp(now_eastern().date())
        expected = today + pd.Timedelta(days=(5 - today.weekday()) % 7)
        if reference != expected:
            raise ValueError(f"Live reference date must be this week's Saturday: {expected.date()}.")
        deadline = datetime.combine((reference - pd.Timedelta(days=3)).date(),
                                    datetime.min.time(), ZoneInfo("America/New_York"))
        deadline = deadline.replace(hour=23)
        if now_eastern() > deadline:
            raise ValueError(f"The Wednesday 11 PM Eastern deadline has passed for {reference.date()}.")
    return reference


def prepare_source_frames(frames, target_column, mode="live"):
    """Parse dates before merging and enforce one South Carolina row per week."""
    cleaned = {}
    for name, original in frames.items():
        frame = original.copy()
        date_col = "Week.Ending.Date" if name == "CDC" else "Week"
        location_col = "Geographic.aggregation" if name == "CDC" else "State"
        required = {date_col, location_col}
        if name == "CDC":
            required.add(target_column)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv is missing columns: {sorted(missing)}")
        locations = frame[location_col].astype(str).str.strip().str.upper()
        frame = frame.loc[locations.isin(["SC", "SOUTH CAROLINA", "45"])].copy()
        if frame.empty:
            raise ValueError(f"{name}.csv has no South Carolina rows.")
        frame[date_col] = pd.to_datetime(frame[date_col], format="mixed", errors="raise").dt.normalize()
        if frame[date_col].duplicated().any():
            raise ValueError(f"{name}.csv has duplicate South Carolina weeks.")
        if (frame[date_col].dt.dayofweek != 5).any():
            raise ValueError(f"{name}.csv observation weeks must end on Saturdays.")
        if mode in ["dataset", "live"] and frame[date_col].max().date() > now_eastern().date():
            raise ValueError(f"{name}.csv contains observation weeks in the future.")
        cleaned[name] = frame.sort_values(date_col).reset_index(drop=True)
    return cleaned


def check_input_dates(frames, merged, reference, mode):
    last = pd.Timestamp(merged["Week"].max())
    dates = merged["Week"].sort_values()
    if dates.duplicated().any() or (dates.diff().dropna() != pd.Timedelta(weeks=1)).any():
        raise ValueError("Merged data have duplicate or missing weeks. Repair the input dates before forecasting.")
    latest = {name: str(frame["Week.Ending.Date" if name == "CDC" else "Week"].max().date())
              for name, frame in frames.items()}
    expected = reference - pd.Timedelta(weeks=1)
    if last != expected:
        raise ValueError(
            f"Inputs are not aligned to reference {reference.date()}. Latest merged week is {last.date()}, "
            f"but T+1 to T+4 require {expected.date()}. Source endings: {latest}. "
            "Use Dataset mode to forecast from the latest common data week; Live mode needs updated data."
        )
    return latest


def latest_known_ratio(history, origin):
    if history.empty:
        return 1.0
    known = history.loc[pd.to_datetime(history["Target_Week"]) <= pd.Timestamp(origin)]
    known = known.loc[np.isfinite(known["Actual"]) & (known["Predicted"] > 0)]
    if known.empty:
        return 1.0
    row = known.sort_values("Target_Week").iloc[-1]
    return float(row["Actual"] / row["Predicted"])


def causal_adjustment(table, alpha=.99):
    adjusted = table.copy().sort_values("Prediction_Week").reset_index(drop=True)
    adjusted["current_ratio"] = np.divide(
        adjusted["Actual"], adjusted["Predicted"],
        out=np.ones(len(adjusted)), where=adjusted["Predicted"].to_numpy() > 0)
    values = []
    for idx, row in adjusted.iterrows():
        ratio = latest_known_ratio(adjusted.iloc[:idx], row["Prediction_Week"])
        values.append(max(float(row["Predicted"]) * ratio * alpha, 0.0))
    adjusted["adjusted_predicted"] = values
    return adjusted


def residual_quantiles(point, history, origin, min_rows=24):
    """Empirical log1p forecast-error quantiles using outcomes known by the origin."""
    known = history.loc[pd.to_datetime(history["Target_Week"]) <= pd.Timestamp(origin)].copy()
    errors = np.log1p(known["Actual"].to_numpy(float)) - np.log1p(
        known["adjusted_predicted"].to_numpy(float))
    errors = errors[np.isfinite(errors)]
    if len(errors) < min_rows:
        raise ValueError(f"Only {len(errors)} mature calibration errors; at least {min_rows} are required.")
    values = np.expm1(np.log1p(max(float(point), 0)) + np.quantile(errors, QUANTILES))
    if not np.isfinite(values).all():
        raise ValueError("Calibration produced nonfinite quantiles.")
    return np.maximum.accumulate(np.rint(np.maximum(values, 0))).astype(np.int64), len(errors)


def weighted_interval_score(actual, values):
    actual = float(actual)
    values = np.asarray(values, dtype=float)
    score = .5 * abs(actual - values[11])
    for idx in range(11):
        alpha = 2 * QUANTILES[idx]
        lower, upper = values[idx], values[-idx - 1]
        interval = upper - lower
        interval += 2 / alpha * max(lower - actual, 0)
        interval += 2 / alpha * max(actual - upper, 0)
        score += alpha / 2 * interval
    return float(score / 11.5)


def temporal_evaluation(modeling_df, feature_cols, horizon, factory, tuner,
                        train_start, train_end, calibration_start, test_start, test_end,
                        origin, target_column, alpha=.99, min_rows=24):
    """Freeze selection/tuning before calibration; refit on mature labels at each origin."""
    frame = modeling_df.copy().sort_values("Week").reset_index(drop=True)
    frame["Target_Week"] = frame["Week"] + pd.Timedelta(weeks=horizon)
    train = frame.loc[frame["Target_Week"].between(train_start, train_end)].copy()
    if len(train) < 80:
        raise ValueError(f"T+{horizon}: at least 80 initial training rows are required.")
    params, cv_score = tuner(train[feature_cols], train[target_column], horizon)
    candidates = frame.loc[frame["Target_Week"].between(calibration_start, origin)]
    records, test_quantiles = [], []
    for number, (_, row) in enumerate(candidates.iterrows(), start=1):
        prediction_origin = pd.Timestamp(row["Week"])
        known_train = frame.loc[frame["Target_Week"] <= prediction_origin]
        if pd.Timestamp(train_end) > prediction_origin:
            raise ValueError("Feature selection or tuning extends beyond a calibration origin.")
        model = factory(params)
        model.fit(known_train[feature_cols], known_train[target_column])
        raw = max(float(model.predict(row[feature_cols].to_frame().T.astype(float))[0]), 0.0)
        history = pd.DataFrame(records, columns=[
            "Prediction_Week", "Target_Week", "Actual", "Predicted",
            "current_ratio", "adjusted_predicted", "train_label_max"])
        adjusted = max(raw * latest_known_ratio(history, prediction_origin) * alpha, 0.0)
        record = {
            "Prediction_Week": prediction_origin, "Target_Week": row["Target_Week"],
            "Actual": float(row[target_column]), "Predicted": raw,
            "current_ratio": float(row[target_column]) / raw if raw > 0 else 1.0,
            "adjusted_predicted": adjusted,
            "train_label_max": known_train["Target_Week"].max(),
        }
        if test_start <= row["Target_Week"] <= test_end:
            quantiles, n_errors = residual_quantiles(adjusted, history, prediction_origin, min_rows)
            test_quantiles.append({
                "Target_Week": row["Target_Week"], "WIS": weighted_interval_score(record["Actual"], quantiles),
                "coverage_50": bool(quantiles[6] <= record["Actual"] <= quantiles[16]),
                "coverage_95": bool(quantiles[1] <= record["Actual"] <= quantiles[-2]),
                "calibration_rows": n_errors,
            })
        records.append(record)
        if number == 1 or number % 10 == 0 or number == len(candidates):
            print(f"T+{horizon}: chronological refits {number}/{len(candidates)}", flush=True)
    history = pd.DataFrame(records)
    test_table = history.loc[history["Target_Week"].between(test_start, test_end)].reset_index(drop=True)
    if test_table.empty:
        raise ValueError(f"T+{horizon}: no held-out test forecasts.")
    cols = ["Prediction_Week", "Target_Week", "Actual", "Predicted"]
    raw_mae = float(np.mean(abs(test_table["Actual"] - test_table["Predicted"])))
    adj_mae = float(np.mean(abs(test_table["Actual"] - test_table["adjusted_predicted"])))
    def pa(prediction):
        mask = test_table["Actual"] >= 15
        actual = test_table.loc[mask, "Actual"].to_numpy(float)
        pred = test_table.loc[mask, prediction].to_numpy(float)
        score = np.divide(np.minimum(actual, pred), np.maximum(actual, pred),
                          out=np.zeros(len(actual)), where=np.maximum(actual, pred) != 0)
        return {"score": float(score.mean()) if len(score) else np.nan,
                "used": int(mask.sum()), "ignored": int((~mask).sum())}
    return {
        "feature_cols": feature_cols, "best_params": params, "best_cv_pa": cv_score,
        "train_df": train.drop(columns="Target_Week"),
        "gap_df": frame.loc[(frame["Target_Week"] > train_end)
                           & (frame["Target_Week"] < calibration_start)].drop(columns="Target_Week"),
        "test_df": frame.loc[frame["Target_Week"].between(test_start, test_end)].drop(columns="Target_Week"),
        "comparison_test": test_table[cols], "adjusted_comparison_test": test_table,
        "test_mae": raw_mae, "adjusted_test_mae": adj_mae,
        "test_pa": pa("Predicted"), "adjusted_test_pa": pa("adjusted_predicted"),
        "calibration_history": history, "quantile_metrics": pd.DataFrame(test_quantiles),
    }


def validate_forecast(frame, reference, model_id, cache_dir=None):
    if list(frame.columns) != COLUMNS or frame.empty:
        raise ValueError("Forecast must have exactly the eight FluSight columns and at least one row.")
    if not re.fullmatch(r"[A-Za-z0-9_+]{1,16}-[A-Za-z0-9_+]{1,16}", model_id):
        raise ValueError("Use a team-model identifier, with 1-16 letters/digits/underscores/plus signs per component.")
    expected_reference = str(pd.Timestamp(reference).date())
    if set(frame["reference_date"]) != {expected_reference}:
        raise ValueError("Forecast dates do not match the filename reference date.")
    if set(frame["target"]) != {TARGET} or set(frame["output_type"]) != {"quantile"}:
        raise ValueError("Only wk inc flu hosp quantiles are supported by this exporter.")
    if set(frame["location"].astype(str)) != {"45"}:
        raise ValueError("These models forecast South Carolina (location 45).")
    for (_, horizon), rows in frame.groupby(["location", "horizon"]):
        if horizon not in [-1, 0, 1, 2, 3]:
            raise ValueError("Hub horizons must be -1 through 3.")
        expected_target = str((pd.Timestamp(reference) + pd.Timedelta(weeks=int(horizon))).date())
        if set(rows["target_end_date"]) != {expected_target}:
            raise ValueError("Target dates must equal reference date plus 7 times horizon.")
        rows = rows.sort_values("output_type_id")
        if len(rows) != 23 or not np.allclose(rows["output_type_id"], QUANTILES):
            raise ValueError("Each location/horizon requires the 23 distinct FluSight quantiles.")
        values = rows["value"].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.rint(values)).any():
            raise ValueError("Hospitalization values must be finite nonnegative integers.")
        if (np.diff(values) < 0).any():
            raise ValueError("Quantiles must not cross.")
    if cache_dir:
        tasks = json.loads((Path(cache_dir) / "tasks.json").read_text(encoding="utf-8"))
        for round_config in tasks["rounds"]:
            for task in round_config["model_tasks"]:
                targets = task["task_ids"]["target"].get("optional") or []
                if TARGET in targets:
                    accepted = task["task_ids"]["reference_date"].get("optional") or []
                    if expected_reference not in accepted:
                        raise ValueError(f"Reference date {expected_reference} is not accepted by the cached hub configuration.")
                    return
        raise ValueError("Hospitalization target not found in cached tasks.json.")


def refresh_hub_config(cache_dir):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in ["tasks.json", "model-metadata-schema.json"]:
        with urllib.request.urlopen(f"{HUB_URL}/hub-config/{name}", timeout=45) as response:
            data = response.read()
        json.loads(data)
        (cache_dir / name).write_bytes(data)
    (cache_dir / "checked_at.json").write_text(
        json.dumps({"checked_at": now_eastern().isoformat(), "repository": HUB_URL}, indent=2), encoding="utf-8")


def build_metadata(settings, model_key, cache_dir):
    import jsonschema
    common = settings["metadata"]
    model = settings["models"][model_key]
    metadata = {key: value for key, value in common.items() if value is not None}
    metadata.update({
        "model_name": model["model_name"], "model_abbr": model["model_abbr"],
        "model_version": "1.0", "methods": model["methods"],
        "methods_long": (
            f"{model['model_name']}: four separate direct horizon regressors for South Carolina. "
            "Inputs are NHSN weekly influenza admissions and local MUSC and PRISMA indicators. "
            "Predictors include lags, trailing summaries, and seasonal terms. MI/VIF selection and "
            "hyperparameter tuning precede chronological calibration. Models are refitted at successive "
            "historical origins using labels no later than the origin. Ratio corrections use only matured "
            "previous forecast errors. Hospitalization quantiles are estimated from empirical log1p forecast "
            "errors by horizon, clipped at zero, and rounded to integers. No dependence across locations is "
            "modeled. Retrospective results use the current historical data extracts, not historical release "
            "vintages; future runs retain input snapshots. These are marginal quantile forecasts, not joint samples."
        ),
        "ensemble_of_models": True, "ensemble_of_hub_models": False,
        "baseline_model": False, "designated_targets": [TARGET],
    })
    schema = json.loads((Path(cache_dir) / "model-metadata-schema.json").read_text(encoding="utf-8"))
    errors = [error.message for error in jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()).iter_errors(metadata)]
    if not settings.get("metadata_confirmed", False):
        errors.append("Confirm model IDs, complete contributor list, website, license, and designation; then set metadata_confirmed=true.")
    if errors:
        raise ValueError("Metadata incomplete: " + "; ".join(errors))
    return metadata


def export_forecast(root, settings, model_key, reference, origin, horizon_outputs, points, mode):
    if mode not in ["dataset", "live", "retrospective"]:
        raise ValueError("Mode must be dataset, live, or retrospective.")
    reference = pd.Timestamp(reference).normalize()
    origin = pd.Timestamp(origin).normalize()
    if mode == "dataset" and reference != origin + pd.Timedelta(weeks=1):
        raise ValueError("Dataset exports must retain the reference date one week after the forecast origin.")
    root = Path(root)
    cache = root / "hub-config-cache"
    model_id = f"{settings['metadata']['team_abbr']}-{settings['models'][model_key]['model_abbr']}"
    records, summaries = [], []
    for local_horizon, result in horizon_outputs.items():
        forecast_origin = pd.Timestamp(result["forecast"]["forecast_origin_week"])
        if forecast_origin != pd.Timestamp(origin):
            raise ValueError(f"T+{local_horizon}: complete predictor data stop before the latest observed week.")
        target_date = forecast_origin + pd.Timedelta(weeks=local_horizon)
        days = (target_date - pd.Timestamp(reference)).days
        if days % 7:
            raise ValueError("Forecast target is not a whole week from the reference date.")
        hub_horizon = days // 7
        values, count = residual_quantiles(points[local_horizon],
            result["holdout"]["calibration_history"], origin, settings["calibration_min_rows"])
        for level, value in zip(QUANTILES, values):
            records.append(dict(zip(COLUMNS, [str(reference.date()), TARGET, hub_horizon,
                str(target_date.date()), "45", "quantile", float(level), int(value)])))
        metrics = result["holdout"]["quantile_metrics"]
        summaries.append({"model": model_key, "notebook_horizon": local_horizon,
            "hub_horizon": hub_horizon, "calibration_errors": count,
            "test_WIS": float(metrics["WIS"].mean()),
            "test_coverage_50": float(metrics["coverage_50"].mean()),
            "test_coverage_95": float(metrics["coverage_95"].mean()),
            "test_weeks": len(metrics)})
    frame = pd.DataFrame(records, columns=COLUMNS)
    # Local data-anchored forecasts can fall outside an active hub round.
    # Only an explicit live submission must match the hub's accepted reference dates.
    validate_forecast(frame, reference, model_id, cache if mode == "live" else None)
    output = root / "outputs" / model_key
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "quantile_forecast.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "quantile_validation_metrics.csv", index=False)
    destination = root / "submissions" / (mode if mode != "live" else "drafts")
    metadata_error = None
    if mode == "live":
        try:
            metadata = build_metadata(settings, model_key, cache)
        except ValueError as error:
            metadata_error = str(error)
        else:
            import yaml
            destination = root / "submissions" / "ready"
            metadata_folder = destination / "model-metadata"
            metadata_folder.mkdir(parents=True, exist_ok=True)
            (metadata_folder / f"{model_id}.yml").write_text(
                yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    forecast_dir = destination / "model-output" / model_id
    forecast_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = forecast_dir / f"{reference.date()}-{model_id}.csv"
    frame.to_csv(forecast_path, index=False)
    manifest = {
        "model_id": model_id, "reference_date": str(reference.date()), "mode": mode,
        "forecast_origin": str(origin.date()),
        "date_anchor": "latest common observation" if mode != "live" else "current submission week",
        "forecast": str(forecast_path), "ready_for_submission": mode == "live" and metadata_error is None,
        "metadata_error": metadata_error, "metrics": summaries,
        "validation": "Local structural checks passed; run official hubValidations in a hub clone before submission.",
        "retrospective_limit": "Historical input release vintages are unavailable; diagnostics use current extracts.",
    }
    (output / "submission_status.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {mode} forecast: {forecast_path}", flush=True)
    if metadata_error:
        print(metadata_error, flush=True)
    if mode != "live":
        print("These dates follow the dataset, not the current submission week. No live submission clearance was issued.", flush=True)
    return frame


def snapshot_inputs(root, data_dir, model_key, reference, mode):
    import shutil
    folder = Path(root) / "outputs" / model_key / "input-snapshots" / f"{mode}-{reference.date()}"
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ["CDC.csv", "MUSC.csv", "PRISMA.csv"]:
        source = Path(data_dir) / name
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        destination = folder / f"{source.stem}-{digest[:12]}.csv"
        if not destination.exists():
            shutil.copy2(source, destination)
        hashes[name] = {"source": str(source), "sha256": digest, "snapshot": str(destination)}
    (folder / "manifest.json").write_text(json.dumps({
        "recorded_at": now_eastern().isoformat(), "inputs": hashes}, indent=2), encoding="utf-8")
    return hashes
