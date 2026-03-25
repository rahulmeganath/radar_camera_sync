"""
validation.py – Input validation and error detection.

All detected issues are logged with appropriate level.
Does not raise exceptions – returns structured error lists instead.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------

def validate_timestamps(ts_array: np.ndarray, label: str = "timestamps") -> list[dict]:
    """Validate a timestamp array. Returns list of issue dicts."""
    issues = []

    nan_count = int(np.isnan(ts_array.astype(float)).sum())
    if nan_count:
        msg = f"{label}: {nan_count} NaN timestamps detected"
        logger.warning(msg)
        issues.append({"level": "WARNING", "type": "nan_timestamps", "detail": msg})

    dup_count = len(ts_array) - len(np.unique(ts_array))
    if dup_count:
        msg = f"{label}: {dup_count} duplicate timestamps detected"
        logger.warning(msg)
        issues.append({"level": "WARNING", "type": "duplicate_timestamps", "detail": msg})

    sorted_ts = np.sort(ts_array.astype(float))
    diffs = np.diff(sorted_ts)
    neg_count = int((diffs < 0).sum())
    if neg_count:
        msg = f"{label}: {neg_count} negative time jumps"
        logger.error(msg)
        issues.append({"level": "ERROR", "type": "negative_time_jump", "detail": msg})

    zero_count = int((sorted_ts == 0).sum())
    if zero_count:
        msg = f"{label}: {zero_count} zero-value timestamps"
        logger.warning(msg)
        issues.append({"level": "WARNING", "type": "zero_timestamps", "detail": msg})

    if len(issues) == 0:
        logger.debug("%s: timestamps valid (%d entries)", label, len(ts_array))

    return issues


# ---------------------------------------------------------------------------
# HDF5 frame validation
# ---------------------------------------------------------------------------

def validate_hdf5_frame(radar_hdf5: Any, msg_id: int) -> list[dict]:
    """Validate that a radar HDF5 frame exists and is non-empty."""
    issues = []
    try:
        df = radar_hdf5.get_frame(int(msg_id))
        if df is None or df.empty:
            msg = f"HDF5 frame msg_id={msg_id}: no targets (empty frame)"
            logger.warning(msg)
            issues.append({"level": "WARNING", "type": "empty_radar_frame",
                           "detail": msg, "msg_id": msg_id})
        else:
            # Check for NaN in critical columns
            for col in ["x", "y", "range"]:
                if col in df.columns:
                    nan_n = int(df[col].isna().sum())
                    if nan_n:
                        msg = f"HDF5 frame msg_id={msg_id}: {nan_n} NaN in column '{col}'"
                        logger.warning(msg)
                        issues.append({"level": "WARNING", "type": "nan_in_radar_data",
                                       "detail": msg, "msg_id": msg_id})
    except Exception as e:
        msg = f"HDF5 frame msg_id={msg_id}: read error – {e}"
        logger.error(msg)
        issues.append({"level": "ERROR", "type": "corrupt_hdf5_entry",
                       "detail": msg, "msg_id": msg_id})
    return issues


# ---------------------------------------------------------------------------
# Video frame validation
# ---------------------------------------------------------------------------

def validate_video_frame(cap: Any, frame_id: int) -> list[dict]:
    """Validate that a camera video frame can be decoded."""
    import cv2
    issues = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ret, frame = cap.read()
        if not ret or frame is None:
            msg = f"Video frame {frame_id}: decode failed"
            logger.error(msg)
            issues.append({"level": "ERROR", "type": "video_decode_error",
                           "detail": msg, "frame_id": frame_id})
        elif frame.sum() == 0:
            msg = f"Video frame {frame_id}: all-black (possibly corrupt)"
            logger.warning(msg)
            issues.append({"level": "WARNING", "type": "black_frame",
                           "detail": msg, "frame_id": frame_id})
    except Exception as e:
        msg = f"Video frame {frame_id}: exception – {e}"
        logger.error(msg)
        issues.append({"level": "ERROR", "type": "video_read_exception",
                       "detail": msg, "frame_id": frame_id})
    return issues


# ---------------------------------------------------------------------------
# DataFrame structural validation
# ---------------------------------------------------------------------------

def validate_dataframe(df: pd.DataFrame,
                        required_cols: list[str],
                        label: str = "DataFrame") -> list[dict]:
    """Validate that a DataFrame has required columns and is non-empty."""
    issues = []
    if df is None or df.empty:
        msg = f"{label}: DataFrame is empty"
        logger.error(msg)
        issues.append({"level": "ERROR", "type": "empty_dataframe", "detail": msg})
        return issues
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        msg = f"{label}: missing columns {missing}"
        logger.error(msg)
        issues.append({"level": "ERROR", "type": "missing_columns",
                       "detail": msg, "columns": missing})
    return issues


# ---------------------------------------------------------------------------
# Full dataset validation report
# ---------------------------------------------------------------------------

def run_full_validation(data_bundle: dict) -> dict:
    """Run all validations on loaded data bundle. Returns report dict."""
    all_issues = []

    # Camera timestamps
    cam_df = data_bundle.get("camera_timestamps")
    if cam_df is not None and not cam_df.empty:
        all_issues += validate_timestamps(cam_df["system_time_ns"].values, "camera_timestamps")
        all_issues += validate_dataframe(cam_df, ["frame_id", "system_time_ns"], "camera_timestamps")

    # Radar timestamps
    rad_df = data_bundle.get("radar_timestamps")
    if rad_df is not None and not rad_df.empty:
        all_issues += validate_timestamps(rad_df["ros_timestamp_ns"].values, "radar_timestamps")
        all_issues += validate_dataframe(rad_df, ["msg_id", "ros_timestamp_ns"], "radar_timestamps")

    # Sync CSV
    sync_df = data_bundle.get("sync_df")
    if sync_df is not None and not sync_df.empty:
        all_issues += validate_timestamps(sync_df["system_time_ns"].values, "sync_csv")

    errors   = [i for i in all_issues if i["level"] == "ERROR"]
    warnings = [i for i in all_issues if i["level"] == "WARNING"]

    report = {
        "total_issues": len(all_issues),
        "errors":        errors,
        "warnings":      warnings,
        "is_valid":      len(errors) == 0,
    }

    logger.info("Validation: %d errors, %d warnings", len(errors), len(warnings))
    return report
