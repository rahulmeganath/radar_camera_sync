"""
alignment_engine.py – Master Timeline construction and frame alignment.

Deterministic, vectorised (numpy) alignment of camera and radar frames
to a unified Master Timeline derived from the sync CSV.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timestamp normalisation
# ---------------------------------------------------------------------------

def normalize_timestamps(ts_ns: np.ndarray) -> np.ndarray:
    """Convert nanosecond integer timestamps to float64 seconds.

    Removes NaN, detects negatives/duplicates, returns sorted array.
    """
    ts_s = ts_ns.astype(np.float64) / 1e9

    nan_mask = np.isnan(ts_s)
    if nan_mask.any():
        logger.warning("Removing %d NaN timestamps", nan_mask.sum())
        ts_s = ts_s[~nan_mask]

    neg_mask = np.diff(ts_s) < 0
    if neg_mask.any():
        logger.warning("Detected %d negative time jumps", neg_mask.sum())

    dup_count = len(ts_s) - len(np.unique(ts_s))
    if dup_count:
        logger.warning("Detected %d duplicate timestamps", dup_count)

    return np.sort(ts_s)


# ---------------------------------------------------------------------------
# Master Timeline
# ---------------------------------------------------------------------------

def build_master_timeline(sync_df: pd.DataFrame) -> np.ndarray:
    """Build the Master Timeline (MTL) from sync CSV system timestamps.

    Uses camera system_time_ns column as the reference clock.
    Returns sorted float64 array in seconds.
    """
    if "system_time_ns" not in sync_df.columns:
        raise ValueError("sync_df missing 'system_time_ns' column")
    ts_ns = sync_df["system_time_ns"].dropna().values.astype(np.int64)
    mtl = normalize_timestamps(ts_ns)
    logger.info("Master Timeline: %d ticks, %.3fs – %.3fs (duration=%.1fs)",
                len(mtl), mtl[0], mtl[-1], mtl[-1] - mtl[0])
    return mtl


# ---------------------------------------------------------------------------
# Nearest-neighbour alignment (vectorised)
# ---------------------------------------------------------------------------

def _nearest_idx(query_times: np.ndarray, reference_times: np.ndarray) -> np.ndarray:
    """For each query time find the index of the nearest reference time."""
    # np.searchsorted gives insertion position; check neighbours
    pos = np.searchsorted(reference_times, query_times)
    pos = np.clip(pos, 0, len(reference_times) - 1)
    # Compare with left neighbour
    pos_left = np.clip(pos - 1, 0, len(reference_times) - 1)
    delta_right = np.abs(reference_times[pos] - query_times)
    delta_left  = np.abs(reference_times[pos_left] - query_times)
    best = np.where(delta_left <= delta_right, pos_left, pos)
    return best


def align_camera_to_mtl(camera_ts_df: pd.DataFrame,
                         mtl: np.ndarray) -> pd.DataFrame:
    """Align every MTL tick to the nearest camera frame.

    Returns DataFrame with columns:
      mtl_time_s, camera_frame_id, camera_time_s, camera_offset_ms
    """
    cam_times = camera_ts_df["camera_time_s"].values
    cam_ids   = camera_ts_df["frame_id"].values

    idx = _nearest_idx(mtl, cam_times)
    selected_times = cam_times[idx]
    selected_ids   = cam_ids[idx]
    offsets_ms     = (mtl - selected_times) * 1000.0

    result = pd.DataFrame({
        "mtl_time_s":       mtl,
        "camera_frame_id":  selected_ids,
        "camera_time_s":    selected_times,
        "camera_offset_ms": offsets_ms,
    })
    logger.info("Camera aligned: %d MTL ticks → %d unique frames",
                len(result), result["camera_frame_id"].nunique())
    return result


def align_radar_to_mtl(radar_ts_df: pd.DataFrame,
                        mtl: np.ndarray) -> pd.DataFrame:
    """Align every MTL tick to the nearest radar frame.

    Returns DataFrame with columns:
      mtl_time_s, radar_msg_id, radar_time_s, radar_offset_ms
    """
    rad_times = radar_ts_df["radar_time_s"].values
    rad_ids   = radar_ts_df["msg_id"].values

    idx = _nearest_idx(mtl, rad_times)
    selected_times = rad_times[idx]
    selected_ids   = rad_ids[idx]
    offsets_ms     = (mtl - selected_times) * 1000.0

    result = pd.DataFrame({
        "mtl_time_s":      mtl,
        "radar_msg_id":    selected_ids,
        "radar_time_s":    selected_times,
        "radar_offset_ms": offsets_ms,
    })
    logger.info("Radar aligned: %d MTL ticks → %d unique frames",
                len(result), result["radar_msg_id"].nunique())
    return result


# ---------------------------------------------------------------------------
# Offset computation
# ---------------------------------------------------------------------------

def compute_offsets(cam_aligned: pd.DataFrame,
                    rad_aligned: pd.DataFrame) -> pd.DataFrame:
    """Compute per-MTL-tick offset = camera_time - radar_time (ms)."""
    assert len(cam_aligned) == len(rad_aligned), "Alignment tables must have same length"
    offset_ms = (cam_aligned["camera_time_s"].values
                 - rad_aligned["radar_time_s"].values) * 1000.0
    merged = cam_aligned[["mtl_time_s", "camera_frame_id", "camera_time_s"]].copy()
    merged["radar_msg_id"]    = rad_aligned["radar_msg_id"].values
    merged["radar_time_s"]    = rad_aligned["radar_time_s"].values
    merged["offset_ms"]       = offset_ms
    merged["abs_offset_ms"]   = np.abs(offset_ms)
    merged["camera_offset_ms"] = cam_aligned["camera_offset_ms"].values
    merged["radar_offset_ms"]  = rad_aligned["radar_offset_ms"].values
    return merged


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def detect_drift(time_s: np.ndarray,
                 offset_ms: np.ndarray) -> dict:
    """Fit linear regression to offset(t) to detect clock drift.

    Returns:
      slope_ms_per_min, intercept_ms, r_squared, p_value
    """
    if len(time_s) < 2:
        return {"slope_ms_per_min": 0.0, "intercept_ms": 0.0,
                "r_squared": 0.0, "p_value": 1.0}
    t_norm = time_s - time_s[0]   # relative seconds
    result = stats.linregress(t_norm, offset_ms)
    slope_ms_per_min = result.slope * 60.0   # per-second → per-minute
    logger.info("Drift: slope=%.4f ms/min, R²=%.4f", slope_ms_per_min, result.rvalue ** 2)
    return {
        "slope_ms_per_min": slope_ms_per_min,
        "intercept_ms":     result.intercept,
        "r_squared":        result.rvalue ** 2,
        "p_value":          result.pvalue,
    }


# ---------------------------------------------------------------------------
# Frame drop detection
# ---------------------------------------------------------------------------

def detect_frame_drops(ts_df: pd.DataFrame,
                        time_col: str,
                        expected_period_s: float,
                        gap_multiplier: float = 1.5) -> pd.DataFrame:
    """Mark frames where the time gap exceeds gap_multiplier × expected_period.

    Adds a 'is_drop' boolean column to a copy of ts_df.
    """
    df = ts_df.copy()
    times = df[time_col].values
    gaps = np.diff(times, prepend=times[0])
    threshold = expected_period_s * gap_multiplier
    df["gap_s"]   = gaps
    df["is_drop"] = gaps > threshold
    n_drops = int(df["is_drop"].sum())
    logger.info("Frame drop detection: %d drops (threshold=%.3fs)", n_drops, threshold)
    return df


# ---------------------------------------------------------------------------
# Aligned frame table
# ---------------------------------------------------------------------------

def build_aligned_frame_table(mtl: np.ndarray,
                               cam_aligned: pd.DataFrame,
                               rad_aligned: pd.DataFrame,
                               sync_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the master aligned frame table (exportable CSV)."""
    table = compute_offsets(cam_aligned, rad_aligned)

    # Elapsed time from start
    table["elapsed_s"] = table["mtl_time_s"] - table["mtl_time_s"].iloc[0]

    # Optional: steering / speed from sync CSV
    if sync_df is not None:
        if "steering_deg" in sync_df.columns:
            table["steering_deg"] = sync_df["steering_deg"].values[:len(table)]
        if "speed" in sync_df.columns:
            table["speed"] = sync_df["speed"].values[:len(table)]
        if "num_targets" in sync_df.columns:
            table["num_radar_targets"] = sync_df["num_targets"].values[:len(table)]

    logger.info("Aligned frame table: %d rows", len(table))
    return table


# ---------------------------------------------------------------------------
# Main alignment pipeline
# ---------------------------------------------------------------------------

def run_alignment(camera_ts_df: pd.DataFrame,
                  radar_ts_df: pd.DataFrame,
                  sync_df: pd.DataFrame,
                  gap_multiplier: float = 1.5) -> dict:
    """Run the full alignment pipeline and return all artefacts."""
    mtl = build_master_timeline(sync_df)

    cam_aligned = align_camera_to_mtl(camera_ts_df, mtl)
    rad_aligned = align_radar_to_mtl(radar_ts_df, mtl)
    aligned_table = build_aligned_frame_table(mtl, cam_aligned, rad_aligned, sync_df)

    drift = detect_drift(aligned_table["mtl_time_s"].values,
                         aligned_table["offset_ms"].values)

    # Frame drop detection
    cam_fps = 30.0
    radar_hz = 18.18
    cam_with_drops = detect_frame_drops(
        camera_ts_df, "camera_time_s", 1.0 / cam_fps, gap_multiplier)
    rad_with_drops = detect_frame_drops(
        radar_ts_df, "radar_time_s", 1.0 / radar_hz, gap_multiplier)

    return {
        "mtl":             mtl,
        "cam_aligned":     cam_aligned,
        "rad_aligned":     rad_aligned,
        "aligned_table":   aligned_table,
        "drift":           drift,
        "cam_with_drops":  cam_with_drops,
        "rad_with_drops":  rad_with_drops,
    }
