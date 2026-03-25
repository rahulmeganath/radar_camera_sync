"""
metadata_parser.py – Parse and structure metadata from all sources.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def parse_camera_session(session_info: dict) -> dict:
    """Structure camera session metadata from SQLite session_info table."""
    def _get(key, default="N/A"):
        return session_info.get(key, default)

    return {
        "session_id":   _get("session_id"),
        "resolution":   _get("resolution"),
        "fps":          _get("fps"),
        "bitrate_bps":  _get("bitrate"),
        "device":       _get("device"),
        "start_time":   _get("start_time"),
        "platform":     _get("platform"),
        "quality_mode": _get("quality_mode"),
        "video_file":   _get("video_file"),
    }


def parse_camera_performance(perf_df: pd.DataFrame) -> dict:
    """Summarise camera performance log."""
    if perf_df.empty:
        return {}
    return {
        "mean_fps":      float(perf_df["fps"].mean()) if "fps" in perf_df.columns else 0.0,
        "min_fps":       float(perf_df["fps"].min()) if "fps" in perf_df.columns else 0.0,
        "max_fps":       float(perf_df["fps"].max()) if "fps" in perf_df.columns else 0.0,
        "total_drops":   int(perf_df["drops"].sum()) if "drops" in perf_df.columns else 0,
        "mean_latency_ms": float(perf_df["max_delta_ms"].mean()) if "max_delta_ms" in perf_df.columns else 0.0,
        "max_latency_ms":  float(perf_df["max_delta_ms"].max()) if "max_delta_ms" in perf_df.columns else 0.0,
        "num_log_entries": int(len(perf_df)),
    }


def parse_radar_log_stats(log_events: list[dict]) -> dict:
    """Extract radar statistics from parsed log events."""
    stat_entries = [e for e in log_events if "stats" in e]

    if not stat_entries:
        return {"num_log_events": len(log_events)}

    hz_vals      = [e["stats"]["hz"]          for e in stat_entries if "hz" in e["stats"]]
    drop_vals    = [e["stats"]["drops"]        for e in stat_entries if "drops" in e["stats"]]
    tgt_vals     = [e["stats"]["tgt_per_msg"]  for e in stat_entries if "tgt_per_msg" in e["stats"]]
    cb_vals      = [e["stats"]["callback_us"]  for e in stat_entries if "callback_us" in e["stats"]]
    delta_vals   = [e["stats"]["delta_ms"]     for e in stat_entries if "delta_ms" in e["stats"]]

    import numpy as np
    def safe_stats(arr):
        if not arr:
            return {"mean": 0, "min": 0, "max": 0}
        return {"mean": float(np.mean(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}

    return {
        "num_log_events":       len(log_events),
        "num_stat_entries":     len(stat_entries),
        "hz":                   safe_stats(hz_vals),
        "total_drops":          int(sum(drop_vals)) if drop_vals else 0,
        "targets_per_msg":      safe_stats(tgt_vals),
        "callback_latency_us":  safe_stats(cb_vals),
        "delta_ms":             safe_stats(delta_vals),
        "warnings":             [e["message"] for e in log_events if e.get("level") == "WARNING"],
        "errors":               [e["message"] for e in log_events if e.get("level") == "ERROR"],
    }


def parse_radar_hdf5_metadata(hdf5_metadata: dict) -> dict:
    """Parse HDF5 metadata group attributes."""
    result = {}
    for k, v in hdf5_metadata.items():
        try:
            if hasattr(v, "item"):
                result[k] = v.item()
            elif isinstance(v, bytes):
                result[k] = v.decode("utf-8", errors="replace")
            else:
                result[k] = v
        except Exception:
            result[k] = str(v)
    return result


def build_metadata_bundle(camera_session: dict,
                           camera_perf: dict,
                           radar_log_stats: dict,
                           radar_hdf5_meta: dict,
                           radar_ts_df: pd.DataFrame,
                           camera_ts_df: pd.DataFrame) -> dict:
    """Combine all metadata into a single structured bundle."""
    import numpy as np

    num_targets_total = int(radar_ts_df["num_targets"].sum()) if "num_targets" in radar_ts_df.columns else 0
    num_targets_mean  = float(radar_ts_df["num_targets"].mean()) if "num_targets" in radar_ts_df.columns else 0.0

    return {
        "camera": {
            "session": camera_session,
            "performance": camera_perf,
            "total_frames": len(camera_ts_df),
        },
        "radar": {
            "hdf5_metadata": radar_hdf5_meta,
            "log_stats": radar_log_stats,
            "total_frames": len(radar_ts_df),
            "avg_targets_per_frame": round(num_targets_mean, 1),
            "total_targets": num_targets_total,
        },
    }
