"""
data_loader.py – Ingest all sensor data files.

Handles:
  - Camera timestamps CSV
  - Camera MP4 (OpenCV handle)
  - Camera metadata SQLite DB
  - Camera performance log
  - Radar timestamps CSV
  - Radar HDF5 (lazy, indexed)
  - Radar log (parsed events)
  - Sync CSV (Master Timeline source)
"""

from __future__ import annotations

import glob
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def load_camera_timestamps(csv_path: str) -> pd.DataFrame:
    """Load camera timestamps CSV → DataFrame with normalised key columns."""
    logger.info("Loading camera timestamps: %s", csv_path)
    df = pd.read_csv(csv_path)
    required = {"frame_id", "system_time_ns"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Camera timestamps CSV missing columns: {missing}")
    df = df.sort_values("frame_id").reset_index(drop=True)
    # Derived
    df["camera_time_s"] = df["system_time_ns"] / 1e9
    logger.info("Loaded %d camera frames", len(df))
    return df


def load_camera_video(mp4_path: str) -> dict[str, Any]:
    """Open camera video with OpenCV and return handle + metadata."""
    logger.info("Opening camera video: %s", mp4_path)
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {mp4_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Video: %dx%d @ %.2f fps, %d frames", width, height, fps, frame_count)
    return {
        "cap": cap,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "path": mp4_path,
    }


def load_camera_metadata_db(db_path: str) -> dict[str, Any]:
    """Parse SQLite metadata DB → session_info dict + frames DataFrame."""
    logger.info("Loading camera metadata DB: %s", db_path)
    conn = sqlite3.connect(db_path)
    try:
        session_info = {}
        try:
            cur = conn.execute("SELECT key, value FROM session_info")
            session_info = dict(cur.fetchall())
        except Exception as e:
            logger.warning("Could not read session_info: %s", e)

        frames_df = pd.DataFrame()
        try:
            frames_df = pd.read_sql("SELECT * FROM frames ORDER BY frame_id", conn)
        except Exception as e:
            logger.warning("Could not read frames table: %s", e)
    finally:
        conn.close()

    logger.info("DB session: %s", session_info.get("session_id", "?"))
    return {"session_info": session_info, "frames": frames_df}


def load_camera_performance_log(log_path: str) -> pd.DataFrame:
    """Parse camera performance log CSV → DataFrame."""
    logger.info("Loading camera performance log: %s", log_path)
    df = pd.read_csv(log_path)
    if "timestamp" in df.columns:
        df["time_s"] = df["timestamp"]
    logger.info("Performance log: %d entries", len(df))
    return df


# ---------------------------------------------------------------------------
# Radar
# ---------------------------------------------------------------------------

def load_radar_timestamps(csv_path: str) -> pd.DataFrame:
    """Load radar timestamps CSV."""
    logger.info("Loading radar timestamps: %s", csv_path)
    df = pd.read_csv(csv_path)
    required = {"msg_id", "ros_timestamp_ns"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Radar timestamps CSV missing columns: {missing}")
    df = df.sort_values("msg_id").reset_index(drop=True)
    df["radar_time_s"] = df["ros_timestamp_ns"] / 1e9
    logger.info("Loaded %d radar frames", len(df))
    return df


class RadarHDF5:
    """Lazy-loading wrapper around radar HDF5.

    Keeps the file open and indexes targets by msg_id on first access.
    """

    def __init__(self, hdf5_path: str):
        self.path = hdf5_path
        self._f: h5py.File | None = None
        self._target_msg_ids: np.ndarray | None = None
        self._msg_ids: np.ndarray | None = None
        self._metadata: dict = {}
        logger.info("Initialising RadarHDF5 handle: %s", hdf5_path)

    def open(self):
        if self._f is None:
            self._f = h5py.File(self.path, "r")
            # Build index
            if "radar" in self._f and "target_msg_id" in self._f["radar"]:
                self._target_msg_ids = self._f["radar"]["target_msg_id"][:]
            if "radar" in self._f and "msg_id" in self._f["radar"]:
                self._msg_ids = self._f["radar"]["msg_id"][:]
            # Metadata attrs
            if "metadata" in self._f:
                for k in self._f["metadata"].keys():
                    try:
                        self._metadata[k] = self._f["metadata"][k][()]
                    except Exception:
                        pass
            logger.info("HDF5 opened. Frames: %d, Targets: %s",
                        len(self._msg_ids) if self._msg_ids is not None else 0,
                        len(self._target_msg_ids) if self._target_msg_ids is not None else 0)

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    @property
    def msg_ids(self) -> np.ndarray:
        self.open()
        return self._msg_ids if self._msg_ids is not None else np.array([])

    @property
    def num_frames(self) -> int:
        return len(self.msg_ids)

    def get_frame(self, msg_id: int) -> pd.DataFrame:
        """Return DataFrame of all targets for the given msg_id."""
        self.open()
        if self._f is None or "radar" not in self._f:
            return pd.DataFrame()
        if self._target_msg_ids is None:
            return pd.DataFrame()
        mask = self._target_msg_ids == int(msg_id)
        if not mask.any():
            logger.debug("No targets for msg_id=%d", msg_id)
            return pd.DataFrame()
        radar_grp = self._f["radar"]
        cols = ["x", "y", "z", "range", "azimuth_angle", "elevation_angle",
                "radial_speed", "rcs", "snr", "noise", "power"]
        data = {}
        for col in cols:
            if col in radar_grp:
                data[col] = radar_grp[col][mask]
        df = pd.DataFrame(data)
        return df

    def get_frame_metadata(self, msg_id: int) -> dict:
        """Return per-frame metadata (num_targets, ros_timestamp_ns, etc.)."""
        self.open()
        if self._f is None or "radar" not in self._f:
            return {}
        radar_grp = self._f["radar"]
        if self._msg_ids is None:
            return {}
        idx_arr = np.where(self._msg_ids == int(msg_id))[0]
        if len(idx_arr) == 0:
            return {}
        idx = idx_arr[0]
        result = {"msg_id": int(msg_id)}
        for field in ["num_targets", "ros_timestamp_ns", "delta_time_ns", "processing_time_us"]:
            if field in radar_grp:
                result[field] = float(radar_grp[field][idx])
        return result

    @property
    def metadata(self) -> dict:
        self.open()
        return self._metadata


def load_radar_hdf5(hdf5_path: str) -> RadarHDF5:
    """Create and open a lazy-loading radar HDF5 handle."""
    rh = RadarHDF5(hdf5_path)
    rh.open()
    return rh


def load_radar_log(log_path: str) -> list[dict]:
    """Parse radar log file → list of event dicts."""
    logger.info("Loading radar log: %s", log_path)
    events = []
    pattern = re.compile(
        r"\[(?P<ts>[\d\-: .]+)\]\s+(?P<level>\w+):\s+(?P<msg>.*)"
    )
    stat_pattern = re.compile(
        r"Msg\s+(?P<msg_id>\d+)\s+\|\s+(?P<elapsed>[\d.]+)s\s+\|\s+"
        r"(?P<hz>[\d.]+)Hz\s+\|\s+Tgt/msg:\s+(?P<tgt_per_msg>[\d.]+)\s+\|\s+"
        r"Δ:\s+(?P<delta_ms>[\d.]+)ms\s+\|\s+Drops:\s+(?P<drops>\d+)\s+\|\s+"
        r"Callback:\s+(?P<callback_us>[\d.]+)µs"
    )
    try:
        with open(log_path) as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    event = {"timestamp": m.group("ts").strip(),
                             "level": m.group("level"),
                             "message": m.group("msg").strip()}
                    sm = stat_pattern.search(m.group("msg"))
                    if sm:
                        event["stats"] = {k: float(v) for k, v in sm.groupdict().items()}
                    events.append(event)
    except Exception as e:
        logger.error("Failed to parse radar log: %s", e)
    logger.info("Parsed %d radar log events", len(events))
    return events


# ---------------------------------------------------------------------------
# Sync CSV (Master Timeline source)
# ---------------------------------------------------------------------------

def load_sync_csv(sync_path: str) -> pd.DataFrame:
    """Load the master sync CSV."""
    logger.info("Loading sync CSV: %s", sync_path)
    df = pd.read_csv(sync_path)
    logger.info("Sync CSV: %d rows, columns: %s", len(df), list(df.columns))
    return df


# ---------------------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------------------

def discover_files(base_dir: str, config: dict) -> dict[str, str]:
    """Discover actual file paths from config patterns."""
    cam_dir = os.path.join(base_dir, config["data"]["camera_dir"])
    rad_dir = os.path.join(base_dir, config["data"]["radar_dir"])

    def find_one(directory, pattern):
        matches = glob.glob(os.path.join(directory, pattern))
        if not matches:
            raise FileNotFoundError(f"No file matching '{pattern}' in '{directory}'")
        if len(matches) > 1:
            logger.warning("Multiple matches for %s; using first: %s", pattern, matches[0])
        return matches[0]

    return {
        "camera_timestamps": find_one(cam_dir, config["data"]["camera_timestamps_pattern"]),
        "camera_video":      find_one(cam_dir, config["data"]["camera_video_pattern"]),
        "camera_metadata":   find_one(cam_dir, config["data"]["camera_metadata_pattern"]),
        "camera_performance":find_one(cam_dir, config["data"]["camera_performance_pattern"]),
        "radar_timestamps":  find_one(rad_dir, config["data"]["radar_timestamps_pattern"]),
        "radar_hdf5":        find_one(rad_dir, config["data"]["radar_hdf5_pattern"]),
        "radar_log":         find_one(rad_dir, config["data"]["radar_log_pattern"]),
        "sync_csv":          os.path.join(base_dir, config["data"]["sync_csv"]),
    }



def _align_timestamps(sensor_df: pd.DataFrame, sync_df: pd.DataFrame,
                      sensor_id_col: str, sync_id_col: str,
                      sensor_ts_col: str, sync_ts_col: str) -> pd.DataFrame:
    """Align sensor timestamps to the Sync CSV's timeline (Unix epoch).

    If sync_ts_col is missing from sync_df, skip alignment silently.
    """
    if sync_ts_col not in sync_df.columns:
        logger.warning("Sync CSV missing column '%s' – skipping timestamp alignment for %s",
                       sync_ts_col, sensor_id_col)
        return sensor_df
    if sync_id_col not in sync_df.columns:
        logger.warning("Sync CSV missing column '%s' – skipping timestamp alignment for %s",
                       sync_id_col, sensor_id_col)
        return sensor_df

    # Prepare sync subset with explicit renaming to avoid collision
    ref_df = sync_df[[sync_id_col, sync_ts_col]].copy()
    ref_df = ref_df.rename(columns={sync_ts_col: "_ref_ts", sync_id_col: "_ref_id"})

    # Merge
    common = pd.merge(sensor_df, ref_df, left_on=sensor_id_col, right_on="_ref_id")

    if common.empty:
        logger.warning("No common frames found for %s alignment – skipping", sensor_id_col)
        return sensor_df

    # Calculate offset (Sync - Sensor)
    diffs = common["_ref_ts"] - common[sensor_ts_col]
    median_offset = diffs.median()

    # If offset is significant (> 1ms), apply it
    if abs(median_offset) > 1e6:
        logger.info("Aligning %s timestamps: adding %.3fs offset (median)",
                    sensor_id_col, median_offset / 1e9)
        sensor_df = sensor_df.copy()
        sensor_df[sensor_ts_col] = sensor_df[sensor_ts_col] + median_offset
        # Re-derive seconds
        if "camera_time_s" in sensor_df.columns and sensor_ts_col == "system_time_ns":
            sensor_df["camera_time_s"] = sensor_df[sensor_ts_col] / 1e9
        if "radar_time_s" in sensor_df.columns and sensor_ts_col == "ros_timestamp_ns":
            sensor_df["radar_time_s"] = sensor_df[sensor_ts_col] / 1e9
    else:
        logger.info("%s timestamps already aligned (median diff=%.3f ms)",
                    sensor_id_col, median_offset / 1e6)

    return sensor_df


def load_all(base_dir: str, config: dict) -> dict[str, Any]:
    """Load everything and return a unified data bundle."""
    paths = discover_files(base_dir, config)
    
    # Load raw
    cam_ts = load_camera_timestamps(paths["camera_timestamps"])
    rad_ts = load_radar_timestamps(paths["radar_timestamps"])
    sync_df = load_sync_csv(paths["sync_csv"])

    # Align timestamps to Sync CSV epoch
    if not sync_df.empty:
        # Camera: frame_id vs frame_id, system_time_ns vs system_time_ns
        cam_ts = _align_timestamps(cam_ts, sync_df,
                                   "frame_id", "frame_id",
                                   "system_time_ns", "system_time_ns")
        
        # Radar: msg_id vs msg_id, ros_timestamp_ns vs radar_time_ns
        rad_ts = _align_timestamps(rad_ts, sync_df,
                                   "msg_id", "msg_id",
                                   "ros_timestamp_ns", "radar_time_ns")

    bundle = {
        "camera_timestamps":    cam_ts,
        "camera_video":         load_camera_video(paths["camera_video"]),
        "camera_metadata_db":   load_camera_metadata_db(paths["camera_metadata"]),
        "camera_performance":   load_camera_performance_log(paths["camera_performance"]),
        "radar_timestamps":     rad_ts,
        "radar_hdf5":           load_radar_hdf5(paths["radar_hdf5"]),
        "radar_log":            load_radar_log(paths["radar_log"]),
        "sync_df":              sync_df,
        "paths":                paths,
    }
    return bundle

