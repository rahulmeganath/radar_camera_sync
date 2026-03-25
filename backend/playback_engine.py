"""
playback_engine.py – Time-driven playback state machine.

Drives synchronized camera + radar frame lookup by monotonically
advancing a current_time pointer. Never uses frame indices directly.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Playback state
# ---------------------------------------------------------------------------

@dataclass
class PlaybackState:
    """Immutable-ish state for the playback engine."""
    current_time: float     # seconds (MTL domain)
    start_time:   float     # seconds (MTL start)
    end_time:     float     # seconds (MTL end)
    speed:        float = 1.0
    is_playing:   bool  = False
    last_tick_wall: float = field(default_factory=time.monotonic)

    def duration(self) -> float:
        return self.end_time - self.start_time

    def elapsed(self) -> float:
        return self.current_time - self.start_time

    def progress(self) -> float:
        """0.0 → 1.0"""
        d = self.duration()
        return self.elapsed() / d if d > 0 else 0.0

    def clamp(self):
        self.current_time = float(np.clip(
            self.current_time, self.start_time, self.end_time))


def make_playback_state(mtl: np.ndarray,
                         speed: float = 1.0) -> PlaybackState:
    return PlaybackState(
        current_time=float(mtl[0]),
        start_time=float(mtl[0]),
        end_time=float(mtl[-1]),
        speed=speed,
        is_playing=False,
        last_tick_wall=time.monotonic(),
    )


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------

def tick(state: PlaybackState, interval_s: float = 0.05) -> PlaybackState:
    """Advance playback by interval_s × speed real-time seconds."""
    if not state.is_playing:
        state.last_tick_wall = time.monotonic()
        return state
    now = time.monotonic()
    delta_real = now - state.last_tick_wall
    state.current_time += delta_real * state.speed
    state.last_tick_wall = now
    state.clamp()
    if state.current_time >= state.end_time:
        state.is_playing = False
        logger.info("Playback reached end of recording")
    return state


# ---------------------------------------------------------------------------
# Frame lookup
# ---------------------------------------------------------------------------

def get_frame_at_time(current_time: float,
                       aligned_table: pd.DataFrame) -> dict:
    """Binary search on MTL for nearest row to current_time.

    Returns dict with camera_frame_id, radar_msg_id, offset_ms, elapsed_s, etc.
    """
    times = aligned_table["mtl_time_s"].values
    idx = int(np.searchsorted(times, current_time, side="left"))
    idx = min(max(idx, 0), len(times) - 1)
    # Check neighbour
    if idx > 0:
        if abs(times[idx-1] - current_time) < abs(times[idx] - current_time):
            idx -= 1

    row = aligned_table.iloc[idx]
    return {
        "idx":             int(idx),
        "camera_frame_id": int(row["camera_frame_id"]),
        "radar_msg_id":    int(row["radar_msg_id"]),
        "mtl_time_s":      float(row["mtl_time_s"]),
        "offset_ms":       float(row.get("offset_ms", 0.0)),
        "elapsed_s":       float(row.get("elapsed_s", 0.0)),
        "camera_time_s":   float(row.get("camera_time_s", 0.0)),
        "radar_time_s":    float(row.get("radar_time_s", 0.0)),
    }


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

def play(state: PlaybackState) -> PlaybackState:
    state.is_playing = True
    state.last_tick_wall = time.monotonic()
    return state


def pause(state: PlaybackState) -> PlaybackState:
    state.is_playing = False
    return state


def jump_to(state: PlaybackState, target_time: float) -> PlaybackState:
    state.current_time = float(np.clip(target_time, state.start_time, state.end_time))
    state.last_tick_wall = time.monotonic()
    return state


def jump_to_progress(state: PlaybackState, progress: float) -> PlaybackState:
    """Jump to a fraction (0–1) of the recording duration."""
    target = state.start_time + progress * state.duration()
    return jump_to(state, target)


def step_forward(state: PlaybackState,
                  aligned_table: pd.DataFrame) -> PlaybackState:
    """Step exactly one MTL tick forward."""
    current = get_frame_at_time(state.current_time, aligned_table)
    idx = min(current["idx"] + 1, len(aligned_table) - 1)
    state.current_time = float(aligned_table["mtl_time_s"].iloc[idx])
    return state


def step_backward(state: PlaybackState,
                   aligned_table: pd.DataFrame) -> PlaybackState:
    """Step exactly one MTL tick backward."""
    current = get_frame_at_time(state.current_time, aligned_table)
    idx = max(current["idx"] - 1, 0)
    state.current_time = float(aligned_table["mtl_time_s"].iloc[idx])
    return state


def set_speed(state: PlaybackState, speed: float,
              min_speed: float = 0.1, max_speed: float = 4.0) -> PlaybackState:
    state.speed = float(np.clip(speed, min_speed, max_speed))
    return state


# ---------------------------------------------------------------------------
# LRU Frame Cache
# ---------------------------------------------------------------------------

class FrameCache:
    """Simple LRU cache for decoded frames (camera or radar)."""

    def __init__(self, max_size: int = 10):
        self._cache: collections.OrderedDict = collections.OrderedDict()
        self._max_size = max_size

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)
