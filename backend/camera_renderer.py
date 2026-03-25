"""
camera_renderer.py – Camera frame extraction and overlay rendering.

Uses OpenCV for random-access frame seeking and overlay annotation.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Overlay appearance
OVERLAY_BG_COLOR   = (0, 0, 0)         # BGR
OVERLAY_TEXT_COLOR = (0, 255, 100)     # Bright green
OVERLAY_WARN_COLOR = (0, 140, 255)    # Orange
OVERLAY_GOOD_COLOR = (0, 200, 80)     # Muted green
FONT               = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE         = 0.55
FONT_THICKNESS     = 1
LINE_HEIGHT        = 22
MARGIN             = 10


def get_camera_frame(cap: cv2.VideoCapture,
                      frame_id: int) -> np.ndarray | None:
    """Seek to frame_id and read the frame. Returns BGR numpy array or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ret, frame = cap.read()
    if not ret or frame is None:
        logger.warning("Failed to read camera frame %d", frame_id)
        return None
    return frame


def render_overlay(frame: np.ndarray, meta: dict) -> np.ndarray:
    """Draw engineering overlay on frame (copy, non-destructive)."""
    out = frame.copy()
    h, w = out.shape[:2]

    offset_ms = meta.get('offset_ms', 0.0)
    abs_offset = abs(offset_ms)

    # Build line entries: (text, color)
    entries = [
        (f"Frame ID   : {meta.get('camera_frame_id', '?')}", OVERLAY_TEXT_COLOR),
        (f"Elapsed    : {meta.get('elapsed_s', 0.0):.3f} s", OVERLAY_TEXT_COLOR),
        (f"Cam Time   : {meta.get('camera_time_s', 0.0):.6f}", OVERLAY_TEXT_COLOR),
        (f"Sync Time  : {meta.get('mtl_time_s', 0.0):.6f}", OVERLAY_TEXT_COLOR),
    ]

    # Offset line with color coding
    if abs_offset > 20:
        offset_color = (0, 0, 255)      # Red for bad sync
    elif abs_offset > 10:
        offset_color = OVERLAY_WARN_COLOR  # Orange for warning
    else:
        offset_color = OVERLAY_GOOD_COLOR  # Green for good
    entries.append((f"Offset     : {offset_ms:+.3f} ms", offset_color))

    if "speed" in meta:
        entries.append((f"Speed      : {meta.get('speed', 0.0):.1f} km/h", OVERLAY_TEXT_COLOR))
    if "steering_deg" in meta:
        entries.append((f"Steering   : {meta.get('steering_deg', 0.0):.2f} deg", OVERLAY_TEXT_COLOR))

    # Panel dimensions
    panel_w = 280
    panel_h = LINE_HEIGHT * len(entries) + MARGIN * 2

    # Semi-transparent background
    overlay = out.copy()
    cv2.rectangle(overlay, (MARGIN, MARGIN),
                  (MARGIN + panel_w, MARGIN + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    # Draw border
    cv2.rectangle(out, (MARGIN, MARGIN),
                  (MARGIN + panel_w, MARGIN + panel_h),
                  (50, 50, 60), 1)

    for i, (text, color) in enumerate(entries):
        y = MARGIN * 2 + i * LINE_HEIGHT + 2
        cv2.putText(out, text, (MARGIN + 8, y), FONT,
                    FONT_SCALE, color, FONT_THICKNESS, cv2.LINE_AA)

    return out


def frame_to_base64(frame: np.ndarray,
                     quality: int = 75,
                     max_width: int = 960) -> str:
    """Encode BGR frame as base64 JPEG string for Dash HTML image display."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))

    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ret, buf = cv2.imencode(".jpg", frame, encode_param)
    if not ret:
        logger.error("JPEG encoding failed")
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def get_placeholder_frame(width: int = 960, height: int = 540) -> np.ndarray:
    """Generate a dark placeholder frame when no video is loaded."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 20, 30)
    cv2.putText(frame, "No Camera Feed", (width // 2 - 120, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 100), 2, cv2.LINE_AA)
    return frame


def render_camera_panel(cap: cv2.VideoCapture | None,
                         frame_id: int,
                         meta: dict,
                         cache: Any | None = None,
                         engineering_mode: bool = True) -> str:
    """High-level: get frame, render overlay, return base64 string."""
    # Check cache
    if cache is not None:
        cached = cache.get(frame_id)
        if cached is not None:
            return cached

    if cap is None:
        frame = get_placeholder_frame()
    else:
        frame = get_camera_frame(cap, frame_id)
        if frame is None:
            frame = get_placeholder_frame()

    if engineering_mode:
        frame = render_overlay(frame, meta)

    result = frame_to_base64(frame)

    if cache is not None:
        cache.put(frame_id, result)

    return result
