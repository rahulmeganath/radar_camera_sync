"""
metrics_engine.py – Synchronisation quality metrics.

Computes statistical metrics from the aligned frame table and
determines PASS / WARNING / FAIL quality status.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_sync_metrics(aligned_table: pd.DataFrame,
                          drift_result: dict,
                          cam_drop_df: pd.DataFrame | None = None,
                          rad_drop_df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compute all synchronisation quality metrics."""
    offsets = aligned_table["offset_ms"].dropna().values

    if len(offsets) == 0:
        logger.error("Empty offset array – cannot compute metrics")
        return {}

    # Delta of offsets (jitter)
    jitter = np.diff(offsets) if len(offsets) > 1 else np.array([0.0])

    # Frame drop rates
    cam_drop_rate = 0.0
    rad_drop_rate = 0.0
    if cam_drop_df is not None and "is_drop" in cam_drop_df.columns:
        cam_drop_rate = float(cam_drop_df["is_drop"].mean())
    if rad_drop_df is not None and "is_drop" in rad_drop_df.columns:
        rad_drop_rate = float(rad_drop_df["is_drop"].mean())

    # Rolling stats
    offset_series = pd.Series(offsets)
    window = min(50, max(1, len(offsets) // 20))
    rolling_mean = offset_series.rolling(window, center=True).mean().bfill().ffill()

    metrics = {
        # Basic
        "mean_offset_ms":      float(np.mean(offsets)),
        "median_offset_ms":    float(np.median(offsets)),
        "std_offset_ms":       float(np.std(offsets)),
        "max_abs_offset_ms":   float(np.max(np.abs(offsets))),
        "min_offset_ms":       float(np.min(offsets)),
        "max_offset_ms":       float(np.max(offsets)),
        "p95_offset_ms":       float(np.percentile(np.abs(offsets), 95)),
        "p99_offset_ms":       float(np.percentile(np.abs(offsets), 99)),

        # Drift
        "drift_slope_ms_per_min": float(drift_result.get("slope_ms_per_min", 0.0)),
        "drift_intercept_ms":     float(drift_result.get("intercept_ms", 0.0)),
        "drift_r_squared":        float(drift_result.get("r_squared", 0.0)),

        # Jitter
        "jitter_std_ms":       float(np.std(jitter)),
        "jitter_max_ms":       float(np.max(np.abs(jitter))) if len(jitter) > 0 else 0.0,

        # Frame drops
        "camera_drop_rate":    cam_drop_rate,
        "radar_drop_rate":     rad_drop_rate,

        # Coverage
        "total_frames":        int(len(aligned_table)),
        "unique_camera_frames": int(aligned_table["camera_frame_id"].nunique()),
        "unique_radar_frames": int(aligned_table["radar_msg_id"].nunique()),

        # Rolling mean array (for plotting)
        "_rolling_mean_offset_ms": rolling_mean.tolist(),
    }

    logger.info(
        "Metrics: mean=%.2f ms, p95=%.2f ms, drift=%.4f ms/min, jitter_std=%.3f ms",
        metrics["mean_offset_ms"], metrics["p95_offset_ms"],
        metrics["drift_slope_ms_per_min"], metrics["jitter_std_ms"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Frame-relative quality helpers
# ---------------------------------------------------------------------------

# Application tiers with frame-period multiplier thresholds
# p95 < good_mult × frame_period → GOOD
# good_mult–warn_mult → WARNING
# > warn_mult → BAD
DEFAULT_TIERS: dict[str, dict[str, float]] = {
    "demo":     {"good_mult": 0.50, "warn_mult": 1.00, "label": "Demo / Visualisation"},
    "fusion":   {"good_mult": 0.25, "warn_mult": 0.50, "label": "Perception Fusion"},
    "tracking": {"good_mult": 0.10, "warn_mult": 0.25, "label": "High-Precision Tracking"},
}

DEFAULT_CAMERA_FPS = 30.0
DEFAULT_DRIFT_THRESHOLD_MS_PER_MIN = 0.5  # any drift above this → WARNING


def compute_frame_relative_metrics(metrics: dict,
                                    camera_fps: float = DEFAULT_CAMERA_FPS,
                                    ) -> dict[str, Any]:
    """Compute frame-relative quality information from raw metrics.

    Returns a dict with:
      - frame_period_ms: ms per frame at the given FPS
      - p95_frames: p95 expressed as fraction of frame period
      - max_abs_frames: max |offset| as fraction of frame period
      - std_frames: std dev as fraction of frame period
      - tier_results: per-tier GOOD/WARNING/BAD ratings
      - overall: overall quality label (PASS/WARNING/FAIL)
      - drift_ok: whether drift is acceptable
    """
    if not metrics:
        return {
            "frame_period_ms": 1000.0 / camera_fps,
            "p95_frames": float("inf"),
            "max_abs_frames": float("inf"),
            "std_frames": float("inf"),
            "tier_results": {},
            "overall": "FAIL",
            "drift_ok": False,
        }

    fp = 1000.0 / camera_fps  # frame period in ms

    p95  = metrics.get("p95_offset_ms", 999.0)
    maxo = metrics.get("max_abs_offset_ms", 999.0)
    std  = metrics.get("std_offset_ms", 999.0)
    drift = abs(metrics.get("drift_slope_ms_per_min", 999.0))

    p95_frames  = p95  / fp
    max_frames  = maxo / fp
    std_frames  = std  / fp
    drift_ok    = drift <= DEFAULT_DRIFT_THRESHOLD_MS_PER_MIN

    tier_results: dict[str, dict] = {}
    for tier_name, tier_cfg in DEFAULT_TIERS.items():
        gm = tier_cfg["good_mult"]
        wm = tier_cfg["warn_mult"]
        if p95_frames < gm:
            rating = "GOOD"
        elif p95_frames < wm:
            rating = "WARNING"
        else:
            rating = "BAD"
        # Drift override: non-negligible drift → at best WARNING
        if not drift_ok and rating == "GOOD":
            rating = "WARNING"
        tier_results[tier_name] = {
            "rating": rating,
            "label": tier_cfg["label"],
            "good_threshold_ms": gm * fp,
            "warn_threshold_ms": wm * fp,
        }

    return {
        "frame_period_ms": fp,
        "p95_frames": p95_frames,
        "max_abs_frames": max_frames,
        "std_frames": std_frames,
        "tier_results": tier_results,
        "overall": _overall_from_tiers(tier_results, drift_ok),
        "drift_ok": drift_ok,
    }


def _overall_from_tiers(tier_results: dict, drift_ok: bool) -> str:
    """Derive a single PASS/WARNING/FAIL from per-tier ratings.

    Logic:
      - Fusion tier GOOD and drift OK → PASS
      - Fusion tier WARNING or drift issue → WARNING
      - Fusion tier BAD → FAIL
    The fusion tier is the primary reference for overall quality.
    """
    fusion = tier_results.get("fusion", {}).get("rating", "BAD")
    if fusion == "GOOD" and drift_ok:
        return "PASS"
    elif fusion == "BAD":
        return "FAIL"
    else:
        return "WARNING"


# ---------------------------------------------------------------------------
# Quality evaluation (public API — backward-compatible signature)
# ---------------------------------------------------------------------------

def evaluate_quality(metrics: dict,
                     thresholds: dict | None = None,
                     camera_fps: float | None = None) -> str:
    """Return 'PASS', 'WARNING', or 'FAIL' using frame-relative thresholds.

    Parameters
    ----------
    metrics : dict
        Output of compute_sync_metrics().
    thresholds : dict | None
        Config dict.  Accepts ``{"camera_fps": 30}`` or legacy format.
        If None, defaults are used.
    camera_fps : float | None
        Override camera FPS.  Wins over thresholds["camera_fps"].
    """
    if not metrics:
        return "FAIL"

    # Resolve FPS
    fps = camera_fps or DEFAULT_CAMERA_FPS
    if thresholds and "camera_fps" in thresholds:
        fps = float(thresholds["camera_fps"])
    if camera_fps is not None:
        fps = camera_fps  # explicit arg always wins

    fr = compute_frame_relative_metrics(metrics, camera_fps=fps)
    return fr["overall"]


# ---------------------------------------------------------------------------
# Offset histogram
# ---------------------------------------------------------------------------

def compute_offset_histogram(offsets: np.ndarray, bins: int = 50) -> dict:
    """Return histogram data for plotting."""
    counts, edges = np.histogram(offsets, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    return {"counts": counts.tolist(), "centers": centers.tolist(), "edges": edges.tolist()}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_metrics_json(metrics: dict, path: str):
    """Export metrics to JSON (excluding private _ keys)."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    exportable = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(exportable, f, indent=2)
    logger.info("Metrics JSON saved: %s", path)


def export_metrics_csv(metrics: dict, path: str):
    """Export metrics to CSV (single-row)."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    exportable = {k: v for k, v in metrics.items() if not k.startswith("_")}
    pd.DataFrame([exportable]).to_csv(path, index=False)
    logger.info("Metrics CSV saved: %s", path)


def export_sync_report_pdf(metrics: dict,
                            aligned_table: pd.DataFrame,
                            quality_label: str,
                            path: str):
    """Generate a PDF sync analysis report using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    doc = SimpleDocTemplate(path, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                  fontSize=20, spaceAfter=12)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                               spaceAfter=6, spaceBefore=12)
    body_style = styles["BodyText"]

    quality_color = {
        "PASS": colors.green,
        "WARNING": colors.orange,
        "FAIL": colors.red,
    }.get(quality_label, colors.black)

    quality_style = ParagraphStyle("Quality", parent=styles["BodyText"],
                                    fontSize=16, textColor=quality_color,
                                    fontName="Helvetica-Bold")

    story = []
    story.append(Paragraph("Radar–Camera Synchronisation Report", title_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Sync Quality: {quality_label}", quality_style))
    story.append(Spacer(1, 0.5*cm))

    # Metrics table
    story.append(Paragraph("Synchronisation Metrics", h2_style))
    metric_rows = [["Metric", "Value"]]
    metric_display = [
        ("Mean Offset",         f"{metrics.get('mean_offset_ms', 0):.3f} ms"),
        ("Median Offset",       f"{metrics.get('median_offset_ms', 0):.3f} ms"),
        ("Std Dev Offset",      f"{metrics.get('std_offset_ms', 0):.3f} ms"),
        ("Max Abs Offset",      f"{metrics.get('max_abs_offset_ms', 0):.3f} ms"),
        ("95th Percentile",     f"{metrics.get('p95_offset_ms', 0):.3f} ms"),
        ("99th Percentile",     f"{metrics.get('p99_offset_ms', 0):.3f} ms"),
        ("Drift Slope",         f"{metrics.get('drift_slope_ms_per_min', 0):.4f} ms/min"),
        ("Drift R²",            f"{metrics.get('drift_r_squared', 0):.4f}"),
        ("Jitter Std Dev",      f"{metrics.get('jitter_std_ms', 0):.3f} ms"),
        ("Camera Drop Rate",    f"{metrics.get('camera_drop_rate', 0)*100:.2f}%"),
        ("Radar Drop Rate",     f"{metrics.get('radar_drop_rate', 0)*100:.2f}%"),
        ("Total MTL Frames",    str(metrics.get("total_frames", 0))),
        ("Unique Camera Frames",str(metrics.get("unique_camera_frames", 0))),
        ("Unique Radar Frames", str(metrics.get("unique_radar_frames", 0))),
    ]
    metric_rows.extend(metric_display)

    t = Table(metric_rows, colWidths=[10*cm, 6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 11),
        ("ALIGN",      (0, 0), (-1, -1), "LEFT"),
        ("ALIGN",      (1, 0), (1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # Aligned frame sample
    story.append(Paragraph("Aligned Frame Sample (first 20 rows)", h2_style))
    sample = aligned_table.head(20)
    sample_cols = ["elapsed_s", "camera_frame_id", "radar_msg_id", "offset_ms"]
    sample_cols = [c for c in sample_cols if c in sample.columns]
    sample_data = [sample_cols] + sample[sample_cols].round(4).values.tolist()
    sample_table = Table(sample_data, colWidths=[4*cm]*len(sample_cols))
    sample_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ]))
    story.append(sample_table)

    doc.build(story)
    logger.info("PDF report saved: %s", path)
