"""
callbacks.py – All Dash callback definitions.

Architecture:
  - Global APP_STATE holds loaded data bundle, playback state, aligned table, metrics.
  - A single dcc.Interval drives the playback loop.
  - Callbacks are side-effectful (write to APP_STATE) but always return pure Dash outputs.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from dash import Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import (
    alignment_engine as ae,
    camera_renderer as cr,
    data_loader as dl,
    metadata_parser as mp_,
    metrics_engine as me,
    playback_engine as pe,
    radar_renderer as rr,
    validation as val,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global mutable application state (loaded once per session)
# ---------------------------------------------------------------------------

APP_STATE: dict = {
    "loaded": False,
    "bundle": None,
    "alignment": None,
    "metrics": None,
    "quality": "—",
    "meta_bundle": None,
    "validation_report": None,
    "playback": None,
    "cam_cache": pe.FrameCache(max_size=10),
    "rad_cache": pe.FrameCache(max_size=10),
    "config": None,
    "base_dir": None,
}

# Colours
C_GREEN  = "#3fb950"
C_ORANGE = "#d29922"
C_RED    = "#f85149"
C_MUTED  = "#8b949e"
C_TEXT   = "#c9d1d9"
C_BG     = "#0d1117"
C_SURFACE= "#161b22"
C_BORDER = "#30363d"


def quality_color(q: str) -> str:
    return {
        "PASS": C_GREEN, "WARNING": C_ORANGE, "FAIL": C_RED,
    }.get(q, C_MUTED)


def _load_config(base_dir: str) -> dict:
    cfg_path = os.path.join(base_dir, "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

_PAPER_BG = "#0d1117"
_PLOT_BG  = "#161b22"
_GRID_CLR = "#30363d"

def _dark_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor=_PAPER_BG, plot_bgcolor=_PLOT_BG,
        font=dict(color=C_TEXT, size=10),
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C_TEXT)),
        **kwargs,
    )


def build_offset_plot(aligned_table: pd.DataFrame) -> go.Figure:
    t = aligned_table["elapsed_s"].values
    o = aligned_table["offset_ms"].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=o, mode="lines",
                              name="Offset", line=dict(color="#58a6ff", width=1)))
    fig.add_hline(y=0, line_color=C_MUTED, line_dash="dash", line_width=0.8)
    fig.update_layout(**_dark_layout(
        title=dict(text="Camera–Radar Offset (ms)", font=dict(color=C_TEXT), x=0.01),
        xaxis=dict(title="Elapsed (s)", gridcolor=_GRID_CLR, color=C_TEXT),
        yaxis=dict(title="Offset (ms)", gridcolor=_GRID_CLR, color=C_TEXT),
        showlegend=False,
    ))
    return fig


def build_rolling_mean_plot(aligned_table: pd.DataFrame, metrics: dict) -> go.Figure:
    t = aligned_table["elapsed_s"].values
    rolling = metrics.get("_rolling_mean_offset_ms", [])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=aligned_table["offset_ms"].values,
                              mode="lines", name="Raw",
                              line=dict(color="#30363d", width=0.8)))
    if len(rolling) == len(t):
        fig.add_trace(go.Scatter(x=t, y=rolling, mode="lines",
                                  name="Rolling Mean",
                                  line=dict(color="#f0883e", width=2)))
    fig.add_hline(y=0, line_color=C_MUTED, line_dash="dash", line_width=0.8)
    fig.update_layout(**_dark_layout(
        title=dict(text="Rolling Mean Offset (ms)", font=dict(color=C_TEXT), x=0.01),
        xaxis=dict(title="Elapsed (s)", gridcolor=_GRID_CLR, color=C_TEXT),
        yaxis=dict(title="Offset (ms)", gridcolor=_GRID_CLR, color=C_TEXT),
    ))
    return fig


def build_timestamps_plot(aligned_table: pd.DataFrame) -> go.Figure:
    t = aligned_table["elapsed_s"].values
    fig = go.Figure()
    cam_t = aligned_table["camera_time_s"].values - aligned_table["camera_time_s"].values[0]
    rad_t = aligned_table["radar_time_s"].values  - aligned_table["radar_time_s"].values[0]
    fig.add_trace(go.Scatter(x=t, y=cam_t, mode="lines", name="Camera",
                              line=dict(color="#58a6ff", width=1.2)))
    fig.add_trace(go.Scatter(x=t, y=rad_t, mode="lines", name="Radar",
                              line=dict(color="#3fb950", width=1.2)))
    fig.update_layout(**_dark_layout(
        title=dict(text="Camera & Radar Timestamps vs MTL", font=dict(color=C_TEXT), x=0.01),
        xaxis=dict(title="Elapsed (s)", gridcolor=_GRID_CLR, color=C_TEXT),
        yaxis=dict(title="Relative Time (s)", gridcolor=_GRID_CLR, color=C_TEXT),
    ))
    return fig


def build_histogram_plot(aligned_table: pd.DataFrame) -> go.Figure:
    offsets = aligned_table["offset_ms"].values
    hist_data = me.compute_offset_histogram(offsets, bins=60)
    fig = go.Figure(go.Bar(
        x=hist_data["centers"], y=hist_data["counts"],
        marker_color="#58a6ff", marker_line_width=0,
        opacity=0.85,
    ))
    fig.update_layout(**_dark_layout(
        title=dict(text="Offset Distribution (ms)", font=dict(color=C_TEXT), x=0.01),
        xaxis=dict(title="Offset (ms)", gridcolor=_GRID_CLR, color=C_TEXT),
        yaxis=dict(title="Count", gridcolor=_GRID_CLR, color=C_TEXT),
        showlegend=False,
    ))
    return fig


def build_drift_plot(aligned_table: pd.DataFrame, drift: dict) -> go.Figure:
    t = aligned_table["elapsed_s"].values
    o = aligned_table["offset_ms"].values
    slope_per_s = drift["slope_ms_per_min"] / 60.0
    intercept   = drift["intercept_ms"]
    trend = intercept + slope_per_s * t

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=o, mode="lines", name="Offset",
                              line=dict(color="#58a6ff", width=0.8), opacity=0.6))
    fig.add_trace(go.Scatter(x=t, y=trend, mode="lines", name="Drift fit",
                              line=dict(color="#f85149", width=2, dash="dot")))
    slope_lbl = f"{drift['slope_ms_per_min']:.4f} ms/min"
    r2_lbl    = f"R²={drift['r_squared']:.4f}"
    fig.add_annotation(x=0.98, y=0.95, xref="paper", yref="paper",
                        text=f"Drift {slope_lbl}  {r2_lbl}",
                        font=dict(color=C_TEXT, size=10),
                        showarrow=False, align="right",
                        bgcolor="rgba(0,0,0,0.5)")
    fig.update_layout(**_dark_layout(
        title=dict(text="Drift Analysis (Linear Fit)", font=dict(color=C_TEXT), x=0.01),
        xaxis=dict(title="Elapsed (s)", gridcolor=_GRID_CLR, color=C_TEXT),
        yaxis=dict(title="Offset (ms)", gridcolor=_GRID_CLR, color=C_TEXT),
    ))
    return fig


# ---------------------------------------------------------------------------
# Register all callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app, base_dir: str):
    """Register all Dash callbacks on the app instance."""

    APP_STATE["base_dir"] = base_dir

    # ------------------------------------------------------------------
    # 1. Load Data
    # ------------------------------------------------------------------
    @app.callback(
        Output("status-dot",       "style"),
        Output("status-text",      "children"),
        Output("store-data-loaded","data"),
        Output("toast-notify",     "children"),
        Output("toast-notify",     "is_open"),
        Output("toast-notify",     "header"),
        Input("btn-load",          "n_clicks"),
        prevent_initial_call=True,
    )
    def load_data(n_clicks):
        if not n_clicks:
            raise PreventUpdate

        dot_loading = {"width": "8px", "height": "8px", "borderRadius": "50%",
                        "background": C_ORANGE, "marginRight": "6px", "marginTop": "2px"}
        dot_ok      = {**dot_loading, "background": C_GREEN}
        dot_err     = {**dot_loading, "background": C_RED}

        try:
            cfg = _load_config(base_dir)
            APP_STATE["config"] = cfg

            bundle = dl.load_all(base_dir, cfg)
            APP_STATE["bundle"] = bundle

            # Alignment
            alignment = ae.run_alignment(
                bundle["camera_timestamps"],
                bundle["radar_timestamps"],
                bundle["sync_df"],
                gap_multiplier=cfg.get("frame_drop", {}).get("gap_multiplier", 1.5),
            )
            APP_STATE["alignment"] = alignment

            # Metrics
            metrics = me.compute_sync_metrics(
                alignment["aligned_table"],
                alignment["drift"],
                alignment["cam_with_drops"],
                alignment["rad_with_drops"],
            )
            thresholds = cfg.get("quality", None)
            quality = me.evaluate_quality(metrics, thresholds)
            APP_STATE["metrics"]  = metrics
            APP_STATE["quality"]  = quality

            # Metadata
            session_info  = bundle["camera_metadata_db"]["session_info"]
            cam_session   = mp_.parse_camera_session(session_info)
            cam_perf      = mp_.parse_camera_performance(bundle["camera_performance"])
            radar_log_st  = mp_.parse_radar_log_stats(bundle["radar_log"])
            radar_hdf5_m  = mp_.parse_radar_hdf5_metadata(bundle["radar_hdf5"].metadata)
            meta_bundle   = mp_.build_metadata_bundle(
                cam_session, cam_perf, radar_log_st, radar_hdf5_m,
                bundle["radar_timestamps"], bundle["camera_timestamps"],
            )
            APP_STATE["meta_bundle"] = meta_bundle

            # Validation
            vrep = val.run_full_validation(bundle)
            APP_STATE["validation_report"] = vrep

            # Playback init
            mtl = alignment["mtl"]
            APP_STATE["playback"] = pe.make_playback_state(mtl, speed=1.0)

            APP_STATE["loaded"] = True

            n_cam = len(bundle["camera_timestamps"])
            n_rad = len(bundle["radar_timestamps"])
            msg = (f"Loaded: {n_cam} camera frames, {n_rad} radar frames. "
                   f"Sync quality: {quality}")
            return dot_ok, f"Loaded – {quality}", True, msg, True, "✅ Data Loaded"

        except Exception as exc:
            logger.exception("Data load failed")
            return dot_err, f"Error: {exc}", False, str(exc), True, "❌ Load Failed"

    # ------------------------------------------------------------------
    # 2. Mode toggle (Engineering ↔ Demo)
    # ------------------------------------------------------------------
    @app.callback(
        Output("engineering-view", "style"),
        Output("demo-view",        "style"),
        Output("store-mode",       "data"),
        Input("mode-toggle",       "value"),
    )
    def toggle_mode(is_demo):
        if is_demo:
            return {"display": "none"}, {"display": "block"}, "demo"
        return {"display": "block"}, {"display": "none"}, "engineering"

    # ------------------------------------------------------------------
    # 3. Playback interval – advance time
    # ------------------------------------------------------------------
    @app.callback(
        Output("playback-interval", "disabled"),
        Output("btn-play",          "children"),
        Input("btn-play",           "n_clicks"),
        State("store-data-loaded",  "data"),
        prevent_initial_call=True,
    )
    def toggle_play(n_clicks, loaded):
        if not loaded or not APP_STATE.get("playback"):
            raise PreventUpdate
        state = APP_STATE["playback"]
        if state.is_playing:
            APP_STATE["playback"] = pe.pause(state)
            return True, "▶"
        else:
            APP_STATE["playback"] = pe.play(state)
            return False, "⏸"

    @app.callback(
        Output("scrub-slider",  "value"),
        Output("time-readout",  "children"),
        # Camera outputs
        Output("camera-frame",     "src"),
        Output("cam-frame-id",     "children"),
        Output("cam-elapsed",      "children"),
        Output("cam-sensor-ts",    "children"),
        Output("cam-sync-ts",      "children"),
        Output("cam-delta",        "children"),
        Output("cam-fps",          "children"),
        Output("cam-resolution",   "children"),
        # Radar info outputs
        Output("rad-msg-id",       "children"),
        Output("rad-elapsed",      "children"),
        Output("rad-sensor-ts",    "children"),
        Output("rad-sync-ts",      "children"),
        Output("rad-delta",        "children"),
        Output("rad-num-targets",  "children"),
        Output("rad-hz",           "children"),
        # Radar graph
        Output("radar-graph",      "figure"),
        # Demo outputs
        Output("demo-camera-frame","src"),
        Output("demo-radar-graph", "figure"),
        Input("playback-interval", "n_intervals"),
        State("store-data-loaded", "data"),
        State("radar-view-selector","value"),
        prevent_initial_call=True,
    )
    def playback_tick(n_intervals, loaded, radar_view):
        if not loaded or not APP_STATE.get("playback"):
            raise PreventUpdate

        # Advance time
        state = APP_STATE["playback"]
        state = pe.tick(state)
        APP_STATE["playback"] = state

        bundle     = APP_STATE["bundle"]
        alignment  = APP_STATE["alignment"]
        cfg        = APP_STATE["config"] or {}
        eng_mode   = APP_STATE.get("mode", "engineering") != "demo"

        frame_info = pe.get_frame_at_time(state.current_time, alignment["aligned_table"])
        cam_id     = frame_info["camera_frame_id"]
        rad_id     = frame_info["radar_msg_id"]
        elapsed    = frame_info["elapsed_s"]
        offset     = frame_info["offset_ms"]

        # Camera
        cam_b64 = cr.render_camera_panel(
            bundle["camera_video"]["cap"], cam_id, frame_info,
            APP_STATE["cam_cache"], engineering_mode=eng_mode,
        )

        # Radar targets
        targets = rr.load_radar_frame(bundle["radar_hdf5"], rad_id, APP_STATE["rad_cache"])
        rad_meta = bundle["radar_hdf5"].get_frame_metadata(rad_id)
        num_tgt  = int(rad_meta.get("num_targets", len(targets)))

        radar_fig = rr.render_radar_panel(
            targets, view=radar_view or "birds_eye",
            msg_id=rad_id, sync_ts=state.current_time,
            config=cfg.get("radar", {}),
        )

        # FPS from performance log
        try:
            fps_val = f"{bundle['camera_video']['fps']:.1f}"
        except Exception:
            fps_val = "—"

        cam_meta = bundle["camera_metadata_db"]["session_info"]
        res = cam_meta.get("resolution", "—")

        duration = state.duration()
        time_str = f"{elapsed:>7.2f}s / {duration:.2f}s"

        # Radar hz from log stats
        rad_hz = "18.2"  # nominal

        return (
            state.progress(),
            time_str,
            # Camera
            cam_b64,
            str(cam_id),
            f"{elapsed:.3f} s",
            f"{frame_info['camera_time_s']:.6f}",
            f"{state.current_time:.6f}",
            f"{offset:.3f} ms",
            fps_val,
            res,
            # Radar
            str(rad_id),
            f"{elapsed:.3f} s",
            f"{frame_info['radar_time_s']:.6f}",
            f"{state.current_time:.6f}",
            f"{-offset:.3f} ms",
            str(num_tgt),
            rad_hz,
            radar_fig,
            # Demo
            cam_b64,
            radar_fig,
        )

    # ------------------------------------------------------------------
    # 4. Step forward / backward
    # ------------------------------------------------------------------
    @app.callback(
        Output("scrub-slider",  "value", allow_duplicate=True),
        Input("btn-fwd",  "n_clicks"),
        Input("btn-back", "n_clicks"),
        State("store-data-loaded", "data"),
        prevent_initial_call=True,
    )
    def step_clicks(fwd, back, loaded):
        if not loaded or not APP_STATE.get("playback"):
            raise PreventUpdate
        ctx = callback_context
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        state = APP_STATE["playback"]
        aligned = APP_STATE["alignment"]["aligned_table"]
        if btn == "btn-fwd":
            APP_STATE["playback"] = pe.step_forward(state, aligned)
        else:
            APP_STATE["playback"] = pe.step_backward(state, aligned)
        return APP_STATE["playback"].progress()

    # ------------------------------------------------------------------
    # 5. Scrub slider → jump to time
    # ------------------------------------------------------------------
    @app.callback(
        Output("time-readout", "children", allow_duplicate=True),
        Input("scrub-slider",  "value"),
        State("store-data-loaded", "data"),
        prevent_initial_call=True,
    )
    def scrub(progress, loaded):
        if not loaded or not APP_STATE.get("playback"):
            raise PreventUpdate
        state = APP_STATE["playback"]
        APP_STATE["playback"] = pe.jump_to_progress(state, float(progress))
        elapsed  = APP_STATE["playback"].elapsed()
        duration = APP_STATE["playback"].duration()
        return f"{elapsed:>7.2f}s / {duration:.2f}s"

    # ------------------------------------------------------------------
    # 6. Speed selector
    # ------------------------------------------------------------------
    @app.callback(
        Output("speed-dropdown", "value"),
        Input("speed-dropdown",  "value"),
        State("store-data-loaded", "data"),
        prevent_initial_call=True,
    )
    def set_speed(speed, loaded):
        if not loaded or not APP_STATE.get("playback"):
            raise PreventUpdate
        APP_STATE["playback"] = pe.set_speed(APP_STATE["playback"], float(speed))
        return speed

    # ------------------------------------------------------------------
    # 7. Sync Analysis tab – populate all plots
    # ------------------------------------------------------------------
    @app.callback(
        Output("plot-offset",     "figure"),
        Output("plot-rolling",    "figure"),
        Output("plot-timestamps", "figure"),
        Output("plot-histogram",  "figure"),
        Output("plot-drift",      "figure"),
        Output("quality-badge",   "children"),
        Output("quality-badge",   "style"),
        Output("metrics-summary-row", "children"),
        Input("main-tabs",        "value"),
        State("store-data-loaded","data"),
    )
    def update_sync_tab(tab, loaded):
        if tab != "sync" or not loaded or not APP_STATE.get("alignment"):
            raise PreventUpdate

        alignment = APP_STATE["alignment"]
        metrics   = APP_STATE["metrics"]
        quality   = APP_STATE["quality"]
        aligned   = alignment["aligned_table"]
        drift     = alignment["drift"]

        from dash import html
        q_color = quality_color(quality)
        badge_style = {
            "fontSize": "24px", "fontWeight": "800", "color": q_color,
            "marginRight": "20px",
        }

        # Mini-metrics
        def _kv(k, v):
            return html.Div([
                html.Div(k, style={"color": C_MUTED, "fontSize": "9px",
                                    "fontWeight": "700", "textTransform": "uppercase",
                                    "letterSpacing": "0.08em"}),
                html.Div(v, style={"color": C_TEXT, "fontSize": "14px",
                                    "fontWeight": "700", "fontFamily": "monospace"}),
            ], style={"marginRight": "24px"})

        summary_row = html.Div([
            _kv("Mean Offset",  f"{metrics.get('mean_offset_ms', 0):.2f} ms"),
            _kv("95th Pct",     f"{metrics.get('p95_offset_ms', 0):.2f} ms"),
            _kv("Drift",        f"{metrics.get('drift_slope_ms_per_min', 0):.4f} ms/min"),
            _kv("Jitter Std",   f"{metrics.get('jitter_std_ms', 0):.3f} ms"),
            _kv("Cam Drops",    f"{metrics.get('camera_drop_rate', 0)*100:.2f}%"),
            _kv("Rad Drops",    f"{metrics.get('radar_drop_rate', 0)*100:.2f}%"),
        ], style={"display": "flex", "flexWrap": "wrap"})

        return (
            build_offset_plot(aligned),
            build_rolling_mean_plot(aligned, metrics),
            build_timestamps_plot(aligned),
            build_histogram_plot(aligned),
            build_drift_plot(aligned, drift),
            quality,
            badge_style,
            summary_row,
        )

    # ------------------------------------------------------------------
    # 8. Demo quality badge
    # ------------------------------------------------------------------
    @app.callback(
        Output("demo-quality-badge", "children"),
        Output("demo-quality-badge", "style"),
        Output("demo-metrics-brief", "children"),
        Input("mode-toggle",         "value"),
        State("store-data-loaded",   "data"),
    )
    def update_demo_badge(is_demo, loaded):
        if not is_demo or not loaded or not APP_STATE.get("metrics"):
            raise PreventUpdate
        quality  = APP_STATE["quality"]
        metrics  = APP_STATE["metrics"]
        q_color  = quality_color(quality)
        badge_style = {
            "fontSize": "56px", "fontWeight": "900", "color": q_color,
            "textAlign": "center", "letterSpacing": "0.06em",
        }
        brief = (
            f"Mean Offset: {metrics.get('mean_offset_ms', 0):.2f} ms  ·  "
            f"95th Pct: {metrics.get('p95_offset_ms', 0):.2f} ms  ·  "
            f"Drift: {metrics.get('drift_slope_ms_per_min', 0):.4f} ms/min"
        )
        return quality, badge_style, brief

    # ------------------------------------------------------------------
    # 9. Metadata tab
    # ------------------------------------------------------------------
    @app.callback(
        Output("meta-camera-session",  "children"),
        Output("meta-radar-stats",     "children"),
        Output("meta-logs",            "children"),
        Output("meta-validation",      "children"),
        Input("main-tabs",             "value"),
        State("store-data-loaded",     "data"),
    )
    def update_metadata_tab(tab, loaded):
        if tab != "metadata" or not loaded or not APP_STATE.get("meta_bundle"):
            raise PreventUpdate

        from dash import html

        meta = APP_STATE["meta_bundle"]
        vrep = APP_STATE["validation_report"] or {}

        def _rows(d: dict) -> list:
            rows = []
            for k, v in d.items():
                if isinstance(v, dict):
                    continue
                rows.append(html.Div([
                    html.Span(str(k) + ": ", style={"color": C_MUTED,
                                                     "fontSize": "11px", "fontWeight": "600"}),
                    html.Span(str(v), style={"color": C_TEXT, "fontSize": "11px",
                                              "fontFamily": "monospace"}),
                ], style={"marginBottom": "4px"}))
            return rows

        cam_rows = _rows(meta["camera"]["session"])
        cam_rows += _rows(meta["camera"]["performance"])
        cam_rows.append(html.Div(f"Total Frames: {meta['camera']['total_frames']}",
                                   style={"color": C_TEXT, "fontSize": "11px"}))

        rad_log = meta["radar"]["log_stats"]
        rad_rows = [
            html.Div(f"Total Frames: {meta['radar']['total_frames']}",
                     style={"color": C_TEXT, "fontSize": "11px", "marginBottom": "4px"}),
            html.Div(f"Avg Targets/Frame: {meta['radar']['avg_targets_per_frame']}",
                     style={"color": C_TEXT, "fontSize": "11px", "marginBottom": "4px"}),
            html.Div(f"Total Targets: {meta['radar']['total_targets']}",
                     style={"color": C_TEXT, "fontSize": "11px", "marginBottom": "4px"}),
            html.Div(f"Mean Hz: {rad_log.get('hz', {}).get('mean', 0):.2f}",
                     style={"color": C_TEXT, "fontSize": "11px", "marginBottom": "4px"}),
            html.Div(f"Total Drops: {rad_log.get('total_drops', 0)}",
                     style={"color": C_TEXT, "fontSize": "11px", "marginBottom": "4px"}),
        ]

        log_rows = []
        for w in rad_log.get("warnings", [])[:5]:
            log_rows.append(html.Div(f"⚠ {w}", style={"color": C_ORANGE,
                                                          "fontSize": "10px",
                                                          "fontFamily": "monospace",
                                                          "marginBottom": "3px"}))
        for e in rad_log.get("errors", [])[:5]:
            log_rows.append(html.Div(f"✕ {e}", style={"color": C_RED,
                                                         "fontSize": "10px",
                                                         "fontFamily": "monospace",
                                                         "marginBottom": "3px"}))
        if not log_rows:
            log_rows = [html.Div("No warnings or errors.", style={"color": C_GREEN,
                                                                     "fontSize": "11px"})]

        n_err  = len(vrep.get("errors", []))
        n_warn = len(vrep.get("warnings", []))
        valid_colour = C_GREEN if vrep.get("is_valid") else C_RED
        val_rows = [
            html.Div(f"Status: {'✓ VALID' if vrep.get('is_valid') else '✗ ISSUES FOUND'}",
                     style={"color": valid_colour, "fontSize": "12px",
                            "fontWeight": "700", "marginBottom": "6px"}),
            html.Div(f"Errors: {n_err}   Warnings: {n_warn}",
                     style={"color": C_TEXT, "fontSize": "11px", "marginBottom": "6px"}),
        ]
        for issue in (vrep.get("errors", []) + vrep.get("warnings", []))[:10]:
            clr = C_RED if issue["level"] == "ERROR" else C_ORANGE
            val_rows.append(html.Div(f"[{issue['level']}] {issue['detail']}",
                                      style={"color": clr, "fontSize": "10px",
                                             "fontFamily": "monospace",
                                             "marginBottom": "2px"}))

        return cam_rows, rad_rows, log_rows, val_rows

    # ------------------------------------------------------------------
    # 10. Export callbacks
    # ------------------------------------------------------------------
    @app.callback(
        Output("export-status",  "children"),
        Output("download-file",  "data"),
        Input("btn-export-aligned","n_clicks"),
        Input("btn-export-json",   "n_clicks"),
        Input("btn-export-csv",    "n_clicks"),
        Input("btn-export-pdf",    "n_clicks"),
        State("store-data-loaded", "data"),
        prevent_initial_call=True,
    )
    def handle_export(n_aligned, n_json, n_csv, n_pdf, loaded):
        if not loaded:
            raise PreventUpdate

        from dash import dcc as _dcc
        ctx = callback_context
        btn = ctx.triggered[0]["prop_id"].split(".")[0]
        cfg = APP_STATE["config"] or {}
        alignment = APP_STATE["alignment"]
        metrics   = APP_STATE["metrics"]
        quality   = APP_STATE["quality"]

        exp_dir = os.path.join(APP_STATE["base_dir"],
                               cfg.get("exports", {}).get("dir", "exports"))
        os.makedirs(exp_dir, exist_ok=True)

        try:
            if btn == "btn-export-aligned":
                path = os.path.join(exp_dir, "aligned_frame_table.csv")
                alignment["aligned_table"].to_csv(path, index=False)
                return f"✓ Saved: {path}", _dcc.send_file(path)

            elif btn == "btn-export-json":
                path = os.path.join(exp_dir, "sync_metrics_summary.json")
                me.export_metrics_json(metrics, path)
                return f"✓ Saved: {path}", _dcc.send_file(path)

            elif btn == "btn-export-csv":
                path = os.path.join(exp_dir, "sync_metrics_summary.csv")
                me.export_metrics_csv(metrics, path)
                return f"✓ Saved: {path}", _dcc.send_file(path)

            elif btn == "btn-export-pdf":
                path = os.path.join(exp_dir, "sync_analysis_report.pdf")
                me.export_sync_report_pdf(metrics, alignment["aligned_table"], quality, path)
                return f"✓ Saved: {path}", _dcc.send_file(path)

        except Exception as e:
            logger.exception("Export failed")
            return f"✗ Export failed: {e}", no_update

        return "No action", no_update
