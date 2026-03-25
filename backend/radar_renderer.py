"""
radar_renderer.py – Radar point cloud visualisation using Plotly.

Supports four views:
  1. Bird's-eye scatter (x vs y)
  2. Range-Doppler heatmap
  3. Velocity vs range scatter
  4. Raw amplitude map (power/SNR)

All plots use uirevision to preserve zoom/pan across frame updates,
and are designed for smooth animation during playback.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Common theme
_PAPER_BG = "#0d1117"
_PLOT_BG  = "#161b22"
_GRID_CLR = "#21262d"
_TEXT_CLR = "#c9d1d9"
_MUTED    = "#8b949e"
_ACCENT   = "#58a6ff"
_GREEN    = "#3fb950"
_ORANGE   = "#f0883e"
_RED      = "#f85149"

_BASE_LAYOUT = dict(
    paper_bgcolor=_PAPER_BG,
    plot_bgcolor=_PLOT_BG,
    font=dict(color=_TEXT_CLR, size=11, family="Inter, system-ui, sans-serif"),
    margin=dict(l=55, r=20, t=44, b=50),
)


def load_radar_frame(radar_hdf5: Any,
                      msg_id: int,
                      cache: Any | None = None) -> pd.DataFrame:
    """Load point cloud targets for msg_id, with optional cache."""
    if cache is not None:
        cached = cache.get(msg_id)
        if cached is not None:
            return cached

    df = radar_hdf5.get_frame(int(msg_id))

    if cache is not None:
        cache.put(msg_id, df)

    return df


def _empty_figure(title: str) -> go.Figure:
    """Create an empty figure with a 'No Data' message."""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=f"<b>{title}</b><br><span style='color:{_MUTED};font-size:12px'>No targets in this frame</span>",
        font=dict(color=_TEXT_CLR, size=16),
        showarrow=False,
    )
    fig.update_layout(
        **_BASE_LAYOUT,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=420,
    )
    return fig


def _annotate_target_count(fig: go.Figure, n_targets: int):
    """Add a target count badge in the top-right corner."""
    fig.add_annotation(
        x=0.98, y=0.97, xref="paper", yref="paper",
        text=f"<b>{n_targets}</b> targets",
        font=dict(color=_GREEN if n_targets > 0 else _RED, size=11),
        showarrow=False,
        bgcolor="rgba(13,17,23,0.75)",
        bordercolor=_GRID_CLR,
        borderwidth=1,
        borderpad=5,
        align="right",
    )


# ---------------------------------------------------------------------------
# 1. Bird's-eye scatter (x vs y)
# ---------------------------------------------------------------------------

def render_birds_eye(targets: pd.DataFrame,
                      max_range: float = 150.0,
                      max_lateral: float = 50.0,
                      msg_id: int = 0,
                      sync_ts: float = 0.0) -> go.Figure:
    """Bird's-eye view: x (forward) vs y (lateral), color = RCS."""
    if targets.empty or "x" not in targets.columns:
        return _empty_figure("Bird's-Eye View")

    n = len(targets)
    color_col = "rcs" if "rcs" in targets.columns else None

    x_vals = targets["x"].values
    y_vals = targets["y"].values
    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))
    x_pad = max((x_max - x_min) * 0.15, 5.0)
    y_pad = max((y_max - y_min) * 0.15, 5.0)

    range_x = [max(-5, x_min - x_pad), min(max_range, x_max + x_pad)]
    range_y = [max(-max_lateral, y_min - y_pad), min(max_lateral, y_max + y_pad)]

    if "snr" in targets.columns:
        snr = targets["snr"].values
        snr_norm = np.clip((snr - snr.min()) / (snr.max() - snr.min() + 1e-9), 0, 1)
        sizes = 5 + snr_norm * 10
    else:
        sizes = np.full(n, 8)

    fig = go.Figure()

    # Range rings
    for r in [25, 50, 75, 100, 125]:
        if r <= max_range:
            theta = np.linspace(-np.pi / 3, np.pi / 3, 60)
            fig.add_trace(go.Scatter(
                x=r * np.cos(theta), y=r * np.sin(theta),
                mode="lines",
                line=dict(color=_GRID_CLR, width=0.5, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_annotation(
                x=r, y=0, text=f"{r}m",
                font=dict(color=_MUTED, size=8),
                showarrow=False, yshift=10,
            )

    # Ego vehicle marker
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=10, color=_GREEN, symbol="triangle-up"),
        text=["EGO"], textposition="bottom center",
        textfont=dict(color=_GREEN, size=8),
        hoverinfo="skip", showlegend=False,
    ))

    # Targets
    fig.add_trace(go.Scattergl(
        x=x_vals, y=y_vals,
        mode="markers",
        marker=dict(
            size=sizes,
            color=targets[color_col].values if color_col else _ACCENT,
            colorscale="Viridis",
            showscale=bool(color_col),
            colorbar=dict(
                title=dict(text="RCS (dBsm)", font=dict(color=_TEXT_CLR, size=10)),
                tickfont=dict(color=_TEXT_CLR, size=9),
                len=0.8, thickness=12,
                bgcolor="rgba(0,0,0,0)",
            ) if color_col else None,
            opacity=0.9,
            line=dict(width=0.5, color="rgba(255,255,255,0.3)"),
        ),
        hovertemplate=(
            "<b>Target</b><br>"
            "X: %{x:.1f} m<br>Y: %{y:.1f} m<br>"
            + (f"RCS: %{{marker.color:.1f}} dBsm<br>" if color_col else "")
            + "<extra></extra>"
        ),
        name="Targets",
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title="",
        xaxis=dict(
            title="X — Forward (m)", range=range_x,
            color=_TEXT_CLR, gridcolor=_GRID_CLR,
            zeroline=True, zerolinecolor=_GRID_CLR, zerolinewidth=1,
            dtick=25,
        ),
        yaxis=dict(
            title="Y — Lateral (m)", range=range_y,
            color=_TEXT_CLR, gridcolor=_GRID_CLR,
            zeroline=True, zerolinecolor=_GRID_CLR, zerolinewidth=1,
            scaleanchor="x", scaleratio=1,
            dtick=10,
        ),
        height=420,
        uirevision="birds_eye_view",
    )

    _annotate_target_count(fig, n)
    return fig


# ---------------------------------------------------------------------------
# 2. Range-Doppler heatmap
# ---------------------------------------------------------------------------

def render_range_doppler(targets: pd.DataFrame,
                          range_bins: int = 60,
                          doppler_bins: int = 40,
                          msg_id: int = 0) -> go.Figure:
    """Heatmap of detections in range x radial_speed space."""
    if targets.empty or "range" not in targets.columns or "radial_speed" not in targets.columns:
        return _empty_figure("Range-Doppler")

    n = len(targets)
    r = targets["range"].values
    v = targets["radial_speed"].values

    fig = go.Figure()

    fig.add_trace(go.Histogram2d(
        x=r, y=v,
        nbinsx=range_bins, nbinsy=doppler_bins,
        colorscale="Hot",
        reversescale=False,
        colorbar=dict(
            title=dict(text="Count", font=dict(color=_TEXT_CLR, size=10)),
            tickfont=dict(color=_TEXT_CLR, size=9),
            len=0.8, thickness=12,
        ),
        hovertemplate="Range: %{x:.1f} m<br>Speed: %{y:.2f} m/s<br>Count: %{z}<extra></extra>",
    ))

    fig.add_trace(go.Scattergl(
        x=r, y=v,
        mode="markers",
        marker=dict(size=4, color=_ACCENT, opacity=0.6,
                    line=dict(width=0.5, color="white")),
        hovertemplate="Range: %{x:.1f} m<br>Speed: %{y:.2f} m/s<extra></extra>",
        name="Detections",
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title="",
        xaxis=dict(title="Range (m)", color=_TEXT_CLR, gridcolor=_GRID_CLR,
                   zeroline=False),
        yaxis=dict(title="Radial Speed (m/s)", color=_TEXT_CLR, gridcolor=_GRID_CLR,
                   zeroline=True, zerolinecolor=_MUTED, zerolinewidth=1),
        height=420,
        uirevision="range_doppler_view",
    )

    _annotate_target_count(fig, n)
    return fig


# ---------------------------------------------------------------------------
# 3. Velocity vs range scatter
# ---------------------------------------------------------------------------

def render_velocity_range(targets: pd.DataFrame, msg_id: int = 0) -> go.Figure:
    """Scatter: range vs radial_speed, colored by SNR."""
    if targets.empty or "range" not in targets.columns or "radial_speed" not in targets.columns:
        return _empty_figure("Velocity vs Range")

    n = len(targets)
    color_col = "snr" if "snr" in targets.columns else None

    if "rcs" in targets.columns:
        rcs = targets["rcs"].values
        rcs_norm = np.clip((rcs - rcs.min()) / (rcs.max() - rcs.min() + 1e-9), 0, 1)
        sizes = 4 + rcs_norm * 10
    else:
        sizes = np.full(n, 6)

    fig = go.Figure()

    fig.add_hline(y=0, line_color=_MUTED, line_width=0.8, line_dash="dash")

    fig.add_annotation(
        x=0.02, y=0.95, xref="paper", yref="paper",
        text="Approaching", font=dict(color=_GREEN, size=9),
        showarrow=False, bgcolor="rgba(0,0,0,0.5)",
    )
    fig.add_annotation(
        x=0.02, y=0.05, xref="paper", yref="paper",
        text="Receding", font=dict(color=_RED, size=9),
        showarrow=False, bgcolor="rgba(0,0,0,0.5)",
    )

    fig.add_trace(go.Scattergl(
        x=targets["range"].values,
        y=targets["radial_speed"].values,
        mode="markers",
        marker=dict(
            size=sizes,
            color=targets[color_col].values if color_col else _ACCENT,
            colorscale="Plasma",
            showscale=bool(color_col),
            colorbar=dict(
                title=dict(text="SNR (dB)", font=dict(color=_TEXT_CLR, size=10)),
                tickfont=dict(color=_TEXT_CLR, size=9),
                len=0.8, thickness=12,
            ) if color_col else None,
            opacity=0.9,
            line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
        ),
        hovertemplate=(
            "<b>Target</b><br>"
            "Range: %{x:.1f} m<br>Speed: %{y:.2f} m/s<br>"
            + (f"SNR: %{{marker.color:.1f}} dB<br>" if color_col else "")
            + "<extra></extra>"
        ),
        name="Targets",
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title="",
        xaxis=dict(title="Range (m)", color=_TEXT_CLR, gridcolor=_GRID_CLR),
        yaxis=dict(title="Radial Speed (m/s)", color=_TEXT_CLR, gridcolor=_GRID_CLR,
                   zeroline=True, zerolinecolor=_MUTED),
        height=420,
        uirevision="velocity_range_view",
    )

    _annotate_target_count(fig, n)
    return fig


# ---------------------------------------------------------------------------
# 4. Raw amplitude map (power / SNR)
# ---------------------------------------------------------------------------

def render_amplitude_map(targets: pd.DataFrame, msg_id: int = 0) -> go.Figure:
    """Scatter: x vs y colored by power, sized by SNR."""
    if targets.empty or "x" not in targets.columns:
        return _empty_figure("Amplitude Map")

    n = len(targets)
    power = targets["power"].values if "power" in targets.columns else np.ones(n)
    snr   = targets["snr"].values   if "snr"   in targets.columns else np.ones(n)

    snr_range = snr.max() - snr.min() if snr.max() != snr.min() else 1.0
    size = np.clip((snr - snr.min()) / snr_range * 14 + 4, 4, 18)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers+text",
        marker=dict(size=10, color=_GREEN, symbol="triangle-up"),
        text=["EGO"], textposition="bottom center",
        textfont=dict(color=_GREEN, size=8),
        hoverinfo="skip", showlegend=False,
    ))

    fig.add_trace(go.Scattergl(
        x=targets["x"].values,
        y=targets["y"].values,
        mode="markers",
        marker=dict(
            size=size,
            color=power,
            colorscale="Inferno",
            showscale=True,
            colorbar=dict(
                title=dict(text="Power (dB)", font=dict(color=_TEXT_CLR, size=10)),
                tickfont=dict(color=_TEXT_CLR, size=9),
                len=0.8, thickness=12,
            ),
            opacity=0.9,
            line=dict(width=0.5, color="rgba(255,255,255,0.2)"),
        ),
        hovertemplate=(
            "<b>Target</b><br>"
            "X: %{x:.1f} m<br>Y: %{y:.1f} m<br>"
            "Power: %{marker.color:.1f} dB<extra></extra>"
        ),
        name="Targets",
    ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title="",
        xaxis=dict(title="X — Forward (m)", color=_TEXT_CLR, gridcolor=_GRID_CLR,
                   zeroline=True, zerolinecolor=_GRID_CLR),
        yaxis=dict(title="Y — Lateral (m)", color=_TEXT_CLR, gridcolor=_GRID_CLR,
                   zeroline=True, zerolinecolor=_GRID_CLR,
                   scaleanchor="x", scaleratio=1),
        height=420,
        uirevision="amplitude_map_view",
    )

    _annotate_target_count(fig, n)
    return fig


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

VIEW_OPTIONS = {
    "birds_eye":     "Bird's-Eye View",
    "range_doppler": "Range-Doppler",
    "velocity_range": "Velocity vs Range",
    "amplitude":     "Amplitude Map",
}


def render_radar_panel(targets: pd.DataFrame,
                        view: str = "birds_eye",
                        msg_id: int = 0,
                        sync_ts: float = 0.0,
                        config: dict | None = None) -> go.Figure:
    """Route to the correct radar renderer based on view name."""
    cfg = config or {}
    max_range   = cfg.get("max_range_m", 150.0)
    max_lateral = cfg.get("max_lateral_m", 50.0)

    if view == "birds_eye":
        return render_birds_eye(targets, max_range, max_lateral, msg_id, sync_ts)
    elif view == "range_doppler":
        return render_range_doppler(targets, range_bins=60, doppler_bins=40, msg_id=msg_id)
    elif view == "velocity_range":
        return render_velocity_range(targets, msg_id=msg_id)
    elif view == "amplitude":
        return render_amplitude_map(targets, msg_id=msg_id)
    else:
        logger.warning("Unknown radar view: %s", view)
        return _empty_figure(f"Unknown view: {view}")
