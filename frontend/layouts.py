"""
layouts.py – Dash application layout.

Two modes:
  Engineering Mode – all raw metrics, timestamps, plots visible
  Demo Mode        – clean UI showing only Sync Quality badge
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

# ---------------------------------------------------------------------------
# Theme colours (dark)
# ---------------------------------------------------------------------------

C_BG       = "#0d1117"
C_SURFACE  = "#161b22"
C_BORDER   = "#30363d"
C_TEXT     = "#c9d1d9"
C_MUTED    = "#8b949e"
C_ACCENT   = "#58a6ff"
C_GREEN    = "#3fb950"
C_ORANGE   = "#d29922"
C_RED      = "#f85149"

STYLE_CARD = {
    "background": C_SURFACE,
    "border": f"1px solid {C_BORDER}",
    "borderRadius": "8px",
    "padding": "12px",
    "marginBottom": "10px",
}

STYLE_LABEL = {
    "color": C_MUTED,
    "fontSize": "10px",
    "fontWeight": "600",
    "textTransform": "uppercase",
    "letterSpacing": "0.08em",
    "marginBottom": "2px",
}

STYLE_VALUE = {
    "color": C_TEXT,
    "fontSize": "13px",
    "fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
}


# ---------------------------------------------------------------------------
# Reusable components
# ---------------------------------------------------------------------------

def stat_card(label: str, value_id: str, unit: str = "") -> html.Div:
    return html.Div([
        html.Div(label, style=STYLE_LABEL),
        html.Div([
            html.Span(id=value_id, style={**STYLE_VALUE, "fontSize": "16px",
                                          "fontWeight": "700"}),
            html.Span(f" {unit}", style={**STYLE_VALUE, "color": C_MUTED}) if unit else None,
        ]),
    ], style={**STYLE_CARD, "minWidth": "130px", "flex": "1"})


def section_header(title: str) -> html.Div:
    return html.Div([
        html.H5(title, style={"color": C_ACCENT, "margin": "0 0 8px 0",
                               "fontSize": "13px", "fontWeight": "700",
                               "textTransform": "uppercase", "letterSpacing": "0.1em"}),
        html.Hr(style={"borderColor": C_BORDER, "margin": "0 0 10px 0"}),
    ])


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------

def build_topbar() -> html.Div:
    return html.Div([
        html.Div([
            html.Div("⬡", style={"fontSize": "24px", "color": C_ACCENT,
                                   "marginRight": "10px"}),
            html.Div([
                html.H4("RADAR–CAMERA SYNC DASHBOARD",
                        style={"margin": 0, "color": C_TEXT, "fontSize": "15px",
                               "fontWeight": "800", "letterSpacing": "0.15em"}),
                html.Div("Deterministic Sensor Synchronisation & Validation Platform",
                         style={"color": C_MUTED, "fontSize": "10px",
                                "letterSpacing": "0.06em"}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),

        html.Div([
            # Mode toggle
            html.Div([
                html.Span("Engineering", style={"color": C_MUTED, "fontSize": "11px",
                                                "marginRight": "6px"}),
                dbc.Switch(id="mode-toggle", value=False,
                           style={"marginBottom": "0"},
                           label="Demo", label_style={"color": C_MUTED,
                                                       "fontSize": "11px"}),
            ], style={"display": "flex", "alignItems": "center", "marginRight": "20px"}),

            # Load data button
            dbc.Button("⏏ Load Data", id="btn-load", size="sm",
                       style={"backgroundColor": "#21262d",
                              "borderColor": C_BORDER, "color": C_TEXT,
                              "marginRight": "8px", "fontSize": "11px"}),

            # Status dot
            html.Div([
                html.Div(id="status-dot", style={
                    "width": "8px", "height": "8px", "borderRadius": "50%",
                    "background": C_BORDER, "marginRight": "6px",
                    "marginTop": "2px",
                }),
                html.Span(id="status-text", children="No data loaded",
                          style={"color": C_MUTED, "fontSize": "11px"}),
            ], style={"display": "flex", "alignItems": "flex-start"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "background": C_SURFACE,
        "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px",
        "padding": "12px 18px",
        "marginBottom": "10px",
    })


# ---------------------------------------------------------------------------
# Playback controls
# ---------------------------------------------------------------------------

def build_playback_bar() -> html.Div:
    btn_style = {
        "backgroundColor": "#21262d",
        "borderColor": C_BORDER,
        "color": C_TEXT,
        "fontSize": "11px",
        "padding": "4px 10px",
    }
    return html.Div([
        html.Div([
            # Transport
            dbc.Button("⏮", id="btn-back",  size="sm", style=btn_style),
            dbc.Button("▶", id="btn-play",  size="sm",
                       style={**btn_style, "width": "56px"}),
            dbc.Button("⏭", id="btn-fwd",   size="sm", style=btn_style),
        ], style={"display": "flex", "gap": "4px", "marginRight": "16px"}),

        # Scrub slider
        html.Div([
            dcc.Slider(id="scrub-slider", min=0, max=1, step=0.0001, value=0,
                       marks=None, tooltip={"always_visible": False},
                       className="scrub-slider"),
        ], style={"flex": "1", "marginRight": "16px", "alignSelf": "center"}),

        # Time readout
        html.Div([
            html.Span(id="time-readout",
                      style={"color": C_TEXT, "fontFamily": "monospace",
                             "fontSize": "12px", "whiteSpace": "nowrap"}),
        ], style={"marginRight": "16px"}),

        # Speed
        html.Div([
            html.Span("Speed:", style={"color": C_MUTED, "fontSize": "11px",
                                        "marginRight": "6px"}),
            dcc.Dropdown(
                id="speed-dropdown",
                options=[
                    {"label": "0.1×", "value": 0.1},
                    {"label": "0.25×", "value": 0.25},
                    {"label": "0.5×", "value": 0.5},
                    {"label": "1×",   "value": 1.0},
                    {"label": "2×",   "value": 2.0},
                    {"label": "4×",   "value": 4.0},
                ],
                value=1.0,
                clearable=False,
                style={"width": "80px", "fontSize": "11px",
                       "backgroundColor": "#21262d",
                       "color": C_TEXT, "border": f"1px solid {C_BORDER}"},
            ),
        ], style={"display": "flex", "alignItems": "center"}),

    ], style={
        "display": "flex",
        "alignItems": "center",
        "background": C_SURFACE,
        "border": f"1px solid {C_BORDER}",
        "borderRadius": "8px",
        "padding": "8px 14px",
        "marginBottom": "10px",
    })


# ---------------------------------------------------------------------------
# Camera tab
# ---------------------------------------------------------------------------

def build_camera_tab() -> dcc.Tab:
    return dcc.Tab(label="Camera", value="camera", style=_tab_style(),
                   selected_style=_tab_selected_style(), children=[
        html.Div([
            # Video panel
            html.Div([
                html.Img(id="camera-frame",
                         style={"width": "100%", "borderRadius": "6px",
                                "border": f"1px solid {C_BORDER}",
                                "background": "#000"}),
            ], style={"flex": "1", "marginRight": "12px", "minWidth": "0"}),

            # Info panel
            html.Div([
                section_header("Camera Info"),
                html.Div(id="camera-info-panel", children=[
                    _info_row("Frame ID",   "cam-frame-id"),
                    _info_row("Elapsed",    "cam-elapsed"),
                    _info_row("Sensor ts",  "cam-sensor-ts"),
                    _info_row("Sync ts",    "cam-sync-ts"),
                    _info_row("Δ Radar",    "cam-delta"),
                    _info_row("FPS",        "cam-fps"),
                    _info_row("Resolution", "cam-resolution"),
                ]),
            ], style={**STYLE_CARD, "width": "230px", "flexShrink": "0"}),
        ], style={"display": "flex", "flexDirection": "row", "alignItems": "flex-start"}),
    ])


# ---------------------------------------------------------------------------
# Radar tab
# ---------------------------------------------------------------------------

def build_radar_tab() -> dcc.Tab:
    return dcc.Tab(label="Radar", value="radar", style=_tab_style(),
                   selected_style=_tab_selected_style(), children=[
        html.Div([
            # Left: plot
            html.Div([
                html.Div([
                    dcc.RadioItems(
                        id="radar-view-selector",
                        options=[
                            {"label": "Bird's-Eye", "value": "birds_eye"},
                            {"label": "Range-Doppler", "value": "range_doppler"},
                            {"label": "Velocity/Range", "value": "velocity_range"},
                            {"label": "Amplitude", "value": "amplitude"},
                        ],
                        value="birds_eye",
                        inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "16px", "fontSize": "12px",
                                    "color": C_TEXT, "cursor": "pointer"},
                        style={"marginBottom": "8px"},
                    ),
                ]),
                dcc.Graph(id="radar-graph", style={"height": "420px"},
                          config={"displayModeBar": True,
                                  "displaylogo": False,
                                  "modeBarButtonsToRemove": ["select2d", "lasso2d"]}),
            ], style={"flex": "1", "marginRight": "12px", "minWidth": "0"}),

            # Right: info panel
            html.Div([
                section_header("Radar Info"),
                html.Div(id="radar-info-panel", children=[
                    _info_row("Msg ID",     "rad-msg-id"),
                    _info_row("Elapsed",    "rad-elapsed"),
                    _info_row("Radar ts",   "rad-sensor-ts"),
                    _info_row("Sync ts",    "rad-sync-ts"),
                    _info_row("Δ Camera",   "rad-delta"),
                    _info_row("Targets",    "rad-num-targets"),
                    _info_row("Hz",         "rad-hz"),
                ]),
            ], style={**STYLE_CARD, "width": "230px", "flexShrink": "0"}),
        ], style={"display": "flex", "flexDirection": "row", "alignItems": "flex-start"}),
    ])


# ---------------------------------------------------------------------------
# Sync Analysis tab
# ---------------------------------------------------------------------------

def build_sync_tab() -> dcc.Tab:
    graph_cfg = {"displayModeBar": False, "displaylogo": False}
    g_style   = {"height": "180px"}
    return dcc.Tab(label="Sync Analysis", value="sync", style=_tab_style(),
                   selected_style=_tab_selected_style(), children=[
        html.Div([
            # Top: quality badge + key numbers
            html.Div(id="quality-banner", children=[
                html.Div(id="quality-badge", children="—",
                         style={"fontSize": "24px", "fontWeight": "800",
                                "color": C_TEXT, "marginRight": "20px"}),
                html.Div(id="metrics-summary-row"),
            ], style={**STYLE_CARD, "display": "flex", "alignItems": "center",
                      "marginBottom": "10px"}),

            # Plots grid
            html.Div([
                html.Div([
                    dcc.Graph(id="plot-offset", style=g_style, config=graph_cfg),
                ], style={"flex": "1", "marginRight": "8px"}),
                html.Div([
                    dcc.Graph(id="plot-rolling", style=g_style, config=graph_cfg),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "marginBottom": "8px"}),

            html.Div([
                html.Div([
                    dcc.Graph(id="plot-timestamps", style=g_style, config=graph_cfg),
                ], style={"flex": "1", "marginRight": "8px"}),
                html.Div([
                    dcc.Graph(id="plot-histogram", style=g_style, config=graph_cfg),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "marginBottom": "8px"}),

            html.Div([
                dcc.Graph(id="plot-drift", style=g_style, config=graph_cfg),
            ]),
        ]),
    ])


# ---------------------------------------------------------------------------
# Metadata tab
# ---------------------------------------------------------------------------

def build_metadata_tab() -> dcc.Tab:
    return dcc.Tab(label="Metadata", value="metadata", style=_tab_style(),
                   selected_style=_tab_selected_style(), children=[
        html.Div([
            html.Div([
                section_header("Camera Session"),
                html.Div(id="meta-camera-session"),
            ], style={**STYLE_CARD, "flex": "1", "marginRight": "8px"}),

            html.Div([
                section_header("Radar Statistics"),
                html.Div(id="meta-radar-stats"),
            ], style={**STYLE_CARD, "flex": "1", "marginRight": "8px"}),

            html.Div([
                section_header("Performance & Logs"),
                html.Div(id="meta-logs"),
            ], style={**STYLE_CARD, "flex": "1"}),
        ], style={"display": "flex"}),

        html.Div([
            section_header("Validation Report"),
            html.Div(id="meta-validation"),
        ], style=STYLE_CARD),
    ])


# ---------------------------------------------------------------------------
# Export tab
# ---------------------------------------------------------------------------

def build_export_tab() -> dcc.Tab:
    btn_style = {
        "width": "100%", "marginBottom": "8px",
        "backgroundColor": "#21262d",
        "borderColor": C_BORDER, "color": C_TEXT, "fontSize": "12px",
    }
    return dcc.Tab(label="Export", value="export", style=_tab_style(),
                   selected_style=_tab_selected_style(), children=[
        html.Div([
            html.Div([
                section_header("Export Data"),
                dbc.Button("⬇ Aligned Frame Table (CSV)",
                           id="btn-export-aligned", style=btn_style, size="sm"),
                dbc.Button("⬇ Metrics Summary (JSON)",
                           id="btn-export-json", style=btn_style, size="sm"),
                dbc.Button("⬇ Metrics Summary (CSV)",
                           id="btn-export-csv", style=btn_style, size="sm"),
                dbc.Button("⬇ Sync Analysis Report (PDF)",
                           id="btn-export-pdf", style=btn_style, size="sm"),
                html.Hr(style={"borderColor": C_BORDER}),
                dbc.Button("⬇ Synchronized Video (MP4)",
                           id="btn-export-video", style=btn_style, size="sm",
                           disabled=True),
                html.Small("Video export is CPU-intensive and runs asynchronously.",
                           style={"color": C_MUTED}),
            ], style={**STYLE_CARD, "maxWidth": "360px"}),

            html.Div([
                section_header("Export Status"),
                html.Div(id="export-status",
                         style={"color": C_TEXT, "fontSize": "12px",
                                "fontFamily": "monospace", "whiteSpace": "pre"}),
                dcc.Download(id="download-file"),
            ], style={**STYLE_CARD, "flex": "1", "marginLeft": "12px"}),
        ], style={"display": "flex"}),
    ])


# ---------------------------------------------------------------------------
# Demo mode overlay
# ---------------------------------------------------------------------------

def build_demo_overlay() -> html.Div:
    """Minimal demo view – shown instead of tabs when demo mode is on."""
    return html.Div([
        html.Div([
            html.Div("SYNC QUALITY", style={
                "color": C_MUTED, "fontSize": "11px", "fontWeight": "700",
                "letterSpacing": "0.12em", "marginBottom": "12px",
            }),
            html.Div(id="demo-quality-badge", children="—", style={
                "fontSize": "56px", "fontWeight": "900",
                "color": C_TEXT, "textAlign": "center",
                "letterSpacing": "0.06em",
            }),
            html.Div(id="demo-metrics-brief", style={
                "color": C_MUTED, "fontSize": "13px",
                "textAlign": "center", "marginTop": "12px",
            }),
        ], style={
            "display": "flex", "flexDirection": "column",
            "alignItems": "center", "justifyContent": "center",
            "padding": "40px",
            **STYLE_CARD,
        }),

        html.Div([
            html.Div([
                html.Img(id="demo-camera-frame",
                         style={"width": "50%", "borderRadius": "6px",
                                "border": f"1px solid {C_BORDER}"}),
                dcc.Graph(id="demo-radar-graph",
                          style={"width": "50%", "height": "360px"},
                          config={"displayModeBar": False}),
            ], style={"display": "flex", "gap": "10px", "marginTop": "10px"}),
        ]),
    ])


# ---------------------------------------------------------------------------
# Root layout
# ---------------------------------------------------------------------------

def build_layout() -> html.Div:
    return html.Div([
        # Interval for playback ticks
        dcc.Interval(id="playback-interval", interval=50, n_intervals=0, disabled=True),

        # Hidden state stores
        dcc.Store(id="store-data-loaded",    data=False),
        dcc.Store(id="store-playback",       data={}),
        dcc.Store(id="store-aligned-table",  data=None),
        dcc.Store(id="store-metrics",        data=None),
        dcc.Store(id="store-mode",           data="engineering"),  # engineering | demo

        # Notification toast
        dbc.Toast(id="toast-notify", header="Notice", is_open=False, duration=4000,
                  style={"position": "fixed", "top": 12, "right": 12,
                         "zIndex": 9999, "background": C_SURFACE,
                         "color": C_TEXT, "border": f"1px solid {C_BORDER}"}),

        # Top bar
        build_topbar(),

        # Playback bar
        build_playback_bar(),

        # Engineering view (tabs)
        html.Div(id="engineering-view", children=[
            dcc.Tabs(id="main-tabs", value="camera",
                     style={"borderBottom": f"1px solid {C_BORDER}"},
                     children=[
                         build_camera_tab(),
                         build_radar_tab(),
                         build_sync_tab(),
                         build_metadata_tab(),
                         build_export_tab(),
                     ]),
        ], style={"display": "block"}),

        # Demo view
        html.Div(id="demo-view",
                 children=[build_demo_overlay()],
                 style={"display": "none"}),

    ], style={
        "background": C_BG,
        "minHeight": "100vh",
        "padding": "12px 16px",
        "fontFamily": "'Inter', 'Segoe UI', system-ui, sans-serif",
        "color": C_TEXT,
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tab_style() -> dict:
    return {
        "backgroundColor": C_BG,
        "color": C_MUTED,
        "border": "none",
        "borderBottom": f"2px solid transparent",
        "fontSize": "12px",
        "fontWeight": "600",
        "padding": "8px 16px",
        "letterSpacing": "0.05em",
    }


def _tab_selected_style() -> dict:
    return {
        **_tab_style(),
        "color": C_ACCENT,
        "borderBottom": f"2px solid {C_ACCENT}",
        "backgroundColor": C_BG,
    }


def _info_row(label: str, value_id: str) -> html.Div:
    return html.Div([
        html.Span(label + ":", style={**STYLE_LABEL, "display": "inline",
                                       "marginRight": "6px"}),
        html.Span(id=value_id, children="—",
                  style={**STYLE_VALUE, "fontSize": "12px"}),
    ], style={"marginBottom": "6px"})
