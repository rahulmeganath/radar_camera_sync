"""
app.py – Dash application entry point.

Usage:
    python frontend/app.py
    # or from project root:
    python -m frontend.app

Then open: http://localhost:8050
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

# Always resolve from this file's location
_HERE     = Path(__file__).resolve().parent
BASE_DIR  = _HERE.parent   # sync_dashboard/

sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sync_dashboard")

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    cfg_path = BASE_DIR / "config.yaml"
    if not cfg_path.exists():
        logger.warning("config.yaml not found at %s", cfg_path)
        return {}
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

external_stylesheets = [
    dbc.themes.DARKLY,
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap",
]

app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,
    title="Radar–Camera Sync Dashboard",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description",
         "content": "Deterministic radar–camera synchronisation and validation platform"},
    ],
)
server = app.server   # expose Flask server for production WSGI servers

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

from frontend.layouts import build_layout  # noqa: E402

app.layout = build_layout()

# ---------------------------------------------------------------------------
# Custom CSS (inline, avoids extra file)
# ---------------------------------------------------------------------------

app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      * { box-sizing: border-box; }
      body { background: #0d1117; margin: 0; }
      .scrub-slider .rc-slider-rail       { background: #30363d; }
      .scrub-slider .rc-slider-track      { background: #58a6ff; }
      .scrub-slider .rc-slider-handle     { border-color: #58a6ff; background: #58a6ff; }
      .Select-control                     { background: #21262d !important; border-color: #30363d !important; color: #c9d1d9 !important; }
      .Select-menu-outer                  { background: #21262d !important; }
      .Select-option                      { color: #c9d1d9 !important; }
      .Select-option:hover                { background: #30363d !important; }
      .Select-value-label                 { color: #c9d1d9 !important; }
      .dash-tab                           { cursor: pointer; }
      /* Scrollbar */
      ::-webkit-scrollbar                 { width: 6px; }
      ::-webkit-scrollbar-track           { background: #0d1117; }
      ::-webkit-scrollbar-thumb           { background: #30363d; border-radius: 3px; }
      ::-webkit-scrollbar-thumb:hover     { background: #58a6ff; }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

# ---------------------------------------------------------------------------
# Register callbacks
# ---------------------------------------------------------------------------

from frontend.callbacks import register_callbacks  # noqa: E402

register_callbacks(app, str(BASE_DIR))

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = _load_config()
    host  = cfg.get("server", {}).get("host",  "0.0.0.0")
    port  = cfg.get("server", {}).get("port",  8050)
    debug = cfg.get("server", {}).get("debug", False)

    logger.info("Starting Radar–Camera Sync Dashboard on http://%s:%s", host, port)
    logger.info("Base directory: %s", BASE_DIR)
    app.run(host=host, port=port, debug=debug)
