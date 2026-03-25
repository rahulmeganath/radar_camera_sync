"""
Radar–Camera Synchronisation Dashboard
PyQt5 desktop application.

Run:
    python app_pyqt.py
"""

from __future__ import annotations

import sys
import time
import json
import base64
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from PyQt5.QtCore import Qt, QTimer, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTableView,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QButtonGroup,
    QScrollArea,
    QSizePolicy,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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


BASE_DIR = Path(__file__).resolve().parent


class PandasTableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df

    def rowCount(self, parent=QModelIndex()):
        return len(self.df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self.df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        v = self.df.iloc[index.row(), index.column()]
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self.df.columns[section])
        return str(self.df.index[section])


class MplCanvas(FigureCanvas):
    def __init__(self, height=3.0):
        self.fig = Figure(figsize=(6, height), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


def _safe_str(v, digits=3):
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


class SyncDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar-Camera Sync Dashboard (PyQt5)")
        self.resize(1800, 1040)

        self.loaded = False
        self.bundle = None
        self.alignment = None
        self.metrics = None
        self.quality = "-"
        self.frame_rel = None
        self.meta_bundle = None
        self.vrep = None
        self.playback_state = None
        self.radar_limits = None
        self.cam_cache = pe.FrameCache(10)
        self.rad_cache = pe.FrameCache(10)

        self._build_ui()
        self._apply_theme()

        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._on_tick)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        self.sidebar = self._build_sidebar()
        splitter.addWidget(self.sidebar)

        main = QWidget()
        splitter.addWidget(main)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1440])

        main_layout = QVBoxLayout(main)

        self.header_title = QLabel("Radar-Camera Synchronisation Dashboard")
        self.header_subtitle = QLabel("Deterministic Sensor Synchronisation and Validation Platform")
        self.header_title.setObjectName("HeaderTitle")
        self.header_subtitle.setObjectName("HeaderSub")
        main_layout.addWidget(self.header_title)
        main_layout.addWidget(self.header_subtitle)

        self.mode_stack = QStackedWidget()
        main_layout.addWidget(self.mode_stack, 1)

        self.engineering_tabs = self._build_engineering_tabs()
        self.demo_page = self._build_demo_page()
        self.mode_stack.addWidget(self.engineering_tabs)
        self.mode_stack.addWidget(self.demo_page)

    def _build_sidebar(self):
        side = QFrame()
        side.setObjectName("Sidebar")
        lay = QVBoxLayout(side)

        title = QLabel("Sync Dashboard")
        title.setObjectName("SidebarTitle")
        lay.addWidget(title)

        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(self.load_data)
        lay.addWidget(self.btn_load)

        mode_box = QGroupBox("Mode")
        mode_l = QHBoxLayout(mode_box)
        self.mode_eng = QRadioButton("Engineering")
        self.mode_demo = QRadioButton("Demo")
        self.mode_eng.setChecked(True)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.mode_eng)
        self.mode_group.addButton(self.mode_demo)
        self.mode_eng.toggled.connect(self._on_mode_changed)
        mode_l.addWidget(self.mode_eng)
        mode_l.addWidget(self.mode_demo)
        lay.addWidget(mode_box)

        pb_box = QGroupBox("Playback")
        pb_l = QVBoxLayout(pb_box)

        self.lbl_time = QLabel("0.00s / 0.00s")
        self.lbl_time.setObjectName("Monospace")
        pb_l.addWidget(self.lbl_time)

        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setRange(0, 10000)
        self.slider_timeline.valueChanged.connect(self._on_scrub)
        pb_l.addWidget(self.slider_timeline)

        tr = QHBoxLayout()
        self.btn_back5 = QPushButton("-5")
        self.btn_prev = QPushButton("Prev")
        self.btn_play = QPushButton("Play")
        self.btn_next = QPushButton("Next")
        self.btn_fwd5 = QPushButton("+5")
        self.btn_back5.clicked.connect(lambda: self._step(-5))
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_next.clicked.connect(lambda: self._step(1))
        self.btn_fwd5.clicked.connect(lambda: self._step(5))
        for b in [self.btn_back5, self.btn_prev, self.btn_play, self.btn_next, self.btn_fwd5]:
            tr.addWidget(b)
        pb_l.addLayout(tr)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Speed"))
        self.cmb_speed = QComboBox()
        self.cmb_speed.addItems(["0.1x", "0.25x", "0.5x", "1.0x", "2.0x", "4.0x"])
        self.cmb_speed.setCurrentText("1.0x")
        self.cmb_speed.currentTextChanged.connect(self._on_speed_changed)
        speed_row.addWidget(self.cmb_speed)
        pb_l.addLayout(speed_row)

        self.lbl_sync = QLabel("Current Sync: -")
        self.lbl_sync.setObjectName("Monospace")
        pb_l.addWidget(self.lbl_sync)
        lay.addWidget(pb_box)

        view_box = QGroupBox("Radar View")
        view_l = QVBoxLayout(view_box)
        self.cmb_radar_view = QComboBox()
        self.radar_view_keys = ["birds_eye", "range_doppler", "velocity_range", "amplitude"]
        self.cmb_radar_view.addItems(["Bird's-Eye", "Range-Doppler", "Velocity-Range", "Amplitude"])
        self.cmb_radar_view.currentIndexChanged.connect(self._update_dynamic_views)
        view_l.addWidget(self.cmb_radar_view)
        lay.addWidget(view_box)

        lay.addStretch(1)
        return side

    def _build_engineering_tabs(self):
        tabs = QTabWidget()
        tabs.addTab(self._build_tab_dashboard(), "Dashboard")
        tabs.addTab(self._build_tab_sync_analysis(), "Sync Analysis")
        tabs.addTab(self._build_tab_metadata(), "Metadata")
        tabs.addTab(self._build_tab_export(), "Export")
        tabs.addTab(self._build_tab_help(), "Help")
        return tabs

    def _build_tab_dashboard(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        self.banner = QLabel("Quality: -")
        self.banner.setObjectName("QualityBanner")
        lay.addWidget(self.banner)

        self.dashboard_splitter = QSplitter(Qt.Horizontal)
        self.dashboard_splitter.setChildrenCollapsible(False)
        lay.addWidget(self.dashboard_splitter, 1)

        cam_panel = QWidget()
        cam_l = QVBoxLayout(cam_panel)
        self.lbl_cam_title = QLabel("Camera")
        self.lbl_camera = QLabel("Load data to start")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setMinimumHeight(320)
        self.lbl_camera.setObjectName("ImagePane")
        self.lbl_camera.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_cam_stats = QLabel("Frame: - | Elapsed: - | Offset: -")
        self.lbl_cam_stats.setObjectName("Monospace")
        cam_l.addWidget(self.lbl_cam_title)
        cam_l.addWidget(self.lbl_camera, 1)
        cam_l.addWidget(self.lbl_cam_stats)
        self.dashboard_splitter.addWidget(cam_panel)

        rad_panel = QWidget()
        rad_l = QVBoxLayout(rad_panel)
        self.lbl_rad_title = QLabel("Radar")
        self.canvas_radar = MplCanvas(height=4.6)
        self.lbl_rad_stats = QLabel("Msg: - | Targets: - | Offset: -")
        self.lbl_rad_stats.setObjectName("Monospace")
        rad_l.addWidget(self.lbl_rad_title)
        rad_l.addWidget(self.canvas_radar, 1)
        rad_l.addWidget(self.lbl_rad_stats)
        self.dashboard_splitter.addWidget(rad_panel)
        self.dashboard_splitter.setStretchFactor(0, 1)
        self.dashboard_splitter.setStretchFactor(1, 1)
        self.dashboard_splitter.setSizes([700, 700])

        self.canvas_live = MplCanvas(height=1.8)
        lay.addWidget(self.canvas_live)

        return page

    def _build_tab_sync_analysis(self):
        page = QWidget()
        out = QVBoxLayout(page)

        self.metrics_box = QGroupBox("Synchronisation Metrics")
        self.metrics_grid = QGridLayout(self.metrics_box)
        out.addWidget(self.metrics_box)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self.sync_plot_layout = QGridLayout(inner)

        self.sync_canvases = [MplCanvas(height=2.6) for _ in range(7)]
        self.sync_plot_layout.addWidget(self.sync_canvases[0], 0, 0)
        self.sync_plot_layout.addWidget(self.sync_canvases[1], 0, 1)
        self.sync_plot_layout.addWidget(self.sync_canvases[2], 1, 0)
        self.sync_plot_layout.addWidget(self.sync_canvases[3], 1, 1)
        self.sync_plot_layout.addWidget(self.sync_canvases[4], 2, 0)
        self.sync_plot_layout.addWidget(self.sync_canvases[5], 2, 1)
        self.sync_plot_layout.addWidget(self.sync_canvases[6], 3, 0, 1, 2)

        scroll.setWidget(inner)
        out.addWidget(scroll, 1)
        return page

    def _build_tab_metadata(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        cols = QSplitter(Qt.Horizontal)
        lay.addWidget(cols, 1)

        self.txt_camera_meta = QTextEdit()
        self.txt_camera_meta.setReadOnly(True)
        self.txt_camera_meta.setPlaceholderText("Camera session and performance metadata")
        cols.addWidget(self.txt_camera_meta)

        self.txt_radar_meta = QTextEdit()
        self.txt_radar_meta.setReadOnly(True)
        self.txt_radar_meta.setPlaceholderText("Radar statistics and metadata")
        cols.addWidget(self.txt_radar_meta)

        self.list_issues = QListWidget()
        cols.addWidget(self.list_issues)

        self.lbl_validation = QLabel("Validation: -")
        self.lbl_validation.setObjectName("Monospace")
        lay.addWidget(self.lbl_validation)

        return page

    def _build_tab_export(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        row = QHBoxLayout()
        self.btn_exp_csv = QPushButton("Download Aligned CSV")
        self.btn_exp_json = QPushButton("Download Metrics JSON")
        self.btn_exp_mcsv = QPushButton("Download Metrics CSV")
        self.btn_exp_pdf = QPushButton("Generate PDF")
        self.btn_exp_all = QPushButton("Save All to exports/")

        self.btn_exp_csv.clicked.connect(self._export_csv)
        self.btn_exp_json.clicked.connect(self._export_json)
        self.btn_exp_mcsv.clicked.connect(self._export_metrics_csv)
        self.btn_exp_pdf.clicked.connect(self._export_pdf)
        self.btn_exp_all.clicked.connect(self._export_all)

        for b in [self.btn_exp_csv, self.btn_exp_json, self.btn_exp_mcsv, self.btn_exp_pdf, self.btn_exp_all]:
            row.addWidget(b)
        lay.addLayout(row)

        self.table_preview = QTableView()
        lay.addWidget(self.table_preview, 1)

        return page

    def _build_tab_help(self):
        page = QWidget()
        lay = QVBoxLayout(page)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setHtml(self._help_html())
        lay.addWidget(txt)
        return page

    def _build_demo_page(self):
        page = QWidget()
        lay = QVBoxLayout(page)

        self.lbl_demo_quality = QLabel("Synchronisation Quality: -")
        self.lbl_demo_quality.setObjectName("DemoQuality")
        self.lbl_demo_quality.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.lbl_demo_quality)

        self.demo_splitter = QSplitter(Qt.Horizontal)
        self.demo_splitter.setChildrenCollapsible(False)
        lay.addWidget(self.demo_splitter, 1)

        self.lbl_demo_camera = QLabel("Load data to start")
        self.lbl_demo_camera.setAlignment(Qt.AlignCenter)
        self.lbl_demo_camera.setObjectName("ImagePane")
        self.lbl_demo_camera.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.demo_splitter.addWidget(self.lbl_demo_camera)

        self.canvas_demo_radar = MplCanvas(height=4.6)
        self.demo_splitter.addWidget(self.canvas_demo_radar)
        self.demo_splitter.setStretchFactor(0, 1)
        self.demo_splitter.setStretchFactor(1, 1)
        self.demo_splitter.setSizes([700, 700])

        self.canvas_demo_offset = MplCanvas(height=2.0)
        lay.addWidget(self.canvas_demo_offset)

        return page

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0d1117;
                color: #c9d1d9;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 12px;
            }
            #Sidebar {
                background: #111827;
                border-right: 1px solid #1f2937;
            }
            #SidebarTitle {
                font-size: 18px;
                font-weight: 700;
                color: #58a6ff;
                padding: 6px 0;
            }
            #HeaderTitle {
                font-size: 18px;
                font-weight: 700;
                color: #58a6ff;
            }
            #HeaderSub {
                font-size: 11px;
                color: #8b949e;
                margin-bottom: 6px;
            }
            #QualityBanner {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                font-weight: 700;
            }
            #DemoQuality {
                border: 1px solid #30363d;
                border-radius: 10px;
                padding: 14px;
                font-size: 20px;
                font-weight: 800;
            }
            #Monospace {
                font-family: Consolas, Menlo, monospace;
                color: #c9d1d9;
            }
            #ImagePane {
                background: #161b22;
                border: 1px solid #30363d;
                border-radius: 8px;
            }
            QGroupBox {
                border: 1px solid #30363d;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: 600;
            }
            QGroupBox::title {
                left: 8px;
                padding: 0 4px;
                color: #58a6ff;
            }
            QPushButton {
                background: #21262d;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 6px 10px;
            }
            QPushButton:hover {
                border-color: #58a6ff;
                color: #58a6ff;
            }
            QTabWidget::pane {
                border: 1px solid #30363d;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #161b22;
                color: #8b949e;
                padding: 8px 14px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                color: #58a6ff;
                border-bottom: 2px solid #58a6ff;
            }
            """
        )

    def _cfg(self) -> dict:
        with open(BASE_DIR / "config.yaml") as f:
            return yaml.safe_load(f)

    def _compute_radar_limits(self, radar_hdf5) -> dict:
        """Compute robust fixed plot limits from full radar dataset."""
        f = getattr(radar_hdf5, "_f", None)
        if f is None or "radar" not in f:
            return {
                "lateral_xlim": (-20.0, 20.0),
                "forward_ylim": (0.0, 22.0),
                "range_xlim": (0.0, 22.0),
                "speed_ylim": (-18.0, 14.0),
            }

        rg = f["radar"]

        def robust_bounds(name: str, p_low: float = 1.0, p_high: float = 99.0):
            if name not in rg:
                return None
            arr = np.asarray(rg[name][:], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None
            lo = float(np.percentile(arr, p_low))
            hi = float(np.percentile(arr, p_high))
            return lo, hi

        bx = robust_bounds("x")
        by = robust_bounds("y")
        br = robust_bounds("range")
        bv = robust_bounds("radial_speed")

        if bx is None or by is None or br is None or bv is None:
            return {
                "lateral_xlim": (-20.0, 20.0),
                "forward_ylim": (0.0, 22.0),
                "range_xlim": (0.0, 22.0),
                "speed_ylim": (-18.0, 14.0),
            }

        # Always include the ego origin (0, 0) in view.
        forward_min = min(0.0, bx[0] - 1.0)
        forward_max = bx[1] + 1.0
        lat_abs = max(abs(by[0]), abs(by[1])) * 1.08
        range_max = br[1] + 1.0
        speed_abs = max(abs(bv[0]), abs(bv[1])) * 1.15

        return {
            "lateral_xlim": (-lat_abs, lat_abs),
            "forward_ylim": (forward_min, forward_max),
            "range_xlim": (0.0, range_max),
            "speed_ylim": (-speed_abs, speed_abs),
        }

    def _sync_color(self, quality: str) -> str:
        return {
            "PASS": "#3fb950",
            "WARNING": "#d29922",
            "FAIL": "#f85149",
        }.get(quality, "#8b949e")

    def load_data(self):
        try:
            cfg = self._cfg()
            bundle = dl.load_all(str(BASE_DIR), cfg)

            alignment = ae.run_alignment(
                bundle["camera_timestamps"],
                bundle["radar_timestamps"],
                bundle["sync_df"],
                gap_multiplier=cfg.get("frame_drop", {}).get("gap_multiplier", 1.5),
            )

            metrics = me.compute_sync_metrics(
                alignment["aligned_table"],
                alignment["drift"],
                alignment["cam_with_drops"],
                alignment["rad_with_drops"],
            )
            thresholds = cfg.get("quality") or {}
            quality = me.evaluate_quality(metrics, thresholds)
            frame_rel = me.compute_frame_relative_metrics(
                metrics,
                camera_fps=float(thresholds.get("camera_fps", me.DEFAULT_CAMERA_FPS)),
            )

            session_info = bundle["camera_metadata_db"]["session_info"]
            cam_session = mp_.parse_camera_session(session_info)
            cam_perf = mp_.parse_camera_performance(bundle["camera_performance"])
            rad_log_st = mp_.parse_radar_log_stats(bundle["radar_log"])
            rad_hdf5_m = mp_.parse_radar_hdf5_metadata(bundle["radar_hdf5"].metadata)
            meta_bundle = mp_.build_metadata_bundle(
                cam_session,
                cam_perf,
                rad_log_st,
                rad_hdf5_m,
                bundle["radar_timestamps"],
                bundle["camera_timestamps"],
            )
            vrep = val.run_full_validation(bundle)

            self.loaded = True
            self.bundle = bundle
            self.alignment = alignment
            self.metrics = metrics
            self.quality = quality
            self.frame_rel = frame_rel
            self.meta_bundle = meta_bundle
            self.vrep = vrep
            self.radar_limits = self._compute_radar_limits(bundle["radar_hdf5"])
            self.playback_state = pe.make_playback_state(alignment["mtl"], speed=1.0)

            self._update_all_views()
            QMessageBox.information(self, "Data Loaded", "Radar and camera data loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", str(e))

    def _on_mode_changed(self):
        self.mode_stack.setCurrentIndex(0 if self.mode_eng.isChecked() else 1)
        self._update_dynamic_views()

    def _on_speed_changed(self, txt: str):
        if not self.playback_state:
            return
        speed = float(txt.replace("x", ""))
        self.playback_state = pe.set_speed(self.playback_state, speed)

    def _toggle_play(self):
        if not self.playback_state:
            return
        if self.playback_state.is_playing:
            self.playback_state = pe.pause(self.playback_state)
            self.btn_play.setText("Play")
            self.timer.stop()
        else:
            self.playback_state = pe.play(self.playback_state)
            self.btn_play.setText("Pause")
            self.timer.start()

    def _on_tick(self):
        if not self.playback_state:
            return
        self.playback_state = pe.tick(self.playback_state)
        self._update_dynamic_views()
        if not self.playback_state.is_playing:
            self.btn_play.setText("Play")
            self.timer.stop()

    def _step(self, n: int):
        if not self.loaded:
            return
        table = self.alignment["aligned_table"]
        info = pe.get_frame_at_time(self.playback_state.current_time, table)
        idx = int(np.clip(info["idx"] + n, 0, len(table) - 1))
        t = float(table["mtl_time_s"].iloc[idx])
        self.playback_state = pe.jump_to(self.playback_state, t)
        self._update_dynamic_views()

    def _on_scrub(self, value: int):
        if not self.playback_state:
            return
        if self.playback_state.duration() <= 0:
            return
        progress = value / 10000.0
        self.playback_state = pe.jump_to_progress(self.playback_state, progress)
        self._update_dynamic_views()

    def _get_frame_info(self):
        if not self.loaded:
            return None
        return pe.get_frame_at_time(self.playback_state.current_time, self.alignment["aligned_table"])

    def _update_all_views(self):
        self._update_static_views()
        self._update_dynamic_views()

    def _update_static_views(self):
        self._update_sync_metrics_grid()
        self._plot_sync_analysis()
        self._update_metadata_tab()
        self._update_export_preview()

    def _update_dynamic_views(self):
        if not self.loaded:
            return
        info = self._get_frame_info()
        self._update_sidebar_status(info)
        self._update_dashboard_tab(info)
        self._update_demo_page(info)

    def _update_sidebar_status(self, info: dict):
        dur = self.playback_state.duration()
        el = self.playback_state.elapsed()
        self.lbl_time.setText(f"{el:7.2f}s / {dur:.2f}s")

        self.slider_timeline.blockSignals(True)
        self.slider_timeline.setValue(int(self.playback_state.progress() * 10000))
        self.slider_timeline.blockSignals(False)

        self.lbl_sync.setText(f"Current Sync: {info['offset_ms']:+.2f} ms")

    def _banner_text(self):
        m = self.metrics or {}
        fr = self.frame_rel or {}
        p95_frames = fr.get("p95_frames")
        p95_suffix = f" ({p95_frames:.3f}xfp)" if p95_frames is not None else ""
        return (
            f"Quality: {self.quality} | Mean: {m.get('mean_offset_ms', 0):.2f} ms | "
            f"P95: {m.get('p95_offset_ms', 0):.2f} ms{p95_suffix} | "
            f"Drift: {m.get('drift_slope_ms_per_min', 0):.4f} ms/min | "
            f"Jitter: {m.get('jitter_std_ms', 0):.3f} ms"
        )

    def _update_dashboard_tab(self, info: dict):
        bundle = self.bundle
        table = self.alignment["aligned_table"]

        self.banner.setText(self._banner_text())
        self.banner.setStyleSheet(
            f"border-left: 4px solid {self._sync_color(self.quality)};"
        )

        cam_id = info["camera_frame_id"]
        rad_id = info["radar_msg_id"]
        offset = info["offset_ms"]
        elapsed = info["elapsed_s"]

        self.lbl_cam_title.setText(f"Camera | Frame {cam_id} | {elapsed:.2f}s")

        row = table.iloc[info["idx"]]
        meta = {
            **info,
            "speed": row.get("speed", np.nan),
            "steering_deg": row.get("steering_deg", np.nan),
        }
        b64 = cr.render_camera_panel(
            bundle["camera_video"]["cap"],
            cam_id,
            meta,
            cache=self.cam_cache,
            engineering_mode=True,
        )
        self._set_image_from_b64(self.lbl_camera, b64)

        self.lbl_cam_stats.setText(
            f"Frame {cam_id} | Elapsed {elapsed:.3f}s | Radar Delta {offset:+.2f} ms"
        )

        radar_view = self.radar_view_keys[self.cmb_radar_view.currentIndex()]
        self.lbl_rad_title.setText(f"Radar | {radar_view} | Msg {rad_id}")

        targets = rr.load_radar_frame(bundle["radar_hdf5"], rad_id, self.rad_cache)
        self._plot_radar_matplotlib(self.canvas_radar, targets, radar_view, title="")

        rad_meta = bundle["radar_hdf5"].get_frame_metadata(rad_id)
        n_tgt = int(rad_meta.get("num_targets", len(targets)))
        self.lbl_rad_stats.setText(
            f"Msg {rad_id} | Targets {n_tgt} | Camera Delta {-offset:+.2f} ms"
        )

        self._plot_live_sync_bar(info)

    def _plot_radar_matplotlib(self, canvas: MplCanvas, targets: pd.DataFrame, view: str, title: str):
        # Reset the whole figure each frame so old colorbar axes do not accumulate.
        canvas.fig.clear()
        ax = canvas.fig.add_subplot(111)
        canvas.ax = ax
        ax.set_facecolor("#161b22")
        canvas.fig.patch.set_facecolor("#0d1117")

        if targets.empty:
            ax.text(0.5, 0.5, "No radar targets", ha="center", va="center", color="#8b949e")
            ax.set_xticks([])
            ax.set_yticks([])
            canvas.draw_idle()
            return

        def add_cbar(scatter, label):
            cbar = canvas.fig.colorbar(
                scatter,
                ax=ax,
                orientation="horizontal",
                fraction=0.09,
                pad=0.14,
            )
            cbar.set_label(label, color="#c9d1d9")
            cbar.ax.tick_params(colors="#c9d1d9", labelsize=8)
            cbar.outline.set_edgecolor("#30363d")

        lim = self.radar_limits or {
            "lateral_xlim": (-20.0, 20.0),
            "forward_ylim": (0.0, 22.0),
            "range_xlim": (0.0, 22.0),
            "speed_ylim": (-18.0, 14.0),
        }

        if view == "birds_eye":
            x = targets.get("x", pd.Series(np.zeros(len(targets)))).values
            y = targets.get("y", pd.Series(np.zeros(len(targets)))).values
            c = targets.get("rcs", pd.Series(np.ones(len(targets)))).values
            c_low, c_high = np.percentile(c, [5, 95]) if len(c) else (0.0, 1.0)
            sc = ax.scatter(
                y,
                x,
                c=c,
                s=52,
                cmap="turbo",
                alpha=0.98,
                edgecolors="#e6edf3",
                linewidths=0.45,
                vmin=c_low,
                vmax=c_high if c_high > c_low else None,
            )
            ax.scatter(
                [0],
                [0],
                marker="^",
                s=180,
                c="#00ff88",
                edgecolors="#ffffff",
                linewidths=1.2,
                zorder=20,
            )
            ax.annotate(
                "EGO",
                (0, 0),
                textcoords="offset points",
                xytext=(8, 8),
                color="#00ff88",
                fontsize=8,
                weight="bold",
                zorder=21,
            )
            ax.set_xlabel("Y Lateral (m)")
            ax.set_ylabel("X Forward (m)")
            ax.grid(color="#4b5563", alpha=0.45)
            ax.set_xlim(*lim["lateral_xlim"])
            ax.set_ylim(*lim["forward_ylim"])
            ax.invert_xaxis()  # keep right-side obstacles on the right after axis swap
            ax.set_aspect("equal", adjustable="box")
            add_cbar(sc, "RCS (dBsm)")
        elif view == "range_doppler":
            r = targets.get("range", pd.Series(np.zeros(len(targets)))).values
            v = targets.get("radial_speed", pd.Series(np.zeros(len(targets)))).values
            ax.hist2d(r, v, bins=[60, 40], cmap="magma")
            ax.scatter(r, v, s=18, c="#8be9fd", alpha=0.7, edgecolors="#111827", linewidths=0.25)
            ax.set_xlabel("Range (m)")
            ax.set_ylabel("Radial Speed (m/s)")
            ax.set_xlim(*lim["range_xlim"])
            ax.set_ylim(*lim["speed_ylim"])
            ax.grid(color="#4b5563", alpha=0.35)
        elif view == "velocity_range":
            r = targets.get("range", pd.Series(np.zeros(len(targets)))).values
            v = targets.get("radial_speed", pd.Series(np.zeros(len(targets)))).values
            c = targets.get("snr", pd.Series(np.ones(len(targets)))).values
            s = 16 + np.clip(targets.get("rcs", pd.Series(np.ones(len(targets)))).values, -10, 20) * 1.2
            c_low, c_high = np.percentile(c, [5, 95]) if len(c) else (0.0, 1.0)
            sc = ax.scatter(
                r,
                v,
                c=c,
                s=s,
                cmap="turbo",
                alpha=0.95,
                edgecolors="#e6edf3",
                linewidths=0.35,
                vmin=c_low,
                vmax=c_high if c_high > c_low else None,
            )
            ax.axhline(0, color="#8b949e", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Range (m)")
            ax.set_ylabel("Radial Speed (m/s)")
            ax.set_xlim(*lim["range_xlim"])
            ax.set_ylim(*lim["speed_ylim"])
            ax.grid(color="#4b5563", alpha=0.35)
            add_cbar(sc, "SNR (dB)")
        else:
            x = targets.get("x", pd.Series(np.zeros(len(targets)))).values
            y = targets.get("y", pd.Series(np.zeros(len(targets)))).values
            p = targets.get("power", pd.Series(np.ones(len(targets)))).values
            snr = targets.get("snr", pd.Series(np.ones(len(targets)))).values
            snr_range = (snr.max() - snr.min()) if len(snr) else 1.0
            s = 14 + (snr - snr.min()) / (snr_range + 1e-9) * 28
            p_low, p_high = np.percentile(p, [5, 95]) if len(p) else (0.0, 1.0)
            sc = ax.scatter(
                y,
                x,
                c=p,
                s=s,
                cmap="turbo",
                alpha=0.98,
                edgecolors="#e6edf3",
                linewidths=0.4,
                vmin=p_low,
                vmax=p_high if p_high > p_low else None,
            )
            ax.scatter(
                [0],
                [0],
                marker="^",
                s=180,
                c="#00ff88",
                edgecolors="#ffffff",
                linewidths=1.2,
                zorder=20,
            )
            ax.annotate(
                "EGO",
                (0, 0),
                textcoords="offset points",
                xytext=(8, 8),
                color="#00ff88",
                fontsize=8,
                weight="bold",
                zorder=21,
            )
            ax.set_xlabel("Y Lateral (m)")
            ax.set_ylabel("X Forward (m)")
            ax.grid(color="#4b5563", alpha=0.45)
            ax.set_xlim(*lim["lateral_xlim"])
            ax.set_ylim(*lim["forward_ylim"])
            ax.invert_xaxis()  # keep right-side obstacles on the right after axis swap
            ax.set_aspect("equal", adjustable="box")
            add_cbar(sc, "Power (dB)")

        ax.set_title(title, color="#c9d1d9", fontsize=10)
        ax.tick_params(colors="#c9d1d9")
        for s in ax.spines.values():
            s.set_color("#30363d")
        canvas.fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.18)
        canvas.draw_idle()

    def _plot_live_sync_bar(self, frame_info: dict):
        ax = self.canvas_live.ax
        ax.clear()
        ax.set_facecolor("#161b22")
        self.canvas_live.fig.patch.set_facecolor("#0d1117")

        aligned = self.alignment["aligned_table"]
        idx = frame_info["idx"]
        start = max(0, idx - 20)
        end = min(len(aligned), idx + 21)
        nearby = aligned.iloc[start:end]

        t = nearby["elapsed_s"].values
        o = nearby["offset_ms"].values

        fp = (self.frame_rel or {}).get("frame_period_ms", 33.3)
        good = 0.25 * fp
        warn = 0.50 * fp

        ax.plot(t, o, color="#58a6ff", linewidth=1.4, marker="o", markersize=2)
        ax.scatter([frame_info["elapsed_s"]], [frame_info["offset_ms"]], c="#f0883e", s=50, marker="D")
        ax.axhline(0, color="#8b949e", linestyle="--", linewidth=0.8)
        ax.axhspan(-good, good, color="#3fb950", alpha=0.08)
        ax.axhspan(good, warn, color="#d29922", alpha=0.08)
        ax.axhspan(-warn, -good, color="#d29922", alpha=0.08)
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Offset (ms)")
        ax.set_title("Live Sync Monitor", color="#c9d1d9", fontsize=10)
        ax.grid(color="#30363d", alpha=0.25)
        ax.tick_params(colors="#c9d1d9")
        for s in ax.spines.values():
            s.set_color("#30363d")
        self.canvas_live.draw_idle()

    def _update_sync_metrics_grid(self):
        while self.metrics_grid.count():
            item = self.metrics_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.loaded:
            self.metrics_grid.addWidget(QLabel("Load data to view metrics"), 0, 0)
            return

        m = self.metrics
        fr = self.frame_rel or {}
        entries = [
            ("Mean Offset", f"{m['mean_offset_ms']:.3f} ms"),
            ("Median Offset", f"{m['median_offset_ms']:.3f} ms"),
            ("Std Dev", f"{m['std_offset_ms']:.3f} ms"),
            ("Max Abs Offset", f"{m['max_abs_offset_ms']:.3f} ms"),
            ("95th Pct", f"{m['p95_offset_ms']:.3f} ms"),
            ("Drift", f"{m['drift_slope_ms_per_min']:.4f} ms/min"),
            ("Drift R2", f"{m['drift_r_squared']:.4f}"),
            ("Jitter Std", f"{m['jitter_std_ms']:.3f} ms"),
            ("Frame Period", f"{fr.get('frame_period_ms', 33.3):.2f} ms"),
            ("P95 Frames", f"{fr.get('p95_frames', 0):.3f} xfp"),
            ("Overall", self.quality),
        ]

        for i, (k, v) in enumerate(entries):
            r = i // 4
            c = (i % 4) * 2
            k_lbl = QLabel(k)
            v_lbl = QLabel(v)
            v_lbl.setObjectName("Monospace")
            if k == "Overall":
                v_lbl.setStyleSheet(f"color: {self._sync_color(self.quality)}; font-weight: 700;")
            self.metrics_grid.addWidget(k_lbl, r, c)
            self.metrics_grid.addWidget(v_lbl, r, c + 1)

    def _plot_sync_analysis(self):
        if not self.loaded:
            return

        aligned = self.alignment["aligned_table"]
        drift = self.alignment["drift"]
        metrics = self.metrics
        fr = self.frame_rel or {}

        t = aligned["elapsed_s"].values
        o = aligned["offset_ms"].values
        good = 0.25 * fr.get("frame_period_ms", 33.3)
        warn = 0.50 * fr.get("frame_period_ms", 33.3)

        ax = self.sync_canvases[0].ax
        ax.clear()
        ax.plot(t, o, color="#58a6ff", linewidth=1)
        ax.axhline(0, color="#8b949e", linestyle="--", linewidth=0.8)
        ax.axhline(good, color="#3fb950", linestyle=":", linewidth=0.8)
        ax.axhline(-good, color="#3fb950", linestyle=":", linewidth=0.8)
        ax.axhline(warn, color="#d29922", linestyle=":", linewidth=0.8)
        ax.axhline(-warn, color="#d29922", linestyle=":", linewidth=0.8)
        ax.set_title("Offset Over Time")
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Offset (ms)")

        ax = self.sync_canvases[1].ax
        ax.clear()
        rolling = np.array(metrics.get("_rolling_mean_offset_ms", []))
        ax.plot(t, o, color="#30363d", linewidth=0.8)
        if len(rolling) == len(t):
            ax.plot(t, rolling, color="#f0883e", linewidth=2)
        ax.axhline(0, color="#8b949e", linestyle="--", linewidth=0.8)
        ax.set_title("Rolling Mean Offset")
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Offset (ms)")

        cam_rel = aligned["camera_time_s"].values - aligned["camera_time_s"].iloc[0]
        rad_rel = aligned["radar_time_s"].values - aligned["radar_time_s"].iloc[0]
        ax = self.sync_canvases[2].ax
        ax.clear()
        ax.plot(t, cam_rel, color="#58a6ff", linewidth=1, label="Camera")
        ax.plot(t, rad_rel, color="#3fb950", linewidth=1, label="Radar")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("Sensor Timestamps vs MTL")
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Relative Time (s)")

        ax = self.sync_canvases[3].ax
        ax.clear()
        ax.hist(o, bins=60, color="#58a6ff", alpha=0.85)
        ax.axvline(metrics["mean_offset_ms"], color="#f0883e", linestyle="--", linewidth=1.5)
        ax.set_title("Offset Distribution")
        ax.set_xlabel("Offset (ms)")
        ax.set_ylabel("Count")

        ax = self.sync_canvases[4].ax
        ax.clear()
        slope_per_s = drift["slope_ms_per_min"] / 60.0
        trend = drift["intercept_ms"] + slope_per_s * t
        ax.plot(t, o, color="#58a6ff", linewidth=0.8, alpha=0.55)
        ax.plot(t, trend, color="#f85149", linestyle=":", linewidth=2)
        ax.set_title("Drift Analysis")
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Offset (ms)")

        ax = self.sync_canvases[5].ax
        ax.clear()
        jitter = np.diff(o) if len(o) > 1 else np.array([0.0])
        jt = t[1:] if len(t) > 1 else t
        ax.plot(jt, jitter, color="#a371f7", linewidth=0.9)
        ax.axhline(0, color="#8b949e", linestyle="--", linewidth=0.8)
        ax.set_title("Frame-to-Frame Jitter")
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Delta Offset (ms)")

        cam_drops = self.alignment["cam_with_drops"]
        rad_drops = self.alignment["rad_with_drops"]
        ax = self.sync_canvases[6].ax
        ax.clear()
        if "is_drop" in cam_drops.columns and cam_drops["is_drop"].any() and "camera_time_s" in cam_drops.columns:
            d = cam_drops[cam_drops["is_drop"]]
            dt = d["camera_time_s"].values - cam_drops["camera_time_s"].iloc[0]
            dg = d["gap_s"].values * 1000
            ax.scatter(dt, dg, c="#f85149", s=22, label=f"Camera ({len(d)})")
        if "is_drop" in rad_drops.columns and rad_drops["is_drop"].any() and "radar_time_s" in rad_drops.columns:
            d = rad_drops[rad_drops["is_drop"]]
            dt = d["radar_time_s"].values - rad_drops["radar_time_s"].iloc[0]
            dg = d["gap_s"].values * 1000
            ax.scatter(dt, dg, c="#d29922", s=22, label=f"Radar ({len(d)})")
        ax.set_title("Frame Drop Timeline")
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Gap (ms)")
        ax.legend(loc="upper right", fontsize=8)

        for cv in self.sync_canvases:
            cv.ax.set_facecolor("#161b22")
            cv.fig.patch.set_facecolor("#0d1117")
            cv.ax.grid(color="#30363d", alpha=0.25)
            cv.ax.tick_params(colors="#c9d1d9")
            cv.ax.xaxis.label.set_color("#c9d1d9")
            cv.ax.yaxis.label.set_color("#c9d1d9")
            cv.ax.title.set_color("#c9d1d9")
            for s in cv.ax.spines.values():
                s.set_color("#30363d")
            cv.draw_idle()

    def _update_metadata_tab(self):
        if not self.loaded:
            self.txt_camera_meta.setPlainText("Load data to view metadata")
            self.txt_radar_meta.clear()
            self.list_issues.clear()
            self.lbl_validation.setText("Validation: -")
            return

        meta = self.meta_bundle
        vrep = self.vrep or {}

        cam = meta["camera"]
        camera_lines = ["[Camera Session]"]
        for k, v in cam["session"].items():
            camera_lines.append(f"{k}: {v}")
        camera_lines.append("\n[Camera Performance]")
        for k, v in cam["performance"].items():
            camera_lines.append(f"{k}: {_safe_str(v)}")
        camera_lines.append(f"total_frames: {cam['total_frames']}")
        self.txt_camera_meta.setPlainText("\n".join(camera_lines))

        rad = meta["radar"]
        radar_lines = ["[Radar Stats]"]
        radar_lines.append(f"total_frames: {rad['total_frames']}")
        radar_lines.append(f"avg_targets_per_frame: {rad['avg_targets_per_frame']}")
        radar_lines.append(f"total_targets: {rad['total_targets']}")
        radar_lines.append("\n[Radar Log Stats]")
        for k, v in rad["log_stats"].items():
            if isinstance(v, dict):
                radar_lines.append(f"{k}:")
                for kk, vv in v.items():
                    radar_lines.append(f"  {kk}: {_safe_str(vv)}")
            elif isinstance(v, list):
                radar_lines.append(f"{k}: {len(v)} entries")
            else:
                radar_lines.append(f"{k}: {_safe_str(v)}")
        self.txt_radar_meta.setPlainText("\n".join(radar_lines))

        self.list_issues.clear()
        issues = (vrep.get("errors", []) + vrep.get("warnings", []))[:50]
        if not issues:
            self.list_issues.addItem(QListWidgetItem("No validation issues."))
        else:
            for issue in issues:
                self.list_issues.addItem(QListWidgetItem(f"[{issue['level']}] {issue['detail']}"))

        self.lbl_validation.setText(
            f"Validation: {'VALID' if vrep.get('is_valid') else 'ISSUES'} | "
            f"Errors {len(vrep.get('errors', []))} | Warnings {len(vrep.get('warnings', []))}"
        )

    def _update_export_preview(self):
        if not self.loaded:
            self.table_preview.setModel(PandasTableModel(pd.DataFrame()))
            return

        table = self.alignment["aligned_table"]
        cols = [
            "elapsed_s",
            "camera_frame_id",
            "radar_msg_id",
            "offset_ms",
            "abs_offset_ms",
            "camera_time_s",
            "radar_time_s",
        ]
        cols = [c for c in cols if c in table.columns]
        preview = table[cols].head(200).copy()
        self.table_preview.setModel(PandasTableModel(preview))

    def _update_demo_page(self, info: dict):
        if not self.loaded:
            return

        self.lbl_demo_quality.setText(
            f"Synchronisation Quality: {self.quality} | "
            f"Mean {self.metrics.get('mean_offset_ms', 0):.2f} ms | "
            f"P95 {self.metrics.get('p95_offset_ms', 0):.2f} ms"
        )
        self.lbl_demo_quality.setStyleSheet(
            f"border: 1px solid {self._sync_color(self.quality)}; color: {self._sync_color(self.quality)};"
        )

        cam_id = info["camera_frame_id"]
        rad_id = info["radar_msg_id"]

        b64 = cr.render_camera_panel(
            self.bundle["camera_video"]["cap"],
            cam_id,
            info,
            cache=self.cam_cache,
            engineering_mode=False,
        )
        self._set_image_from_b64(self.lbl_demo_camera, b64)

        targets = rr.load_radar_frame(self.bundle["radar_hdf5"], rad_id, self.rad_cache)
        self._plot_radar_matplotlib(self.canvas_demo_radar, targets, "birds_eye", title="Demo Radar")

        ax = self.canvas_demo_offset.ax
        ax.clear()
        ax.set_facecolor("#161b22")
        self.canvas_demo_offset.fig.patch.set_facecolor("#0d1117")

        aligned = self.alignment["aligned_table"]
        t = aligned["elapsed_s"].values
        o = aligned["offset_ms"].values
        ax.plot(t, o, color="#58a6ff", linewidth=1)
        ax.scatter([info["elapsed_s"]], [info["offset_ms"]], c="#f0883e", marker="D", s=45)
        ax.axhline(0, color="#8b949e", linestyle="--", linewidth=0.8)
        fp = (self.frame_rel or {}).get("frame_period_ms", 33.3)
        ax.axhspan(-0.25 * fp, 0.25 * fp, color="#3fb950", alpha=0.08)
        ax.set_title("Sync Offset Over Recording")
        ax.set_xlabel("Elapsed (s)")
        ax.set_ylabel("Offset (ms)")
        ax.grid(color="#30363d", alpha=0.25)
        ax.tick_params(colors="#c9d1d9")
        for s in ax.spines.values():
            s.set_color("#30363d")
        self.canvas_demo_offset.draw_idle()

    def _set_image_from_b64(self, label: QLabel, b64_data: str):
        if not b64_data:
            label.setText("Frame decode failed")
            label.setPixmap(QPixmap())
            return
        try:
            raw = base64.b64decode(b64_data.split(",", 1)[1])
            arr = np.frombuffer(raw, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                label.setText("Invalid image")
                label.setPixmap(QPixmap())
                return
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qimg)
            target_size = label.contentsRect().size()
            if target_size.width() <= 1 or target_size.height() <= 1:
                target_size = label.size()
            pix = pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setText("")
            label.setPixmap(pix)
        except Exception:
            label.setText("Image render error")
            label.setPixmap(QPixmap())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.loaded:
            info = self._get_frame_info()
            if info:
                self._update_dashboard_tab(info)
                self._update_demo_page(info)

    def _export_dir(self):
        p = BASE_DIR / "exports"
        p.mkdir(exist_ok=True)
        return p

    def _choose_path(self, caption: str, default_name: str, filter_str: str):
        path, _ = QFileDialog.getSaveFileName(self, caption, str(BASE_DIR / default_name), filter_str)
        return path

    def _export_csv(self):
        if not self.loaded:
            return
        path = self._choose_path("Save aligned table", "aligned_frame_table.csv", "CSV (*.csv)")
        if not path:
            return
        self.alignment["aligned_table"].to_csv(path, index=False)
        QMessageBox.information(self, "Export", f"Saved: {path}")

    def _export_json(self):
        if not self.loaded:
            return
        path = self._choose_path("Save metrics JSON", "sync_metrics_summary.json", "JSON (*.json)")
        if not path:
            return
        me.export_metrics_json(self.metrics, path)
        QMessageBox.information(self, "Export", f"Saved: {path}")

    def _export_metrics_csv(self):
        if not self.loaded:
            return
        path = self._choose_path("Save metrics CSV", "sync_metrics_summary.csv", "CSV (*.csv)")
        if not path:
            return
        me.export_metrics_csv(self.metrics, path)
        QMessageBox.information(self, "Export", f"Saved: {path}")

    def _export_pdf(self):
        if not self.loaded:
            return
        path = self._choose_path("Save PDF report", "sync_analysis_report.pdf", "PDF (*.pdf)")
        if not path:
            return
        try:
            me.export_sync_report_pdf(self.metrics, self.alignment["aligned_table"], self.quality, path)
            QMessageBox.information(self, "Export", f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "PDF Export Failed", str(e))

    def _export_all(self):
        if not self.loaded:
            return
        d = self._export_dir()
        self.alignment["aligned_table"].to_csv(d / "aligned_frame_table.csv", index=False)
        me.export_metrics_json(self.metrics, str(d / "sync_metrics_summary.json"))
        me.export_metrics_csv(self.metrics, str(d / "sync_metrics_summary.csv"))
        me.export_sync_report_pdf(
            self.metrics,
            self.alignment["aligned_table"],
            self.quality,
            str(d / "sync_analysis_report.pdf"),
        )
        QMessageBox.information(self, "Export", f"All exports written to {d}")

    def _help_html(self) -> str:
        return """
        <h2 style='color:#58a6ff;'>Sync Quality - Frame-Relative Thresholds</h2>
        <p>Thresholds scale with camera FPS using frame period (<b>fp</b>), not fixed milliseconds.</p>
        <ul>
          <li><b>Fusion tier</b> drives overall quality (PASS/WARNING/FAIL).</li>
          <li><b>GOOD:</b> p95 &lt; 0.25 x fp</li>
          <li><b>WARNING:</b> p95 &lt; 0.50 x fp</li>
          <li><b>FAIL:</b> p95 >= 0.50 x fp</li>
        </ul>
        <h3 style='color:#58a6ff;'>Core Metrics</h3>
        <ul>
          <li><b>Mean / Median Offset:</b> systematic timing bias</li>
          <li><b>P95 Offset:</b> primary reliability signal</li>
          <li><b>Drift Slope:</b> long-term clock divergence (ms/min)</li>
          <li><b>Jitter Std:</b> frame-to-frame timing instability</li>
          <li><b>Drop Events:</b> data continuity failures</li>
        </ul>
        <h3 style='color:#58a6ff;'>Workflow</h3>
        <ol>
          <li>Load data from sidebar.</li>
          <li>Use playback and timeline scrub to inspect aligned pairs.</li>
          <li>Switch radar views for different target-domain diagnostics.</li>
          <li>Validate quality and export reports from Export tab.</li>
        </ol>
        """

    def closeEvent(self, event):
        try:
            if self.bundle is not None:
                vid = self.bundle.get("camera_video", {})
                cap = vid.get("cap")
                if cap is not None:
                    cap.release()
                rh = self.bundle.get("radar_hdf5")
                if rh is not None:
                    rh.close()
        finally:
            super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    w = SyncDashboard()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
