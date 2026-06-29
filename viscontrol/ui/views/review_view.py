"""REVIEW view — browse historical fault images and the event CSV.

The main window seeds the view with paths from ``storage.log_dir`` and
``storage.defect_image_dir``. Selecting an image previews it; selecting a CSV
opens it as a plain text table.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from viscontrol.ui.theme import (
    BORDER,
    FONT_NORMAL,
    FONT_SMALL,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from viscontrol.ui.widgets.camera_view import CameraView, CameraViewState


class ReviewView(QWidget):
    """List of recent defect images on the left, preview + CSV on the right."""

    export_logs_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._defect_dir: Path | None = None
        self._log_dir: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(16)

        header = QFrame()
        header.setObjectName("card")
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(20, 12, 20, 12)
        title = QLabel(self.tr("Review"))
        title.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_NORMAL}pt; font-weight: 600;"
        )
        h_layout.addWidget(title, 1)
        self._export_btn = QPushButton(self.tr("Export logs…"))
        self._export_btn.setObjectName("secondary")
        self._export_btn.clicked.connect(self.export_logs_clicked.emit)
        h_layout.addWidget(self._export_btn, 0)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # Left: defect list.
        list_frame = QFrame()
        list_frame.setObjectName("card")
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(12, 12, 12, 12)
        defects_label = QLabel(self.tr("Defect images"))
        defects_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_NORMAL}pt; font-weight: 600;"
        )
        self._defect_list = QListWidget()
        self._defect_list.setStyleSheet(
            f"QListWidget {{ border: 1px solid {BORDER}; }}"
        )
        self._defect_list.currentItemChanged.connect(self._on_defect_selected)
        logs_label = QLabel(self.tr("Daily event logs"))
        logs_label.setStyleSheet(
            f"color: {TEXT_PRIMARY}; font-size: {FONT_NORMAL}pt; font-weight: 600;"
        )
        self._log_list = QListWidget()
        self._log_list.setStyleSheet(
            f"QListWidget {{ border: 1px solid {BORDER}; }}"
        )
        self._log_list.currentItemChanged.connect(self._on_log_selected)
        list_layout.addWidget(defects_label)
        list_layout.addWidget(self._defect_list, 1)
        list_layout.addWidget(logs_label)
        list_layout.addWidget(self._log_list, 1)
        splitter.addWidget(list_frame)

        # Right: preview + CSV text.
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)
        self._preview = CameraView(self.tr("Selected defect"))
        self._csv = QPlainTextEdit()
        self._csv.setReadOnly(True)
        self._csv.setStyleSheet(
            f"QPlainTextEdit {{ font-family: 'Consolas','Courier New',monospace;"
            f" font-size: {FONT_SMALL}pt; border: 1px solid {BORDER};"
            f" background: white; color: {TEXT_PRIMARY}; padding: 6px; }}"
        )
        right_layout.addWidget(self._preview, 2)
        right_layout.addWidget(self._csv, 1)
        splitter.addWidget(right_frame)
        splitter.setSizes([300, 700])

    def set_dirs(self, defect_dir: Path, log_dir: Path) -> None:
        self._defect_dir = defect_dir
        self._log_dir = log_dir
        self.refresh()

    def refresh(self) -> None:
        self._defect_list.clear()
        if self._defect_dir and self._defect_dir.exists():
            for p in sorted(self._defect_dir.iterdir(), reverse=True):
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    item = QListWidgetItem(p.name)
                    item.setData(Qt.ItemDataRole.UserRole, str(p))
                    self._defect_list.addItem(item)
        self._log_list.clear()
        if self._log_dir and self._log_dir.exists():
            for p in sorted(self._log_dir.glob("events-*.csv"), reverse=True):
                item = QListWidgetItem(p.name)
                item.setData(Qt.ItemDataRole.UserRole, str(p))
                self._log_list.addItem(item)

    def pick_export_dir(self) -> Path | None:
        path = QFileDialog.getExistingDirectory(self, self.tr("Export logs to…"))
        return Path(path) if path else None

    # ---------- handlers ----------

    def _on_defect_selected(self, current: QListWidgetItem | None) -> None:
        if not current:
            self._preview.set_state(CameraViewState())
            return
        path = Path(current.data(Qt.ItemDataRole.UserRole))
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return
        self._preview.set_state(CameraViewState(image=img))

    def _on_log_selected(self, current: QListWidgetItem | None) -> None:
        if not current:
            self._csv.setPlainText("")
            return
        path = Path(current.data(Qt.ItemDataRole.UserRole))
        try:
            self._csv.setPlainText(path.read_text(encoding="utf-8"))
        except OSError as e:
            self._csv.setPlainText(f"failed to read {path}: {e}")
