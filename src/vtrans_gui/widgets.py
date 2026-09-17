"""Small shared widgets."""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)


class Card(QFrame):
    """A titled group of controls."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setFrameShape(QFrame.NoFrame)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(14, 12, 14, 14)
        self._layout.setSpacing(9)

        heading = QLabel(title)
        heading.setObjectName("cardTitle")
        self._layout.addWidget(heading)

    def add(self, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        return widget

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)

    def add_hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("hint")
        label.setWordWrap(True)
        self._layout.addWidget(label)
        return label


class Collapsible(QWidget):
    """A disclosure triangle hiding advanced controls."""

    toggled_open = Signal(bool)

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._button = QToolButton()
        self._button.setText(title)
        self._button.setCheckable(True)
        self._button.setChecked(False)
        self._button.setStyleSheet("QToolButton { border: none; font-weight: 600; }")
        self._button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._button.setArrowType(Qt.RightArrow)
        self._button.clicked.connect(self._on_click)

        self.content = QWidget()
        self.content.setVisible(False)
        self._content_layout = QVBoxLayout(self.content)
        self._content_layout.setContentsMargins(0, 6, 0, 0)
        self._content_layout.setSpacing(9)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._button)
        layout.addWidget(self.content)

    def add(self, widget: QWidget) -> QWidget:
        self._content_layout.addWidget(widget)
        return widget

    def add_layout(self, layout) -> None:
        self._content_layout.addLayout(layout)

    def set_open(self, is_open: bool) -> None:
        self._button.setChecked(is_open)
        self._on_click()

    def is_open(self) -> bool:
        return self._button.isChecked()

    def _on_click(self) -> None:
        opened = self._button.isChecked()
        self._button.setArrowType(Qt.DownArrow if opened else Qt.RightArrow)
        self.content.setVisible(opened)
        self.toggled_open.emit(opened)


class Badge(QLabel):
    """Colour-coded status chip: fits / tight / will not fit."""

    LEVELS = {
        "ok": ("#1f4620", "#8fdc8f"),
        "warn": ("#4a3c14", "#f0cf72"),
        "bad": ("#4c1f1f", "#f0999a"),
        "muted": ("#2e2e2e", "#a8a8a8"),
    }

    def __init__(self, text: str = "", level: str = "muted",
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.set_level(level)

    def set_level(self, level: str) -> None:
        background, foreground = self.LEVELS.get(level, self.LEVELS["muted"])
        self.setStyleSheet(
            f"QLabel {{ background: {background}; color: {foreground};"
            f" border-radius: 3px; padding: 2px 7px; font-size: 11px;"
            f" font-weight: 600; }}"
        )

    def update_badge(self, text: str, level: str) -> None:
        self.setText(text)
        self.set_level(level)


def hline() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("QFrame { color: #3a3a3a; }")
    return line


def row(*widgets: QWidget, stretch_last: bool = True) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    for index, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if (stretch_last and index == len(widgets) - 1) else 0)
    return layout
