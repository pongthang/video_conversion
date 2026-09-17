"""Application styling.

A dark palette on purpose: the window is mostly a video frame preview, and a
light chrome around it skews how the caption contrast reads.
"""

STYLESHEET = """
QWidget {
    background: #202124;
    color: #e4e4e6;
    font-size: 13px;
}
/* Labels, radios and checkboxes sit on top of a card, so they must not paint
   the window colour over it. */
QLabel, QRadioButton, QCheckBox, QToolButton {
    background: transparent;
}
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget {
    background: #202124;
}
QFrame#card {
    background: #2a2b2e;
    border: 1px solid #3a3b3f;
    border-radius: 6px;
}
QLabel#cardTitle {
    font-size: 13px;
    font-weight: 700;
    color: #f0f0f2;
    padding-bottom: 2px;
}
QLabel#hint {
    color: #9b9ba0;
    font-size: 11px;
}
QLabel#stageLabel {
    color: #c9c9ce;
    font-size: 12px;
}
QPushButton {
    background: #3a3b40;
    border: 1px solid #4a4b50;
    border-radius: 5px;
    padding: 6px 14px;
}
QPushButton:hover   { background: #45464c; }
QPushButton:pressed { background: #35363a; }
QPushButton:disabled {
    background: #2a2b2e;
    color: #6a6a70;
    border-color: #3a3b3f;
}
QPushButton#primary {
    background: #2f6fd0;
    border: 1px solid #3b7fe0;
    color: #ffffff;
    font-weight: 700;
    padding: 9px 20px;
    font-size: 14px;
}
QPushButton#primary:hover    { background: #3b7fe0; }
QPushButton#primary:disabled { background: #2a3a52; color: #7f8fa5; border-color: #33465f; }
QPushButton#danger {
    background: #8c3a3a;
    border: 1px solid #a04646;
    color: #ffffff;
    font-weight: 700;
    padding: 9px 20px;
    font-size: 14px;
}
QPushButton#danger:hover { background: #a04646; }
QComboBox, QLineEdit, QSpinBox {
    background: #1b1c1f;
    border: 1px solid #45464c;
    border-radius: 4px;
    padding: 5px 8px;
    selection-background-color: #2f6fd0;
}
QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled { color: #6a6a70; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: #1b1c1f;
    border: 1px solid #45464c;
    selection-background-color: #2f6fd0;
}
QRadioButton, QCheckBox { spacing: 7px; }
QProgressBar {
    background: #1b1c1f;
    border: 1px solid #3a3b3f;
    border-radius: 4px;
    height: 20px;
    text-align: center;
    color: #e4e4e6;
}
QProgressBar::chunk {
    background: #2f6fd0;
    border-radius: 3px;
}
QProgressBar#stageBar {
    height: 6px;
    border: none;
    background: #313236;
}
QProgressBar#stageBar::chunk { background: #5a9bf0; }
QPlainTextEdit {
    background: #17181a;
    border: 1px solid #3a3b3f;
    border-radius: 4px;
    font-family: "DejaVu Sans Mono", "Cascadia Mono", Consolas, monospace;
    font-size: 11px;
    color: #c3c3c8;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #45464c;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #2f6fd0;
    width: 15px;
    height: 15px;
    margin: -6px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #2f6fd0; border-radius: 2px; }
QToolButton { color: #e4e4e6; }
QScrollBar:vertical {
    background: #202124; width: 10px; margin: 0;
}
QScrollBar::handle:vertical {
    background: #45464c; border-radius: 5px; min-height: 30px;
}
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QToolTip {
    background: #1b1c1f; color: #e4e4e6;
    border: 1px solid #45464c; padding: 4px;
}
"""
