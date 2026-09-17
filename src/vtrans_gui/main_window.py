"""The application window."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFileDialog,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QRadioButton, QScrollArea, QSizePolicy, QSlider,
                               QSplitter, QVBoxLayout, QWidget)

from vtrans.media import is_url
from vtrans.utils import fmt_duration

from vtrans.config import Config

from .assets import missing_for, summarise
from .catalog import (ASR_MODELS, PRESETS, TRANSLATE_MODELS, TTS_MODELS,
                      default_voice, find_option, preset_requirements,
                      voices_for)
from .hardware import Hardware, detect
from .previewpane import PreviewPane
from .settings import Settings
from .widgets import Badge, Card, Collapsible, hline, row
from .worker import ConversionWorker, DownloadWorker

VIDEO_FILTER = ("Video files (*.mp4 *.mkv *.mov *.avi *.webm *.flv *.wmv *.m4v *.ts);;"
                "All files (*)")


class MainWindow(QWidget):
    """Single-window UI: options on the left, caption preview on the right."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Miko Video Translator - Chinese to English")
        self.setMinimumSize(1120, 760)
        self.setAcceptDrops(True)

        self.settings = Settings.load()
        self.hardware: Hardware = detect()
        # Building and restoring both fire the widgets' change signals, and
        # those handlers save. Without this guard the first one to run writes
        # a half-populated window over the user's real settings: every value
        # still at its constructed minimum.
        self._restoring = True
        self.worker: Optional[ConversionWorker] = None
        self.downloader: Optional[DownloadWorker] = None
        self.source: Optional[str] = None
        self.started_at: float = 0.0
        self.models_dir = Config.load().resolve_dir("general.models_dir")

        self._build()
        self._restore()
        self._restoring = False
        self._refresh_requirements()
        self._request_preview()

    # =================================================================== UI

    def _build(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # ---- left: options, scrollable so the window can be small ----
        options = QWidget()
        self.options_layout = QVBoxLayout(options)
        self.options_layout.setContentsMargins(14, 14, 10, 14)
        self.options_layout.setSpacing(12)

        self.options_layout.addWidget(self._build_source_card())
        self.options_layout.addWidget(self._build_voice_card())
        self.options_layout.addWidget(self._build_quality_card())
        self.options_layout.addWidget(self._build_device_card())
        self.options_layout.addWidget(self._build_subtitle_card())
        self.options_layout.addWidget(self._build_output_card())
        self.options_layout.addStretch(1)

        # The cards need this much width before their badges and hints start
        # being cut off; without a floor the splitter happily shrinks them.
        options.setMinimumWidth(430)

        scroll = QScrollArea()
        scroll.setWidget(options)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(455)

        # ---- right: preview and run ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 14, 14, 14)
        right_layout.setSpacing(12)
        right_layout.addWidget(self._build_preview_card(), 1)
        right_layout.addWidget(self._build_run_card())

        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setSizes([470, 700])
        splitter.setStretchFactor(1, 1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

    # ------------------------------------------------------------- source

    def _build_source_card(self) -> Card:
        card = Card("1 · Video")

        self.source_label = QLabel("No video chosen")
        self.source_label.setWordWrap(True)
        self.source_label.setObjectName("hint")

        browse = QPushButton("Choose video...")
        browse.clicked.connect(self._choose_file)

        card.add_layout(row(browse, self.source_label))
        card.add(hline())

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("...or paste a YouTube / video URL")
        self.url_edit.editingFinished.connect(self._url_entered)
        card.add(self.url_edit)
        card.add_hint("You can also drag a video file onto this window.")
        return card

    # -------------------------------------------------------------- voice

    def _build_voice_card(self) -> Card:
        card = Card("2 · Voice")

        self.female_radio = QRadioButton("Female")
        self.male_radio = QRadioButton("Male")
        self.gender_group = QButtonGroup(self)
        self.gender_group.addButton(self.female_radio)
        self.gender_group.addButton(self.male_radio)
        self.female_radio.toggled.connect(self._gender_changed)

        gender_row = QHBoxLayout()
        gender_row.setContentsMargins(0, 0, 0, 0)
        gender_row.addWidget(self.female_radio)
        gender_row.addWidget(self.male_radio)
        gender_row.addStretch(1)
        card.add_layout(gender_row)

        self.voice_combo = QComboBox()
        self.voice_combo.currentIndexChanged.connect(self._voice_changed)
        card.add(self.voice_combo)
        self.voice_note = card.add_hint("")
        return card

    # ------------------------------------------------------------ quality

    def _build_quality_card(self) -> Card:
        card = Card("3 · Quality and models")

        self.preset_combo = QComboBox()
        for preset in PRESETS:
            self.preset_combo.addItem(preset.label, preset.key)
        self.preset_combo.currentIndexChanged.connect(self._preset_changed)
        card.add(self.preset_combo)

        self.preset_note = card.add_hint("")

        self.preset_badge = Badge("", "muted")
        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.addWidget(self.preset_badge)
        badge_row.addStretch(1)
        card.add_layout(badge_row)

        self.advanced = Collapsible("Advanced: choose each model")
        self.asr_combo = self._model_combo(self.advanced, "Transcription (Whisper)", ASR_MODELS)
        self.mt_combo = self._model_combo(self.advanced, "Translation", TRANSLATE_MODELS)
        self.tts_combo = self._model_combo(self.advanced, "Speech", TTS_MODELS)
        self.advanced.toggled_open.connect(lambda _: self._refresh_requirements())
        card.add(self.advanced)
        return card

    def _model_combo(self, parent: Collapsible, title: str, options) -> QComboBox:
        label = QLabel(title)
        label.setStyleSheet("font-weight: 600; margin-top: 4px;")
        parent.add(label)

        combo = QComboBox()
        for option in options:
            combo.addItem(option.label, option.key)
        parent.add(combo)

        requirement = QLabel("")
        requirement.setObjectName("hint")
        requirement.setWordWrap(True)
        parent.add(requirement)

        badge = Badge("", "muted")
        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.addWidget(badge)
        badge_row.addStretch(1)
        parent.add_layout(badge_row)

        combo.requirement_label = requirement   # type: ignore[attr-defined]
        combo.badge = badge                     # type: ignore[attr-defined]
        combo.options = options                 # type: ignore[attr-defined]
        combo.currentIndexChanged.connect(self._model_changed)
        return combo

    # ------------------------------------------------------------- device

    def _build_device_card(self) -> Card:
        card = Card("4 · Where to run")

        self.auto_radio = QRadioButton("Automatic")
        self.gpu_radio = QRadioButton("GPU")
        self.cpu_radio = QRadioButton("CPU only")
        self.device_group = QButtonGroup(self)
        for button in (self.auto_radio, self.gpu_radio, self.cpu_radio):
            self.device_group.addButton(button)
            button.toggled.connect(self._device_changed)

        device_row = QHBoxLayout()
        device_row.setContentsMargins(0, 0, 0, 0)
        for button in (self.auto_radio, self.gpu_radio, self.cpu_radio):
            device_row.addWidget(button)
        device_row.addStretch(1)
        card.add_layout(device_row)

        if not self.hardware.has_gpu:
            self.gpu_radio.setEnabled(False)
            self.gpu_radio.setToolTip("No CUDA GPU was detected on this machine.")

        gpu_badge = Badge(self.hardware.summary,
                          "ok" if self.hardware.has_gpu else "muted")
        cpu_badge = Badge(self.hardware.cpu_summary, "muted")
        hardware_row = QHBoxLayout()
        hardware_row.setContentsMargins(0, 0, 0, 0)
        hardware_row.setSpacing(6)
        hardware_row.addWidget(gpu_badge)
        hardware_row.addWidget(cpu_badge)
        hardware_row.addStretch(1)
        card.add_layout(hardware_row)

        self.fallback_check = QCheckBox("If the GPU fails, finish that step on the CPU")
        self.fallback_check.setToolTip(
            "Each step is retried on the CPU independently, so one step running "
            "out of video memory does not slow down the whole conversion."
        )
        card.add(self.fallback_check)
        return card

    # ---------------------------------------------------------- subtitles

    def _build_subtitle_card(self) -> Card:
        card = Card("5 · Subtitles")

        self.subs_combo = QComboBox()
        for label, key in [("Burned into the picture", "burn"),
                           ("Separate track you can switch off", "soft"),
                           ("Both", "both"),
                           ("No subtitles", "none")]:
            self.subs_combo.addItem(label, key)
        self.subs_combo.currentIndexChanged.connect(self._subs_mode_changed)
        card.add(self.subs_combo)

        # -- size
        self.size_value = QLabel("22")
        self.size_value.setFixedWidth(26)
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(12, 44)
        self.size_slider.valueChanged.connect(self._font_size_changed)
        size_label = QLabel("Text size")
        size_label.setFixedWidth(96)
        card.add_layout(row(size_label, self.size_value, self.size_slider))

        # -- vertical position
        self.margin_value = QLabel("28")
        self.margin_value.setFixedWidth(26)
        self.margin_slider = QSlider(Qt.Horizontal)
        self.margin_slider.setRange(6, 120)
        self.margin_slider.valueChanged.connect(self._margin_changed)
        margin_label = QLabel("Height above bottom")
        margin_label.setFixedWidth(96)
        margin_label.setWordWrap(True)
        card.add_layout(row(margin_label, self.margin_value, self.margin_slider))
        card.add_hint(
            "Raise the captions if your source video already has its own "
            "subtitles burned into the bottom of the picture."
        )

        # -- wrap width
        self.wrap_value = QLabel("42")
        self.wrap_value.setFixedWidth(26)
        self.wrap_slider = QSlider(Qt.Horizontal)
        self.wrap_slider.setRange(24, 70)
        self.wrap_slider.valueChanged.connect(self._wrap_changed)
        wrap_label = QLabel("Line length")
        wrap_label.setFixedWidth(96)
        card.add_layout(row(wrap_label, self.wrap_value, self.wrap_slider))
        return card

    # ------------------------------------------------------------- output

    def _build_output_card(self) -> Card:
        card = Card("6 · Save to")
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        pick = QPushButton("Change...")
        pick.clicked.connect(self._choose_output_dir)
        card.add_layout(row(self.output_edit, pick, stretch_last=False))
        self.output_name = card.add_hint("")
        return card

    # ------------------------------------------------------------ preview

    def _build_preview_card(self) -> Card:
        card = Card("Caption preview")
        self.preview = PreviewPane()
        card.add(self.preview)
        return card

    # ---------------------------------------------------------------- run

    def _build_run_card(self) -> Card:
        card = Card("Convert")

        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self._start)
        self.start_button.setEnabled(False)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.setVisible(False)

        self.open_button = QPushButton("Open folder")
        self.open_button.clicked.connect(self._open_output_dir)
        self.open_button.setVisible(False)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.open_button)
        button_row.addStretch(1)
        self.elapsed_label = QLabel("")
        self.elapsed_label.setObjectName("hint")
        button_row.addWidget(self.elapsed_label)
        card.add_layout(button_row)

        self.overall_bar = QProgressBar()
        self.overall_bar.setRange(0, 1000)
        self.overall_bar.setValue(0)
        self.overall_bar.setFormat("%p%")
        card.add(self.overall_bar)

        self.stage_label = QLabel("Ready")
        self.stage_label.setObjectName("stageLabel")
        card.add(self.stage_label)

        self.stage_bar = QProgressBar()
        self.stage_bar.setObjectName("stageBar")
        self.stage_bar.setRange(0, 1000)
        self.stage_bar.setTextVisible(False)
        card.add(self.stage_bar)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setMinimumHeight(120)
        self.log_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        card.add(self.log_view)
        return card

    # ============================================================== state

    def _restore(self) -> None:
        settings = self.settings
        (self.female_radio if settings.voice_gender == "female" else self.male_radio).setChecked(True)
        self._set_combo(self.preset_combo, settings.preset)
        self._set_combo(self.asr_combo, settings.asr_model)
        self._set_combo(self.mt_combo, settings.translate_backend)
        self._set_combo(self.tts_combo, settings.tts_backend)
        self._set_combo(self.subs_combo, settings.subtitles_mode)

        device = settings.device
        if device == "cuda" and self.hardware.has_gpu:
            self.gpu_radio.setChecked(True)
        elif device == "cpu":
            self.cpu_radio.setChecked(True)
        else:
            self.auto_radio.setChecked(True)

        self.fallback_check.setChecked(settings.cpu_fallback)
        self.size_slider.setValue(settings.font_size)
        self.margin_slider.setValue(settings.margin_v)
        self.wrap_slider.setValue(settings.max_line_chars)
        self.output_edit.setText(settings.output_dir)
        self.advanced.set_open(settings.advanced_open)
        self._rebuild_voices()

    @staticmethod
    def _set_combo(combo: QComboBox, key: str) -> None:
        index = combo.findData(key)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _persist(self) -> None:
        if self._restoring:
            return
        s = self.settings
        s.voice_gender = "female" if self.female_radio.isChecked() else "male"
        s.voice = self.voice_combo.currentData() or s.voice
        s.preset = self.preset_combo.currentData()
        s.asr_model = self.asr_combo.currentData()
        s.translate_backend = self.mt_combo.currentData()
        s.tts_backend = self.tts_combo.currentData()
        s.subtitles_mode = self.subs_combo.currentData()
        s.device = ("cuda" if self.gpu_radio.isChecked()
                    else "cpu" if self.cpu_radio.isChecked() else "auto")
        s.cpu_fallback = self.fallback_check.isChecked()
        s.font_size = self.size_slider.value()
        s.margin_v = self.margin_slider.value()
        s.max_line_chars = self.wrap_slider.value()
        s.output_dir = self.output_edit.text()
        s.advanced_open = self.advanced.is_open()
        s.save()

    # ============================================================ handlers

    def _choose_file(self) -> None:
        start = self.settings.last_input_dir or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Choose a video", start, VIDEO_FILTER)
        if path:
            self._set_source(path)

    def _url_entered(self) -> None:
        text = self.url_edit.text().strip()
        if text and is_url(text):
            self._set_source(text)

    def _set_source(self, source: str) -> None:
        self.source = source
        if is_url(source):
            self.source_label.setText("URL - the video will be downloaded first")
            self.preview.set_video(None)
        else:
            path = Path(source)
            self.settings.last_input_dir = str(path.parent)
            size_mb = path.stat().st_size / (1024 ** 2) if path.is_file() else 0
            self.source_label.setText(f"{path.name}  ({size_mb:.0f} MB)")
            self.url_edit.clear()
            self.preview.set_video(path)
        self.start_button.setEnabled(True)
        self._update_output_name()
        self._request_preview()

    def _gender_changed(self) -> None:
        self._rebuild_voices()
        self._persist()

    def _rebuild_voices(self) -> None:
        gender = "female" if self.female_radio.isChecked() else "male"
        backend = self.tts_combo.currentData() or "piper"
        voices = voices_for(gender, backend)

        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        for voice in voices:
            self.voice_combo.addItem(voice.label, voice.key)
        index = self.voice_combo.findData(self.settings.voice)
        self.voice_combo.setCurrentIndex(index if index >= 0 else 0)
        self.voice_combo.blockSignals(False)

        if self.voice_combo.currentIndex() < 0 and voices:
            self._set_combo(self.voice_combo, default_voice(gender, backend))
        self._voice_changed()

    def _voice_changed(self) -> None:
        key = self.voice_combo.currentData()
        gender = "female" if self.female_radio.isChecked() else "male"
        backend = self.tts_combo.currentData() or "piper"
        note = ""
        for voice in voices_for(gender, backend):
            if voice.key == key:
                note = voice.note or ""
                if backend == "piper" and voice.disk_mb:
                    from .assets import piper_voice_present
                    if not piper_voice_present(key, self.models_dir):
                        note += (f"\nNot downloaded yet - about {voice.disk_mb:.0f} MB "
                                 f"will be fetched when you press Start.")
                break
        self.voice_note.setText(note)
        self._persist()

    def _preset_changed(self) -> None:
        key = self.preset_combo.currentData()
        preset = next((p for p in PRESETS if p.key == key), None)
        if not preset:
            return
        self.preset_note.setText(preset.description)
        # A preset drives the advanced combos, so the two never disagree.
        for combo, value in ((self.asr_combo, preset.asr),
                             (self.mt_combo, preset.translate),
                             (self.tts_combo, preset.tts)):
            combo.blockSignals(True)
            self._set_combo(combo, value)
            combo.blockSignals(False)
        self._rebuild_voices()
        self._refresh_requirements()
        self._persist()

    def _model_changed(self) -> None:
        self._rebuild_voices()
        self._refresh_requirements()
        self._persist()

    def _device_changed(self) -> None:
        self.fallback_check.setEnabled(not self.cpu_radio.isChecked())
        self._refresh_requirements()
        self._persist()

    def _subs_mode_changed(self) -> None:
        drawing = self.subs_combo.currentData() != "none"
        for widget in (self.size_slider, self.margin_slider, self.wrap_slider):
            widget.setEnabled(drawing)
        self._persist()
        self._request_preview()

    def _font_size_changed(self, value: int) -> None:
        self.size_value.setText(str(value))
        self._persist()
        self._request_preview()

    def _margin_changed(self, value: int) -> None:
        self.margin_value.setText(str(value))
        self._persist()
        self._request_preview()

    def _wrap_changed(self, value: int) -> None:
        self.wrap_value.setText(str(value))
        self._persist()
        self._request_preview()

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Save converted videos to",
                                                self.output_edit.text() or str(Path.home()))
        if path:
            self.output_edit.setText(path)
            self._update_output_name()
            self._persist()

    def _open_output_dir(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_edit.text()))

    # ------------------------------------------------------- drag and drop

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self._set_source(url.toLocalFile())
                break

    # ============================================================ derived

    def _on_gpu(self) -> bool:
        if self.cpu_radio.isChecked():
            return False
        return self.hardware.has_gpu

    def _refresh_requirements(self) -> None:
        on_gpu = self._on_gpu()
        vram = self.hardware.vram_gb if on_gpu else 0.0

        for combo in (self.asr_combo, self.mt_combo, self.tts_combo):
            option = find_option(combo.options, combo.currentData())  # type: ignore[attr-defined]
            if not option:
                continue
            combo.requirement_label.setText(  # type: ignore[attr-defined]
                f"{option.requirement_text(on_gpu)} · quality: {option.quality}"
                + (f"\n{option.note}" if option.note else ""))
            combo.badge.update_badge(*self._fit_badge(option, vram, on_gpu))  # type: ignore[attr-defined]

        preset = next((p for p in PRESETS if p.key == self.preset_combo.currentData()), None)
        if preset:
            requirements = preset_requirements(preset)
            if on_gpu:
                text = (f"peak {requirements['peak_vram_gb']:.1f} GB VRAM · "
                        f"{requirements['disk_gb']:.1f} GB download")
                level = ("ok" if vram >= requirements["peak_vram_gb"] + 0.4
                         else "warn" if vram >= requirements["peak_vram_gb"]
                         else "bad")
            else:
                text = (f"peak {requirements['peak_ram_gb']:.1f} GB RAM · "
                        f"{requirements['disk_gb']:.1f} GB download")
                level = "ok" if self.hardware.ram_gb >= requirements["peak_ram_gb"] else "warn"
            self.preset_badge.update_badge(text, level)

    def _fit_badge(self, option, vram: float, on_gpu: bool):
        if not on_gpu:
            if option.ram_gb <= self.hardware.ram_gb:
                return "runs on CPU", "ok"
            return f"needs {option.ram_gb:.0f} GB RAM", "bad"
        if option.vram_gb == 0.0:
            return "CPU only - GPU not used", "muted"
        headroom = vram - option.vram_gb
        if headroom >= 0.4:
            return "fits comfortably", "ok"
        if headroom >= 0.0:
            # device.whisper_compute_type quantises below 6 GB, which is what
            # makes this survivable rather than an out-of-memory crash.
            return "tight - will be quantised", "warn"
        return f"needs {option.vram_gb:.1f} GB VRAM", "bad"

    def _update_output_name(self) -> None:
        if not self.source:
            self.output_name.setText("")
            return
        self.output_name.setText(f"Will be saved as  {self._output_path().name}")

    def _output_path(self) -> Path:
        directory = Path(self.output_edit.text() or Path.home())
        if self.source and not is_url(self.source):
            stem = Path(self.source).stem
        else:
            stem = "video"
        candidate = directory / f"{stem}.en.mp4"
        # Never silently overwrite a previous conversion.
        counter = 2
        while candidate.exists():
            candidate = directory / f"{stem}.en.{counter}.mp4"
            counter += 1
        return candidate

    def _request_preview(self) -> None:
        if self.subs_combo.currentData() == "none":
            return
        self.preview.request(
            font_size=self.size_slider.value(),
            margin_v=self.margin_slider.value(),
            max_line_chars=self.wrap_slider.value(),
            font=self.settings.extra.get("font", "DejaVu Sans"),
        )

    # =============================================================== run

    def _overrides(self) -> dict:
        return {
            "general.device": ("cuda" if self.gpu_radio.isChecked()
                               else "cpu" if self.cpu_radio.isChecked() else "auto"),
            "asr.model": self.asr_combo.currentData(),
            "translate.backend": self.mt_combo.currentData(),
            "tts.backend": self.tts_combo.currentData(),
            "tts.voice" if self.tts_combo.currentData() == "piper" else "tts.kokoro_voice":
                self.voice_combo.currentData(),
            "separate.enabled": self.settings.keep_background,
            "subtitles.mode": self.subs_combo.currentData(),
            "subtitles.font_size": self.size_slider.value(),
            "subtitles.margin_v": self.margin_slider.value(),
            "subtitles.max_line_chars": self.wrap_slider.value(),
        }

    def _missing_assets(self):
        voice_key = self.voice_combo.currentData() or ""
        size = 65.0
        for voice in voices_for("female" if self.female_radio.isChecked() else "male",
                                self.tts_combo.currentData() or "piper"):
            if voice.key == voice_key:
                size = voice.disk_mb or 65.0
                break
        option = find_option(ASR_MODELS, self.asr_combo.currentData())
        return missing_for(
            voice=voice_key,
            tts_backend=self.tts_combo.currentData() or "piper",
            asr_model=self.asr_combo.currentData() or "small",
            translate_backend=self.mt_combo.currentData() or "nllb",
            models_dir=self.models_dir,
            voice_size_mb=size,
            asr_size_mb=(option.disk_gb * 1024) if option else 0.0,
        )

    def _start(self) -> None:
        if not self.source:
            return

        # Anything missing is fetched first. Discovering it at the stage that
        # needs it means failing after the user has already waited through
        # transcription, which is the worst possible moment.
        missing = self._missing_assets()
        if missing:
            self.log_view.clear()
            self._set_running(True)
            self.stage_label.setText("Downloading models...")
            self.overall_bar.setRange(0, 0)     # indeterminate
            self.started_at = time.time()
            self._on_log("First run with these settings. Downloading "
                         + summarise(missing) + ".", "info")

            args: list[str] = []
            for item in missing:
                args += item.download_args
            self.downloader = DownloadWorker(self)
            self.downloader.logged.connect(self._on_log)
            self.downloader.finished.connect(self._on_download_finished)
            self.downloader.start(args)
            return

        self.log_view.clear()
        self._begin_conversion()

    def _on_download_finished(self, ok: bool, message: str) -> None:
        self.overall_bar.setRange(0, 1000)
        if not ok:
            self._set_running(False)
            self.stage_label.setText("Download failed")
            self._on_log(message, "error")
            QMessageBox.critical(self, "Could not download models", message)
            return
        self._on_log("Models ready.", "info")
        self._begin_conversion()

    def _begin_conversion(self) -> None:
        if not self.source:
            return
        out_path = self._output_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        self.overall_bar.setValue(0)
        self.stage_bar.setValue(0)
        self.stage_label.setText("Starting...")
        self.started_at = time.time()
        self._set_running(True)

        self.worker = ConversionWorker(self)
        self.worker.overall.connect(self._on_overall)
        self.worker.stage.connect(self._on_stage)
        self.worker.logged.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start(
            source=self.source,
            out_path=out_path,
            overrides=self._overrides(),
            cpu_fallback=self.fallback_check.isChecked(),
        )
        self.last_output = out_path

    def _cancel(self) -> None:
        if self.downloader and self.downloader.running:
            self.downloader.cancel()
            self.overall_bar.setRange(0, 1000)
            self._set_running(False)
            self.stage_label.setText("Cancelled")
            return
        if self.worker:
            self.cancel_button.setEnabled(False)
            self.cancel_button.setText("Stopping...")
            self.worker.cancel()

    def _set_running(self, running: bool) -> None:
        self.start_button.setVisible(not running)
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running)
        self.cancel_button.setText("Cancel")
        self.open_button.setVisible(False)
        for widget in (self.preset_combo, self.asr_combo, self.mt_combo, self.tts_combo,
                       self.voice_combo, self.female_radio, self.male_radio,
                       self.auto_radio, self.gpu_radio, self.cpu_radio,
                       self.subs_combo, self.size_slider, self.margin_slider,
                       self.wrap_slider, self.url_edit, self.fallback_check):
            widget.setEnabled(not running)
        if not running and self.hardware.has_gpu is False:
            self.gpu_radio.setEnabled(False)

    # ------------------------------------------------------------ signals

    def _on_overall(self, value: float) -> None:
        self.overall_bar.setValue(int(value * 1000))
        elapsed = time.time() - self.started_at
        if value > 0.02:
            remaining = elapsed * (1 - value) / value
            self.elapsed_label.setText(
                f"{fmt_duration(elapsed)} elapsed · about {fmt_duration(remaining)} left")
        else:
            self.elapsed_label.setText(f"{fmt_duration(elapsed)} elapsed")

    def _on_stage(self, name: str, title: str, fraction: float) -> None:
        self.stage_label.setText(title)
        self.stage_bar.setValue(int(fraction * 1000))

    def _on_log(self, message: str, level: str) -> None:
        if level in ("debug",):
            return
        prefix = {"warning": "! ", "error": "X "}.get(level, "")
        self.log_view.appendPlainText(prefix + message)

    def _on_finished(self, message: dict) -> None:
        self._set_running(False)
        if message.get("t") == "cancelled":
            self.stage_label.setText("Cancelled")
            self.elapsed_label.setText("")
            self._on_log("Cancelled. Starting the same video again will resume "
                         "from the last finished step.", "warning")
            return

        self.overall_bar.setValue(1000)
        self.stage_bar.setValue(1000)
        self.stage_label.setText("Finished")
        self.open_button.setVisible(True)
        demoted = message.get("demoted") or []
        if demoted:
            self._on_log("These steps ran on the CPU after the GPU failed: "
                         + ", ".join(demoted), "warning")
        self.elapsed_label.setText(f"Done in {fmt_duration(time.time() - self.started_at)}")
        self._on_log(f"Saved to {message.get('output', '')}", "info")

    def _on_failed(self, reason: str) -> None:
        self._set_running(False)
        self.stage_label.setText("Failed")
        self.elapsed_label.setText("")
        self._on_log(reason, "error")
        QMessageBox.critical(self, "Conversion failed", reason)

    # ------------------------------------------------------------- close

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker and self.worker.running:
            answer = QMessageBox.question(
                self, "Conversion in progress",
                "A conversion is still running. Stop it and quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            self.worker.kill()
        self.preview.shutdown()
        self._persist()
        event.accept()
