"""Startup overlay that reflects resident service preload progress."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget


SERVICE_DISPLAY_NAMES = {
    "detection": "Detection",
    "ocr": "OCR",
    "translation": "Translation",
    "inpaint": "Inpaint",
    "render": "Render",
    "export": "Export",
    "process": "Process",
}


class StartupOverlay(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("StartupOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            """
            QFrame#StartupOverlay {
                background: rgba(15, 23, 42, 220);
            }
            QFrame#StartupOverlayCard {
                background: rgba(17, 24, 39, 245);
                border: 1px solid rgba(59, 130, 246, 140);
                border-radius: 18px;
            }
            QLabel#StartupOverlayTitle {
                color: #e5e7eb;
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#StartupOverlayMessage {
                color: #cbd5e1;
                font-size: 12px;
            }
            QLabel#StartupOverlayServiceName {
                color: #e5e7eb;
                font-weight: 600;
            }
            QLabel#StartupOverlayServiceState {
                color: #cbd5e1;
            }
            """
        )
        self._rows: dict[str, tuple[QLabel, QLabel]] = {}

        shell = QVBoxLayout(self)
        shell.setContentsMargins(24, 24, 24, 24)
        shell.setSpacing(18)

        card = QFrame(self)
        card.setObjectName("StartupOverlayCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(16)

        self.title_label = QLabel("Starting resident services...")
        self.title_label.setObjectName("StartupOverlayTitle")
        card_layout.addWidget(self.title_label)

        self.message_label = QLabel("Loading models and workers...")
        self.message_label.setObjectName("StartupOverlayMessage")
        self.message_label.setWordWrap(True)
        card_layout.addWidget(self.message_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(8)
        row = 0
        for key in ("detection", "ocr", "translation", "inpaint", "render", "export", "process"):
            name_label = QLabel(SERVICE_DISPLAY_NAMES[key])
            name_label.setObjectName("StartupOverlayServiceName")
            state_label = QLabel("Starting...")
            state_label.setObjectName("StartupOverlayServiceState")
            grid.addWidget(name_label, row, 0)
            grid.addWidget(state_label, row, 1)
            self._rows[key] = (name_label, state_label)
            row += 1
        card_layout.addLayout(grid)
        shell.addStretch(1)
        shell.addWidget(card, 0, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        shell.addStretch(1)

    def set_overall_message(self, message: str) -> None:
        self.message_label.setText(str(message or "Loading models and workers..."))

    def set_service_status(self, service_name: str, state: str, message: str | None = None) -> None:
        normalized = str(service_name or "").strip().lower()
        row = self._rows.get(normalized)
        if row is None:
            return
        _name_label, state_label = row
        display_state = str(state or "").strip().lower() or "starting"
        state_text = display_state.replace("_", " ").title()
        detail_text = str(message or "").strip()
        state_label.setText(f"{state_text}: {detail_text}" if detail_text else state_text)
        state_label.setProperty("startupState", display_state)
        state_label.style().unpolish(state_label)
        state_label.style().polish(state_label)


__all__ = ["StartupOverlay", "SERVICE_DISPLAY_NAMES"]
