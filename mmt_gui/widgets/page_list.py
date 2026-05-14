"""Page list widget for project source images."""

from __future__ import annotations

from collections.abc import Sequence

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QListWidget, QListWidgetItem


class PageListWidget(QListWidget):
    """Lists imported page filenames and emits row selections."""

    page_selected = pyqtSignal(int)

    def __init__(self, parent: QListWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.currentRowChanged.connect(self._emit_page_selected)

    def set_pages(self, page_names: Sequence[str], selected_index: int | None = None) -> None:
        self.clear()

        for page_name in page_names:
            self.addItem(QListWidgetItem(page_name))

        if not page_names:
            return

        if selected_index is None or selected_index < 0 or selected_index >= len(page_names):
            selected_index = 0

        self.setCurrentRow(selected_index)

    def _emit_page_selected(self, row: int) -> None:
        if row >= 0:
            self.page_selected.emit(row)
