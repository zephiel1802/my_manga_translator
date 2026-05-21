"""Render stage inspector panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
)

from mmt_core import (
    RenderConfig,
    has_active_style_overrides,
    list_project_fonts,
    normalize_render_style_overrides,
    parse_color_value,
    summarize_render_edit_state,
    summarize_render_json,
)
from mmt_gui.widgets import CollapsibleSection, StaticSection, TextItemEditorWidget
from mmt_gui.widgets.settings_card import style_button

from .base_panel import StagePanel


class RenderPanel(StagePanel):
    """Inspector panel for render settings, metadata, and render box editing."""

    prepare_selected_requested = pyqtSignal()
    reprepare_selected_requested = pyqtSignal()
    prepare_all_requested = pyqtSignal()
    reprepare_all_requested = pyqtSignal()
    run_selected_requested = pyqtSignal()
    rerun_selected_requested = pyqtSignal()
    run_all_requested = pyqtSignal()
    rerun_all_requested = pyqtSignal()
    reload_requested = pyqtSignal()
    clear_preview_requested = pyqtSignal()
    box_edit_mode_toggled = pyqtSignal(bool)
    save_box_edits_requested = pyqtSignal()
    cancel_box_edits_requested = pyqtSignal()
    exclude_selected_box_requested = pyqtSignal()
    restore_selected_box_requested = pyqtSignal()
    show_excluded_items_toggled = pyqtSignal(bool)
    reload_box_cache_requested = pyqtSignal()
    current_item_changed = pyqtSignal(int)
    save_style_edits_requested = pyqtSignal()
    translated_text_double_clicked = pyqtSignal(int)

    def __init__(self, workspace_root: Path, parent: object | None = None) -> None:
        super().__init__("Render", parent)
        self.workspace_root = workspace_root
        self._all_items: list[dict[str, Any]] = []
        self._items: list[dict[str, Any]] = []
        self._actions_enabled = True
        self._box_edit_dirty = False
        self._selected_box: dict[str, Any] | None = None
        self._style_editor_guard = False
        self._style_editor_dirty = False
        self._style_editor_loaded_item_id: int | None = None
        self._style_editor_loaded_snapshot: dict[str, Any] | None = None
        self._style_editor_translated_text = ""
        self._pending_style_snapshots_by_item_id: dict[int, dict[str, Any]] = {}

        actions_card = StaticSection("Render Action", expanded=True)
        self.actions_section = actions_card
        actions_layout = QGridLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setHorizontalSpacing(8)
        actions_layout.setVerticalSpacing(8)

        prepare_current_label = QLabel("Prepare Current")
        prepare_current_label.setProperty("role", "muted")
        actions_layout.addWidget(prepare_current_label, 0, 0)

        self.prepare_selected_button = QPushButton("Prepare")
        style_button(self.prepare_selected_button, "primary")
        self.prepare_selected_button.clicked.connect(self.prepare_selected_requested.emit)
        self.prepare_selected_button.setToolTip("Prepare render metadata for the current page and reuse cache when available.")
        actions_layout.addWidget(self.prepare_selected_button, 0, 1)

        self.reprepare_selected_button = QPushButton("Re-prepare")
        style_button(self.reprepare_selected_button, "rerun")
        self.reprepare_selected_button.clicked.connect(self.reprepare_selected_requested.emit)
        self.reprepare_selected_button.setToolTip("Force the current page to rebuild render metadata.")
        actions_layout.addWidget(self.reprepare_selected_button, 0, 2)

        prepare_all_label = QLabel("Prepare All")
        prepare_all_label.setProperty("role", "muted")
        actions_layout.addWidget(prepare_all_label, 1, 0)

        self.prepare_all_button = QPushButton("Prepare All")
        style_button(self.prepare_all_button, "primary")
        self.prepare_all_button.clicked.connect(self.prepare_all_requested.emit)
        self.prepare_all_button.setToolTip("Prepare render metadata for every page and reuse cache when available.")
        actions_layout.addWidget(self.prepare_all_button, 1, 1)

        self.reprepare_all_button = QPushButton("Re-prepare All")
        style_button(self.reprepare_all_button, "rerun")
        self.reprepare_all_button.clicked.connect(self.reprepare_all_requested.emit)
        self.reprepare_all_button.setToolTip("Force every page to rebuild render metadata.")
        actions_layout.addWidget(self.reprepare_all_button, 1, 2)

        render_current_label = QLabel("Render Current")
        render_current_label.setProperty("role", "muted")
        actions_layout.addWidget(render_current_label, 2, 0)

        self.run_selected_button = QPushButton("Render")
        style_button(self.run_selected_button, "primary")
        self.run_selected_button.clicked.connect(self.run_selected_requested.emit)
        self.run_selected_button.setToolTip("Render the current page and reuse cached output when available.")
        actions_layout.addWidget(self.run_selected_button, 2, 1)

        self.rerun_selected_button = QPushButton("Re-render")
        style_button(self.rerun_selected_button, "rerun")
        self.rerun_selected_button.clicked.connect(self.rerun_selected_requested.emit)
        self.rerun_selected_button.setToolTip("Force the current page to regenerate rendered output.")
        actions_layout.addWidget(self.rerun_selected_button, 2, 2)

        render_all_label = QLabel("Render All")
        render_all_label.setProperty("role", "muted")
        actions_layout.addWidget(render_all_label, 3, 0)

        self.run_all_button = QPushButton("Render All")
        style_button(self.run_all_button, "primary")
        self.run_all_button.clicked.connect(self.run_all_requested.emit)
        self.run_all_button.setToolTip("Render every page and reuse cached output when available.")
        actions_layout.addWidget(self.run_all_button, 3, 1)

        self.rerun_all_button = QPushButton("Re-render All")
        style_button(self.rerun_all_button, "rerun")
        self.rerun_all_button.clicked.connect(self.rerun_all_requested.emit)
        self.rerun_all_button.setToolTip("Force every page to regenerate rendered output.")
        actions_layout.addWidget(self.rerun_all_button, 3, 2)

        cache_label = QLabel("Cache")
        cache_label.setProperty("role", "muted")
        actions_layout.addWidget(cache_label, 4, 0)

        self.reload_button = QPushButton("Reload Cache")
        style_button(self.reload_button, "secondary")
        self.reload_button.clicked.connect(self.reload_requested.emit)
        self.reload_button.setToolTip("Reload render data from disk.")
        actions_layout.addWidget(self.reload_button, 4, 1)

        self.clear_preview_button = QPushButton("Clear Render Preview")
        style_button(self.clear_preview_button, "danger")
        self.clear_preview_button.clicked.connect(self.clear_preview_requested.emit)
        self.clear_preview_button.setToolTip("Return the preview to the source page.")
        actions_layout.addWidget(self.clear_preview_button, 4, 2)
        actions_card.content_layout.addLayout(actions_layout)
        self.content_layout.addWidget(actions_card)

        settings_card = CollapsibleSection("Render Settings", expanded=False)
        self.settings_section = settings_card
        settings_form = QFormLayout()
        settings_form.setContentsMargins(0, 0, 0, 0)
        settings_form.setSpacing(8)

        self.font_name_input = QComboBox()
        self.font_name_input.setEditable(True)
        self.font_name_input.addItem("")
        for display_name, _font_path in list_project_fonts(workspace_root):
            self.font_name_input.addItem(display_name)
        if self.font_name_input.count() > 1:
            self.font_name_input.setCurrentIndex(1)

        self.font_path_input = QLineEdit()
        self.font_path_input.setPlaceholderText("Optional explicit font file path.")

        self.min_font_size_input = QSpinBox()
        self.min_font_size_input.setRange(6, 256)
        self.min_font_size_input.setValue(12)
        self.max_font_size_input = QSpinBox()
        self.max_font_size_input.setRange(6, 512)
        self.max_font_size_input.setValue(72)

        self.stroke_enabled_checkbox = QCheckBox("Stroke enabled")
        self.stroke_enabled_checkbox.setChecked(True)
        self.stroke_width_input = QDoubleSpinBox()
        self.stroke_width_input.setRange(0.0, 20.0)
        self.stroke_width_input.setSingleStep(0.5)
        self.stroke_width_input.setValue(0.0)
        self.stroke_width_input.setSpecialValueText("Auto")

        self.text_color_input = QLineEdit("auto")
        self.stroke_color_input = QLineEdit("auto")

        self.auto_color_checkbox = QCheckBox("Auto color")
        self.auto_color_checkbox.setChecked(True)
        self.auto_direction_checkbox = QCheckBox("Auto direction")
        self.auto_direction_checkbox.setChecked(True)
        self.vertical_cjk_checkbox = QCheckBox("Vertical CJK")
        self.vertical_cjk_checkbox.setChecked(True)
        self.save_sprites_checkbox = QCheckBox("Save sprites")
        self.save_sprites_checkbox.setChecked(True)

        settings_form.addRow("Font:", self.font_name_input)
        settings_form.addRow("Font Path:", self.font_path_input)
        settings_form.addRow("Min Font Size:", self.min_font_size_input)
        settings_form.addRow("Max Font Size:", self.max_font_size_input)
        settings_form.addRow("Stroke Enabled:", self.stroke_enabled_checkbox)
        settings_form.addRow("Stroke Width:", self.stroke_width_input)
        settings_form.addRow("Text Color:", self.text_color_input)
        settings_form.addRow("Stroke Color:", self.stroke_color_input)
        settings_form.addRow("Auto Color:", self.auto_color_checkbox)
        settings_form.addRow("Auto Direction:", self.auto_direction_checkbox)
        settings_form.addRow("Vertical CJK:", self.vertical_cjk_checkbox)
        settings_form.addRow("Save Sprites:", self.save_sprites_checkbox)
        settings_card.content_layout.addLayout(settings_form)
        self.content_layout.addWidget(settings_card)

        items_card = CollapsibleSection("Render Items", expanded=True)
        self.items_section = items_card
        self.items_table = QTableWidget(0, 8)
        self.items_table.setProperty("stageTable", True)
        self.items_table.setHorizontalHeaderLabels(
            ["id", "kind", "writing_mode", "font_size", "status", "translated_text", "override_text", "override_status"]
        )
        self.items_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.items_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.items_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.items_table.setAlternatingRowColors(True)
        self.items_table.itemSelectionChanged.connect(self._on_item_selected)
        self.items_table.cellDoubleClicked.connect(self._on_item_double_clicked)
        header = self.items_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        self.items_table.setMinimumHeight(280)
        items_card.content_layout.addWidget(self.items_table)
        self.content_layout.addWidget(items_card)

        self.item_editor_section = StaticSection("Render Item Editor", expanded=True)
        item_editor_info_layout = QGridLayout()
        item_editor_info_layout.setContentsMargins(0, 0, 0, 0)
        item_editor_info_layout.setHorizontalSpacing(8)
        item_editor_info_layout.setVerticalSpacing(6)

        self.style_editor_details_label = QLabel(
            "Select a render item to override rendered text, font, size, direction, and spacing for that item only."
        )
        self.style_editor_details_label.setWordWrap(True)
        self.style_editor_details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        item_editor_info_layout.addWidget(self.style_editor_details_label, 0, 0, 1, 3)

        self.style_dirty_label = QLabel("Saved")
        self.style_dirty_label.setProperty("role", "muted")
        item_editor_info_layout.addWidget(self.style_dirty_label, 1, 0)

        self.style_override_hint_label = QLabel("")
        self.style_override_hint_label.setProperty("role", "muted")
        self.style_override_hint_label.setWordWrap(True)
        self.style_override_hint_label.setVisible(False)
        item_editor_info_layout.addWidget(self.style_override_hint_label, 1, 1, 1, 2)
        self.item_editor_section.content_layout.addLayout(item_editor_info_layout)

        item_editor_actions_layout = QGridLayout()
        item_editor_actions_layout.setContentsMargins(0, 0, 0, 0)
        item_editor_actions_layout.setHorizontalSpacing(8)
        item_editor_actions_layout.setVerticalSpacing(8)

        self.save_style_edits_button = QPushButton("Save Item Style")
        style_button(self.save_style_edits_button, "primary")
        self.save_style_edits_button.clicked.connect(self.save_style_edits_requested.emit)
        item_editor_actions_layout.addWidget(self.save_style_edits_button, 0, 0)

        self.revert_style_edits_button = QPushButton("Revert Unsaved Style")
        style_button(self.revert_style_edits_button, "secondary")
        self.revert_style_edits_button.clicked.connect(self.revert_current_style_editor)
        item_editor_actions_layout.addWidget(self.revert_style_edits_button, 0, 1)
        self.item_editor_section.content_layout.addLayout(item_editor_actions_layout)

        self.override_enabled_checkbox = QCheckBox("Enable Item Overrides")
        self.override_enabled_checkbox.toggled.connect(self._on_style_editor_changed)
        self.item_editor_section.content_layout.addWidget(self.override_enabled_checkbox)

        self.render_text_override_editor = TextItemEditorWidget(
            "Render Text Override",
            placeholder="Leave blank to render the translated text from Translation.",
        )
        self.render_text_override_editor.dirty_changed.connect(self._on_style_editor_changed)
        self.item_editor_section.content_layout.addWidget(self.render_text_override_editor)

        item_style_form = QFormLayout()
        item_style_form.setContentsMargins(0, 0, 0, 0)
        item_style_form.setSpacing(8)

        self.item_font_name_input = QComboBox()
        self.item_font_name_input.setEditable(True)
        self.item_font_name_input.addItem("")
        for display_name, _font_path in list_project_fonts(workspace_root):
            self.item_font_name_input.addItem(display_name)
        self.item_font_name_input.currentTextChanged.connect(self._on_style_editor_changed)

        self.item_font_path_input = QLineEdit()
        self.item_font_path_input.setPlaceholderText("Optional explicit font file path for this item.")
        self.item_font_path_input.textChanged.connect(self._on_style_editor_changed)

        self.item_font_size_mode_input = QComboBox()
        self.item_font_size_mode_input.addItem("Inherit Global Fit", "inherit")
        self.item_font_size_mode_input.addItem("Fit In This Box", "fit")
        self.item_font_size_mode_input.addItem("Fixed Font Size", "fixed")
        self.item_font_size_mode_input.currentIndexChanged.connect(self._on_style_editor_changed)
        self.item_font_size_mode_input.currentIndexChanged.connect(self._update_style_editor_field_visibility)

        self.item_fixed_font_size_input = QSpinBox()
        self.item_fixed_font_size_input.setRange(0, 512)
        self.item_fixed_font_size_input.setSpecialValueText("Inherit")
        self.item_fixed_font_size_input.valueChanged.connect(self._on_style_editor_changed)

        self.item_min_font_size_input = QSpinBox()
        self.item_min_font_size_input.setRange(0, 256)
        self.item_min_font_size_input.setSpecialValueText("Inherit")
        self.item_min_font_size_input.valueChanged.connect(self._on_style_editor_changed)

        self.item_max_font_size_input = QSpinBox()
        self.item_max_font_size_input.setRange(0, 512)
        self.item_max_font_size_input.setSpecialValueText("Inherit")
        self.item_max_font_size_input.valueChanged.connect(self._on_style_editor_changed)

        self.item_writing_mode_input = QComboBox()
        self.item_writing_mode_input.addItem("Inherit Global", "inherit")
        self.item_writing_mode_input.addItem("Auto For This Item", "auto")
        self.item_writing_mode_input.addItem("Horizontal", "horizontal")
        self.item_writing_mode_input.addItem("Vertical RL", "vertical_rl")
        self.item_writing_mode_input.currentIndexChanged.connect(self._on_style_editor_changed)

        self.item_stroke_enabled_input = QComboBox()
        self.item_stroke_enabled_input.addItem("Inherit", None)
        self.item_stroke_enabled_input.addItem("Enabled", True)
        self.item_stroke_enabled_input.addItem("Disabled", False)
        self.item_stroke_enabled_input.currentIndexChanged.connect(self._on_style_editor_changed)

        self.item_stroke_width_input = QDoubleSpinBox()
        self.item_stroke_width_input.setRange(0.0, 20.0)
        self.item_stroke_width_input.setSingleStep(0.5)
        self.item_stroke_width_input.setValue(0.0)
        self.item_stroke_width_input.setSpecialValueText("Inherit")
        self.item_stroke_width_input.valueChanged.connect(self._on_style_editor_changed)

        self.item_text_color_mode_input = QComboBox()
        self.item_text_color_mode_input.addItem("Inherit Global", "inherit")
        self.item_text_color_mode_input.addItem("Auto For This Item", "auto")
        self.item_text_color_mode_input.addItem("Custom Colors", "custom")
        self.item_text_color_mode_input.currentIndexChanged.connect(self._on_style_editor_changed)
        self.item_text_color_mode_input.currentIndexChanged.connect(self._update_style_editor_field_visibility)

        self.item_text_color_input = QLineEdit()
        self.item_text_color_input.setPlaceholderText("auto or #RRGGBB or R,G,B")
        self.item_text_color_input.textChanged.connect(self._on_style_editor_changed)

        self.item_stroke_color_input = QLineEdit()
        self.item_stroke_color_input.setPlaceholderText("auto or #RRGGBB or R,G,B")
        self.item_stroke_color_input.textChanged.connect(self._on_style_editor_changed)

        self.item_line_spacing_ratio_input = QDoubleSpinBox()
        self.item_line_spacing_ratio_input.setDecimals(2)
        self.item_line_spacing_ratio_input.setRange(0.0, 1.5)
        self.item_line_spacing_ratio_input.setSingleStep(0.02)
        self.item_line_spacing_ratio_input.setValue(0.0)
        self.item_line_spacing_ratio_input.setSpecialValueText("Inherit")
        self.item_line_spacing_ratio_input.valueChanged.connect(self._on_style_editor_changed)

        item_style_form.addRow("Font:", self.item_font_name_input)
        item_style_form.addRow("Font Path:", self.item_font_path_input)
        item_style_form.addRow("Font Size Mode:", self.item_font_size_mode_input)
        item_style_form.addRow("Fixed Font Size:", self.item_fixed_font_size_input)
        item_style_form.addRow("Min Font Size:", self.item_min_font_size_input)
        item_style_form.addRow("Max Font Size:", self.item_max_font_size_input)
        item_style_form.addRow("Writing Mode:", self.item_writing_mode_input)
        item_style_form.addRow("Stroke:", self.item_stroke_enabled_input)
        item_style_form.addRow("Stroke Width:", self.item_stroke_width_input)
        item_style_form.addRow("Color Mode:", self.item_text_color_mode_input)
        item_style_form.addRow("Text Color:", self.item_text_color_input)
        item_style_form.addRow("Stroke Color:", self.item_stroke_color_input)
        item_style_form.addRow("Line Spacing Ratio:", self.item_line_spacing_ratio_input)
        self.item_editor_section.content_layout.addLayout(item_style_form)

        self.content_layout.addWidget(self.item_editor_section)

        self.box_editor_section = CollapsibleSection("Render Box Editor", expanded=False)
        box_editor_layout = QGridLayout()
        box_editor_layout.setContentsMargins(0, 0, 0, 0)
        box_editor_layout.setHorizontalSpacing(8)
        box_editor_layout.setVerticalSpacing(8)

        self.enable_box_edit_checkbox = QCheckBox("Enable Render Box Editing")
        self.enable_box_edit_checkbox.toggled.connect(self.box_edit_mode_toggled.emit)
        box_editor_layout.addWidget(self.enable_box_edit_checkbox, 0, 0, 1, 2)

        self.show_excluded_items_checkbox = QCheckBox("Show Excluded Render Items")
        self.show_excluded_items_checkbox.toggled.connect(self._on_show_excluded_toggled)
        box_editor_layout.addWidget(self.show_excluded_items_checkbox, 0, 2)

        self.save_box_edits_button = QPushButton("Save Box Edits")
        style_button(self.save_box_edits_button, "primary")
        self.save_box_edits_button.clicked.connect(self.save_box_edits_requested.emit)
        self.save_box_edits_button.setToolTip("Write edited render boxes back to the cached render JSON.")
        box_editor_layout.addWidget(self.save_box_edits_button, 1, 0)

        self.cancel_box_edits_button = QPushButton("Cancel Unsaved Edits")
        style_button(self.cancel_box_edits_button, "secondary")
        self.cancel_box_edits_button.clicked.connect(self.cancel_box_edits_requested.emit)
        self.cancel_box_edits_button.setToolTip("Discard in-memory render box edits and reload the cached render JSON.")
        box_editor_layout.addWidget(self.cancel_box_edits_button, 1, 1)

        self.reload_box_cache_button = QPushButton("Reload Boxes")
        style_button(self.reload_box_cache_button, "secondary")
        self.reload_box_cache_button.clicked.connect(self.reload_box_cache_requested.emit)
        self.reload_box_cache_button.setToolTip("Reload editable render boxes from the cached render JSON.")
        box_editor_layout.addWidget(self.reload_box_cache_button, 1, 2)

        self.exclude_selected_box_button = QPushButton("Delete / Exclude")
        style_button(self.exclude_selected_box_button, "danger")
        self.exclude_selected_box_button.clicked.connect(self.exclude_selected_box_requested.emit)
        self.exclude_selected_box_button.setToolTip("Soft-delete the selected render item by marking it excluded.")
        box_editor_layout.addWidget(self.exclude_selected_box_button, 2, 0)

        self.restore_selected_box_button = QPushButton("Restore Selected")
        style_button(self.restore_selected_box_button, "secondary")
        self.restore_selected_box_button.clicked.connect(self.restore_selected_box_requested.emit)
        self.restore_selected_box_button.setToolTip("Restore an excluded render item when Show Excluded Render Items is enabled.")
        box_editor_layout.addWidget(self.restore_selected_box_button, 2, 1)
        self.box_editor_section.content_layout.addLayout(box_editor_layout)

        self.box_dirty_label = QLabel("")
        self.box_dirty_label.setProperty("role", "muted")
        self.box_dirty_label.setVisible(False)
        self.box_editor_section.content_layout.addWidget(self.box_dirty_label)

        self.box_warning_label = QLabel("")
        self.box_warning_label.setProperty("role", "muted")
        self.box_warning_label.setWordWrap(True)
        self.box_warning_label.setVisible(False)
        self.box_editor_section.content_layout.addWidget(self.box_warning_label)

        selected_form = QFormLayout()
        selected_form.setContentsMargins(0, 0, 0, 0)
        selected_form.setSpacing(8)
        self.selected_box_id_value = QLabel("-")
        self.selected_box_translation_id_value = QLabel("-")
        self.selected_box_ocr_id_value = QLabel("-")
        self.selected_box_kind_value = QLabel("-")
        self.selected_box_render_bbox_value = QLabel("-")
        self.selected_box_writing_mode_value = QLabel("-")
        self.selected_box_font_size_value = QLabel("-")
        self.selected_box_status_value = QLabel("-")
        self.selected_box_sprite_value = QLabel("-")
        self.selected_box_excluded_value = QLabel("-")
        self.selected_box_needs_render_value = QLabel("-")
        self.selected_box_error_value = QLabel("-")
        self.selected_box_error_value.setWordWrap(True)

        selected_form.addRow("Selected ID:", self.selected_box_id_value)
        selected_form.addRow("Translation Item:", self.selected_box_translation_id_value)
        selected_form.addRow("OCR Item:", self.selected_box_ocr_id_value)
        selected_form.addRow("Kind:", self.selected_box_kind_value)
        selected_form.addRow("Render BBox:", self.selected_box_render_bbox_value)
        selected_form.addRow("Writing Mode:", self.selected_box_writing_mode_value)
        selected_form.addRow("Font Size:", self.selected_box_font_size_value)
        selected_form.addRow("Status:", self.selected_box_status_value)
        selected_form.addRow("Sprite Path:", self.selected_box_sprite_value)
        selected_form.addRow("Excluded:", self.selected_box_excluded_value)
        selected_form.addRow("Needs Render:", self.selected_box_needs_render_value)
        selected_form.addRow("Error:", self.selected_box_error_value)
        self.box_editor_section.content_layout.addLayout(selected_form)
        self.content_layout.addWidget(self.box_editor_section)

        self._update_box_editor_state()

    def config_sections(self) -> list[QWidget]:
        return [self.settings_section]

    def simplify_for_config_stage(self) -> None:
        # ConfigPanel already reparents Render Settings into the shared Config stage.
        # Keep render-specific inspector sections attached so they remain visible
        # on the dedicated Render tab.
        return

    def config(self, *, force_override: bool | None = None) -> RenderConfig:
        try:
            parsed_text_color = parse_color_value(self.text_color_input.text())
        except ValueError as exc:
            raise ValueError(f"Invalid text color value. {exc}") from exc

        try:
            parsed_stroke_color = parse_color_value(self.stroke_color_input.text())
        except ValueError as exc:
            raise ValueError(f"Invalid stroke color value. {exc}") from exc

        stroke_width_value = self.stroke_width_input.value()
        return RenderConfig(
            font_name=self.font_name_input.currentText().strip(),
            font_path=self.font_path_input.text().strip(),
            min_font_size=self.min_font_size_input.value(),
            max_font_size=max(self.min_font_size_input.value(), self.max_font_size_input.value()),
            stroke_enabled=self.stroke_enabled_checkbox.isChecked(),
            stroke_width=stroke_width_value if stroke_width_value > 0 else None,
            text_color=parsed_text_color,
            stroke_color=parsed_stroke_color,
            auto_color=self.auto_color_checkbox.isChecked(),
            auto_direction=self.auto_direction_checkbox.isChecked(),
            vertical_cjk=self.vertical_cjk_checkbox.isChecked(),
            save_sprites=self.save_sprites_checkbox.isChecked(),
            force=bool(force_override) if force_override is not None else False,
        )

    def set_data(
        self,
        render_data: dict[str, Any] | None,
        output_image: str | None = None,
        *,
        preserve_pending_style_edits: bool = False,
    ) -> None:
        payload = render_data or {}
        previous_item_id = self.current_table_item_id()
        if not preserve_pending_style_edits:
            self.clear_pending_style_edit()
            self._set_style_editor_dirty(False)
        self._all_items = list(payload.get("items", [])) if isinstance(payload.get("items"), list) else []
        self._rebuild_items_table(previous_item_id=previous_item_id)

        edit_summary = summarize_render_edit_state(payload if payload else {"items": self._all_items})
        if edit_summary.get("needs_render"):
            self.set_box_warning("Render edits changed. Re-render is recommended.")
        else:
            self.set_box_warning(None)

    def clear_view(self, *, output_image: str | None = None) -> None:
        self.clear_pending_style_edit()
        self._all_items = []
        self._items = []
        self.items_table.blockSignals(True)
        self.items_table.setRowCount(0)
        self.items_table.blockSignals(False)
        self.set_selected_box(None)
        self.set_box_dirty(False)
        self._set_style_editor_dirty(False)
        self.set_box_warning(None)
        self.box_editor_section.set_expanded(False)

    def set_actions_enabled(self, enabled: bool) -> None:
        self._actions_enabled = bool(enabled)
        for widget in (
            self.prepare_selected_button,
            self.reprepare_selected_button,
            self.prepare_all_button,
            self.reprepare_all_button,
            self.run_selected_button,
            self.rerun_selected_button,
            self.run_all_button,
            self.rerun_all_button,
            self.reload_button,
            self.clear_preview_button,
            self.font_name_input,
            self.font_path_input,
            self.min_font_size_input,
            self.max_font_size_input,
            self.stroke_enabled_checkbox,
            self.stroke_width_input,
            self.text_color_input,
            self.stroke_color_input,
            self.auto_color_checkbox,
            self.auto_direction_checkbox,
            self.vertical_cjk_checkbox,
            self.save_sprites_checkbox,
            self.items_table,
            self.enable_box_edit_checkbox,
            self.show_excluded_items_checkbox,
            self.save_box_edits_button,
            self.cancel_box_edits_button,
            self.reload_box_cache_button,
            self.exclude_selected_box_button,
            self.restore_selected_box_button,
        ):
            widget.setEnabled(enabled)
        self._update_box_editor_state()

    def set_box_edit_mode_checked(self, enabled: bool) -> None:
        self.enable_box_edit_checkbox.blockSignals(True)
        self.enable_box_edit_checkbox.setChecked(bool(enabled))
        self.enable_box_edit_checkbox.blockSignals(False)
        self._update_box_editor_state()

    def box_edit_mode_enabled(self) -> bool:
        return self.enable_box_edit_checkbox.isChecked()

    def set_show_excluded_items_checked(self, enabled: bool) -> None:
        self.show_excluded_items_checkbox.blockSignals(True)
        self.show_excluded_items_checkbox.setChecked(bool(enabled))
        self.show_excluded_items_checkbox.blockSignals(False)
        self._rebuild_items_table(previous_item_id=self.current_table_item_id())
        self._update_box_editor_state()

    def show_excluded_items_enabled(self) -> bool:
        return self.show_excluded_items_checkbox.isChecked()

    def set_box_dirty(self, dirty: bool) -> None:
        self._box_edit_dirty = bool(dirty)
        self.box_dirty_label.setVisible(self._box_edit_dirty)
        self.box_dirty_label.setText("Unsaved render box edits")
        self._update_box_editor_state()

    def has_unsaved_box_edits(self) -> bool:
        return self._box_edit_dirty

    def select_item_by_id(self, item_id: int | None) -> bool:
        row_index = self._row_for_item_id(item_id)
        if row_index is not None:
            self.items_table.setCurrentCell(row_index, 0)
            return self.current_table_item_id() == item_id
        return False

    def current_table_item_id(self) -> int | None:
        row_index = self.items_table.currentRow()
        if row_index < 0 or row_index >= len(self._items):
            return None
        return int(self._items[row_index].get("id", row_index))

    def set_selected_box(self, box_data: dict[str, Any] | None) -> None:
        self._selected_box = dict(box_data) if isinstance(box_data, dict) else None
        if self._selected_box is None:
            self.selected_box_id_value.setText("-")
            self.selected_box_translation_id_value.setText("-")
            self.selected_box_ocr_id_value.setText("-")
            self.selected_box_kind_value.setText("-")
            self.selected_box_render_bbox_value.setText("-")
            self.selected_box_writing_mode_value.setText("-")
            self.selected_box_font_size_value.setText("-")
            self.selected_box_status_value.setText("-")
            self.selected_box_sprite_value.setText("-")
            self.selected_box_excluded_value.setText("-")
            self.selected_box_needs_render_value.setText("-")
            self.selected_box_error_value.setText("-")
            self._update_box_editor_state()
            return

        item = self._selected_box
        self.selected_box_id_value.setText(str(item.get("id", "-")))
        self.selected_box_translation_id_value.setText(str(item.get("translation_item_id", "-")))
        self.selected_box_ocr_id_value.setText(str(item.get("ocr_item_id", "-")))
        self.selected_box_kind_value.setText(str(item.get("kind", "-")))
        self.selected_box_render_bbox_value.setText(self._format_bbox(item.get("render_bbox")))
        self.selected_box_writing_mode_value.setText(str(item.get("writing_mode", "-") or "-"))
        self.selected_box_font_size_value.setText(str(item.get("font_size", "-") or "-"))
        self.selected_box_status_value.setText(str(item.get("status", "-") or "-"))
        self.selected_box_sprite_value.setText(str(item.get("sprite_path", "-") or "-"))
        self.selected_box_excluded_value.setText("Yes" if bool(item.get("excluded", False)) else "No")
        self.selected_box_needs_render_value.setText("Yes" if bool(item.get("needs_render", False)) else "No")
        self.selected_box_error_value.setText(str(item.get("error", "-") or "-"))
        self._update_box_editor_state()

    def selected_box(self) -> dict[str, Any] | None:
        return dict(self._selected_box) if self._selected_box is not None else None

    def set_box_warning(self, text: str | None) -> None:
        normalized = str(text or "").strip()
        self.box_warning_label.setText(normalized)
        self.box_warning_label.setVisible(bool(normalized))

    def settings_snapshot(self) -> dict[str, Any]:
        return {
            "font_name": self.font_name_input.currentText().strip(),
            "font_path": self.font_path_input.text().strip(),
            "min_font_size": self.min_font_size_input.value(),
            "max_font_size": self.max_font_size_input.value(),
            "stroke_enabled": self.stroke_enabled_checkbox.isChecked(),
            "stroke_width": self.stroke_width_input.value(),
            "text_color": self.text_color_input.text().strip(),
            "stroke_color": self.stroke_color_input.text().strip(),
            "auto_color": self.auto_color_checkbox.isChecked(),
            "auto_direction": self.auto_direction_checkbox.isChecked(),
            "vertical_cjk": self.vertical_cjk_checkbox.isChecked(),
            "save_sprites": self.save_sprites_checkbox.isChecked(),
        }

    def apply_settings(self, settings: dict[str, Any]) -> None:
        if not isinstance(settings, dict):
            return
        self.font_name_input.setCurrentText(str(settings.get("font_name", "") or self.font_name_input.currentText()))
        self.font_path_input.setText(str(settings.get("font_path", "") or ""))
        try:
            self.min_font_size_input.setValue(int(settings.get("min_font_size", self.min_font_size_input.value())))
        except Exception:
            pass
        try:
            self.max_font_size_input.setValue(int(settings.get("max_font_size", self.max_font_size_input.value())))
        except Exception:
            pass
        self.stroke_enabled_checkbox.setChecked(
            bool(settings.get("stroke_enabled", self.stroke_enabled_checkbox.isChecked()))
        )
        try:
            self.stroke_width_input.setValue(float(settings.get("stroke_width", self.stroke_width_input.value())))
        except Exception:
            pass
        self.text_color_input.setText(str(settings.get("text_color", "") or self.text_color_input.text()))
        self.stroke_color_input.setText(str(settings.get("stroke_color", "") or self.stroke_color_input.text()))
        self.auto_color_checkbox.setChecked(bool(settings.get("auto_color", self.auto_color_checkbox.isChecked())))
        self.auto_direction_checkbox.setChecked(
            bool(settings.get("auto_direction", self.auto_direction_checkbox.isChecked()))
        )
        self.vertical_cjk_checkbox.setChecked(
            bool(settings.get("vertical_cjk", self.vertical_cjk_checkbox.isChecked()))
        )
        self.save_sprites_checkbox.setChecked(
            bool(settings.get("save_sprites", self.save_sprites_checkbox.isChecked()))
        )

    def _rebuild_items_table(self, *, previous_item_id: int | None) -> None:
        show_excluded = self.show_excluded_items_checkbox.isChecked()
        self._items = [
            dict(item)
            for item in self._all_items
            if show_excluded or not bool(item.get("excluded", False))
        ]

        self.items_table.blockSignals(True)
        self.items_table.setRowCount(len(self._items))
        for row_index, item in enumerate(self._items):
            status_text = "excluded" if bool(item.get("excluded", False)) else str(item.get("status", ""))
            override_text = str(
                normalize_render_style_overrides(item.get("style_overrides")).get("render_text_override", "") or ""
            ).strip()
            override_status = self._override_status_text(item)
            row_values = [
                str(item.get("id", "")),
                str(item.get("kind", "")),
                str(item.get("writing_mode", "")),
                str(item.get("font_size", "")),
                status_text,
                self._display_text(item.get("translated_text")),
                self._display_text(override_text or "-"),
                override_status,
            ]
            error_text = str(item.get("error", "") or "").strip()
            tooltip_lines = []
            translated_text = str(item.get("translated_text", "") or "").strip()
            if translated_text:
                tooltip_lines.append(translated_text)
            if override_text:
                tooltip_lines.append(f"Override text: {override_text}")
            if bool(item.get("needs_render", False)):
                tooltip_lines.append("Render box changed. Re-render is recommended.")
            if has_active_style_overrides(item.get("style_overrides")):
                tooltip_lines.append("Manual render style overrides are active for this item.")
            if bool(item.get("excluded", False)):
                tooltip_lines.append("This render item is excluded.")
            if error_text:
                tooltip_lines.append(f"Error: {error_text}")
            tooltip = "\n\n".join(tooltip_lines) if tooltip_lines else ""
            for column_index, value in enumerate(row_values):
                table_item = QTableWidgetItem(value)
                if tooltip:
                    table_item.setToolTip(tooltip)
                if column_index == 0:
                    table_item.setData(Qt.ItemDataRole.UserRole, int(item.get("id", row_index)))
                self.items_table.setItem(row_index, column_index, table_item)
        self.items_table.blockSignals(False)

        self.box_editor_section.set_expanded(bool(self._all_items))
        if self._items:
            target_row = self._row_for_item_id(previous_item_id)
            if target_row is None:
                target_row = 0
            self.items_table.setCurrentCell(target_row, 0)
            self._update_item_details(target_row)
        else:
            self._update_item_details(None)

    def _on_show_excluded_toggled(self, enabled: bool) -> None:
        self._rebuild_items_table(previous_item_id=self.current_table_item_id())
        self.show_excluded_items_toggled.emit(bool(enabled))
        self._update_box_editor_state()

    def _on_item_selected(self) -> None:
        row_index = self.items_table.currentRow()
        self._update_item_details(row_index)
        current_id = self.current_table_item_id()
        if current_id is not None:
            self.current_item_changed.emit(current_id)

    def _on_item_double_clicked(self, row_index: int, column_index: int) -> None:
        if row_index < 0 or row_index >= len(self._items):
            return
        if column_index not in {5, 6}:
            return
        item_id = int(self._items[row_index].get("id", row_index))
        self._scroll_to_item_editor()
        self.render_text_override_editor.editor.setFocus()
        self.translated_text_double_clicked.emit(item_id)

    def _update_item_details(self, row_index: int | None) -> None:
        if row_index is None or row_index < 0 or row_index >= len(self._items):
            self.set_selected_box(None)
            self._load_style_editor_for_item(None)
            return
        item = self._items[row_index]
        self.set_selected_box(item)
        self._load_style_editor_for_item(item)

    def _row_for_item_id(self, item_id: int | None) -> int | None:
        if item_id is None:
            return None
        for row_index, item in enumerate(self._items):
            if int(item.get("id", row_index)) == int(item_id):
                return row_index
        return None

    def _update_box_editor_state(self) -> None:
        edit_enabled = self.box_edit_mode_enabled() and self._actions_enabled and bool(self._all_items)
        has_selection = self._selected_box is not None
        selected_excluded = bool(self._selected_box and self._selected_box.get("excluded", False))
        self.save_box_edits_button.setEnabled(edit_enabled and self._box_edit_dirty)
        self.cancel_box_edits_button.setEnabled(edit_enabled and self._box_edit_dirty)
        self.reload_box_cache_button.setEnabled(edit_enabled)
        self.exclude_selected_box_button.setEnabled(edit_enabled and has_selection and not selected_excluded)
        self.restore_selected_box_button.setEnabled(edit_enabled and has_selection and selected_excluded)
        self._update_style_editor_state()

    def has_unsaved_style_edits(self) -> bool:
        return bool(self._style_editor_dirty)

    def current_style_editor_item_id(self) -> int | None:
        if self._style_editor_loaded_item_id is not None:
            return int(self._style_editor_loaded_item_id)
        return self.current_table_item_id()

    def clear_pending_style_edit(self, item_id: int | None = None) -> None:
        if item_id is None:
            self._pending_style_snapshots_by_item_id.clear()
            return
        self._pending_style_snapshots_by_item_id.pop(int(item_id), None)

    def current_style_overrides(self) -> dict[str, Any]:
        text_color_value = self.item_text_color_input.text().strip()
        stroke_color_value = self.item_stroke_color_input.text().strip()
        try:
            parsed_text_color = parse_color_value(text_color_value)
        except ValueError as exc:
            raise ValueError(f"Invalid item text color value. {exc}") from exc

        try:
            parsed_stroke_color = parse_color_value(stroke_color_value)
        except ValueError as exc:
            raise ValueError(f"Invalid item stroke color value. {exc}") from exc

        raw_payload = self._current_style_editor_snapshot()
        translated_text = str(self._style_editor_translated_text or "")
        editor_text = str(raw_payload.get("render_text_override", "") or "")
        if editor_text == translated_text or not editor_text.strip():
            persisted_override_text = ""
        else:
            persisted_override_text = editor_text
        stroke_mode = self.item_stroke_enabled_input.currentData()
        stroke_width_value = self.item_stroke_width_input.value()
        line_spacing_value = self.item_line_spacing_ratio_input.value()
        return normalize_render_style_overrides(
            {
                "enabled": bool(raw_payload.get("enabled", False)),
                "render_text_override": persisted_override_text,
                "font_name": str(raw_payload.get("font_name", "") or ""),
                "font_path": str(raw_payload.get("font_path", "") or ""),
                "font_size_mode": str(raw_payload.get("font_size_mode", "inherit") or "inherit"),
                "fixed_font_size": int(raw_payload.get("fixed_font_size", 0) or 0),
                "min_font_size": int(raw_payload.get("min_font_size", 0) or 0),
                "max_font_size": int(raw_payload.get("max_font_size", 0) or 0),
                "writing_mode": str(raw_payload.get("writing_mode", "inherit") or "inherit"),
                "stroke_enabled": stroke_mode,
                "stroke_width": stroke_width_value if stroke_width_value > 0 else None,
                "text_color_mode": str(raw_payload.get("text_color_mode", "inherit") or "inherit"),
                "text_color": list(parsed_text_color) if parsed_text_color is not None else None,
                "stroke_color": list(parsed_stroke_color) if parsed_stroke_color is not None else None,
                "line_spacing_ratio": line_spacing_value if line_spacing_value > 0 else None,
            }
        )

    def revert_current_style_editor(self) -> None:
        item_id = self.current_style_editor_item_id()
        if item_id is not None:
            self.clear_pending_style_edit(item_id)
        current_row = self.items_table.currentRow()
        if current_row < 0 or current_row >= len(self._items):
            self._load_style_editor_for_item(None)
            return
        self._load_style_editor_for_item(self._items[current_row])

    @staticmethod
    def _display_text(value: Any, limit: int = 72) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return f"{text[: limit - 1]}..."

    @staticmethod
    def _format_bbox(bbox: Any) -> str:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return "-"
        return f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"

    def _on_style_editor_changed(self, _value: object = None) -> None:
        if self._style_editor_guard:
            return
        item_id = self.current_style_editor_item_id()
        if item_id is None:
            self._set_style_editor_dirty(False)
            return
        current_snapshot = self._current_style_editor_snapshot()
        loaded_snapshot = self._style_editor_loaded_snapshot or self._empty_style_editor_snapshot()
        if current_snapshot != loaded_snapshot:
            self._pending_style_snapshots_by_item_id[int(item_id)] = dict(current_snapshot)
            self._set_style_editor_dirty(True)
        else:
            self.clear_pending_style_edit(item_id)
            self._set_style_editor_dirty(False)
        self._update_style_editor_state()

    def _set_style_editor_dirty(self, dirty: bool) -> None:
        self._style_editor_dirty = bool(dirty)
        self.style_dirty_label.setText("Unsaved item style edits" if dirty else "Saved")

    def _load_style_editor_for_item(self, item: dict[str, Any] | None) -> None:
        previous_item_id = self._style_editor_loaded_item_id
        if previous_item_id is not None and self._style_editor_dirty:
            self._pending_style_snapshots_by_item_id[int(previous_item_id)] = self._current_style_editor_snapshot()

        self._style_editor_guard = True
        try:
            if not isinstance(item, dict):
                self._style_editor_loaded_item_id = None
                self._style_editor_loaded_snapshot = None
                self._style_editor_translated_text = ""
                self._apply_style_editor_snapshot(self._empty_style_editor_snapshot())
                self.render_text_override_editor.set_enabled_for_item(False, message="No render item selected.")
                self.style_editor_details_label.setText(
                    "Select a render item to override rendered text, font, size, direction, and spacing for that item only."
                )
                self.style_override_hint_label.setVisible(False)
                self._set_style_editor_dirty(False)
                self._update_style_editor_state()
                return

            item_id = int(item.get("id", 0))
            self._style_editor_loaded_item_id = item_id
            source_text = str(item.get("translated_text", "") or "").strip()
            self._style_editor_translated_text = str(item.get("translated_text", "") or "")
            self.render_text_override_editor.set_enabled_for_item(
                True,
                message="Edit the translated text directly here. If it matches Translation, no text override is saved.",
            )
            self.style_editor_details_label.setText(
                f"Render item {item_id} ({str(item.get('kind', '') or 'bubble')}) will use Translation text by default unless you override it here."
            )
            saved_snapshot = self._snapshot_from_item(item)
            pending_snapshot = self._pending_style_snapshots_by_item_id.get(item_id)
            self._style_editor_loaded_snapshot = dict(saved_snapshot)
            self._apply_style_editor_snapshot(pending_snapshot or saved_snapshot)
            self.style_override_hint_label.setText(
                f"Current Translation text: {self._display_text(source_text, limit=160) or '(empty)'}"
            )
            self.style_override_hint_label.setVisible(True)
            self._set_style_editor_dirty(item_id in self._pending_style_snapshots_by_item_id)
            self._update_style_editor_state()
        finally:
            self._style_editor_guard = False

    def _snapshot_from_item(self, item: dict[str, Any]) -> dict[str, Any]:
        overrides = normalize_render_style_overrides(item.get("style_overrides"))
        translated_text = str(item.get("translated_text", "") or "")
        override_text = str(overrides.get("render_text_override", "") or "")
        return {
            "enabled": bool(overrides.get("enabled", False)),
            "render_text_override": override_text if override_text else translated_text,
            "font_name": str(overrides.get("font_name", "") or ""),
            "font_path": str(overrides.get("font_path", "") or ""),
            "font_size_mode": str(overrides.get("font_size_mode", "inherit") or "inherit"),
            "fixed_font_size": int(overrides.get("fixed_font_size", 0) or 0),
            "min_font_size": int(overrides.get("min_font_size", 0) or 0),
            "max_font_size": int(overrides.get("max_font_size", 0) or 0),
            "writing_mode": str(overrides.get("writing_mode", "inherit") or "inherit"),
            "stroke_enabled": overrides.get("stroke_enabled"),
            "stroke_width": float(overrides.get("stroke_width", 0.0) or 0.0),
            "text_color_mode": str(overrides.get("text_color_mode", "inherit") or "inherit"),
            "text_color": self._color_to_string(overrides.get("text_color")),
            "stroke_color": self._color_to_string(overrides.get("stroke_color")),
            "line_spacing_ratio": float(overrides.get("line_spacing_ratio", 0.0) or 0.0),
        }

    def _apply_style_editor_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.override_enabled_checkbox.setChecked(bool(snapshot.get("enabled", False)))
        self.render_text_override_editor.set_loaded_text(
            str(snapshot.get("render_text_override", "") or ""),
            status_text="Edit the translated text directly here. Matching Translation means no text override is saved.",
        )
        self.item_font_name_input.setCurrentText(str(snapshot.get("font_name", "") or ""))
        self.item_font_path_input.setText(str(snapshot.get("font_path", "") or ""))
        self._set_combo_data(self.item_font_size_mode_input, str(snapshot.get("font_size_mode", "inherit") or "inherit"))
        self.item_fixed_font_size_input.setValue(int(snapshot.get("fixed_font_size", 0) or 0))
        self.item_min_font_size_input.setValue(int(snapshot.get("min_font_size", 0) or 0))
        self.item_max_font_size_input.setValue(int(snapshot.get("max_font_size", 0) or 0))
        self._set_combo_data(self.item_writing_mode_input, str(snapshot.get("writing_mode", "inherit") or "inherit"))
        self._set_combo_data(self.item_stroke_enabled_input, snapshot.get("stroke_enabled"))
        self.item_stroke_width_input.setValue(float(snapshot.get("stroke_width", 0.0) or 0.0))
        self._set_combo_data(
            self.item_text_color_mode_input,
            str(snapshot.get("text_color_mode", "inherit") or "inherit"),
        )
        self.item_text_color_input.setText(str(snapshot.get("text_color", "") or ""))
        self.item_stroke_color_input.setText(str(snapshot.get("stroke_color", "") or ""))
        self.item_line_spacing_ratio_input.setValue(float(snapshot.get("line_spacing_ratio", 0.0) or 0.0))
        self._update_style_editor_field_visibility()

    def _empty_style_editor_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "render_text_override": "",
            "font_name": "",
            "font_path": "",
            "font_size_mode": "inherit",
            "fixed_font_size": 0,
            "min_font_size": 0,
            "max_font_size": 0,
            "writing_mode": "inherit",
            "stroke_enabled": None,
            "stroke_width": 0.0,
            "text_color_mode": "inherit",
            "text_color": "",
            "stroke_color": "",
            "line_spacing_ratio": 0.0,
        }

    def _current_style_editor_snapshot(self) -> dict[str, Any]:
        return {
            "enabled": self.override_enabled_checkbox.isChecked(),
            "render_text_override": self.render_text_override_editor.text(),
            "font_name": self.item_font_name_input.currentText().strip(),
            "font_path": self.item_font_path_input.text().strip(),
            "font_size_mode": self.item_font_size_mode_input.currentData(),
            "fixed_font_size": self.item_fixed_font_size_input.value(),
            "min_font_size": self.item_min_font_size_input.value(),
            "max_font_size": self.item_max_font_size_input.value(),
            "writing_mode": self.item_writing_mode_input.currentData(),
            "stroke_enabled": self.item_stroke_enabled_input.currentData(),
            "stroke_width": self.item_stroke_width_input.value(),
            "text_color_mode": self.item_text_color_mode_input.currentData(),
            "text_color": self.item_text_color_input.text().strip(),
            "stroke_color": self.item_stroke_color_input.text().strip(),
            "line_spacing_ratio": self.item_line_spacing_ratio_input.value(),
        }

    def _update_style_editor_field_visibility(self) -> None:
        font_size_mode = str(self.item_font_size_mode_input.currentData() or "inherit")
        fixed_enabled = font_size_mode == "fixed"
        self.item_fixed_font_size_input.setEnabled(fixed_enabled and self._actions_enabled and self._style_editor_loaded_item_id is not None)
        range_enabled = font_size_mode != "fixed"
        self.item_min_font_size_input.setEnabled(range_enabled and self._actions_enabled and self._style_editor_loaded_item_id is not None)
        self.item_max_font_size_input.setEnabled(range_enabled and self._actions_enabled and self._style_editor_loaded_item_id is not None)

        custom_colors = str(self.item_text_color_mode_input.currentData() or "inherit") == "custom"
        self.item_text_color_input.setEnabled(custom_colors and self._actions_enabled and self._style_editor_loaded_item_id is not None)
        self.item_stroke_color_input.setEnabled(custom_colors and self._actions_enabled and self._style_editor_loaded_item_id is not None)

    def _update_style_editor_state(self) -> None:
        has_item = self._style_editor_loaded_item_id is not None
        enabled = self._actions_enabled and has_item
        for widget in (
            self.override_enabled_checkbox,
            self.item_font_name_input,
            self.item_font_path_input,
            self.item_font_size_mode_input,
            self.item_writing_mode_input,
            self.item_stroke_enabled_input,
            self.item_stroke_width_input,
            self.item_text_color_mode_input,
            self.item_line_spacing_ratio_input,
            self.save_style_edits_button,
            self.revert_style_edits_button,
        ):
            widget.setEnabled(enabled)
        self.render_text_override_editor.set_enabled_for_item(
            enabled,
            message="Edit the translated text directly here. Matching Translation means no text override is saved."
            if enabled
            else "No render item selected.",
        )
        self.save_style_edits_button.setEnabled(enabled and self._style_editor_dirty)
        self.revert_style_edits_button.setEnabled(enabled and self._style_editor_dirty)
        self._update_style_editor_field_visibility()

    def _scroll_to_item_editor(self) -> None:
        self.item_editor_section.setVisible(True)
        self.ensureWidgetVisible(self.item_editor_section, 0, 48)

    @staticmethod
    def _set_combo_data(combo_box: QComboBox, expected_value: Any) -> None:
        for index in range(combo_box.count()):
            if combo_box.itemData(index) == expected_value:
                combo_box.setCurrentIndex(index)
                return
        if isinstance(expected_value, str):
            combo_box.setCurrentText(expected_value)

    @staticmethod
    def _color_to_string(value: Any) -> str:
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return ""
        return f"{int(value[0])},{int(value[1])},{int(value[2])}"

    @staticmethod
    def _override_status_text(item: dict[str, Any]) -> str:
        if has_active_style_overrides(item.get("style_overrides")):
            return "active"
        return "inherit"


__all__ = ["RenderPanel"]
