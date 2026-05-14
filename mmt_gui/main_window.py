"""Main window for the PyQt6 desktop shell."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from mmt_core import (
    LANGUAGE_CHOICES,
    LlamaServerManager,
    RenderConfig,
    STYLE_PROMPTS,
    TRANSLATOR_CHOICES,
    TranslationConfig,
    detection_json_path,
    get_lama_model_manager,
    inpaint_image_path,
    inpaint_json_path,
    inpaint_preview_mask_path,
    load_detection_json,
    load_inpaint_json,
    load_ocr_json,
    load_render_json,
    load_translation_json,
    list_project_fonts,
    ocr_json_path,
    parse_color_value,
    render_image_path,
    render_json_path,
    resolve_font_path,
    save_ocr_payload,
    save_translation_json,
    summarize_ocr_items,
    summarize_inpaint_json,
    summarize_render_json,
    summarize_translation_json,
    translation_json_path,
)

from . import APP_NAME
from .project import MangaProject, PROJECT_FILENAME
from .widgets import ImagePreviewWidget, PageListWidget
from .workers import (
    DetectionTask,
    DetectionWorkerResult,
    InpaintMaskTask,
    InpaintMaskWorkerResult,
    InpaintTask,
    InpaintWorkerResult,
    LamaModelTask,
    LamaModelTaskResult,
    LlamaServerTask,
    LlamaServerTaskResult,
    OCRInferenceTask,
    OCRInferenceWorkerResult,
    OCRPreparationTask,
    OCRPreparationWorkerResult,
    RenderPreparationTask,
    RenderPreparationWorkerResult,
    RenderTask,
    RenderWorkerResult,
    TaskWorker,
    TranslationInitializationTask,
    TranslationInitializationWorkerResult,
    TranslationPageResult,
    TranslationTask,
    TranslationWorkerResult,
    create_inpaint_mask_worker,
    create_inpaint_worker,
    create_lama_model_worker,
    create_render_preparation_worker,
    create_render_worker,
    create_detection_worker,
    create_ocr_inference_worker,
    create_llama_server_worker,
    create_ocr_preparation_worker,
    create_translation_initialization_worker,
    create_translation_worker,
)

PLACEHOLDER_TABS = (
    ("Project", "Project-level controls and metadata will be expanded in a later task."),
    ("Export", "Export controls and packaging options will be added in a later task."),
)

IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.webp)"
PROJECT_FILTER = f"Project Files ({PROJECT_FILENAME});;JSON Files (*.json)"
SERVER_STATE_UNKNOWN = "Unknown"
SERVER_STATE_STOPPED = "Stopped"
SERVER_STATE_STARTING = "Starting"
SERVER_STATE_READY = "Ready"
SERVER_STATE_ERROR = "Error"
OCR_TEXT_COLUMN = 5
TRANSLATION_SOURCE_TEXT_COLUMN = 4
TRANSLATION_TEXT_COLUMN = 5
RENDER_STATUS_COLUMN = 4
INPAINT_PREVIEW_SOURCE = "Source"
INPAINT_PREVIEW_MASK = "Mask Overlay"
INPAINT_PREVIEW_RESULT = "Inpaint Result"
RENDER_PREVIEW_SOURCE = "Source"
RENDER_PREVIEW_INPAINT = "Inpaint Result"
RENDER_PREVIEW_RESULT = "Render Result"


class MainWindow(QMainWindow):
    """Desktop shell that manages project files, detection, OCR, and preview state."""

    def __init__(self) -> None:
        super().__init__()
        self.workspace_root = Path(__file__).resolve().parents[1]
        self.current_project: MangaProject | None = None
        self.current_detection_data: dict[str, Any] | None = None
        self.current_ocr_data: dict[str, Any] | None = None
        self.current_ocr_items: list[dict[str, Any]] = []
        self.current_ocr_cache_path: Path | None = None
        self.current_translation_data: dict[str, Any] | None = None
        self.current_translation_items: list[dict[str, Any]] = []
        self.current_translation_cache_path: Path | None = None
        self.current_inpaint_data: dict[str, Any] | None = None
        self.current_inpaint_cache_path: Path | None = None
        self.current_render_data: dict[str, Any] | None = None
        self.current_render_items: list[dict[str, Any]] = []
        self.current_render_cache_path: Path | None = None
        self.thread_pool = QThreadPool.globalInstance()
        self._active_workers: list[TaskWorker] = []

        default_server_url = "http://127.0.0.1:8080"
        default_model_path = self.workspace_root / "model" / "paddleocr_vl" / "model.gguf"
        default_mmproj_path = self.workspace_root / "model" / "paddleocr_vl" / "mmproj.gguf"
        default_llama_cpp_dir = self.workspace_root / "tools" / "llama.cpp"
        self.llama_server_manager = LlamaServerManager(
            server_url=default_server_url,
            model_path=default_model_path,
            mmproj_path=default_mmproj_path,
            llama_cpp_dir=default_llama_cpp_dir,
            gpu_layers=99,
            ctx_size=8192,
        )

        self.setWindowTitle(APP_NAME)
        self.resize(1500, 980)

        self._build_ui()
        self._build_menu()
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

    def _build_ui(self) -> None:
        self.page_list = PageListWidget()
        self.page_list.setMinimumWidth(220)
        self.page_list.page_selected.connect(self._on_page_selected)

        self.image_preview = ImagePreviewWidget()
        self.image_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        workspace_splitter = QSplitter(Qt.Orientation.Horizontal)
        workspace_splitter.addWidget(self.page_list)
        workspace_splitter.addWidget(self.image_preview)
        workspace_splitter.setStretchFactor(0, 0)
        workspace_splitter.setStretchFactor(1, 1)

        self.stage_tabs = QTabWidget()
        self.stage_tabs.setDocumentMode(True)
        self.stage_tabs.setTabPosition(QTabWidget.TabPosition.North)

        self.project_tab_index = self.stage_tabs.addTab(self._create_placeholder_tab(PLACEHOLDER_TABS[0][1]), PLACEHOLDER_TABS[0][0])
        self.detection_tab_index = self.stage_tabs.addTab(self._create_detection_tab(), "Detection")
        self.ocr_tab_index = self.stage_tabs.addTab(self._create_ocr_tab(), "OCR")
        self.translation_tab_index = self.stage_tabs.addTab(self._create_translation_tab(), "Translation")
        self.inpaint_tab_index = self.stage_tabs.addTab(self._create_inpaint_tab(), "Inpaint")
        self.render_tab_index = self.stage_tabs.addTab(self._create_render_tab(), "Render")
        for title, description in PLACEHOLDER_TABS[1:]:
            self.stage_tabs.addTab(self._create_placeholder_tab(description), title)
        self.stage_tabs.currentChanged.connect(lambda _index: self._refresh_preview_for_current_page())

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setPlaceholderText("Log output will appear here.")
        self.log_output.document().setMaximumBlockCount(1000)

        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self.stage_tabs)
        top_layout.addWidget(workspace_splitter, 1)

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_container)
        main_splitter.addWidget(self.log_output)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setSizes([760, 190])

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(main_splitter)
        self.setCentralWidget(container)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        assert isinstance(file_menu, QMenu)

        new_action = file_menu.addAction("New Project")
        new_action.triggered.connect(self.new_project)

        open_action = file_menu.addAction("Open Project")
        open_action.triggered.connect(self.open_project)

        save_action = file_menu.addAction("Save Project")
        save_action.triggered.connect(self.save_project)

        file_menu.addSeparator()

        import_action = file_menu.addAction("Import Images")
        import_action.triggered.connect(self.import_images)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

    def _create_detection_tab(self) -> QWidget:
        detection_tab = QWidget()
        layout = QVBoxLayout(detection_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro_label = QLabel(
            "Run page detection in the background and preview cached bubble, text, and layout overlays."
        )
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self.run_detection_selected_button = QPushButton("Run Detection for Selected Page")
        self.run_detection_selected_button.clicked.connect(self.run_detection_for_selected_page)
        button_row.addWidget(self.run_detection_selected_button)

        self.run_detection_all_button = QPushButton("Run Detection for All Pages")
        self.run_detection_all_button.clicked.connect(self.run_detection_for_all_pages)
        button_row.addWidget(self.run_detection_all_button)

        self.reload_detection_button = QPushButton("Reload Cached Detection")
        self.reload_detection_button.clicked.connect(self.reload_cached_detection)
        button_row.addWidget(self.reload_detection_button)

        self.clear_overlay_button = QPushButton("Clear Overlay")
        self.clear_overlay_button.clicked.connect(self.clear_detection_overlay)
        button_row.addWidget(self.clear_overlay_button)

        button_row.addStretch(1)
        layout.addLayout(button_row)

        stats_container = QWidget()
        stats_form = QFormLayout(stats_container)
        stats_form.setContentsMargins(0, 0, 0, 0)
        stats_form.setSpacing(6)

        self.detection_bubbles_value = QLabel("0")
        self.detection_text_regions_value = QLabel("0")
        self.detection_layout_regions_value = QLabel("0")
        self.detection_method_value = QLabel("-")
        self.detection_cache_path_value = QLabel("-")
        self.detection_cache_path_value.setWordWrap(True)
        self.detection_cache_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        stats_form.addRow("Bubbles:", self.detection_bubbles_value)
        stats_form.addRow("Text Regions:", self.detection_text_regions_value)
        stats_form.addRow("Layout Regions:", self.detection_layout_regions_value)
        stats_form.addRow("Method:", self.detection_method_value)
        stats_form.addRow("Cache JSON:", self.detection_cache_path_value)

        layout.addWidget(stats_container)
        layout.addStretch(1)
        return detection_tab

    def _create_ocr_tab(self) -> QWidget:
        ocr_tab = QWidget()
        layout = QVBoxLayout(ocr_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        server_group = QGroupBox("PaddleOCR-VL llama.cpp Server")
        server_layout = QVBoxLayout(server_group)

        server_form = QFormLayout()
        self.server_url_input = QLineEdit(self.llama_server_manager.server_url)
        self.server_model_path_input = QLineEdit(self.llama_server_manager.model_path)
        self.server_mmproj_path_input = QLineEdit(self.llama_server_manager.mmproj_path)
        self.server_llama_cpp_dir_input = QLineEdit(self.llama_server_manager.llama_cpp_dir)

        self.server_gpu_layers_input = QSpinBox()
        self.server_gpu_layers_input.setRange(-1, 999)
        self.server_gpu_layers_input.setValue(self.llama_server_manager.gpu_layers)

        self.server_ctx_size_input = QSpinBox()
        self.server_ctx_size_input.setRange(512, 131072)
        self.server_ctx_size_input.setSingleStep(512)
        self.server_ctx_size_input.setValue(self.llama_server_manager.ctx_size)

        server_form.addRow("Server URL:", self.server_url_input)
        server_form.addRow("Model Path:", self.server_model_path_input)
        server_form.addRow("mmproj Path:", self.server_mmproj_path_input)
        server_form.addRow("llama.cpp Dir:", self.server_llama_cpp_dir_input)
        server_form.addRow("GPU Layers:", self.server_gpu_layers_input)
        server_form.addRow("Context Size:", self.server_ctx_size_input)
        server_layout.addLayout(server_form)

        server_button_row = QHBoxLayout()
        self.check_server_button = QPushButton("Check Server")
        self.check_server_button.clicked.connect(self.check_llama_server)
        server_button_row.addWidget(self.check_server_button)

        self.start_server_button = QPushButton("Start Server")
        self.start_server_button.clicked.connect(self.start_llama_server)
        server_button_row.addWidget(self.start_server_button)

        self.stop_server_button = QPushButton("Stop Server")
        self.stop_server_button.clicked.connect(self.stop_llama_server)
        server_button_row.addWidget(self.stop_server_button)
        server_button_row.addStretch(1)
        server_layout.addLayout(server_button_row)

        self.server_status_value = QLabel(SERVER_STATE_UNKNOWN)
        server_status_form = QFormLayout()
        server_status_form.setContentsMargins(0, 0, 0, 0)
        server_status_form.addRow("Status:", self.server_status_value)
        server_layout.addLayout(server_status_form)

        items_group = QGroupBox("OCR Items")
        items_layout = QVBoxLayout(items_group)

        items_button_row = QHBoxLayout()
        self.prepare_ocr_selected_button = QPushButton("Prepare OCR Items for Selected Page")
        self.prepare_ocr_selected_button.clicked.connect(self.prepare_ocr_items_for_selected_page)
        items_button_row.addWidget(self.prepare_ocr_selected_button)

        self.prepare_ocr_all_button = QPushButton("Prepare OCR Items for All Pages")
        self.prepare_ocr_all_button.clicked.connect(self.prepare_ocr_items_for_all_pages)
        items_button_row.addWidget(self.prepare_ocr_all_button)

        self.run_ocr_selected_page_button = QPushButton("Run OCR for Selected Page")
        self.run_ocr_selected_page_button.clicked.connect(self.run_ocr_for_selected_page)
        items_button_row.addWidget(self.run_ocr_selected_page_button)

        self.run_ocr_all_button = QPushButton("Run OCR for All Pages")
        self.run_ocr_all_button.clicked.connect(self.run_ocr_for_all_pages)
        items_button_row.addWidget(self.run_ocr_all_button)

        self.run_ocr_selected_items_button = QPushButton("Run OCR for Selected Item(s)")
        self.run_ocr_selected_items_button.clicked.connect(self.run_ocr_for_selected_items)
        items_button_row.addWidget(self.run_ocr_selected_items_button)

        items_button_row.addStretch(1)
        items_layout.addLayout(items_button_row)

        edit_button_row = QHBoxLayout()
        self.force_reocr_checkbox = QCheckBox("Force Re-OCR")
        edit_button_row.addWidget(self.force_reocr_checkbox)

        self.save_ocr_text_button = QPushButton("Save Edited OCR Text")
        self.save_ocr_text_button.clicked.connect(self.save_edited_ocr_text)
        edit_button_row.addWidget(self.save_ocr_text_button)

        self.reload_ocr_button = QPushButton("Reload Cached OCR Items")
        self.reload_ocr_button.clicked.connect(self.reload_cached_ocr_items)
        edit_button_row.addWidget(self.reload_ocr_button)
        edit_button_row.addStretch(1)
        items_layout.addLayout(edit_button_row)

        ocr_details_container = QWidget()
        ocr_details_form = QFormLayout(ocr_details_container)
        ocr_details_form.setContentsMargins(0, 0, 0, 0)
        ocr_details_form.setSpacing(6)

        self.ocr_total_items_value = QLabel("0")
        self.ocr_prepared_items_value = QLabel("0")
        self.ocr_done_items_value = QLabel("0")
        self.ocr_error_items_value = QLabel("0")
        self.ocr_cache_path_value = QLabel("-")
        self.ocr_cache_path_value.setWordWrap(True)
        self.ocr_cache_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        ocr_details_form.addRow("Total Items:", self.ocr_total_items_value)
        ocr_details_form.addRow("Prepared:", self.ocr_prepared_items_value)
        ocr_details_form.addRow("Done:", self.ocr_done_items_value)
        ocr_details_form.addRow("Error:", self.ocr_error_items_value)
        ocr_details_form.addRow("OCR JSON:", self.ocr_cache_path_value)
        items_layout.addWidget(ocr_details_container)

        items_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.ocr_items_table = QTableWidget(0, 6)
        self.ocr_items_table.setHorizontalHeaderLabels(["id", "kind", "bbox", "ocr_bbox", "status", "text"])
        self.ocr_items_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.ocr_items_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.ocr_items_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.ocr_items_table.setAlternatingRowColors(True)
        self.ocr_items_table.itemSelectionChanged.connect(self._on_ocr_item_selected)
        header = self.ocr_items_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        items_splitter.addWidget(self.ocr_items_table)

        crop_preview_container = QWidget()
        crop_preview_layout = QVBoxLayout(crop_preview_container)
        crop_preview_layout.setContentsMargins(0, 0, 0, 0)
        crop_preview_layout.setSpacing(6)
        crop_preview_layout.addWidget(QLabel("Crop Preview"))

        self.ocr_crop_preview = ImagePreviewWidget()
        self.ocr_crop_preview.setMinimumWidth(260)
        self.ocr_crop_preview.setMinimumHeight(220)
        crop_preview_layout.addWidget(self.ocr_crop_preview, 1)

        self.ocr_item_details_value = QLabel("Select an OCR item to preview its crop.")
        self.ocr_item_details_value.setWordWrap(True)
        self.ocr_item_details_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        crop_preview_layout.addWidget(self.ocr_item_details_value)

        items_splitter.addWidget(crop_preview_container)
        items_splitter.setStretchFactor(0, 1)
        items_splitter.setStretchFactor(1, 0)
        items_layout.addWidget(items_splitter, 1)

        layout.addWidget(server_group)
        layout.addWidget(items_group, 1)
        return ocr_tab

    def _create_inpaint_tab(self) -> QWidget:
        inpaint_tab = QWidget()
        layout = QVBoxLayout(inpaint_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        controls_group = QGroupBox("Inpaint Controls")
        controls_layout = QVBoxLayout(controls_group)

        controls_button_row = QHBoxLayout()
        self.prepare_inpaint_mask_selected_button = QPushButton("Prepare Mask for Selected Page")
        self.prepare_inpaint_mask_selected_button.clicked.connect(self.prepare_inpaint_mask_for_selected_page)
        controls_button_row.addWidget(self.prepare_inpaint_mask_selected_button)

        self.prepare_inpaint_mask_all_button = QPushButton("Prepare Mask for All Pages")
        self.prepare_inpaint_mask_all_button.clicked.connect(self.prepare_inpaint_mask_for_all_pages)
        controls_button_row.addWidget(self.prepare_inpaint_mask_all_button)

        self.run_inpaint_selected_button = QPushButton("Run Inpaint for Selected Page")
        self.run_inpaint_selected_button.clicked.connect(self.run_inpaint_for_selected_page)
        controls_button_row.addWidget(self.run_inpaint_selected_button)

        self.run_inpaint_all_button = QPushButton("Run Inpaint for All Pages")
        self.run_inpaint_all_button.clicked.connect(self.run_inpaint_for_all_pages)
        controls_button_row.addWidget(self.run_inpaint_all_button)
        controls_button_row.addStretch(1)
        controls_layout.addLayout(controls_button_row)

        tools_button_row = QHBoxLayout()
        self.reload_inpaint_button = QPushButton("Reload Cached Inpaint")
        self.reload_inpaint_button.clicked.connect(self.reload_cached_inpaint)
        tools_button_row.addWidget(self.reload_inpaint_button)

        self.clear_inpaint_preview_button = QPushButton("Clear Inpaint Preview")
        self.clear_inpaint_preview_button.clicked.connect(self.clear_inpaint_preview)
        tools_button_row.addWidget(self.clear_inpaint_preview_button)

        self.load_lama_model_button = QPushButton("Load LaMa Model")
        self.load_lama_model_button.clicked.connect(self.load_lama_model)
        tools_button_row.addWidget(self.load_lama_model_button)

        self.unload_lama_model_button = QPushButton("Unload LaMa Model")
        self.unload_lama_model_button.clicked.connect(self.unload_lama_model)
        tools_button_row.addWidget(self.unload_lama_model_button)
        tools_button_row.addStretch(1)
        controls_layout.addLayout(tools_button_row)

        settings_form = QFormLayout()
        self.inpaint_mask_padding_input = QSpinBox()
        self.inpaint_mask_padding_input.setRange(0, 128)
        self.inpaint_mask_padding_input.setValue(8)

        self.inpaint_use_bubble_mask_checkbox = QCheckBox("Use bubble mask guidance")
        self.inpaint_use_bubble_mask_checkbox.setChecked(True)

        self.inpaint_use_crop_windows_checkbox = QCheckBox("Use crop windows")
        self.inpaint_use_crop_windows_checkbox.setChecked(True)

        self.inpaint_force_checkbox = QCheckBox("Force re-inpaint")

        inpaint_options_widget = QWidget()
        inpaint_options_layout = QHBoxLayout(inpaint_options_widget)
        inpaint_options_layout.setContentsMargins(0, 0, 0, 0)
        inpaint_options_layout.addWidget(self.inpaint_use_bubble_mask_checkbox)
        inpaint_options_layout.addWidget(self.inpaint_use_crop_windows_checkbox)
        inpaint_options_layout.addWidget(self.inpaint_force_checkbox)
        inpaint_options_layout.addStretch(1)

        self.inpaint_device_input = QComboBox()
        self.inpaint_device_input.setEditable(True)
        self.inpaint_device_input.addItems(["auto", "cpu", "cuda", "cuda:0"])
        self.inpaint_device_input.setCurrentText("auto")

        self.inpaint_preview_mode_input = QComboBox()
        self.inpaint_preview_mode_input.addItems(
            [INPAINT_PREVIEW_SOURCE, INPAINT_PREVIEW_MASK, INPAINT_PREVIEW_RESULT]
        )
        self.inpaint_preview_mode_input.currentTextChanged.connect(
            lambda _text: self._refresh_preview_for_current_page()
        )

        self.lama_model_status_value = QLabel("Not loaded")

        settings_form.addRow("Mask Padding:", self.inpaint_mask_padding_input)
        settings_form.addRow("Options:", inpaint_options_widget)
        settings_form.addRow("Device:", self.inpaint_device_input)
        settings_form.addRow("Preview Mode:", self.inpaint_preview_mode_input)
        settings_form.addRow("LaMa Model:", self.lama_model_status_value)
        controls_layout.addLayout(settings_form)

        details_group = QGroupBox("Inpaint Details")
        details_form = QFormLayout(details_group)
        details_form.setContentsMargins(12, 12, 12, 12)
        details_form.setSpacing(6)

        self.inpaint_source_path_value = QLabel("-")
        self.inpaint_source_path_value.setWordWrap(True)
        self.inpaint_source_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.inpaint_ocr_cache_path_value = QLabel("-")
        self.inpaint_ocr_cache_path_value.setWordWrap(True)
        self.inpaint_ocr_cache_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.inpaint_text_mask_path_value = QLabel("-")
        self.inpaint_text_mask_path_value.setWordWrap(True)
        self.inpaint_text_mask_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.inpaint_bubble_mask_path_value = QLabel("-")
        self.inpaint_bubble_mask_path_value.setWordWrap(True)
        self.inpaint_bubble_mask_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.inpaint_output_path_value = QLabel("-")
        self.inpaint_output_path_value.setWordWrap(True)
        self.inpaint_output_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.inpaint_item_count_value = QLabel("0")
        self.inpaint_masked_pixels_value = QLabel("0")
        self.inpaint_status_value = QLabel("-")
        self.inpaint_error_value = QLabel("-")
        self.inpaint_error_value.setWordWrap(True)
        self.inpaint_error_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        details_form.addRow("Source Image:", self.inpaint_source_path_value)
        details_form.addRow("OCR Cache:", self.inpaint_ocr_cache_path_value)
        details_form.addRow("Text Mask:", self.inpaint_text_mask_path_value)
        details_form.addRow("Bubble Mask:", self.inpaint_bubble_mask_path_value)
        details_form.addRow("Output Image:", self.inpaint_output_path_value)
        details_form.addRow("Item Count:", self.inpaint_item_count_value)
        details_form.addRow("Masked Pixels:", self.inpaint_masked_pixels_value)
        details_form.addRow("Status:", self.inpaint_status_value)
        details_form.addRow("Error:", self.inpaint_error_value)

        layout.addWidget(controls_group)
        layout.addWidget(details_group)
        layout.addStretch(1)
        return inpaint_tab

    def _create_render_tab(self) -> QWidget:
        render_tab = QWidget()
        layout = QVBoxLayout(render_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        settings_group = QGroupBox("Render Settings")
        settings_layout = QVBoxLayout(settings_group)

        settings_form = QFormLayout()
        self.render_font_name_input = QComboBox()
        self.render_font_name_input.setEditable(True)
        self.render_font_name_input.addItem("")
        for display_name, _font_path in list_project_fonts(self.workspace_root):
            self.render_font_name_input.addItem(display_name)
        if self.render_font_name_input.count() > 1:
            self.render_font_name_input.setCurrentIndex(1)

        self.render_font_path_input = QLineEdit()
        self.render_font_path_input.setPlaceholderText("Optional explicit font file path.")

        self.render_min_font_size_input = QSpinBox()
        self.render_min_font_size_input.setRange(6, 256)
        self.render_min_font_size_input.setValue(12)

        self.render_max_font_size_input = QSpinBox()
        self.render_max_font_size_input.setRange(6, 512)
        self.render_max_font_size_input.setValue(72)

        self.render_stroke_enabled_checkbox = QCheckBox("Stroke enabled")
        self.render_stroke_enabled_checkbox.setChecked(True)

        self.render_stroke_width_input = QDoubleSpinBox()
        self.render_stroke_width_input.setRange(0.0, 20.0)
        self.render_stroke_width_input.setSingleStep(0.5)
        self.render_stroke_width_input.setValue(0.0)
        self.render_stroke_width_input.setSpecialValueText("Auto")

        self.render_text_color_input = QLineEdit("auto")
        self.render_stroke_color_input = QLineEdit("auto")

        self.render_auto_color_checkbox = QCheckBox("Auto color")
        self.render_auto_color_checkbox.setChecked(True)
        self.render_auto_direction_checkbox = QCheckBox("Auto direction")
        self.render_auto_direction_checkbox.setChecked(True)
        self.render_vertical_cjk_checkbox = QCheckBox("Vertical CJK")
        self.render_vertical_cjk_checkbox.setChecked(True)
        self.render_save_sprites_checkbox = QCheckBox("Save sprites")
        self.render_save_sprites_checkbox.setChecked(True)
        self.render_force_checkbox = QCheckBox("Force re-render")

        render_options_widget = QWidget()
        render_options_layout = QHBoxLayout(render_options_widget)
        render_options_layout.setContentsMargins(0, 0, 0, 0)
        render_options_layout.addWidget(self.render_auto_color_checkbox)
        render_options_layout.addWidget(self.render_auto_direction_checkbox)
        render_options_layout.addWidget(self.render_vertical_cjk_checkbox)
        render_options_layout.addWidget(self.render_save_sprites_checkbox)
        render_options_layout.addWidget(self.render_force_checkbox)
        render_options_layout.addStretch(1)

        self.render_preview_mode_input = QComboBox()
        self.render_preview_mode_input.addItems(
            [RENDER_PREVIEW_SOURCE, RENDER_PREVIEW_INPAINT, RENDER_PREVIEW_RESULT]
        )
        self.render_preview_mode_input.currentTextChanged.connect(
            lambda _text: self._refresh_preview_for_current_page()
        )

        settings_form.addRow("Font:", self.render_font_name_input)
        settings_form.addRow("Font Path:", self.render_font_path_input)
        settings_form.addRow("Min Font Size:", self.render_min_font_size_input)
        settings_form.addRow("Max Font Size:", self.render_max_font_size_input)
        settings_form.addRow("Stroke Width:", self.render_stroke_width_input)
        settings_form.addRow("Text Color:", self.render_text_color_input)
        settings_form.addRow("Stroke Color:", self.render_stroke_color_input)
        settings_form.addRow("Options:", render_options_widget)
        settings_form.addRow("Preview Mode:", self.render_preview_mode_input)
        settings_layout.addLayout(settings_form)

        items_group = QGroupBox("Render Items")
        items_layout = QVBoxLayout(items_group)

        prepare_button_row = QHBoxLayout()
        self.prepare_render_selected_button = QPushButton("Prepare Render for Selected Page")
        self.prepare_render_selected_button.clicked.connect(self.prepare_render_for_selected_page)
        prepare_button_row.addWidget(self.prepare_render_selected_button)

        self.prepare_render_all_button = QPushButton("Prepare Render for All Pages")
        self.prepare_render_all_button.clicked.connect(self.prepare_render_for_all_pages)
        prepare_button_row.addWidget(self.prepare_render_all_button)

        self.run_render_selected_button = QPushButton("Run Render for Selected Page")
        self.run_render_selected_button.clicked.connect(self.run_render_for_selected_page)
        prepare_button_row.addWidget(self.run_render_selected_button)

        self.run_render_all_button = QPushButton("Run Render for All Pages")
        self.run_render_all_button.clicked.connect(self.run_render_for_all_pages)
        prepare_button_row.addWidget(self.run_render_all_button)
        prepare_button_row.addStretch(1)
        items_layout.addLayout(prepare_button_row)

        render_tools_row = QHBoxLayout()
        self.reload_render_button = QPushButton("Reload Cached Render")
        self.reload_render_button.clicked.connect(self.reload_cached_render)
        render_tools_row.addWidget(self.reload_render_button)

        self.clear_render_preview_button = QPushButton("Clear Render Preview")
        self.clear_render_preview_button.clicked.connect(self.clear_render_preview)
        render_tools_row.addWidget(self.clear_render_preview_button)
        render_tools_row.addStretch(1)
        items_layout.addLayout(render_tools_row)

        render_details_container = QWidget()
        render_details_form = QFormLayout(render_details_container)
        render_details_form.setContentsMargins(0, 0, 0, 0)
        render_details_form.setSpacing(6)

        self.render_translation_cache_path_value = QLabel("-")
        self.render_translation_cache_path_value.setWordWrap(True)
        self.render_translation_cache_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.render_inpaint_image_path_value = QLabel("-")
        self.render_inpaint_image_path_value.setWordWrap(True)
        self.render_inpaint_image_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.render_output_path_value = QLabel("-")
        self.render_output_path_value.setWordWrap(True)
        self.render_output_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.render_rendered_item_count_value = QLabel("0")
        self.render_skipped_item_count_value = QLabel("0")
        self.render_status_value = QLabel("-")
        self.render_error_value = QLabel("-")
        self.render_error_value.setWordWrap(True)
        self.render_error_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        render_details_form.addRow("Translation Cache:", self.render_translation_cache_path_value)
        render_details_form.addRow("Inpaint Image:", self.render_inpaint_image_path_value)
        render_details_form.addRow("Render Output:", self.render_output_path_value)
        render_details_form.addRow("Rendered Items:", self.render_rendered_item_count_value)
        render_details_form.addRow("Skipped Items:", self.render_skipped_item_count_value)
        render_details_form.addRow("Status:", self.render_status_value)
        render_details_form.addRow("Error:", self.render_error_value)
        items_layout.addWidget(render_details_container)

        render_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.render_items_table = QTableWidget(0, 8)
        self.render_items_table.setHorizontalHeaderLabels(
            ["id", "kind", "writing_mode", "font_size", "status", "translated_text", "render_bbox", "sprite_path"]
        )
        self.render_items_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.render_items_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.render_items_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.render_items_table.setAlternatingRowColors(True)
        self.render_items_table.itemSelectionChanged.connect(self._on_render_item_selected)
        render_header = self.render_items_table.horizontalHeader()
        render_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        render_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        render_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        render_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        render_header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        render_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        render_header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        render_header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        render_splitter.addWidget(self.render_items_table)

        render_side_panel = QWidget()
        render_side_layout = QVBoxLayout(render_side_panel)
        render_side_layout.setContentsMargins(0, 0, 0, 0)
        render_side_layout.setSpacing(6)
        render_side_layout.addWidget(QLabel("Selected Render Item"))
        self.render_item_details_value = QLabel("Select a render item to inspect it.")
        self.render_item_details_value.setWordWrap(True)
        self.render_item_details_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        render_side_layout.addWidget(self.render_item_details_value)
        render_side_layout.addStretch(1)
        render_splitter.addWidget(render_side_panel)
        render_splitter.setStretchFactor(0, 1)
        render_splitter.setStretchFactor(1, 0)
        items_layout.addWidget(render_splitter, 1)

        layout.addWidget(settings_group)
        layout.addWidget(items_group, 1)
        return render_tab

    def _create_translation_tab(self) -> QWidget:
        translation_tab = QWidget()
        layout = QVBoxLayout(translation_tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        config_group = QGroupBox("Translation Settings")
        config_layout = QVBoxLayout(config_group)

        config_form = QFormLayout()
        self.translation_source_language_input = QComboBox()
        self.translation_source_language_input.setEditable(True)
        self.translation_source_language_input.addItems(LANGUAGE_CHOICES)
        self.translation_source_language_input.setCurrentText("ja")

        self.translation_target_language_input = QComboBox()
        self.translation_target_language_input.setEditable(True)
        self.translation_target_language_input.addItems(LANGUAGE_CHOICES)
        self.translation_target_language_input.setCurrentText("en")

        self.translation_translator_input = QComboBox()
        self.translation_translator_input.addItems(TRANSLATOR_CHOICES)
        self.translation_translator_input.setCurrentText("Google")

        self.translation_style_input = QComboBox()
        self.translation_style_input.addItems(STYLE_PROMPTS.keys())
        self.translation_style_input.setCurrentText("Default")

        self.translation_batch_size_input = QSpinBox()
        self.translation_batch_size_input.setRange(1, 50)
        self.translation_batch_size_input.setValue(3)

        self.translation_use_context_memory_checkbox = QCheckBox("Use context memory")
        self.translation_force_checkbox = QCheckBox("Force Re-translate")
        provider_flags = QWidget()
        provider_flags_layout = QHBoxLayout(provider_flags)
        provider_flags_layout.setContentsMargins(0, 0, 0, 0)
        provider_flags_layout.addWidget(self.translation_use_context_memory_checkbox)
        provider_flags_layout.addWidget(self.translation_force_checkbox)
        provider_flags_layout.addStretch(1)

        self.translation_custom_prompt_input = QPlainTextEdit()
        self.translation_custom_prompt_input.setPlaceholderText("Optional additional translation instructions.")
        self.translation_custom_prompt_input.setMaximumHeight(90)

        self.translation_gemini_api_key_input = QLineEdit()
        self.translation_gemini_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.translation_local_llm_server_url_input = QLineEdit("http://127.0.0.1:8080")
        self.translation_local_llm_model_input = QLineEdit("gpt-4o")
        self.translation_deepseek_api_key_input = QLineEdit()
        self.translation_deepseek_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.translation_deepseek_model_input = QLineEdit("deepseek-v4-flash")
        self.translation_deepseek_thinking_checkbox = QCheckBox("Enable DeepSeek thinking")

        config_form.addRow("Source Language:", self.translation_source_language_input)
        config_form.addRow("Target Language:", self.translation_target_language_input)
        config_form.addRow("Translator:", self.translation_translator_input)
        config_form.addRow("Style:", self.translation_style_input)
        config_form.addRow("Custom Prompt:", self.translation_custom_prompt_input)
        config_form.addRow("Batch Size (pages):", self.translation_batch_size_input)
        config_form.addRow("Options:", provider_flags)
        config_form.addRow("Gemini API Key:", self.translation_gemini_api_key_input)
        config_form.addRow("Local LLM Server URL:", self.translation_local_llm_server_url_input)
        config_form.addRow("Local LLM Model:", self.translation_local_llm_model_input)
        config_form.addRow("DeepSeek API Key:", self.translation_deepseek_api_key_input)
        config_form.addRow("DeepSeek Model:", self.translation_deepseek_model_input)
        config_form.addRow("DeepSeek Options:", self.translation_deepseek_thinking_checkbox)
        config_layout.addLayout(config_form)

        items_group = QGroupBox("Translation Items")
        items_layout = QVBoxLayout(items_group)

        init_button_row = QHBoxLayout()
        self.initialize_translation_selected_button = QPushButton("Initialize Translation for Selected Page")
        self.initialize_translation_selected_button.clicked.connect(
            self.initialize_translation_for_selected_page
        )
        init_button_row.addWidget(self.initialize_translation_selected_button)

        self.initialize_translation_all_button = QPushButton("Initialize Translation for All Pages")
        self.initialize_translation_all_button.clicked.connect(self.initialize_translation_for_all_pages)
        init_button_row.addWidget(self.initialize_translation_all_button)

        self.run_translation_selected_page_button = QPushButton("Run Translation for Selected Page")
        self.run_translation_selected_page_button.clicked.connect(self.run_translation_for_selected_page)
        init_button_row.addWidget(self.run_translation_selected_page_button)

        self.run_translation_all_button = QPushButton("Run Translation for All Pages")
        self.run_translation_all_button.clicked.connect(self.run_translation_for_all_pages)
        init_button_row.addWidget(self.run_translation_all_button)
        init_button_row.addStretch(1)
        items_layout.addLayout(init_button_row)

        translate_button_row = QHBoxLayout()
        self.run_translation_selected_items_button = QPushButton("Run Translation for Selected Item(s)")
        self.run_translation_selected_items_button.clicked.connect(self.run_translation_for_selected_items)
        translate_button_row.addWidget(self.run_translation_selected_items_button)

        self.reload_translation_button = QPushButton("Reload Cached Translation")
        self.reload_translation_button.clicked.connect(self.reload_cached_translation)
        translate_button_row.addWidget(self.reload_translation_button)

        self.save_translation_text_button = QPushButton("Save Edited Translation Text")
        self.save_translation_text_button.clicked.connect(self.save_edited_translation_text)
        translate_button_row.addWidget(self.save_translation_text_button)
        translate_button_row.addStretch(1)
        items_layout.addLayout(translate_button_row)

        translation_details_container = QWidget()
        translation_details_form = QFormLayout(translation_details_container)
        translation_details_form.setContentsMargins(0, 0, 0, 0)
        translation_details_form.setSpacing(6)

        self.translation_total_items_value = QLabel("0")
        self.translation_pending_items_value = QLabel("0")
        self.translation_done_items_value = QLabel("0")
        self.translation_error_items_value = QLabel("0")
        self.translation_cache_path_value = QLabel("-")
        self.translation_cache_path_value.setWordWrap(True)
        self.translation_cache_path_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        translation_details_form.addRow("Total Items:", self.translation_total_items_value)
        translation_details_form.addRow("Pending:", self.translation_pending_items_value)
        translation_details_form.addRow("Done:", self.translation_done_items_value)
        translation_details_form.addRow("Error:", self.translation_error_items_value)
        translation_details_form.addRow("Translation JSON:", self.translation_cache_path_value)
        items_layout.addWidget(translation_details_container)

        translation_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.translation_items_table = QTableWidget(0, 6)
        self.translation_items_table.setHorizontalHeaderLabels(
            ["id", "ocr_item_id", "kind", "status", "source_text", "translated_text"]
        )
        self.translation_items_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.translation_items_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.translation_items_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.translation_items_table.setAlternatingRowColors(True)
        self.translation_items_table.itemSelectionChanged.connect(self._on_translation_item_selected)
        translation_header = self.translation_items_table.horizontalHeader()
        translation_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        translation_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        translation_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        translation_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        translation_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        translation_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        translation_splitter.addWidget(self.translation_items_table)

        translation_details_panel = QWidget()
        translation_details_layout = QVBoxLayout(translation_details_panel)
        translation_details_layout.setContentsMargins(0, 0, 0, 0)
        translation_details_layout.setSpacing(6)
        translation_details_layout.addWidget(QLabel("Selected Item"))
        self.translation_item_details_value = QLabel("Select a translation item to inspect it.")
        self.translation_item_details_value.setWordWrap(True)
        self.translation_item_details_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        translation_details_layout.addWidget(self.translation_item_details_value)
        translation_details_layout.addStretch(1)
        translation_splitter.addWidget(translation_details_panel)
        translation_splitter.setStretchFactor(0, 1)
        translation_splitter.setStretchFactor(1, 0)
        items_layout.addWidget(translation_splitter, 1)

        layout.addWidget(config_group)
        layout.addWidget(items_group, 1)
        return translation_tab

    def _create_placeholder_tab(self, description: str) -> QWidget:
        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        layout.setContentsMargins(12, 12, 12, 12)

        label = QLabel(description)
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addStretch(1)
        return placeholder

    def new_project(self) -> None:
        project_dir = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if not project_dir:
            return

        project_root = Path(project_dir)
        project_file = project_root / PROJECT_FILENAME
        if project_file.exists():
            response = QMessageBox.question(
                self,
                "Project Already Exists",
                f"{project_file} already exists.\n\nOverwrite the existing project file?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return

        try:
            self.current_project = MangaProject.create(project_root)
        except Exception as exc:  # pragma: no cover - GUI error path.
            self._show_error("Failed to create project", str(exc))
            return

        self.stage_tabs.setCurrentIndex(self.project_tab_index)
        self._refresh_project_view()
        self._log_message(f"Created project at {self.current_project.root_dir}")
        self.statusBar().showMessage("Project created")

    def open_project(self) -> None:
        project_file, _ = QFileDialog.getOpenFileName(self, "Open Project", "", PROJECT_FILTER)
        if not project_file:
            return

        try:
            self.current_project = MangaProject.load(Path(project_file))
        except Exception as exc:  # pragma: no cover - GUI error path.
            self._show_error("Failed to open project", str(exc))
            return

        self._refresh_project_view()
        self._log_message(f"Opened project from {self.current_project.project_file}")
        self.statusBar().showMessage("Project loaded")

    def save_project(self) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before saving.")
            return

        try:
            self.current_project.save()
        except Exception as exc:  # pragma: no cover - GUI error path.
            self._show_error("Failed to save project", str(exc))
            return

        self._log_message(f"Saved project to {self.current_project.project_file}")
        self.statusBar().showMessage("Project saved")

    def import_images(self) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before importing images.")
            return

        file_names, _ = QFileDialog.getOpenFileNames(self, "Import Images", "", IMAGE_FILTER)
        if not file_names:
            return

        try:
            imported_images = self.current_project.import_images([Path(file_name) for file_name in file_names])
            self.current_project.save()
        except Exception as exc:  # pragma: no cover - GUI error path.
            self._show_error("Failed to import images", str(exc))
            return

        if not imported_images:
            self._show_error("No images imported", "No supported image files were selected.")
            return

        self._refresh_project_view()
        self._log_message(f"Imported {len(imported_images)} image(s) into {self.current_project.source_dir}")
        self.statusBar().showMessage(f"Imported {len(imported_images)} image(s)")

    def run_detection_for_selected_page(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, _, image_path = selected_context
        self.stage_tabs.setCurrentIndex(self.detection_tab_index)
        self._start_detection_task([image_path], task_name=f"Detection: {image_path.name}")

    def run_detection_for_all_pages(self) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before running detection.")
            return

        all_image_paths = self.current_project.all_image_paths()
        missing_image_paths = [path for path in all_image_paths if not path.exists()]
        image_paths = [path for path in all_image_paths if path.exists()]
        if not image_paths:
            self._show_error("No source images", "Import images before running detection.")
            return

        for missing_image_path in missing_image_paths:
            self._log_message(f"Skipping missing source image during batch detection: {missing_image_path}")

        self.stage_tabs.setCurrentIndex(self.detection_tab_index)
        self._start_detection_task(
            image_paths,
            task_name=f"Detection: {len(image_paths)} page(s)",
        )

    def reload_cached_detection(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None:
            return

        index, _, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.detection_tab_index)
        self._load_cached_detection_for_index(index, show_errors=True, persist_stage_status=True)

    def clear_detection_overlay(self) -> None:
        self.image_preview.clear_overlays()
        self.current_detection_data = None
        self.statusBar().showMessage("Detection overlay cleared")
        self._log_message("Cleared detection overlay preview.")

    def start_llama_server(self) -> None:
        if not self._apply_server_inputs_to_manager():
            return
        self.stage_tabs.setCurrentIndex(self.ocr_tab_index)
        self._set_server_status(SERVER_STATE_STARTING)
        self._start_llama_server_action("start", timeout_seconds=60.0)

    def check_llama_server(self) -> None:
        if not self._apply_server_inputs_to_manager():
            return
        self.stage_tabs.setCurrentIndex(self.ocr_tab_index)
        self._start_llama_server_action("check", timeout_seconds=5.0)

    def stop_llama_server(self) -> None:
        if not self._apply_server_inputs_to_manager():
            return
        self.stage_tabs.setCurrentIndex(self.ocr_tab_index)
        self._start_llama_server_action("stop", timeout_seconds=10.0)

    def prepare_ocr_items_for_selected_page(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.ocr_tab_index)
        self._start_ocr_preparation_task(
            [image_relative_path],
            task_name=f"OCR Prep: {Path(image_relative_path).name}",
        )

    def prepare_ocr_items_for_all_pages(self) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before preparing OCR items.")
            return

        image_relative_paths = [
            relative_path
            for relative_path in (
                self.current_project.page_relative_path_for_index(index)
                for index in range(self.current_project.page_count)
            )
            if relative_path is not None
        ]
        if not image_relative_paths:
            self._show_error("No source images", "Import images before preparing OCR items.")
            return

        self.stage_tabs.setCurrentIndex(self.ocr_tab_index)
        self._start_ocr_preparation_task(
            image_relative_paths,
            task_name=f"OCR Prep: {len(image_relative_paths)} page(s)",
        )

    def reload_cached_ocr_items(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None:
            return

        index, _, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.ocr_tab_index)
        self._load_cached_ocr_for_index(index, show_errors=True, persist_stage_status=True)

    def run_ocr_for_selected_page(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        if not self._apply_server_inputs_to_manager():
            return

        _, image_relative_path, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.ocr_tab_index)
        self._start_ocr_inference_task(
            [image_relative_path],
            task_name=f"OCR: {Path(image_relative_path).name}",
            selected_item_ids_by_page=None,
        )

    def run_ocr_for_all_pages(self) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before running OCR.")
            return

        if not self._apply_server_inputs_to_manager():
            return

        image_relative_paths = [
            relative_path
            for relative_path in (
                self.current_project.page_relative_path_for_index(index)
                for index in range(self.current_project.page_count)
            )
            if relative_path is not None
        ]
        if not image_relative_paths:
            self._show_error("No source images", "Import images and prepare OCR items before running OCR.")
            return

        self.stage_tabs.setCurrentIndex(self.ocr_tab_index)
        self._start_ocr_inference_task(
            image_relative_paths,
            task_name=f"OCR: {len(image_relative_paths)} page(s)",
            selected_item_ids_by_page=None,
        )

    def run_ocr_for_selected_items(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        if self.current_ocr_data is None:
            self._show_error("OCR items not prepared", "Prepare OCR items before running OCR.")
            return

        if not self._apply_server_inputs_to_manager():
            return

        selected_item_ids = self._selected_ocr_item_ids(show_error=True)
        if not selected_item_ids:
            return

        _, image_relative_path, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.ocr_tab_index)
        self._start_ocr_inference_task(
            [image_relative_path],
            task_name=f"OCR Items: {Path(image_relative_path).name}",
            selected_item_ids_by_page={image_relative_path: selected_item_ids},
        )

    def save_edited_ocr_text(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        index, image_relative_path, _ = selected_context
        cache_path = ocr_json_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self._show_error("OCR items not prepared", "Prepare OCR items before saving OCR text.")
            return

        try:
            self._sync_current_ocr_data_from_table()
            if self.current_ocr_data is None:
                raise RuntimeError("No OCR data is loaded for the selected page.")
            save_ocr_payload(self.current_ocr_data, cache_path)
            self.current_ocr_cache_path = cache_path
            self._update_project_ocr_stage_status(image_relative_path, self.current_ocr_data)
            self.current_project.save()
        except Exception as exc:
            self._show_error("Failed to save OCR text", str(exc))
            return

        self._load_cached_ocr_for_index(index, show_errors=False)
        self.statusBar().showMessage("Saved OCR text edits")
        self._log_message(f"Saved OCR text edits to {cache_path}")

    def initialize_translation_for_selected_page(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.translation_tab_index)
        self._start_translation_initialization_task(
            [image_relative_path],
            task_name=f"Translation Init: {Path(image_relative_path).name}",
        )

    def initialize_translation_for_all_pages(self) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before initializing translation.")
            return

        image_relative_paths = [
            relative_path
            for relative_path in (
                self.current_project.page_relative_path_for_index(index)
                for index in range(self.current_project.page_count)
            )
            if relative_path is not None
        ]
        if not image_relative_paths:
            self._show_error("No source images", "Import images and prepare OCR before initializing translation.")
            return

        self.stage_tabs.setCurrentIndex(self.translation_tab_index)
        self._start_translation_initialization_task(
            image_relative_paths,
            task_name=f"Translation Init: {len(image_relative_paths)} page(s)",
        )

    def run_translation_for_selected_page(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.translation_tab_index)
        self._start_translation_task(
            [image_relative_path],
            task_name=f"Translation: {Path(image_relative_path).name}",
            selected_item_ids_by_page=None,
        )

    def run_translation_for_all_pages(self) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before running translation.")
            return

        image_relative_paths = [
            relative_path
            for relative_path in (
                self.current_project.page_relative_path_for_index(index)
                for index in range(self.current_project.page_count)
            )
            if relative_path is not None
        ]
        if not image_relative_paths:
            self._show_error("No source images", "Import images and prepare OCR before running translation.")
            return

        self.stage_tabs.setCurrentIndex(self.translation_tab_index)
        self._start_translation_task(
            image_relative_paths,
            task_name=f"Translation: {len(image_relative_paths)} page(s)",
            selected_item_ids_by_page=None,
        )

    def run_translation_for_selected_items(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        if self.current_translation_data is None:
            self._show_error("Translation not initialized", "Initialize translation for this page first.")
            return

        selected_item_ids = self._selected_translation_item_ids(show_error=True)
        if not selected_item_ids:
            return

        _, image_relative_path, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.translation_tab_index)
        self._start_translation_task(
            [image_relative_path],
            task_name=f"Translation Items: {Path(image_relative_path).name}",
            selected_item_ids_by_page={image_relative_path: selected_item_ids},
        )

    def reload_cached_translation(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None:
            return

        index, _, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.translation_tab_index)
        self._load_cached_translation_for_index(index, show_errors=True, persist_stage_status=True)

    def save_edited_translation_text(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        index, image_relative_path, _ = selected_context
        cache_path = translation_json_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self._show_error("Translation not initialized", "Initialize translation before saving translation text.")
            return

        try:
            self._sync_current_translation_data_from_table()
            if self.current_translation_data is None:
                raise RuntimeError("No translation data is loaded for the selected page.")
            save_translation_json(cache_path, self.current_translation_data)
            self.current_translation_cache_path = cache_path
            self._update_project_translation_stage_status(image_relative_path, self.current_translation_data)
            self.current_project.save()
        except Exception as exc:
            self._show_error("Failed to save translation text", str(exc))
            return

        self._load_cached_translation_for_index(index, show_errors=False)
        self.statusBar().showMessage("Saved translation text edits")
        self._log_message(f"Saved translation text edits to {cache_path}")

    def prepare_inpaint_mask_for_selected_page(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.inpaint_tab_index)
        self._start_inpaint_mask_task(
            [image_relative_path],
            task_name=f"Inpaint Mask: {Path(image_relative_path).name}",
        )

    def prepare_inpaint_mask_for_all_pages(self) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before preparing inpaint masks.")
            return

        image_relative_paths = [
            relative_path
            for relative_path in (
                self.current_project.page_relative_path_for_index(index)
                for index in range(self.current_project.page_count)
            )
            if relative_path is not None
        ]
        if not image_relative_paths:
            self._show_error("No source images", "Import images before preparing inpaint masks.")
            return

        self.stage_tabs.setCurrentIndex(self.inpaint_tab_index)
        self._start_inpaint_mask_task(
            image_relative_paths,
            task_name=f"Inpaint Mask: {len(image_relative_paths)} page(s)",
        )

    def run_inpaint_for_selected_page(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.inpaint_tab_index)
        self._start_inpaint_task(
            [image_relative_path],
            task_name=f"Inpaint: {Path(image_relative_path).name}",
        )

    def run_inpaint_for_all_pages(self) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before running inpaint.")
            return

        image_relative_paths = [
            relative_path
            for relative_path in (
                self.current_project.page_relative_path_for_index(index)
                for index in range(self.current_project.page_count)
            )
            if relative_path is not None
        ]
        if not image_relative_paths:
            self._show_error("No source images", "Import images before running inpaint.")
            return

        self.stage_tabs.setCurrentIndex(self.inpaint_tab_index)
        self._start_inpaint_task(
            image_relative_paths,
            task_name=f"Inpaint: {len(image_relative_paths)} page(s)",
        )

    def reload_cached_inpaint(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None:
            return

        index, _, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.inpaint_tab_index)
        self._load_cached_inpaint_for_index(index, show_errors=True, persist_stage_status=True)
        self._refresh_preview_for_current_page()

    def clear_inpaint_preview(self) -> None:
        self.inpaint_preview_mode_input.setCurrentText(INPAINT_PREVIEW_SOURCE)
        self.image_preview.clear_mask_overlay()
        self._refresh_preview_for_current_page()
        self.statusBar().showMessage("Inpaint preview cleared")
        self._log_message("Cleared inpaint preview overlay.")

    def load_lama_model(self) -> None:
        self.stage_tabs.setCurrentIndex(self.inpaint_tab_index)
        self._start_lama_model_task("load")

    def unload_lama_model(self) -> None:
        self.stage_tabs.setCurrentIndex(self.inpaint_tab_index)
        self._start_lama_model_task("unload")

    def prepare_render_for_selected_page(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.render_tab_index)
        self._start_render_preparation_task(
            [image_relative_path],
            task_name=f"Render Prep: {Path(image_relative_path).name}",
        )

    def prepare_render_for_all_pages(self) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before preparing render data.")
            return

        image_relative_paths = [
            relative_path
            for relative_path in (
                self.current_project.page_relative_path_for_index(index)
                for index in range(self.current_project.page_count)
            )
            if relative_path is not None
        ]
        if not image_relative_paths:
            self._show_error("No source images", "Import images before preparing render data.")
            return

        self.stage_tabs.setCurrentIndex(self.render_tab_index)
        self._start_render_preparation_task(
            image_relative_paths,
            task_name=f"Render Prep: {len(image_relative_paths)} page(s)",
        )

    def run_render_for_selected_page(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None or self.current_project is None:
            return

        _, image_relative_path, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.render_tab_index)
        self._start_render_task(
            [image_relative_path],
            task_name=f"Render: {Path(image_relative_path).name}",
        )

    def run_render_for_all_pages(self) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before running render.")
            return

        image_relative_paths = [
            relative_path
            for relative_path in (
                self.current_project.page_relative_path_for_index(index)
                for index in range(self.current_project.page_count)
            )
            if relative_path is not None
        ]
        if not image_relative_paths:
            self._show_error("No source images", "Import images before running render.")
            return

        self.stage_tabs.setCurrentIndex(self.render_tab_index)
        self._start_render_task(
            image_relative_paths,
            task_name=f"Render: {len(image_relative_paths)} page(s)",
        )

    def reload_cached_render(self) -> None:
        selected_context = self._selected_page_context(show_error=True)
        if selected_context is None:
            return

        index, _, _ = selected_context
        self.stage_tabs.setCurrentIndex(self.render_tab_index)
        self._load_cached_render_for_index(index, show_errors=True, persist_stage_status=True)
        self._refresh_preview_for_current_page()

    def clear_render_preview(self) -> None:
        self.render_preview_mode_input.setCurrentText(RENDER_PREVIEW_SOURCE)
        self._refresh_preview_for_current_page()
        self.statusBar().showMessage("Render preview cleared")
        self._log_message("Cleared render preview.")

    def _refresh_project_view(self) -> None:
        if self.current_project is None:
            self.page_list.clear()
            self.image_preview.clear_image()
            self.current_detection_data = None
            self.current_ocr_data = None
            self.current_ocr_cache_path = None
            self.current_translation_data = None
            self.current_translation_cache_path = None
            self.current_inpaint_data = None
            self.current_inpaint_cache_path = None
            self.current_render_data = None
            self.current_render_items = []
            self.current_render_cache_path = None
            self._reset_detection_stats()
            self._clear_ocr_view()
            self._clear_translation_view()
            self._clear_inpaint_view()
            self._clear_render_view()
            self._update_window_title()
            return

        selected_index = self.current_project.data.current_page_index
        self.page_list.set_pages(self.current_project.page_display_names(), selected_index)

        if self.current_project.page_count == 0:
            self.image_preview.clear_image()
            self.current_detection_data = None
            self.current_ocr_data = None
            self.current_ocr_cache_path = None
            self.current_translation_data = None
            self.current_translation_cache_path = None
            self.current_inpaint_data = None
            self.current_inpaint_cache_path = None
            self.current_render_data = None
            self.current_render_items = []
            self.current_render_cache_path = None
            self._reset_detection_stats()
            self._clear_ocr_view()
            self._clear_translation_view()
            self._clear_inpaint_view()
            self._clear_render_view()
            self.statusBar().showMessage("Project ready")

        self._update_window_title()

    def _on_page_selected(self, index: int) -> None:
        if self.current_project is None:
            return

        self.current_project.set_current_page(index)
        page_path = self.current_project.image_path_for_index(index)
        page_name = self.current_project.page_display_names()[index]

        if page_path is None or not page_path.exists():
            self.image_preview.clear_image()
            self.current_detection_data = None
            self.current_ocr_data = None
            self.current_ocr_cache_path = None
            self.current_translation_data = None
            self.current_translation_cache_path = None
            self.current_inpaint_data = None
            self.current_inpaint_cache_path = None
            self.current_render_data = None
            self.current_render_items = []
            self.current_render_cache_path = None
            self._reset_detection_stats()
            self._clear_ocr_view()
            self._clear_translation_view()
            self._clear_inpaint_view()
            self._clear_render_view()
            self._log_message(f"Missing source image: {page_name}")
            self.statusBar().showMessage("Source image is missing")
            return

        has_overlay = self._load_cached_detection_for_index(index, show_errors=False)
        has_ocr = self._load_cached_ocr_for_index(index, show_errors=False)
        has_translation = self._load_cached_translation_for_index(index, show_errors=False)
        has_inpaint = self._load_cached_inpaint_for_index(index, show_errors=False)
        has_render = self._load_cached_render_for_index(index, show_errors=False)

        if not self._refresh_preview_for_current_page():
            self.current_detection_data = None
            self.current_ocr_data = None
            self.current_ocr_cache_path = None
            self.current_translation_data = None
            self.current_translation_cache_path = None
            self.current_inpaint_data = None
            self.current_inpaint_cache_path = None
            self.current_render_data = None
            self.current_render_items = []
            self.current_render_cache_path = None
            self._reset_detection_stats()
            self._clear_ocr_view()
            self._clear_translation_view()
            self._clear_inpaint_view()
            self._clear_render_view()
            self._log_message(f"Unable to preview page: {page_name}")
            self.statusBar().showMessage("Preview unavailable")
            return
        if has_overlay and has_ocr and has_translation and has_inpaint and has_render:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with cached render output"
            )
        elif has_overlay and has_ocr and has_translation and has_inpaint:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with detection, OCR, translation, and inpaint cache"
            )
        elif has_overlay and has_ocr and has_translation:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with detection overlay, OCR items, and translation"
            )
        elif has_inpaint and has_overlay and has_ocr:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with detection, OCR items, and inpaint cache"
            )
        elif has_overlay and has_ocr:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with detection overlay and OCR items"
            )
        elif has_inpaint:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with cached inpaint output"
            )
        elif has_render:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with cached render metadata"
            )
        elif has_overlay:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with detection overlay"
            )
        elif has_ocr:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with cached OCR items"
            )
        elif has_translation:
            self.statusBar().showMessage(
                f"Showing page {index + 1} of {self.current_project.page_count} with cached translation"
            )
        else:
            self.statusBar().showMessage(f"Showing page {index + 1} of {self.current_project.page_count}")

        self._log_message(f"Selected page: {page_name}")

    def _load_cached_detection_for_index(
        self,
        index: int,
        *,
        show_errors: bool,
        persist_stage_status: bool = False,
    ) -> bool:
        if self.current_project is None:
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        if image_relative_path is None:
            self.image_preview.clear_overlays()
            self._reset_detection_stats()
            return False

        cache_path = detection_json_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self.image_preview.clear_overlays()
            self.current_detection_data = None
            self._reset_detection_stats(cache_path=cache_path)
            return False

        try:
            detection_data = load_detection_json(cache_path)
        except Exception as exc:
            self.image_preview.clear_overlays()
            self.current_detection_data = None
            self._reset_detection_stats(cache_path=cache_path)
            self._log_message(f"Failed to load detection cache {cache_path}: {exc}")
            if show_errors:
                self._show_error("Invalid detection cache", str(exc))
            return False

        self.current_detection_data = detection_data
        self.image_preview.set_detection_overlay(detection_data)
        self._update_detection_stats(detection_data, cache_path)

        if persist_stage_status:
            relative_cache_path = self._relative_project_path(cache_path)
            self.current_project.update_stage_status(
                image_relative_path,
                "detection",
                status="done",
                cache_path=relative_cache_path,
            )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                if show_errors:
                    self._show_error("Failed to save project", str(exc))
                else:
                    self._log_message(f"Failed to save project after loading detection cache: {exc}")

        return True

    def _load_cached_ocr_for_index(
        self,
        index: int,
        *,
        show_errors: bool,
        persist_stage_status: bool = False,
    ) -> bool:
        if self.current_project is None:
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        if image_relative_path is None:
            self._clear_ocr_view()
            return False

        cache_path = ocr_json_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self.current_ocr_data = None
            self.current_ocr_cache_path = None
            self._clear_ocr_view()
            return False

        try:
            ocr_data = load_ocr_json(cache_path)
        except Exception as exc:
            self.current_ocr_data = None
            self.current_ocr_cache_path = None
            self._clear_ocr_view()
            self._log_message(f"Failed to load OCR cache {cache_path}: {exc}")
            if show_errors:
                self._show_error("Invalid OCR cache", str(exc))
            return False

        self.current_ocr_data = ocr_data
        self.current_ocr_cache_path = cache_path
        self._populate_ocr_items_table(ocr_data.get("items", []))

        if persist_stage_status:
            self._update_project_ocr_stage_status(image_relative_path, ocr_data)
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                if show_errors:
                    self._show_error("Failed to save project", str(exc))
                else:
                    self._log_message(f"Failed to save project after loading OCR cache: {exc}")

        return True

    def _load_cached_translation_for_index(
        self,
        index: int,
        *,
        show_errors: bool,
        persist_stage_status: bool = False,
    ) -> bool:
        if self.current_project is None:
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        if image_relative_path is None:
            self._clear_translation_view()
            return False

        cache_path = translation_json_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self.current_translation_data = None
            self.current_translation_cache_path = None
            self._clear_translation_view()
            return False

        try:
            translation_data = load_translation_json(cache_path)
        except Exception as exc:
            self.current_translation_data = None
            self.current_translation_cache_path = None
            self._clear_translation_view()
            self._log_message(f"Failed to load translation cache {cache_path}: {exc}")
            if show_errors:
                self._show_error("Invalid translation cache", str(exc))
            return False

        self.current_translation_data = translation_data
        self.current_translation_cache_path = cache_path
        self._populate_translation_items_table(translation_data.get("items", []))

        if persist_stage_status:
            self._update_project_translation_stage_status(image_relative_path, translation_data)
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                if show_errors:
                    self._show_error("Failed to save project", str(exc))
                else:
                    self._log_message(f"Failed to save project after loading translation cache: {exc}")

        return True

    def _load_cached_inpaint_for_index(
        self,
        index: int,
        *,
        show_errors: bool,
        persist_stage_status: bool = False,
    ) -> bool:
        if self.current_project is None:
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        if image_relative_path is None:
            self._clear_inpaint_view()
            return False

        cache_path = inpaint_json_path(self.current_project, image_relative_path)
        expected_output_path = inpaint_image_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self.current_inpaint_data = None
            self.current_inpaint_cache_path = None
            self._reset_inpaint_details(
                source_image=image_relative_path,
                output_image=expected_output_path if expected_output_path.exists() else None,
            )
            return expected_output_path.exists()

        try:
            inpaint_data = load_inpaint_json(cache_path)
        except Exception as exc:
            self.current_inpaint_data = None
            self.current_inpaint_cache_path = None
            self._reset_inpaint_details(
                source_image=image_relative_path,
                output_image=expected_output_path if expected_output_path.exists() else None,
            )
            self._log_message(f"Failed to load inpaint cache {cache_path}: {exc}")
            if show_errors:
                self._show_error("Invalid inpaint cache", str(exc))
            return expected_output_path.exists()

        self.current_inpaint_data = inpaint_data
        self.current_inpaint_cache_path = cache_path
        self._update_inpaint_details(inpaint_data, cache_path)

        if persist_stage_status:
            self._update_project_inpaint_stage_status(image_relative_path, inpaint_data)
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                if show_errors:
                    self._show_error("Failed to save project", str(exc))
                else:
                    self._log_message(f"Failed to save project after loading inpaint cache: {exc}")

        return True

    def _load_cached_render_for_index(
        self,
        index: int,
        *,
        show_errors: bool,
        persist_stage_status: bool = False,
    ) -> bool:
        if self.current_project is None:
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        if image_relative_path is None:
            self._clear_render_view()
            return False

        cache_path = render_json_path(self.current_project, image_relative_path)
        expected_output_path = render_image_path(self.current_project, image_relative_path)
        if not cache_path.exists():
            self.current_render_data = None
            self.current_render_items = []
            self.current_render_cache_path = None
            self._reset_render_details(output_image=expected_output_path if expected_output_path.exists() else None)
            return expected_output_path.exists()

        try:
            render_data = load_render_json(cache_path)
        except Exception as exc:
            self.current_render_data = None
            self.current_render_items = []
            self.current_render_cache_path = None
            self._reset_render_details(output_image=expected_output_path if expected_output_path.exists() else None)
            self._log_message(f"Failed to load render cache {cache_path}: {exc}")
            if show_errors:
                self._show_error("Invalid render cache", str(exc))
            return expected_output_path.exists()

        self.current_render_data = render_data
        self.current_render_cache_path = cache_path
        self._populate_render_items_table(render_data.get("items", []))

        if persist_stage_status:
            self._update_project_render_stage_status(image_relative_path, render_data)
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                if show_errors:
                    self._show_error("Failed to save project", str(exc))
                else:
                    self._log_message(f"Failed to save project after loading render cache: {exc}")

        return True

    def _start_detection_task(self, image_paths: list[Path], *, task_name: str) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before running detection.")
            return

        task = DetectionTask(
            name=task_name,
            stage="detection",
            image_paths=image_paths,
            detection_cache_dir=self.current_project.cache_dir / "detection",
            masks_cache_dir=self.current_project.cache_dir / "masks",
            force=True,
        )
        worker = create_detection_worker(task)
        worker.signals.started.connect(self._on_detection_worker_started)
        worker.signals.progress.connect(self._on_detection_worker_progress)
        worker.signals.message.connect(self._log_message)
        worker.signals.finished.connect(
            lambda result, active_worker=worker: self._on_detection_worker_finished(result, active_worker)
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker, active_task=task: self._on_detection_worker_failed(
                message,
                active_worker,
                active_task,
            )
        )

        self._active_workers.append(worker)
        self._set_detection_actions_enabled(False)
        self.statusBar().showMessage("Detection is running...")
        self.thread_pool.start(worker)

    def _start_ocr_preparation_task(self, image_relative_paths: list[str], *, task_name: str) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before preparing OCR items.")
            return

        task = OCRPreparationTask(
            name=task_name,
            stage="ocr",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            force=True,
            save_crops=True,
        )
        worker = create_ocr_preparation_worker(task)
        worker.signals.started.connect(self._on_ocr_worker_started)
        worker.signals.progress.connect(self._on_ocr_worker_progress)
        worker.signals.message.connect(self._log_message)
        worker.signals.finished.connect(
            lambda result, active_worker=worker: self._on_ocr_worker_finished(result, active_worker)
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker, active_task=task: self._on_ocr_worker_failed(
                message,
                active_worker,
                active_task,
            )
        )

        self._active_workers.append(worker)
        self._set_ocr_actions_enabled(False)
        self.statusBar().showMessage("Preparing OCR items...")
        self.thread_pool.start(worker)

    def _start_ocr_inference_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
        selected_item_ids_by_page: dict[str, list[int]] | None,
    ) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before running OCR.")
            return

        task = OCRInferenceTask(
            name=task_name,
            stage="ocr",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            server_url=self.llama_server_manager.server_url,
            force=self.force_reocr_checkbox.isChecked(),
            selected_item_ids_by_page=selected_item_ids_by_page or {},
            timeout=120.0,
        )
        worker = create_ocr_inference_worker(task)
        worker.signals.started.connect(self._on_ocr_inference_worker_started)
        worker.signals.progress.connect(self._on_ocr_inference_worker_progress)
        worker.signals.message.connect(self._log_message)
        worker.signals.finished.connect(
            lambda result, active_worker=worker: self._on_ocr_inference_worker_finished(result, active_worker)
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker, active_task=task: self._on_ocr_inference_worker_failed(
                message,
                active_worker,
                active_task,
            )
        )

        self._active_workers.append(worker)
        self._set_ocr_actions_enabled(False)
        self.statusBar().showMessage("Running OCR...")
        self.thread_pool.start(worker)

    def _start_translation_initialization_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
    ) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before initializing translation.")
            return

        config = self._translation_config_from_inputs()
        task = TranslationInitializationTask(
            name=task_name,
            stage="translation",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            config=config,
            force=self.translation_force_checkbox.isChecked(),
        )
        worker = create_translation_initialization_worker(task)
        worker.signals.started.connect(self._on_translation_init_worker_started)
        worker.signals.progress.connect(self._on_translation_init_worker_progress)
        worker.signals.message.connect(self._log_message)
        worker.signals.finished.connect(
            lambda result, active_worker=worker: self._on_translation_init_worker_finished(result, active_worker)
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker, active_task=task: self._on_translation_init_worker_failed(
                message,
                active_worker,
                active_task,
            )
        )

        self._active_workers.append(worker)
        self._set_translation_actions_enabled(False)
        self.statusBar().showMessage("Initializing translation...")
        self.thread_pool.start(worker)

    def _start_translation_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
        selected_item_ids_by_page: dict[str, list[int]] | None,
    ) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before running translation.")
            return

        config = self._translation_config_from_inputs()
        task = TranslationTask(
            name=task_name,
            stage="translation",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            config=config,
            force=self.translation_force_checkbox.isChecked(),
            selected_item_ids_by_page=selected_item_ids_by_page or {},
        )
        worker = create_translation_worker(task)
        worker.signals.started.connect(self._on_translation_worker_started)
        worker.signals.progress.connect(self._on_translation_worker_progress)
        worker.signals.message.connect(self._log_message)
        worker.signals.finished.connect(
            lambda result, active_worker=worker: self._on_translation_worker_finished(result, active_worker)
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker, active_task=task: self._on_translation_worker_failed(
                message,
                active_worker,
                active_task,
            )
        )

        self._active_workers.append(worker)
        self._set_translation_actions_enabled(False)
        self.statusBar().showMessage("Running translation...")
        self.thread_pool.start(worker)

    def _start_inpaint_mask_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
    ) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before preparing inpaint masks.")
            return

        task = InpaintMaskTask(
            name=task_name,
            stage="inpaint",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            force=self.inpaint_force_checkbox.isChecked(),
            mask_padding=self.inpaint_mask_padding_input.value(),
            use_bubble_mask=self.inpaint_use_bubble_mask_checkbox.isChecked(),
        )
        worker = create_inpaint_mask_worker(task)
        worker.signals.started.connect(self._on_inpaint_mask_worker_started)
        worker.signals.progress.connect(self._on_inpaint_mask_worker_progress)
        worker.signals.message.connect(self._log_message)
        worker.signals.finished.connect(
            lambda result, active_worker=worker: self._on_inpaint_mask_worker_finished(result, active_worker)
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker, active_task=task: self._on_inpaint_mask_worker_failed(
                message,
                active_worker,
                active_task,
            )
        )

        self._active_workers.append(worker)
        self._set_inpaint_actions_enabled(False)
        self.statusBar().showMessage("Preparing inpaint masks...")
        self.thread_pool.start(worker)

    def _start_inpaint_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
    ) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before running inpaint.")
            return

        task = InpaintTask(
            name=task_name,
            stage="inpaint",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            force=self.inpaint_force_checkbox.isChecked(),
            mask_padding=self.inpaint_mask_padding_input.value(),
            use_bubble_mask=self.inpaint_use_bubble_mask_checkbox.isChecked(),
            use_crop_windows=self.inpaint_use_crop_windows_checkbox.isChecked(),
            device=self._inpaint_device_value(),
        )
        worker = create_inpaint_worker(task)
        worker.signals.started.connect(self._on_inpaint_worker_started)
        worker.signals.progress.connect(self._on_inpaint_worker_progress)
        worker.signals.message.connect(self._log_message)
        worker.signals.finished.connect(
            lambda result, active_worker=worker: self._on_inpaint_worker_finished(result, active_worker)
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker, active_task=task: self._on_inpaint_worker_failed(
                message,
                active_worker,
                active_task,
            )
        )

        self._active_workers.append(worker)
        self._set_inpaint_actions_enabled(False)
        self.statusBar().showMessage("Running inpaint...")
        self.thread_pool.start(worker)

    def _start_lama_model_task(self, action: str) -> None:
        task = LamaModelTask(
            name=f"LaMa model: {action}",
            stage="inpaint",
            action=action,
            device=self._inpaint_device_value(),
        )
        worker = create_lama_model_worker(task)
        worker.signals.started.connect(self._on_lama_model_worker_started)
        worker.signals.message.connect(self._log_message)
        worker.signals.finished.connect(
            lambda result, active_worker=worker, active_action=action: self._on_lama_model_worker_finished(
                result,
                active_worker,
                active_action,
            )
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker, active_action=action: self._on_lama_model_worker_failed(
                message,
                active_worker,
                active_action,
            )
        )

        self._active_workers.append(worker)
        self._set_inpaint_actions_enabled(False)
        self.thread_pool.start(worker)

    def _start_render_preparation_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
    ) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before preparing render data.")
            return

        task = RenderPreparationTask(
            name=task_name,
            stage="render",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            force=self.render_force_checkbox.isChecked(),
        )
        worker = create_render_preparation_worker(task)
        worker.signals.started.connect(self._on_render_prep_worker_started)
        worker.signals.progress.connect(self._on_render_prep_worker_progress)
        worker.signals.message.connect(self._log_message)
        worker.signals.finished.connect(
            lambda result, active_worker=worker: self._on_render_prep_worker_finished(result, active_worker)
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker, active_task=task: self._on_render_prep_worker_failed(
                message,
                active_worker,
                active_task,
            )
        )

        self._active_workers.append(worker)
        self._set_render_actions_enabled(False)
        self.statusBar().showMessage("Preparing render data...")
        self.thread_pool.start(worker)

    def _start_render_task(
        self,
        image_relative_paths: list[str],
        *,
        task_name: str,
    ) -> None:
        if self.current_project is None:
            self._show_error("No project open", "Create or open a project before running render.")
            return

        try:
            render_config = self._render_config_from_inputs()
        except Exception as exc:
            self._show_error("Invalid render settings", str(exc))
            return

        task = RenderTask(
            name=task_name,
            stage="render",
            project=self.current_project,
            image_relative_paths=image_relative_paths,
            config=render_config,
            force=self.render_force_checkbox.isChecked(),
        )
        worker = create_render_worker(task)
        worker.signals.started.connect(self._on_render_worker_started)
        worker.signals.progress.connect(self._on_render_worker_progress)
        worker.signals.message.connect(self._log_message)
        worker.signals.finished.connect(
            lambda result, active_worker=worker: self._on_render_worker_finished(result, active_worker)
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker, active_task=task: self._on_render_worker_failed(
                message,
                active_worker,
                active_task,
            )
        )

        self._active_workers.append(worker)
        self._set_render_actions_enabled(False)
        self.statusBar().showMessage("Rendering translated pages...")
        self.thread_pool.start(worker)

    def _start_llama_server_action(self, action: str, *, timeout_seconds: float) -> None:
        task = LlamaServerTask(
            name=f"llama.cpp server: {action}",
            stage="ocr",
            manager=self.llama_server_manager,
            action=action,
            timeout_seconds=timeout_seconds,
        )
        worker = create_llama_server_worker(task)
        worker.signals.started.connect(self._on_llama_worker_started)
        worker.signals.message.connect(self._log_message)
        worker.signals.finished.connect(
            lambda result, active_worker=worker, active_action=action: self._on_llama_worker_finished(
                result,
                active_worker,
                active_action,
            )
        )
        worker.signals.failed.connect(
            lambda message, active_worker=worker, active_action=action: self._on_llama_worker_failed(
                message,
                active_worker,
                active_action,
            )
        )

        self._active_workers.append(worker)
        self._set_server_actions_enabled(False)
        self.thread_pool.start(worker)

    def _on_detection_worker_started(self, task_name: str) -> None:
        self._log_message(f"Started background task: {task_name}")

    def _on_detection_worker_progress(self, progress_value: int) -> None:
        self.statusBar().showMessage(f"Detection progress: {progress_value}%")

    def _on_detection_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._release_worker(worker)
        self._set_detection_actions_enabled(True)

        if not isinstance(result, DetectionWorkerResult):
            self.statusBar().showMessage("Detection finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                relative_source_path = self.current_project.relative_source_path(page_result.image_path)
                if relative_source_path is None:
                    continue

                if page_result.json_path is not None:
                    self.current_project.update_stage_status(
                        relative_source_path,
                        "detection",
                        status="done",
                        cache_path=self._relative_project_path(page_result.json_path),
                    )
                else:
                    self.current_project.update_stage_status(
                        relative_source_path,
                        "detection",
                        status="failed",
                        error=page_result.error or "Unknown detection failure.",
                    )

            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._show_error("Failed to save project", str(exc))

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_detection_for_index(current_index, show_errors=False)

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Detection finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Detection finished for {success_count} page(s).")

    def _on_detection_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: DetectionTask,
    ) -> None:
        self._release_worker(worker)
        self._set_detection_actions_enabled(True)

        if self.current_project is not None:
            for image_path in task.image_paths:
                relative_source_path = self.current_project.relative_source_path(image_path)
                if relative_source_path is None:
                    continue
                self.current_project.update_stage_status(
                    relative_source_path,
                    "detection",
                    status="failed",
                    error=message,
                )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._log_message(f"Failed to save project after detection error: {exc}")

        self.statusBar().showMessage("Detection failed")
        self._show_error("Detection failed", message)

    def _on_ocr_worker_started(self, task_name: str) -> None:
        self._log_message(f"Started background task: {task_name}")

    def _on_ocr_worker_progress(self, progress_value: int) -> None:
        self.statusBar().showMessage(f"OCR item preparation progress: {progress_value}%")

    def _on_ocr_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._release_worker(worker)
        self._set_ocr_actions_enabled(True)

        if not isinstance(result, OCRPreparationWorkerResult):
            self.statusBar().showMessage("OCR item preparation finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.json_path is not None:
                    try:
                        ocr_data = load_ocr_json(page_result.json_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "ocr",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_ocr_stage_status(page_result.image_relative_path, ocr_data)
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "ocr",
                        status="failed",
                        error=page_result.error or "Unknown OCR preparation failure.",
                    )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._show_error("Failed to save project", str(exc))

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_ocr_for_index(current_index, show_errors=False)

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"OCR item preparation finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"OCR item preparation finished for {success_count} page(s).")

    def _on_ocr_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: OCRPreparationTask,
    ) -> None:
        self._release_worker(worker)
        self._set_ocr_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "ocr",
                    status="failed",
                    error=message,
                )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._log_message(f"Failed to save project after OCR preparation error: {exc}")

        self.statusBar().showMessage("OCR item preparation failed")
        self._show_error("OCR item preparation failed", message)

    def _on_ocr_inference_worker_started(self, task_name: str) -> None:
        self._log_message(f"Started background task: {task_name}")

    def _on_ocr_inference_worker_progress(self, progress_value: int) -> None:
        self.statusBar().showMessage(f"OCR progress: {progress_value}%")

    def _on_ocr_inference_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._release_worker(worker)
        self._set_ocr_actions_enabled(True)

        if not isinstance(result, OCRInferenceWorkerResult):
            self.statusBar().showMessage("OCR finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.json_path is not None:
                    try:
                        ocr_data = load_ocr_json(page_result.json_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "ocr",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_ocr_stage_status(page_result.image_relative_path, ocr_data)
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "ocr",
                        status="failed",
                        error=page_result.error or "Unknown OCR inference failure.",
                    )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._show_error("Failed to save project", str(exc))

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_ocr_for_index(current_index, show_errors=False)

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"OCR finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"OCR finished for {success_count} page(s).")

    def _on_ocr_inference_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: OCRInferenceTask,
    ) -> None:
        self._release_worker(worker)
        self._set_ocr_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "ocr",
                    status="failed",
                    error=message,
                )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._log_message(f"Failed to save project after OCR error: {exc}")

        self.statusBar().showMessage("OCR failed")
        self._show_error("OCR failed", message)

    def _on_translation_init_worker_started(self, task_name: str) -> None:
        self._log_message(f"Started background task: {task_name}")

    def _on_translation_init_worker_progress(self, progress_value: int) -> None:
        self.statusBar().showMessage(f"Translation initialization progress: {progress_value}%")

    def _on_translation_init_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._release_worker(worker)
        self._set_translation_actions_enabled(True)

        if not isinstance(result, TranslationInitializationWorkerResult):
            self.statusBar().showMessage("Translation initialization finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.json_path is not None:
                    try:
                        translation_data = load_translation_json(page_result.json_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "translation",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_translation_stage_status(
                            page_result.image_relative_path,
                            translation_data,
                        )
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "translation",
                        status="failed",
                        error=page_result.error or "Unknown translation initialization failure.",
                    )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._show_error("Failed to save project", str(exc))

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_translation_for_index(current_index, show_errors=False)

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Translation initialization finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Translation initialization finished for {success_count} page(s).")

    def _on_translation_init_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: TranslationInitializationTask,
    ) -> None:
        self._release_worker(worker)
        self._set_translation_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "translation",
                    status="failed",
                    error=message,
                )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._log_message(f"Failed to save project after translation init error: {exc}")

        self.statusBar().showMessage("Translation initialization failed")
        self._show_error("Translation initialization failed", message)

    def _on_translation_worker_started(self, task_name: str) -> None:
        self._log_message(f"Started background task: {task_name}")

    def _on_translation_worker_progress(self, progress_value: int) -> None:
        self.statusBar().showMessage(f"Translation progress: {progress_value}%")

    def _on_translation_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._release_worker(worker)
        self._set_translation_actions_enabled(True)

        if not isinstance(result, TranslationWorkerResult):
            self.statusBar().showMessage("Translation finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.json_path is not None:
                    try:
                        translation_data = load_translation_json(page_result.json_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "translation",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_translation_stage_status(
                            page_result.image_relative_path,
                            translation_data,
                        )
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "translation",
                        status="failed",
                        error=page_result.error or "Unknown translation failure.",
                    )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._show_error("Failed to save project", str(exc))

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_translation_for_index(current_index, show_errors=False)

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Translation finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Translation finished for {success_count} page(s).")

    def _on_translation_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: TranslationTask,
    ) -> None:
        self._release_worker(worker)
        self._set_translation_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "translation",
                    status="failed",
                    error=message,
                )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._log_message(f"Failed to save project after translation error: {exc}")

        self.statusBar().showMessage("Translation failed")
        self._show_error("Translation failed", message)

    def _on_inpaint_mask_worker_started(self, task_name: str) -> None:
        self._log_message(f"Started background task: {task_name}")

    def _on_inpaint_mask_worker_progress(self, progress_value: int) -> None:
        self.statusBar().showMessage(f"Inpaint mask preparation progress: {progress_value}%")

    def _on_inpaint_mask_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._release_worker(worker)
        self._set_inpaint_actions_enabled(True)

        if not isinstance(result, InpaintMaskWorkerResult):
            self.statusBar().showMessage("Inpaint mask preparation finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.mask_path is not None:
                    metadata_path = inpaint_json_path(self.current_project, page_result.image_relative_path)
                    try:
                        inpaint_data = load_inpaint_json(metadata_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "inpaint",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_inpaint_stage_status(
                            page_result.image_relative_path,
                            inpaint_data,
                        )
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "inpaint",
                        status="failed",
                        error=page_result.error or "Unknown inpaint mask preparation failure.",
                    )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._show_error("Failed to save project", str(exc))

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_inpaint_for_index(current_index, show_errors=False)
            self._refresh_preview_for_current_page()

        success_count = len(result.mask_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Inpaint mask preparation finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Inpaint mask preparation finished for {success_count} page(s).")

    def _on_inpaint_mask_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: InpaintMaskTask,
    ) -> None:
        self._release_worker(worker)
        self._set_inpaint_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "inpaint",
                    status="failed",
                    error=message,
                )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._log_message(f"Failed to save project after inpaint mask error: {exc}")

        self.statusBar().showMessage("Inpaint mask preparation failed")
        self._show_error("Inpaint mask preparation failed", message)

    def _on_inpaint_worker_started(self, task_name: str) -> None:
        self._log_message(f"Started background task: {task_name}")

    def _on_inpaint_worker_progress(self, progress_value: int) -> None:
        self.statusBar().showMessage(f"Inpaint progress: {progress_value}%")

    def _on_inpaint_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._release_worker(worker)
        self._set_inpaint_actions_enabled(True)

        if not isinstance(result, InpaintWorkerResult):
            self.statusBar().showMessage("Inpaint finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.image_path is not None:
                    metadata_path = inpaint_json_path(self.current_project, page_result.image_relative_path)
                    try:
                        inpaint_data = load_inpaint_json(metadata_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "inpaint",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_inpaint_stage_status(
                            page_result.image_relative_path,
                            inpaint_data,
                        )
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "inpaint",
                        status="failed",
                        error=page_result.error or "Unknown inpaint failure.",
                    )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._show_error("Failed to save project", str(exc))

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_inpaint_for_index(current_index, show_errors=False)
            self._refresh_preview_for_current_page()

        success_count = len(result.image_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Inpaint finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Inpaint finished for {success_count} page(s).")

    def _on_inpaint_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: InpaintTask,
    ) -> None:
        self._release_worker(worker)
        self._set_inpaint_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "inpaint",
                    status="failed",
                    error=message,
                )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._log_message(f"Failed to save project after inpaint error: {exc}")

        self.statusBar().showMessage("Inpaint failed")
        self._show_error("Inpaint failed", message)

    def _on_lama_model_worker_started(self, task_name: str) -> None:
        self._log_message(f"Started background task: {task_name}")

    def _on_lama_model_worker_finished(
        self,
        result: object,
        worker: TaskWorker,
        action: str,
    ) -> None:
        self._release_worker(worker)
        self._set_inpaint_actions_enabled(True)

        if not isinstance(result, LamaModelTaskResult):
            self.statusBar().showMessage("LaMa model action finished")
            return

        if result.loaded:
            status_text = f"Loaded ({result.device or 'auto'})"
        else:
            status_text = "Not loaded"
        self.lama_model_status_value.setText(status_text)
        self.statusBar().showMessage(result.message)
        self._log_message(result.message)

    def _on_lama_model_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        action: str,
    ) -> None:
        self._release_worker(worker)
        self._set_inpaint_actions_enabled(True)
        self.lama_model_status_value.setText("Error")
        self.statusBar().showMessage("LaMa model action failed")
        self._show_error("LaMa model error", message)

    def _on_render_prep_worker_started(self, task_name: str) -> None:
        self._log_message(f"Started background task: {task_name}")

    def _on_render_prep_worker_progress(self, progress_value: int) -> None:
        self.statusBar().showMessage(f"Render preparation progress: {progress_value}%")

    def _on_render_prep_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._release_worker(worker)
        self._set_render_actions_enabled(True)

        if not isinstance(result, RenderPreparationWorkerResult):
            self.statusBar().showMessage("Render preparation finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.json_path is not None:
                    try:
                        render_data = load_render_json(page_result.json_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "render",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_render_stage_status(
                            page_result.image_relative_path,
                            render_data,
                        )
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "render",
                        status="failed",
                        error=page_result.error or "Unknown render preparation failure.",
                    )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._show_error("Failed to save project", str(exc))

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_render_for_index(current_index, show_errors=False)
            self._refresh_preview_for_current_page()

        success_count = len(result.json_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Render preparation finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Render preparation finished for {success_count} page(s).")

    def _on_render_prep_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: RenderPreparationTask,
    ) -> None:
        self._release_worker(worker)
        self._set_render_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "render",
                    status="failed",
                    error=message,
                )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._log_message(f"Failed to save project after render preparation error: {exc}")

        self.statusBar().showMessage("Render preparation failed")
        self._show_error("Render preparation failed", message)

    def _on_render_worker_started(self, task_name: str) -> None:
        self._log_message(f"Started background task: {task_name}")

    def _on_render_worker_progress(self, progress_value: int) -> None:
        self.statusBar().showMessage(f"Render progress: {progress_value}%")

    def _on_render_worker_finished(self, result: object, worker: TaskWorker) -> None:
        self._release_worker(worker)
        self._set_render_actions_enabled(True)

        if not isinstance(result, RenderWorkerResult):
            self.statusBar().showMessage("Render finished")
            return

        if self.current_project is not None:
            for page_result in result.page_results:
                if page_result.image_path is not None:
                    metadata_path = render_json_path(self.current_project, page_result.image_relative_path)
                    try:
                        render_data = load_render_json(metadata_path)
                    except Exception as exc:
                        self.current_project.update_stage_status(
                            page_result.image_relative_path,
                            "render",
                            status="failed",
                            error=str(exc),
                        )
                    else:
                        self._update_project_render_stage_status(
                            page_result.image_relative_path,
                            render_data,
                        )
                else:
                    self.current_project.update_stage_status(
                        page_result.image_relative_path,
                        "render",
                        status="failed",
                        error=page_result.error or "Unknown render failure.",
                    )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._show_error("Failed to save project", str(exc))

        current_index = self._current_page_index()
        if current_index is not None:
            self._load_cached_render_for_index(current_index, show_errors=False)
            self._refresh_preview_for_current_page()

        success_count = len(result.image_paths)
        failure_count = len(result.failures)
        if failure_count:
            self.statusBar().showMessage(
                f"Render finished with {success_count} success(es) and {failure_count} failure(s)."
            )
        else:
            self.statusBar().showMessage(f"Render finished for {success_count} page(s).")

    def _on_render_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        task: RenderTask,
    ) -> None:
        self._release_worker(worker)
        self._set_render_actions_enabled(True)

        if self.current_project is not None:
            for image_relative_path in task.image_relative_paths:
                self.current_project.update_stage_status(
                    image_relative_path,
                    "render",
                    status="failed",
                    error=message,
                )
            try:
                self.current_project.save()
            except Exception as exc:  # pragma: no cover - GUI error path.
                self._log_message(f"Failed to save project after render error: {exc}")

        self.statusBar().showMessage("Render failed")
        self._show_error("Render failed", message)

    def _on_llama_worker_started(self, task_name: str) -> None:
        self._log_message(f"Started background task: {task_name}")

    def _on_llama_worker_finished(
        self,
        result: object,
        worker: TaskWorker,
        action: str,
    ) -> None:
        self._release_worker(worker)
        self._set_server_actions_enabled(True)

        if not isinstance(result, LlamaServerTaskResult):
            self.statusBar().showMessage("llama.cpp server action finished")
            return

        self._set_server_status(result.state)
        self.statusBar().showMessage(result.message)
        self._log_message(result.message)

    def _on_llama_worker_failed(
        self,
        message: str,
        worker: TaskWorker,
        action: str,
    ) -> None:
        self._release_worker(worker)
        self._set_server_actions_enabled(True)
        if action == "stop":
            self._set_server_status(SERVER_STATE_ERROR)
        elif action == "check":
            self._set_server_status(SERVER_STATE_ERROR)
        else:
            self._set_server_status(SERVER_STATE_ERROR)
        self.statusBar().showMessage("llama.cpp server action failed")
        self._show_error("llama.cpp server error", message)

    def _set_detection_actions_enabled(self, enabled: bool) -> None:
        self.run_detection_selected_button.setEnabled(enabled)
        self.run_detection_all_button.setEnabled(enabled)
        self.reload_detection_button.setEnabled(enabled)

    def _set_ocr_actions_enabled(self, enabled: bool) -> None:
        self.prepare_ocr_selected_button.setEnabled(enabled)
        self.prepare_ocr_all_button.setEnabled(enabled)
        self.run_ocr_selected_page_button.setEnabled(enabled)
        self.run_ocr_all_button.setEnabled(enabled)
        self.run_ocr_selected_items_button.setEnabled(enabled)
        self.save_ocr_text_button.setEnabled(enabled)
        self.reload_ocr_button.setEnabled(enabled)
        self.force_reocr_checkbox.setEnabled(enabled)
        self.ocr_items_table.setEnabled(enabled)

    def _set_translation_actions_enabled(self, enabled: bool) -> None:
        self.initialize_translation_selected_button.setEnabled(enabled)
        self.initialize_translation_all_button.setEnabled(enabled)
        self.run_translation_selected_page_button.setEnabled(enabled)
        self.run_translation_all_button.setEnabled(enabled)
        self.run_translation_selected_items_button.setEnabled(enabled)
        self.reload_translation_button.setEnabled(enabled)
        self.save_translation_text_button.setEnabled(enabled)
        self.translation_force_checkbox.setEnabled(enabled)
        self.translation_items_table.setEnabled(enabled)

    def _set_inpaint_actions_enabled(self, enabled: bool) -> None:
        self.prepare_inpaint_mask_selected_button.setEnabled(enabled)
        self.prepare_inpaint_mask_all_button.setEnabled(enabled)
        self.run_inpaint_selected_button.setEnabled(enabled)
        self.run_inpaint_all_button.setEnabled(enabled)
        self.reload_inpaint_button.setEnabled(enabled)
        self.clear_inpaint_preview_button.setEnabled(enabled)
        self.load_lama_model_button.setEnabled(enabled)
        self.unload_lama_model_button.setEnabled(enabled)
        self.inpaint_mask_padding_input.setEnabled(enabled)
        self.inpaint_use_bubble_mask_checkbox.setEnabled(enabled)
        self.inpaint_use_crop_windows_checkbox.setEnabled(enabled)
        self.inpaint_force_checkbox.setEnabled(enabled)
        self.inpaint_device_input.setEnabled(enabled)
        self.inpaint_preview_mode_input.setEnabled(enabled)

    def _set_render_actions_enabled(self, enabled: bool) -> None:
        self.prepare_render_selected_button.setEnabled(enabled)
        self.prepare_render_all_button.setEnabled(enabled)
        self.run_render_selected_button.setEnabled(enabled)
        self.run_render_all_button.setEnabled(enabled)
        self.reload_render_button.setEnabled(enabled)
        self.clear_render_preview_button.setEnabled(enabled)
        self.render_font_name_input.setEnabled(enabled)
        self.render_font_path_input.setEnabled(enabled)
        self.render_min_font_size_input.setEnabled(enabled)
        self.render_max_font_size_input.setEnabled(enabled)
        self.render_stroke_enabled_checkbox.setEnabled(enabled)
        self.render_stroke_width_input.setEnabled(enabled)
        self.render_text_color_input.setEnabled(enabled)
        self.render_stroke_color_input.setEnabled(enabled)
        self.render_auto_color_checkbox.setEnabled(enabled)
        self.render_auto_direction_checkbox.setEnabled(enabled)
        self.render_vertical_cjk_checkbox.setEnabled(enabled)
        self.render_save_sprites_checkbox.setEnabled(enabled)
        self.render_force_checkbox.setEnabled(enabled)
        self.render_preview_mode_input.setEnabled(enabled)
        self.render_items_table.setEnabled(enabled)

    def _set_server_actions_enabled(self, enabled: bool) -> None:
        self.check_server_button.setEnabled(enabled)
        self.start_server_button.setEnabled(enabled)
        self.stop_server_button.setEnabled(enabled)

    def _set_server_status(self, status: str) -> None:
        self.server_status_value.setText(status)

    def _apply_server_inputs_to_manager(self) -> bool:
        try:
            self.llama_server_manager.update_config(
                server_url=self.server_url_input.text().strip(),
                model_path=self.server_model_path_input.text().strip(),
                mmproj_path=self.server_mmproj_path_input.text().strip(),
                llama_cpp_dir=self.server_llama_cpp_dir_input.text().strip(),
                gpu_layers=self.server_gpu_layers_input.value(),
                ctx_size=self.server_ctx_size_input.value(),
            )
        except Exception as exc:
            self._set_server_status(SERVER_STATE_ERROR)
            self._show_error("Invalid server settings", str(exc))
            return False
        return True

    def _translation_config_from_inputs(self) -> TranslationConfig:
        return TranslationConfig(
            source_language=self.translation_source_language_input.currentText().strip() or "ja",
            target_language=self.translation_target_language_input.currentText().strip() or "en",
            translator=self.translation_translator_input.currentText().strip() or "Google",
            style=self.translation_style_input.currentText().strip() or "Default",
            custom_prompt=self.translation_custom_prompt_input.toPlainText().strip(),
            batch_size_pages=self.translation_batch_size_input.value(),
            use_context_memory=self.translation_use_context_memory_checkbox.isChecked(),
            local_llm_server_url=self.translation_local_llm_server_url_input.text().strip() or "http://127.0.0.1:8080",
            local_llm_model=self.translation_local_llm_model_input.text().strip() or "gpt-4o",
            gemini_api_key=self.translation_gemini_api_key_input.text().strip(),
            deepseek_api_key=self.translation_deepseek_api_key_input.text().strip(),
            deepseek_model=self.translation_deepseek_model_input.text().strip() or "deepseek-v4-flash",
            deepseek_thinking=self.translation_deepseek_thinking_checkbox.isChecked(),
        )

    def _inpaint_device_value(self) -> str | None:
        device_value = self.inpaint_device_input.currentText().strip()
        if not device_value or device_value.lower() == "auto":
            return None
        return device_value

    def _render_config_from_inputs(self) -> dict[str, Any]:
        try:
            parsed_text_color = parse_color_value(self.render_text_color_input.text())
        except ValueError as exc:
            raise ValueError(f"Invalid text color value. {exc}") from exc

        try:
            parsed_stroke_color = parse_color_value(self.render_stroke_color_input.text())
        except ValueError as exc:
            raise ValueError(f"Invalid stroke color value. {exc}") from exc

        manual_font_path = self.render_font_path_input.text().strip()
        if manual_font_path:
            try:
                resolve_font_path(
                    self.workspace_root,
                    font_name=self.render_font_name_input.currentText().strip(),
                    font_path=manual_font_path,
                )
            except Exception as exc:
                raise ValueError(str(exc)) from exc

        stroke_width_value = self.render_stroke_width_input.value()
        return RenderConfig(
            font_name=self.render_font_name_input.currentText().strip(),
            font_path=manual_font_path,
            min_font_size=self.render_min_font_size_input.value(),
            max_font_size=max(
                self.render_min_font_size_input.value(),
                self.render_max_font_size_input.value(),
            ),
            stroke_enabled=self.render_stroke_enabled_checkbox.isChecked(),
            stroke_width=stroke_width_value if stroke_width_value > 0 else None,
            text_color=parsed_text_color,
            stroke_color=parsed_stroke_color,
            auto_color=self.render_auto_color_checkbox.isChecked(),
            auto_direction=self.render_auto_direction_checkbox.isChecked(),
            vertical_cjk=self.render_vertical_cjk_checkbox.isChecked(),
            save_sprites=self.render_save_sprites_checkbox.isChecked(),
            force=self.render_force_checkbox.isChecked(),
        ).to_metadata()

    def _selected_page_context(self, *, show_error: bool) -> tuple[int, str, Path] | None:
        if self.current_project is None:
            if show_error:
                self._show_error("No project open", "Create or open a project before working with this stage.")
            return None

        if self.current_project.page_count == 0:
            if show_error:
                self._show_error("No page selected", "Import images before using this stage.")
            return None

        index = self._current_page_index()
        if index is None:
            if show_error:
                self._show_error("No page selected", "Select a page from the page list first.")
            return None

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        image_path = self.current_project.image_path_for_index(index)
        if image_relative_path is None or image_path is None:
            if show_error:
                self._show_error("No page selected", "Select a valid page from the page list first.")
            return None

        if not image_path.exists():
            if show_error:
                self._show_error("Missing source image", f"Source image not found:\n{image_path}")
            return None

        return index, image_relative_path, image_path

    def _current_page_index(self) -> int | None:
        if self.current_project is None or self.current_project.page_count == 0:
            return None

        index = self.page_list.currentRow()
        if index < 0:
            index = self.current_project.data.current_page_index

        if index < 0 or index >= self.current_project.page_count:
            return None
        return index

    def _update_detection_stats(self, detection_data: dict[str, Any], cache_path: Path) -> None:
        self.detection_bubbles_value.setText(str(len(detection_data.get("bubbles", []))))
        self.detection_text_regions_value.setText(str(len(detection_data.get("text_regions", []))))
        self.detection_layout_regions_value.setText(str(len(detection_data.get("layout_regions", []))))
        self.detection_method_value.setText(str(detection_data.get("method") or "-"))
        self.detection_cache_path_value.setText(str(cache_path))

    def _reset_detection_stats(self, *, cache_path: Path | None = None) -> None:
        self.detection_bubbles_value.setText("0")
        self.detection_text_regions_value.setText("0")
        self.detection_layout_regions_value.setText("0")
        self.detection_method_value.setText("-")
        self.detection_cache_path_value.setText(str(cache_path) if cache_path is not None else "-")

    def _populate_ocr_items_table(self, items: list[dict[str, Any]]) -> None:
        self.current_ocr_items = list(items)
        self.ocr_items_table.blockSignals(True)
        self.ocr_items_table.setRowCount(len(self.current_ocr_items))

        for row_index, item in enumerate(self.current_ocr_items):
            row_values = [
                str(item.get("id", "")),
                str(item.get("kind", "")),
                self._format_bbox(item.get("bbox")),
                self._format_bbox(item.get("ocr_bbox")),
                str(item.get("status", "")),
                str(item.get("text", "")),
            ]
            error_text = str(item.get("error", "") or "").strip()

            for column_index, value in enumerate(row_values):
                table_item = QTableWidgetItem(value)
                if column_index == 0:
                    table_item.setData(Qt.ItemDataRole.UserRole, int(item.get("id", row_index)))
                if column_index != OCR_TEXT_COLUMN:
                    table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if error_text:
                    table_item.setToolTip(error_text)
                self.ocr_items_table.setItem(row_index, column_index, table_item)

        self.ocr_items_table.blockSignals(False)
        self._update_ocr_details(self.current_ocr_items, self.current_ocr_cache_path)

        if self.current_ocr_items:
            self.ocr_items_table.setCurrentCell(0, 0)
            self._update_ocr_crop_preview_for_row(0)
        else:
            self._update_ocr_crop_preview_for_row(None)

    def _clear_ocr_view(self) -> None:
        self.current_ocr_data = None
        self.current_ocr_items = []
        self.current_ocr_cache_path = None
        self.ocr_items_table.blockSignals(True)
        self.ocr_items_table.setRowCount(0)
        self.ocr_items_table.blockSignals(False)
        self._update_ocr_crop_preview_for_row(None)
        self._reset_ocr_details()

    def _on_ocr_item_selected(self) -> None:
        selected_row = self.ocr_items_table.currentRow()
        if selected_row < 0:
            self._update_ocr_crop_preview_for_row(None)
            return

        self._update_ocr_crop_preview_for_row(selected_row)

    def _update_ocr_crop_preview_for_row(self, row_index: int | None) -> None:
        if (
            row_index is None
            or row_index < 0
            or row_index >= len(self.current_ocr_items)
            or self.current_project is None
        ):
            self.ocr_crop_preview.clear_image()
            self.ocr_item_details_value.setText("Select an OCR item to preview its crop.")
            return

        item = self.current_ocr_items[row_index]
        crop_path = item.get("crop_path")
        item_id = int(item.get("id", row_index))
        status = str(item.get("status", "") or "-")
        error_text = str(item.get("error", "") or "").strip()
        details_text = f"Item {item_id} status: {status}"
        if error_text:
            details_text = f"{details_text}\nError: {error_text}"
        self.ocr_item_details_value.setText(details_text)
        if not crop_path:
            self.ocr_crop_preview.clear_image()
            return

        crop_file = self.current_project.root_dir / str(crop_path)
        if not crop_file.exists():
            self.ocr_crop_preview.clear_image()
            if error_text:
                self.ocr_item_details_value.setText(f"{details_text}\nCrop missing: {crop_file}")
            else:
                self.ocr_item_details_value.setText(f"{details_text}\nCrop missing: {crop_file}")
            return

        self.ocr_crop_preview.set_image(crop_file)

    def _sync_current_ocr_data_from_table(self) -> None:
        if self.current_ocr_data is None:
            raise RuntimeError("No OCR cache is loaded for the selected page.")

        items = self.current_ocr_data.get("items")
        if not isinstance(items, list):
            raise ValueError("The loaded OCR cache is invalid.")

        for row_index, item in enumerate(items):
            if row_index >= self.ocr_items_table.rowCount():
                break
            text_item = self.ocr_items_table.item(row_index, OCR_TEXT_COLUMN)
            status_item = self.ocr_items_table.item(row_index, 4)
            item["text"] = text_item.text() if text_item is not None else str(item.get("text", ""))
            item["status"] = status_item.text() if status_item is not None else str(item.get("status", "prepared"))
            item["updated_at"] = item.get("updated_at", "")

        self.current_ocr_items = list(items)
        self._update_ocr_details(self.current_ocr_items, self.current_ocr_cache_path)

    def _selected_ocr_item_ids(self, *, show_error: bool) -> list[int]:
        selection_model = self.ocr_items_table.selectionModel()
        if selection_model is None:
            if show_error:
                self._show_error("No OCR items selected", "Select one or more OCR rows first.")
            return []

        selected_indexes = selection_model.selectedRows()
        if not selected_indexes:
            if show_error:
                self._show_error("No OCR items selected", "Select one or more OCR rows first.")
            return []

        selected_ids: list[int] = []
        for model_index in selected_indexes:
            row_index = model_index.row()
            if row_index < 0 or row_index >= len(self.current_ocr_items):
                continue
            selected_ids.append(int(self.current_ocr_items[row_index].get("id", row_index)))

        if not selected_ids and show_error:
            self._show_error("No OCR items selected", "Select one or more OCR rows first.")
        return selected_ids

    def _update_ocr_details(self, items: list[dict[str, Any]], cache_path: Path | None) -> None:
        summary = summarize_ocr_items(items)
        self.ocr_total_items_value.setText(str(summary.get("total", 0)))
        self.ocr_prepared_items_value.setText(str(summary.get("prepared", 0)))
        self.ocr_done_items_value.setText(str(summary.get("done", 0)))
        self.ocr_error_items_value.setText(str(summary.get("error", 0)))
        self.ocr_cache_path_value.setText(str(cache_path) if cache_path is not None else "-")

    def _reset_ocr_details(self) -> None:
        self.ocr_total_items_value.setText("0")
        self.ocr_prepared_items_value.setText("0")
        self.ocr_done_items_value.setText("0")
        self.ocr_error_items_value.setText("0")
        self.ocr_cache_path_value.setText("-")
        self.ocr_item_details_value.setText("Select an OCR item to preview its crop.")

    def _ocr_stage_status_from_data(self, ocr_data: dict[str, Any]) -> str:
        summary = summarize_ocr_items(ocr_data.get("items", []))
        total_items = summary.get("total", 0)
        done_items = summary.get("done", 0)
        error_items = summary.get("error", 0)

        if total_items == 0:
            return "prepared"
        if done_items == total_items and error_items == 0:
            return "done"
        if done_items > 0 or error_items > 0:
            return "partial" if error_items > 0 or done_items < total_items else "done"
        return "prepared"

    def _update_project_ocr_stage_status(self, image_relative_path: str, ocr_data: dict[str, Any]) -> None:
        if self.current_project is None:
            return

        cache_path = ocr_json_path(self.current_project, image_relative_path)
        self.current_project.update_stage_status(
            image_relative_path,
            "ocr",
            status=self._ocr_stage_status_from_data(ocr_data),
            cache_path=self._relative_project_path(cache_path),
        )

    def _populate_translation_items_table(self, items: list[dict[str, Any]]) -> None:
        self.current_translation_items = list(items)
        self.translation_items_table.blockSignals(True)
        self.translation_items_table.setRowCount(len(self.current_translation_items))

        for row_index, item in enumerate(self.current_translation_items):
            row_values = [
                str(item.get("id", "")),
                str(item.get("ocr_item_id", "")),
                str(item.get("kind", "")),
                str(item.get("status", "")),
                str(item.get("source_text", "")),
                str(item.get("translated_text", "")),
            ]
            error_text = str(item.get("error", "") or "").strip()

            for column_index, value in enumerate(row_values):
                table_item = QTableWidgetItem(value)
                if column_index == 0:
                    table_item.setData(Qt.ItemDataRole.UserRole, int(item.get("id", row_index)))
                if column_index != TRANSLATION_TEXT_COLUMN:
                    table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if error_text:
                    table_item.setToolTip(error_text)
                self.translation_items_table.setItem(row_index, column_index, table_item)

        self.translation_items_table.blockSignals(False)
        self._update_translation_details(self.current_translation_items, self.current_translation_cache_path)

        if self.current_translation_items:
            self.translation_items_table.setCurrentCell(0, 0)
            self._update_translation_item_details_for_row(0)
        else:
            self._update_translation_item_details_for_row(None)

    def _clear_translation_view(self) -> None:
        self.current_translation_data = None
        self.current_translation_items = []
        self.current_translation_cache_path = None
        self.translation_items_table.blockSignals(True)
        self.translation_items_table.setRowCount(0)
        self.translation_items_table.blockSignals(False)
        self._reset_translation_details()

    def _on_translation_item_selected(self) -> None:
        selected_row = self.translation_items_table.currentRow()
        if selected_row < 0:
            self._update_translation_item_details_for_row(None)
            return

        self._update_translation_item_details_for_row(selected_row)

    def _update_translation_item_details_for_row(self, row_index: int | None) -> None:
        if row_index is None or row_index < 0 or row_index >= len(self.current_translation_items):
            self.translation_item_details_value.setText("Select a translation item to inspect it.")
            return

        item = self.current_translation_items[row_index]
        item_id = int(item.get("id", row_index))
        ocr_item_id = int(item.get("ocr_item_id", item_id))
        status = str(item.get("status", "") or "-")
        error_text = str(item.get("error", "") or "").strip()
        detail_lines = [
            f"Translation item {item_id}",
            f"OCR item: {ocr_item_id}",
            f"Status: {status}",
            f"BBox: {self._format_bbox(item.get('bbox'))}",
        ]
        if error_text:
            detail_lines.append(f"Error: {error_text}")
        self.translation_item_details_value.setText("\n".join(detail_lines))
        self._update_ocr_crop_preview_for_ocr_item_id(ocr_item_id)

    def _update_ocr_crop_preview_for_ocr_item_id(self, ocr_item_id: int) -> None:
        for row_index, item in enumerate(self.current_ocr_items):
            if int(item.get("id", row_index)) == int(ocr_item_id):
                self._update_ocr_crop_preview_for_row(row_index)
                return

    def _sync_current_translation_data_from_table(self) -> None:
        if self.current_translation_data is None:
            raise RuntimeError("No translation cache is loaded for the selected page.")

        items = self.current_translation_data.get("items")
        if not isinstance(items, list):
            raise ValueError("The loaded translation cache is invalid.")

        timestamp = datetime.now().replace(microsecond=0).isoformat()
        for row_index, item in enumerate(items):
            if row_index >= self.translation_items_table.rowCount():
                break

            translated_item = self.translation_items_table.item(row_index, TRANSLATION_TEXT_COLUMN)
            status_item = self.translation_items_table.item(row_index, 3)
            new_text = translated_item.text() if translated_item is not None else str(item.get("translated_text", ""))
            previous_text = str(item.get("translated_text", "") or "")
            current_status = status_item.text() if status_item is not None else str(item.get("status", "pending") or "pending")
            if new_text != previous_text:
                item["translated_text"] = new_text
                item["status"] = "manually_edited"
                item["error"] = ""
                item["updated_at"] = timestamp
                item["translator"] = "manual_edit"
            else:
                item["translated_text"] = new_text
                item["status"] = current_status
                item["updated_at"] = item.get("updated_at", "")

        self.current_translation_data["updated_at"] = timestamp
        self.current_translation_items = list(items)
        self._update_translation_details(self.current_translation_items, self.current_translation_cache_path)

    def _selected_translation_item_ids(self, *, show_error: bool) -> list[int]:
        selection_model = self.translation_items_table.selectionModel()
        if selection_model is None:
            if show_error:
                self._show_error("No translation items selected", "Select one or more translation rows first.")
            return []

        selected_indexes = selection_model.selectedRows()
        if not selected_indexes:
            if show_error:
                self._show_error("No translation items selected", "Select one or more translation rows first.")
            return []

        selected_ids: list[int] = []
        for model_index in selected_indexes:
            row_index = model_index.row()
            if row_index < 0 or row_index >= len(self.current_translation_items):
                continue
            selected_ids.append(int(self.current_translation_items[row_index].get("id", row_index)))

        if not selected_ids and show_error:
            self._show_error("No translation items selected", "Select one or more translation rows first.")
        return selected_ids

    def _update_translation_details(self, items: list[dict[str, Any]], cache_path: Path | None) -> None:
        summary = summarize_translation_json({"items": items})
        self.translation_total_items_value.setText(str(summary.get("total", 0)))
        self.translation_pending_items_value.setText(str(summary.get("pending", 0)))
        self.translation_done_items_value.setText(str(summary.get("done", 0)))
        self.translation_error_items_value.setText(str(summary.get("error", 0)))
        self.translation_cache_path_value.setText(str(cache_path) if cache_path is not None else "-")

    def _reset_translation_details(self) -> None:
        self.translation_total_items_value.setText("0")
        self.translation_pending_items_value.setText("0")
        self.translation_done_items_value.setText("0")
        self.translation_error_items_value.setText("0")
        self.translation_cache_path_value.setText("-")
        self.translation_item_details_value.setText("Select a translation item to inspect it.")

    def _translation_stage_status_from_data(self, translation_data: dict[str, Any]) -> str:
        summary = summarize_translation_json(translation_data)
        total_items = summary.get("total", 0)
        done_items = summary.get("done", 0)
        error_items = summary.get("error", 0)
        pending_items = summary.get("pending", 0)
        manual_items = summary.get("manually_edited", 0)
        skipped_items = summary.get("skipped", 0)

        if total_items == 0:
            return "initialized"
        if done_items + manual_items + skipped_items == total_items and error_items == 0 and pending_items == 0:
            return "done"
        if error_items == total_items:
            return "failed"
        if done_items > 0 or manual_items > 0 or error_items > 0:
            return "partial"
        return "initialized"

    def _update_project_translation_stage_status(
        self,
        image_relative_path: str,
        translation_data: dict[str, Any],
    ) -> None:
        if self.current_project is None:
            return

        cache_path = translation_json_path(self.current_project, image_relative_path)
        self.current_project.update_stage_status(
            image_relative_path,
            "translation",
            status=self._translation_stage_status_from_data(translation_data),
            cache_path=self._relative_project_path(cache_path),
        )

    def _update_inpaint_details(self, inpaint_data: dict[str, Any], cache_path: Path | None) -> None:
        summary = summarize_inpaint_json(inpaint_data)
        self.inpaint_source_path_value.setText(str(inpaint_data.get("source_image", "") or "-"))
        self.inpaint_ocr_cache_path_value.setText(str(inpaint_data.get("ocr_cache_path", "") or "-"))
        self.inpaint_text_mask_path_value.setText(str(inpaint_data.get("text_mask_path", "") or "-"))
        self.inpaint_bubble_mask_path_value.setText(str(inpaint_data.get("bubble_mask_path", "") or "-"))
        self.inpaint_output_path_value.setText(str(inpaint_data.get("output_image_path", "") or "-"))
        self.inpaint_item_count_value.setText(str(summary.get("item_count", 0)))
        self.inpaint_masked_pixels_value.setText(str(summary.get("masked_pixel_count", 0)))
        self.inpaint_status_value.setText(str(summary.get("status", "-") or "-"))
        self.inpaint_error_value.setText(str(summary.get("error", "") or "-"))

        model_status = get_lama_model_manager().status()
        if model_status.get("loaded"):
            self.lama_model_status_value.setText(f"Loaded ({model_status.get('device', 'auto') or 'auto'})")
        else:
            self.lama_model_status_value.setText("Not loaded")

    def _clear_inpaint_view(self) -> None:
        self.current_inpaint_data = None
        self.current_inpaint_cache_path = None
        self._reset_inpaint_details()

    def _reset_inpaint_details(
        self,
        *,
        source_image: str | None = None,
        output_image: Path | None = None,
    ) -> None:
        self.inpaint_source_path_value.setText(source_image or "-")
        self.inpaint_ocr_cache_path_value.setText("-")
        self.inpaint_text_mask_path_value.setText("-")
        self.inpaint_bubble_mask_path_value.setText("-")
        self.inpaint_output_path_value.setText(str(output_image) if output_image is not None else "-")
        self.inpaint_item_count_value.setText("0")
        self.inpaint_masked_pixels_value.setText("0")
        self.inpaint_status_value.setText("-")
        self.inpaint_error_value.setText("-")
        model_status = get_lama_model_manager().status()
        if model_status.get("loaded"):
            self.lama_model_status_value.setText(f"Loaded ({model_status.get('device', 'auto') or 'auto'})")
        else:
            self.lama_model_status_value.setText("Not loaded")

    def _inpaint_stage_status_from_data(self, inpaint_data: dict[str, Any]) -> str:
        status = str(inpaint_data.get("status", "pending") or "pending").strip().lower()
        if status == "done":
            return "done"
        if status == "error":
            return "failed"
        if status == "running":
            return "running"
        return "prepared"

    def _update_project_inpaint_stage_status(
        self,
        image_relative_path: str,
        inpaint_data: dict[str, Any],
    ) -> None:
        if self.current_project is None:
            return

        cache_path = inpaint_json_path(self.current_project, image_relative_path)
        self.current_project.update_stage_status(
            image_relative_path,
            "inpaint",
            status=self._inpaint_stage_status_from_data(inpaint_data),
            cache_path=self._relative_project_path(cache_path),
        )

    def _populate_render_items_table(self, items: list[dict[str, Any]]) -> None:
        self.current_render_items = list(items)
        self.render_items_table.blockSignals(True)
        self.render_items_table.setRowCount(len(self.current_render_items))

        for row_index, item in enumerate(self.current_render_items):
            row_values = [
                str(item.get("id", "")),
                str(item.get("kind", "")),
                str(item.get("writing_mode", "")),
                str(item.get("font_size", "")),
                str(item.get("status", "")),
                str(item.get("translated_text", "")),
                self._format_bbox(item.get("render_bbox")),
                str(item.get("sprite_path", "")),
            ]
            error_text = str(item.get("error", "") or "").strip()

            for column_index, value in enumerate(row_values):
                table_item = QTableWidgetItem(value)
                if column_index == 0:
                    table_item.setData(Qt.ItemDataRole.UserRole, int(item.get("id", row_index)))
                if error_text:
                    table_item.setToolTip(error_text)
                self.render_items_table.setItem(row_index, column_index, table_item)

        self.render_items_table.blockSignals(False)
        self._update_render_details(self.current_render_items, self.current_render_cache_path)

        if self.current_render_items:
            self.render_items_table.setCurrentCell(0, 0)
            self._update_render_item_details_for_row(0)
        else:
            self._update_render_item_details_for_row(None)

    def _clear_render_view(self) -> None:
        self.current_render_data = None
        self.current_render_items = []
        self.current_render_cache_path = None
        self.render_items_table.blockSignals(True)
        self.render_items_table.setRowCount(0)
        self.render_items_table.blockSignals(False)
        self._reset_render_details()

    def _on_render_item_selected(self) -> None:
        selected_row = self.render_items_table.currentRow()
        if selected_row < 0:
            self._update_render_item_details_for_row(None)
            return

        self._update_render_item_details_for_row(selected_row)

    def _update_render_item_details_for_row(self, row_index: int | None) -> None:
        if row_index is None or row_index < 0 or row_index >= len(self.current_render_items):
            self.render_item_details_value.setText("Select a render item to inspect it.")
            return

        item = self.current_render_items[row_index]
        item_id = int(item.get("id", row_index))
        ocr_item_id = int(item.get("ocr_item_id", item_id))
        status = str(item.get("status", "") or "-")
        error_text = str(item.get("error", "") or "").strip()
        detail_lines = [
            f"Render item {item_id}",
            f"OCR item: {ocr_item_id}",
            f"Status: {status}",
            f"Writing mode: {str(item.get('writing_mode', '') or '-')}",
            f"Render box: {self._format_bbox(item.get('render_bbox'))}",
        ]
        if error_text:
            detail_lines.append(f"Error: {error_text}")
        sprite_path = str(item.get("sprite_path", "") or "").strip()
        if sprite_path:
            detail_lines.append(f"Sprite: {sprite_path}")
        self.render_item_details_value.setText("\n".join(detail_lines))
        self._update_ocr_crop_preview_for_ocr_item_id(ocr_item_id)

    def _update_render_details(self, items: list[dict[str, Any]], cache_path: Path | None) -> None:
        summary = summarize_render_json({"items": items, "status": self.current_render_data.get("status", "pending") if isinstance(self.current_render_data, dict) else "pending"})
        self.render_rendered_item_count_value.setText(str(summary.get("rendered", 0)))
        self.render_skipped_item_count_value.setText(str(summary.get("skipped", 0)))
        self.render_status_value.setText(str(summary.get("status", "-") or "-"))
        self.render_error_value.setText(
            str(self.current_render_data.get("error", "") or "-") if isinstance(self.current_render_data, dict) else "-"
        )
        self.render_translation_cache_path_value.setText(
            str(self.current_render_data.get("translation_cache_path", "") or "-")
            if isinstance(self.current_render_data, dict)
            else "-"
        )
        self.render_inpaint_image_path_value.setText(
            str(self.current_render_data.get("inpaint_image_path", "") or "-")
            if isinstance(self.current_render_data, dict)
            else "-"
        )
        self.render_output_path_value.setText(
            str(self.current_render_data.get("output_image_path", "") or "-")
            if isinstance(self.current_render_data, dict)
            else "-"
        )

    def _reset_render_details(self, *, output_image: Path | None = None) -> None:
        self.render_translation_cache_path_value.setText("-")
        self.render_inpaint_image_path_value.setText("-")
        self.render_output_path_value.setText(str(output_image) if output_image is not None else "-")
        self.render_rendered_item_count_value.setText("0")
        self.render_skipped_item_count_value.setText("0")
        self.render_status_value.setText("-")
        self.render_error_value.setText("-")
        self.render_item_details_value.setText("Select a render item to inspect it.")

    def _render_stage_status_from_data(self, render_data: dict[str, Any]) -> str:
        summary = summarize_render_json(render_data)
        total_items = int(summary.get("total", 0) or 0)
        rendered_items = int(summary.get("rendered", 0) or 0)
        error_items = int(summary.get("error", 0) or 0)
        status = str(render_data.get("status", "pending") or "pending").lower()

        if status == "done" and rendered_items > 0:
            return "done"
        if status == "error" and rendered_items == 0:
            return "failed"
        if rendered_items > 0 and rendered_items < max(total_items, 1):
            return "partial"
        if rendered_items > 0:
            return "done"
        if error_items > 0:
            return "failed"
        return "prepared"

    def _update_project_render_stage_status(
        self,
        image_relative_path: str,
        render_data: dict[str, Any],
    ) -> None:
        if self.current_project is None:
            return

        cache_path = render_json_path(self.current_project, image_relative_path)
        self.current_project.update_stage_status(
            image_relative_path,
            "render",
            status=self._render_stage_status_from_data(render_data),
            cache_path=self._relative_project_path(cache_path),
        )

    def _refresh_preview_for_current_page(self) -> bool:
        if self.current_project is None:
            self.image_preview.clear_image()
            return False

        index = self._current_page_index()
        if index is None:
            self.image_preview.clear_image()
            return False

        image_relative_path = self.current_project.page_relative_path_for_index(index)
        page_path = self.current_project.image_path_for_index(index)
        if image_relative_path is None or page_path is None or not page_path.exists():
            self.image_preview.clear_image()
            return False

        preview_image_path = page_path
        mask_overlay_path: Path | None = None

        if self.stage_tabs.currentIndex() == self.render_tab_index:
            preview_mode = self.render_preview_mode_input.currentText().strip() or RENDER_PREVIEW_SOURCE
            if preview_mode == RENDER_PREVIEW_INPAINT:
                cached_inpaint_path = inpaint_image_path(self.current_project, image_relative_path)
                if cached_inpaint_path.exists():
                    preview_image_path = cached_inpaint_path
            elif preview_mode == RENDER_PREVIEW_RESULT:
                cached_render_path = render_image_path(self.current_project, image_relative_path)
                if cached_render_path.exists():
                    preview_image_path = cached_render_path
        elif self.stage_tabs.currentIndex() == self.inpaint_tab_index:
            preview_mode = self.inpaint_preview_mode_input.currentText().strip() or INPAINT_PREVIEW_SOURCE
            if preview_mode == INPAINT_PREVIEW_RESULT:
                cached_result_path = inpaint_image_path(self.current_project, image_relative_path)
                if cached_result_path.exists():
                    preview_image_path = cached_result_path
            elif preview_mode == INPAINT_PREVIEW_MASK:
                cached_mask_path = inpaint_preview_mask_path(self.current_project, image_relative_path)
                if cached_mask_path.exists():
                    mask_overlay_path = cached_mask_path

        if not self.image_preview.set_image(preview_image_path):
            return False

        if mask_overlay_path is not None and mask_overlay_path.exists():
            self.image_preview.set_mask_overlay(mask_overlay_path)
        else:
            self.image_preview.clear_mask_overlay()

        if self.current_detection_data is not None:
            self.image_preview.set_detection_overlay(self.current_detection_data)

        return True

    def _format_bbox(self, bbox: Any) -> str:
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return "-"
        return f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"

    def _relative_project_path(self, path: Path) -> str:
        if self.current_project is None:
            return str(path)

        try:
            return path.resolve().relative_to(self.current_project.root_dir).as_posix()
        except ValueError:
            return str(path)

    def _release_worker(self, worker: TaskWorker) -> None:
        if worker in self._active_workers:
            self._active_workers.remove(worker)

    def _update_window_title(self) -> None:
        if self.current_project is None:
            self.setWindowTitle(APP_NAME)
            return

        self.setWindowTitle(f"{APP_NAME} - {self.current_project.data.name}")

    def _log_message(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message}")

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
        self._log_message(f"{title}: {message}")
