from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any

from PySide6.QtCore import QSize, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QStandardItem
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.model_fetcher import (
    AuthError,
    EmptyResultError,
    FetchError,
    GeminiModelFetcher,
    ModelEntry,
    ModelFetcher,
    NetworkError,
    OpenAICompatibleModelFetcher,
    RateLimitError,
    ServerError,
)
from core.model_profiles import (
    ALLOWED_PROVIDERS,
    generate_profile_id,
    normalize_api_key_list,
    normalize_profiles_payload,
    sanitize_profile_id,
)
from core.text_utils import build_tool_ui_labels
from ui.theme import TEXT_MUTED, TEXT_PRIMARY
from .foundation import (
    CollapsibleSection,
    CopySafePlainTextEdit,
    _fa_icon,
    _make_mono_font,
    _sync_plain_text_height,
    format_approval_detail_text,
)


class ModelLoadState(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    FALLBACK = "fallback"
    ERROR = "error"


def _fetch_error_message(error: FetchError) -> str:
    if isinstance(error, AuthError):
        return "Неверный API Key. Проверьте ключ."
    if isinstance(error, RateLimitError):
        return "Превышен лимит запросов. Подождите."
    if isinstance(error, ServerError):
        return "Ошибка сервера. Попробуйте позже."
    if isinstance(error, NetworkError):
        return "Нет соединения. Проверьте сеть."
    if isinstance(error, EmptyResultError):
        return "Нет доступных моделей для этого ключа."
    return "Не удалось загрузить модели."


class ModelFetchWorker(QThread):
    fetched = Signal(int, object)
    failed = Signal(int, str)

    def __init__(
        self,
        request_id: int,
        fetcher: ModelFetcher,
        api_key: str,
        base_url: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(None)
        self._request_id = int(request_id)
        self._fetcher = fetcher
        self._api_key = str(api_key or "").strip()
        self._base_url = str(base_url or "").strip()

    def run(self) -> None:
        try:
            result = asyncio.run(self._fetcher.fetch(self._api_key, self._base_url))
        except FetchError as error:
            self.failed.emit(self._request_id, _fetch_error_message(error))
            return
        except Exception:
            self.failed.emit(self._fetch_request_id, "Не удалось загрузить модели.")
            return
        self.fetched.emit(self._request_id, result)


class ModelSettingsDialog(QDialog):
    profiles_saved = Signal(object)

    def __init__(self, payload: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ModelSettingsDialog")
        self.setWindowTitle("Model Profiles")
        self.setModal(False)
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1050, 670)
        self.setMinimumSize(760, 420)

        normalized = normalize_profiles_payload(payload or {})
        self._profiles: list[dict[str, Any]] = [dict(item) for item in normalized.get("profiles", [])]
        self._active_profile = str(normalized.get("active_profile") or "").strip()
        self._name_manual_flags: list[bool] = []
        self._selected_row = -1
        self._loading_form = False
        self._filter_text = ""
        self._result_payload = normalized
        self._form_enabled = False
        self._model_state = ModelLoadState.IDLE
        self._model_cache: dict[tuple[str, ...], list[ModelEntry]] = {}
        self._model_entries_by_id: dict[str, ModelEntry] = {}
        self._model_workers: list[ModelFetchWorker] = []
        self._fetch_request_id = 0
        self._fetch_debounce = QTimer(self)
        self._fetch_debounce.setSingleShot(True)
        self._fetch_debounce.setInterval(600)
        self._fetch_debounce.timeout.connect(self._start_fetch)
        self._name_manual_flags = self._compute_initial_name_manual_flags()

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        hero_card = QFrame()
        hero_card.setObjectName("ModelSettingsHeroCard")
        hero_layout = QHBoxLayout(hero_card)
        hero_layout.setContentsMargins(10, 8, 10, 8)
        hero_layout.setSpacing(10)

        hero_copy = QVBoxLayout()
        hero_copy.setContentsMargins(0, 0, 0, 0)
        hero_copy.setSpacing(3)

        header_title = QLabel("Model Profiles")
        header_title.setObjectName("ModelSettingsTitle")
        hero_copy.addWidget(header_title, 0, Qt.AlignLeft | Qt.AlignTop)

        header_hint = QLabel(
            "Keep providers tidy, switch the active profile for new runs, and tune image support without losing your place."
        )
        header_hint.setObjectName("ModelSettingsSubtitle")
        header_hint.setWordWrap(True)
        hero_copy.addWidget(header_hint)

        active_name = str(self._active_profile or "").strip() or "none"
        self.active_profile_label = QLabel(f"Active now: {active_name}")
        self.active_profile_label.setObjectName("ModelSettingsMeta")
        self.active_profile_label.setWordWrap(True)
        hero_copy.addWidget(self.active_profile_label)
        hero_layout.addLayout(hero_copy, 1)

        hero_stats = QVBoxLayout()
        hero_stats.setContentsMargins(0, 0, 0, 0)
        hero_stats.setSpacing(4)

        self.profile_count_chip = QLabel("")
        self.profile_count_chip.setObjectName("ModelSettingsChip")
        hero_stats.addWidget(self.profile_count_chip, 0, Qt.AlignRight)

        self.left_meta_chip = QLabel("")
        self.left_meta_chip.setObjectName("ModelSettingsChip")
        hero_stats.addWidget(self.left_meta_chip, 0, Qt.AlignRight)
        hero_stats.addStretch(1)
        hero_layout.addLayout(hero_stats, 0)
        root.addWidget(hero_card)

        self.body_splitter = QSplitter(Qt.Horizontal)
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.setHandleWidth(8)

        left_container = QFrame()
        left_container.setObjectName("ModelSettingsPane")
        left = QVBoxLayout(left_container)
        left.setContentsMargins(8, 8, 8, 8)
        left.setSpacing(6)

        left_header = QHBoxLayout()
        left_header.setContentsMargins(0, 0, 0, 0)
        left_header.setSpacing(4)
        left_label = QLabel("Library")
        left_label.setObjectName("ModelSettingsSectionTitle")
        left_header.addWidget(left_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
        left_header.addStretch(1)
        left.addLayout(left_header)

        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("ModelSettingsSearchField")
        self.search_edit.setPlaceholderText("Search by name, provider, or model")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setAccessibleName("Profile search")
        self.search_edit.setAccessibleDescription("Filter profiles by name, provider, or model")
        self.search_edit.addAction(_fa_icon("fa5s.search", color=TEXT_MUTED, size=10), QLineEdit.LeadingPosition)
        left.addWidget(self.search_edit)

        self.profile_list = QListWidget()
        self.profile_list.setObjectName("ModelProfileList")
        self.profile_list.setMinimumWidth(280)
        self.profile_list.setAccessibleName("Profile list")
        self.profile_list.setAccessibleDescription("Select a model profile to edit")
        self.profile_list.currentRowChanged.connect(self._on_selection_changed)
        left.addWidget(self.profile_list, 1)

        left_buttons = QHBoxLayout()
        left_buttons.setSpacing(4)
        self.add_button = QPushButton("New Profile")
        self.add_button.setObjectName("SettingsAddButton")
        self.add_button.setIcon(_fa_icon("fa5s.plus", color=TEXT_PRIMARY, size=10))
        self.delete_button = QPushButton("Remove")
        self.delete_button.setObjectName("SettingsDeleteButton")
        self.delete_button.setIcon(_fa_icon("fa5s.trash", color="#F08F8F", size=10))
        left_buttons.addWidget(self.add_button)
        left_buttons.addWidget(self.delete_button)
        left.addLayout(left_buttons)

        left_hint = QLabel("Tip: leave Name empty and it will be generated from the model automatically.")
        left_hint.setObjectName("ModelSettingsMeta")
        left_hint.setWordWrap(True)
        left.addWidget(left_hint)

        right_container = QFrame()
        right_container.setObjectName("ModelSettingsPane")
        right = QVBoxLayout(right_container)
        right.setContentsMargins(8, 8, 8, 8)
        right.setSpacing(6)

        right_header = QHBoxLayout()
        right_header.setContentsMargins(0, 0, 0, 0)
        right_header.setSpacing(4)
        right_label = QLabel("Editor")
        right_label.setObjectName("ModelSettingsSectionTitle")
        right_header.addWidget(right_label, 0, Qt.AlignLeft | Qt.AlignVCenter)

        self.profile_state_chip = QLabel("No profile selected")
        self.profile_state_chip.setObjectName("ModelSettingsChip")
        right_header.addWidget(self.profile_state_chip, 0, Qt.AlignLeft | Qt.AlignVCenter)
        right_header.addStretch(1)

        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.setObjectName("ModelSettingsInlineButton")
        self.duplicate_button.setIcon(_fa_icon("fa5s.copy", color=TEXT_PRIMARY, size=10))
        self.duplicate_button.setEnabled(False)
        right_header.addWidget(self.duplicate_button, 0, Qt.AlignRight | Qt.AlignVCenter)
        right.addLayout(right_header)

        self.form_hint = QLabel("Select a profile to review credentials, key rotation, model settings, and image support.")
        self.form_hint.setObjectName("ModelSettingsMeta")
        self.form_hint.setWordWrap(True)
        right.addWidget(self.form_hint)

        self.save_state_label = QLabel("")
        self.save_state_label.setObjectName("ModelSettingsMeta")
        self.save_state_label.setWordWrap(True)
        self.save_state_label.setVisible(False)
        right.addWidget(self.save_state_label)

        self.summary_card = QFrame()
        self.summary_card.setObjectName("ModelSettingsSummaryCard")
        summary_layout = QHBoxLayout(self.summary_card)
        summary_layout.setContentsMargins(6, 4, 6, 4)
        summary_layout.setSpacing(4)

        self.summary_provider = QLabel("Provider: —")
        self.summary_provider.setObjectName("ModelSettingsSummaryLabel")
        self.summary_model = QLabel("Model: —")
        self.summary_model.setObjectName("ModelSettingsSummaryLabel")
        self.summary_images = QLabel("Image input: off")
        self.summary_images.setObjectName("ModelSettingsSummaryLabel")
        summary_layout.addWidget(self.summary_provider)
        summary_layout.addWidget(self.summary_model, 1)
        summary_layout.addWidget(self.summary_images)
        right.addWidget(self.summary_card)

        editor_scroll = QScrollArea()
        editor_scroll.setObjectName("ModelSettingsScrollArea")
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setFrameShape(QFrame.NoFrame)

        editor_content = QWidget()
        editor_content.setObjectName("ModelSettingsEditorContent")
        editor_layout = QVBoxLayout(editor_content)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(6)

        form_frame = QFrame()
        form_frame.setObjectName("ModelSettingsFormCard")
        form_layout = QFormLayout(form_frame)
        form_layout.setContentsMargins(6, 6, 6, 6)
        form_layout.setHorizontalSpacing(6)
        form_layout.setVerticalSpacing(5)
        form_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        form_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form_layout.setFormAlignment(Qt.AlignTop)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("profile-id")
        self.name_edit.setClearButtonEnabled(True)
        self.name_edit.setAccessibleName("Profile name")

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["openai", "gemini"])
        self.provider_combo.setAccessibleName("Provider")

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("ModelSettingsModelCombo")
        self.model_combo.setAccessibleName("Model")
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.model_combo.setEditable(False)
        self.model_combo.setMaxVisibleItems(14)

        self.model_text_edit = QLineEdit()
        self.model_text_edit.setPlaceholderText("Введите название модели вручную")
        self.model_text_edit.setClearButtonEnabled(True)
        self.model_text_edit.setAccessibleName("Model")
        self.model_edit = self.model_text_edit

        self.model_reload_button = QToolButton()
        self.model_reload_button.setObjectName("ModelSettingsInlineToolButton")
        self.model_reload_button.setIcon(_fa_icon("fa5s.redo-alt", color=TEXT_MUTED, size=10))
        self.model_reload_button.setToolTip("Повторить загрузку")
        self.model_reload_button.setAccessibleName("Retry model loading")
        self.model_reload_button.setCursor(Qt.PointingHandCursor)

        self.model_popup_button = QToolButton()
        self.model_popup_button.setObjectName("ModelSettingsInlineToolButton")
        self.model_popup_button.setIcon(_fa_icon("fa5s.caret-down", color=TEXT_MUTED, size=10))
        self.model_popup_button.setToolTip("Показать список моделей")
        self.model_popup_button.setAccessibleName("Show model list")
        self.model_popup_button.setCursor(Qt.PointingHandCursor)

        self.model_loading_label = QLabel("")
        self.model_loading_label.setObjectName("ModelSettingsHintText")
        self.model_loading_label.setWordWrap(True)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("API key")
        self.api_key_edit.setAccessibleName("API key")

        self.api_key_reveal_button = QToolButton()
        self.api_key_reveal_button.setObjectName("ModelSettingsInlineToolButton")
        self.api_key_reveal_button.setIcon(_fa_icon("fa5s.eye", color=TEXT_MUTED, size=10))
        self.api_key_reveal_button.setToolTip("Show or hide API key")
        self.api_key_reveal_button.setAccessibleName("Toggle API key visibility")

        self.api_key_copy_button = QToolButton()
        self.api_key_copy_button.setObjectName("ModelSettingsInlineToolButton")
        self.api_key_copy_button.setIcon(_fa_icon("fa5s.copy", color=TEXT_MUTED, size=10))
        self.api_key_copy_button.setToolTip("Copy API key")
        self.api_key_copy_button.setAccessibleName("Copy API key")

        self.api_key_rotation_button = QToolButton()
        self.api_key_rotation_button.setObjectName("ModelSettingsInlineToolButton")
        self.api_key_rotation_button.setIcon(_fa_icon("fa5s.sync-alt", color=TEXT_MUTED, size=10))
        self.api_key_rotation_button.setToolTip("Configure API key rotation")
        self.api_key_rotation_button.setAccessibleName("Configure API key rotation")

        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("https://api.openai.com/v1")
        self.base_url_edit.setClearButtonEnabled(True)
        self.base_url_edit.setAccessibleName("Base URL")

        self.supports_images_checkbox = QCheckBox("Image input support")
        self.supports_images_checkbox.setObjectName("ModelSupportsImagesCheckbox")
        self.supports_images_checkbox.setToolTip("Allow image attachments for this profile.")
        self.supports_images_checkbox.setAccessibleName("Image input support")

        self.images_hint_label = QLabel("Enables sending images with prompts when the selected model supports vision input.")
        self.images_hint_label.setObjectName("ModelSettingsHintText")
        self.images_hint_label.setWordWrap(True)

        api_key_row = QWidget()
        api_key_row.setObjectName("ModelSettingsFieldRow")
        api_key_layout = QHBoxLayout(api_key_row)
        api_key_layout.setContentsMargins(0, 0, 0, 0)
        api_key_layout.setSpacing(6)
        api_key_layout.addWidget(self.api_key_edit, 1)
        api_key_layout.addWidget(self.api_key_reveal_button, 0, Qt.AlignVCenter)
        api_key_layout.addWidget(self.api_key_copy_button, 0, Qt.AlignVCenter)
        api_key_layout.addWidget(self.api_key_rotation_button, 0, Qt.AlignVCenter)

        model_row = QWidget()
        model_row.setObjectName("ModelSettingsFieldRow")
        model_row_layout = QHBoxLayout(model_row)
        model_row_layout.setContentsMargins(0, 0, 0, 0)
        model_row_layout.setSpacing(6)
        model_row_layout.addWidget(self.model_combo, 1)
        model_row_layout.addWidget(self.model_text_edit, 1)
        model_row_layout.addWidget(self.model_popup_button, 0, Qt.AlignVCenter)
        model_row_layout.addWidget(self.model_reload_button, 0, Qt.AlignVCenter)

        model_field = QWidget()
        model_field_layout = QVBoxLayout(model_field)
        model_field_layout.setContentsMargins(0, 0, 0, 0)
        model_field_layout.setSpacing(4)
        model_field_layout.addWidget(model_row)
        model_field_layout.addWidget(self.model_loading_label)

        images_row = QWidget()
        images_layout = QVBoxLayout(images_row)
        images_layout.setContentsMargins(0, 2, 0, 2)
        images_layout.setSpacing(4)
        images_layout.addWidget(self.supports_images_checkbox)
        images_layout.addWidget(self.images_hint_label)

        label_width = 68
        name_label = QLabel("&Name")
        provider_label = QLabel("&Provider")
        model_label = QLabel("&Model")
        api_key_label = QLabel("&API Key")
        base_url_label = QLabel("Base &URL")
        images_label = QLabel("I&mages")
        for label in (name_label, provider_label, model_label, api_key_label, base_url_label, images_label):
            label.setObjectName("ModelSettingsFieldLabel")
            label.setFixedWidth(label_width)

        name_label.setBuddy(self.name_edit)
        provider_label.setBuddy(self.provider_combo)
        model_label.setBuddy(self.model_text_edit)
        api_key_label.setBuddy(self.api_key_edit)
        base_url_label.setBuddy(self.base_url_edit)
        images_label.setBuddy(self.supports_images_checkbox)
        self.model_field_label = model_label

        form_layout.addRow(name_label, self.name_edit)
        form_layout.addRow(provider_label, self.provider_combo)
        form_layout.addRow(model_label, model_field)
        form_layout.addRow(api_key_label, api_key_row)
        form_layout.addRow(base_url_label, self.base_url_edit)
        form_layout.addRow(images_label, images_row)
        editor_layout.addWidget(form_frame)

        rotation_content = QWidget()
        rotation_content_layout = QVBoxLayout(rotation_content)
        rotation_content_layout.setContentsMargins(0, 0, 0, 0)
        rotation_content_layout.setSpacing(6)

        rotation_card = QFrame()
        rotation_card.setObjectName("ModelSettingsFormCard")
        rotation_card_layout = QVBoxLayout(rotation_card)
        rotation_card_layout.setContentsMargins(8, 8, 8, 8)
        rotation_card_layout.setSpacing(6)

        rotation_meta_row = QHBoxLayout()
        rotation_meta_row.setContentsMargins(0, 0, 0, 0)
        rotation_meta_row.setSpacing(6)

        rotation_helper = QLabel("Add one API key per line. The active key field stays in sync with this pool.")
        rotation_helper.setObjectName("ModelSettingsHintText")
        rotation_helper.setWordWrap(True)
        rotation_meta_row.addWidget(rotation_helper, 1)

        self.api_key_rotation_count_chip = QLabel("")
        self.api_key_rotation_count_chip.setObjectName("ModelSettingsChip")
        rotation_meta_row.addWidget(self.api_key_rotation_count_chip, 0, Qt.AlignRight | Qt.AlignTop)
        rotation_card_layout.addLayout(rotation_meta_row)

        self.api_key_rotation_editor = CopySafePlainTextEdit()
        self.api_key_rotation_editor.setPlaceholderText("sk-key-1\nsk-key-2\ngm-key-3")
        self.api_key_rotation_editor.setObjectName("InlineCodeView")
        self.api_key_rotation_editor.setFont(_make_mono_font(10))
        self.api_key_rotation_editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.api_key_rotation_editor.setMinimumHeight(120)
        rotation_card_layout.addWidget(self.api_key_rotation_editor, 1)

        self.api_key_rotation_status_label = QLabel("")
        self.api_key_rotation_status_label.setObjectName("ModelSettingsMeta")
        self.api_key_rotation_status_label.setWordWrap(True)
        rotation_card_layout.addWidget(self.api_key_rotation_status_label)

        rotation_content_layout.addWidget(rotation_card)
        self.api_key_rotation_section = CollapsibleSection(
            "API key rotation",
            rotation_content,
            expanded=True,
            content_margins=(0, 0, 0, 0),
        )
        editor_layout.addWidget(self.api_key_rotation_section)

        helper_card = QFrame()
        helper_card.setObjectName("ModelSettingsHelperCard")
        helper_layout = QVBoxLayout(helper_card)
        helper_layout.setContentsMargins(8, 6, 8, 6)
        helper_layout.setSpacing(3)

        helper_title = QLabel("Editing notes")
        helper_title.setObjectName("ModelSettingsHelperTitle")
        helper_body = QLabel(
            "Only enabled profiles can become active. Toggling a model keeps your current place in the list and preserves the editor state."
        )
        helper_body.setObjectName("ModelSettingsHintText")
        helper_body.setWordWrap(True)
        helper_layout.addWidget(helper_title)
        helper_layout.addWidget(helper_body)
        editor_layout.addWidget(helper_card)
        editor_layout.addStretch(1)

        editor_scroll.setWidget(editor_content)
        right.addWidget(editor_scroll, 1)

        left_container.setMinimumWidth(260)
        right_container.setMinimumWidth(360)
        self.body_splitter.addWidget(left_container)
        self.body_splitter.addWidget(right_container)
        self.body_splitter.setSizes([270, 500])
        root.addWidget(self.body_splitter, 1)

        actions = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        actions.setObjectName("ModelSettingsActions")
        self.save_button = actions.button(QDialogButtonBox.StandardButton.Save)
        if self.save_button is not None:
            self.save_button.setObjectName("PrimaryButton")
            self.save_button.setIcon(_fa_icon("fa5s.save", color="#FFFFFF", size=11))
            self.save_button.setMinimumHeight(24)
        self.close_button = actions.button(QDialogButtonBox.StandardButton.Close)
        if self.close_button is not None:
            self.close_button.setMinimumHeight(24)
        root.addWidget(actions)

        self.add_button.clicked.connect(self._add_profile)
        self.delete_button.clicked.connect(self._delete_selected_profile)
        self.duplicate_button.clicked.connect(self._duplicate_selected_profile)
        self.search_edit.textChanged.connect(self._apply_profile_filter)
        if self.save_button is not None:
            self.save_button.clicked.connect(self._save_and_accept)
        if self.close_button is not None:
            self.close_button.clicked.connect(self.reject)

        self.name_edit.textEdited.connect(self._on_name_edited)
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.model_text_edit.textChanged.connect(self._on_model_changed)
        self.api_key_edit.textChanged.connect(self._on_api_key_changed)
        self.api_key_reveal_button.clicked.connect(self._toggle_api_key_visibility)
        self.api_key_copy_button.clicked.connect(self._copy_api_key)
        self.api_key_rotation_button.clicked.connect(self._edit_api_key_rotation)
        self.api_key_rotation_editor.textChanged.connect(self._on_api_key_rotation_text_changed)
        self.base_url_edit.textChanged.connect(self._on_base_url_changed)
        self.model_popup_button.clicked.connect(self._show_model_popup)
        self.model_reload_button.clicked.connect(self._reload_models)
        self.supports_images_checkbox.checkStateChanged.connect(self._on_form_changed)

        self._set_model_state(ModelLoadState.IDLE)
        self._update_api_key_rotation_summary(api_keys=[])
        self._refresh_profile_list()
        if self.profile_list.count() > 0:
            self.profile_list.setCurrentRow(self._preferred_row_for_open())
        else:
            self._set_form_enabled(False)
        self._refresh_profile_counts()

    def result_payload(self) -> dict[str, Any]:
        return dict(self._result_payload)

    def closeEvent(self, event) -> None:
        self._fetch_debounce.stop()
        self._fetch_request_id += 1
        for worker in list(self._model_workers):
            if worker.isRunning():
                worker.wait(3000)
            if worker in self._model_workers:
                self._model_workers.remove(worker)
            worker.deleteLater()
        super().closeEvent(event)

    def _current_row(self) -> int:
        return self.profile_list.currentRow()

    def _set_form_enabled(self, enabled: bool) -> None:
        self._form_enabled = bool(enabled)
        for widget in (
            self.name_edit,
            self.provider_combo,
            self.api_key_edit,
            self.api_key_rotation_button,
            self.api_key_rotation_editor,
            self.api_key_rotation_section.toggle_button,
            self.base_url_edit,
            self.supports_images_checkbox,
        ):
            widget.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)
        if enabled:
            self._update_base_url_field_state(self.provider_combo.currentText())
        self._set_model_state(self._model_state, message=self.model_loading_label.text())

    def _normalized_provider(self) -> str:
        provider = str(self.provider_combo.currentText() or "").strip().lower()
        return provider if provider in ALLOWED_PROVIDERS else "openai"

    def _invalidate_pending_fetches(self) -> None:
        self._fetch_debounce.stop()
        self._fetch_request_id += 1

    def _clear_model_options(self) -> None:
        self._model_entries_by_id = {}
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.blockSignals(False)

    def _set_combo_placeholder(self, text: str) -> None:
        placeholder = str(text or "").strip()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if placeholder:
            self.model_combo.addItem(placeholder)
            self.model_combo.setCurrentIndex(0)
        self.model_combo.blockSignals(False)

    def _set_model_state(self, state: ModelLoadState, *, message: str = "") -> None:
        self._model_state = state
        provider = self._normalized_provider()
        is_openai = provider == "openai"
        show_combo = state in {ModelLoadState.LOADING, ModelLoadState.LOADED, ModelLoadState.ERROR}
        show_text = not show_combo
        status_text = str(message or "").strip()
        has_fetch_inputs = self._current_fetch_inputs() is not None

        self.model_combo.setVisible(show_combo)
        self.model_text_edit.setVisible(show_text)
        self.model_combo.setEditable(is_openai and state in {ModelLoadState.LOADED, ModelLoadState.FALLBACK})
        self.model_combo.setEnabled(self._form_enabled and state in {ModelLoadState.LOADED, ModelLoadState.FALLBACK})
        self.model_text_edit.setEnabled(self._form_enabled and state == ModelLoadState.FALLBACK)
        self.model_popup_button.setVisible(self._form_enabled and show_combo)
        self.model_popup_button.setEnabled(self._form_enabled and state == ModelLoadState.LOADED and self.model_combo.count() > 0)
        self.model_reload_button.setVisible(self._form_enabled)
        self.model_reload_button.setEnabled(self._form_enabled and has_fetch_inputs and state != ModelLoadState.LOADING)
        self.model_loading_label.setText(status_text)
        self.model_loading_label.setVisible(bool(status_text))
        if hasattr(self, "model_field_label"):
            self.model_field_label.setBuddy(self.model_combo if show_combo else self.model_text_edit)

    def _set_current_model_widgets_text(self, value: str) -> None:
        text = str(value or "").strip()
        self.model_text_edit.blockSignals(True)
        self.model_text_edit.setText(text)
        self.model_text_edit.blockSignals(False)

        self.model_combo.blockSignals(True)
        if self.model_combo.count() == 0:
            self.model_combo.setCurrentIndex(-1)
        if self.model_combo.isEditable():
            self.model_combo.setEditText(text)
        else:
            index = self.model_combo.findText(text)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)

    def _show_model_popup(self) -> None:
        if not self.model_combo.isVisible() or not self.model_combo.isEnabled() or self.model_combo.count() <= 0:
            return
        self.model_combo.setFocus(Qt.OtherFocusReason)
        self.model_combo.showPopup()

    def _get_current_model_value(self) -> str:
        if self._model_state == ModelLoadState.LOADED:
            return str(self.model_combo.currentText() or "").strip()
        return str(self.model_text_edit.text() or "").strip()

    def _current_fetch_inputs(self) -> tuple[str, str, str, ModelFetcher, tuple[str, ...]] | None:
        provider = self._normalized_provider()
        api_key = str(self.api_key_edit.text() or "").strip()
        if not api_key:
            return None
        if provider == "gemini":
            return provider, api_key, "", GeminiModelFetcher(), (provider, api_key)
        base_url = str(self.base_url_edit.text() or "").strip().rstrip("/")
        if not base_url:
            return None
        return provider, api_key, base_url, OpenAICompatibleModelFetcher(), (provider, api_key, base_url)

    def _schedule_fetch(self, delay_ms: int = 600) -> None:
        if self._current_row() < 0:
            return
        if self._current_fetch_inputs() is None:
            self._set_model_state(ModelLoadState.IDLE)
            return
        self._fetch_debounce.setInterval(max(0, int(delay_ms)))
        self._fetch_debounce.start()

    def _cleanup_model_worker(self, worker: ModelFetchWorker) -> None:
        if worker in self._model_workers:
            self._model_workers.remove(worker)
        worker.deleteLater()

    def _sync_api_key_field_from_rotation_editor(self) -> tuple[list[str], int, str]:
        row = self._current_row()
        api_keys = self._current_rotation_api_keys()
        current_key = str(self.api_key_edit.text() or "").strip()
        preferred_index = self._profile_api_key_index(row)
        api_key_index, active_api_key = self._resolve_api_key_selection(
            api_keys,
            current_key=current_key,
            preferred_index=preferred_index,
        )
        self._update_api_key_rotation_summary(api_keys=api_keys, active_key=active_api_key)
        if self.api_key_edit.text() != active_api_key:
            self._loading_form = True
            self.api_key_edit.setText(active_api_key)
            self._loading_form = False
        return api_keys, api_key_index, active_api_key

    def _sync_rotation_editor_from_active_key(self) -> list[str]:
        row = self._current_row()
        api_keys = self._current_rotation_api_keys()
        current_key = str(self.api_key_edit.text() or "").strip()
        preferred_index = self._profile_api_key_index(row)
        if api_keys:
            preferred_index = max(0, min(preferred_index, len(api_keys) - 1))
        if current_key:
            if api_keys:
                api_keys[preferred_index] = current_key
            else:
                api_keys = [current_key]
        elif api_keys:
            api_keys.pop(preferred_index)
        normalized_keys = normalize_api_key_list(api_keys)
        self._set_api_key_rotation_editor_text(normalized_keys)
        return normalized_keys

    def _append_gemini_model_items(self, entries: list[ModelEntry]) -> list[str]:
        ordered_ids: list[str] = []
        gemini_ids = sorted((entry.id for entry in entries if entry.family == "gemini"), reverse=True)
        gemma_ids = sorted((entry.id for entry in entries if entry.family == "gemma"), reverse=True)
        for model_id in gemini_ids:
            self.model_combo.addItem(model_id)
            ordered_ids.append(model_id)
        if gemma_ids and gemini_ids:
            separator = QStandardItem("── Gemma ──")
            separator.setFlags(Qt.ItemFlag.NoItemFlags)
            self.model_combo.model().appendRow(separator)
        for model_id in gemma_ids:
            self.model_combo.addItem(model_id)
            ordered_ids.append(model_id)
        return ordered_ids

    def _apply_entry_image_support(self, model_id: str) -> None:
        entry = self._model_entries_by_id.get(str(model_id or "").strip())
        if entry is None:
            return
        self.supports_images_checkbox.setChecked(bool(entry.supports_image_input))

    def _sync_image_support_after_model_load(self, model_id: str) -> None:
        current_row = self._current_row()
        if 0 <= current_row < len(self._profiles):
            profile = self._profiles[current_row]
            profile_model = str(profile.get("model") or "").strip()
            if profile_model and profile_model == str(model_id or "").strip():
                self.supports_images_checkbox.setChecked(bool(profile.get("supports_image_input")))
                return
        self._apply_entry_image_support(model_id)

    def _apply_loaded_models(self, entries: list[ModelEntry]) -> None:
        provider = self._normalized_provider()
        current_value = self._get_current_model_value()
        self._model_entries_by_id = {entry.id: entry for entry in entries}

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        if provider == "gemini":
            ordered_ids = self._append_gemini_model_items(entries)
        else:
            ordered_ids = sorted(entry.id for entry in entries)
            for model_id in ordered_ids:
                self.model_combo.addItem(model_id)

        selected_model = current_value if current_value in self._model_entries_by_id else (ordered_ids[0] if ordered_ids else "")
        if selected_model:
            selected_index = self.model_combo.findText(selected_model)
            if selected_index >= 0:
                self.model_combo.setCurrentIndex(selected_index)
            elif provider == "openai":
                self.model_combo.setEditText(selected_model)
        self.model_combo.blockSignals(False)

        self._set_current_model_widgets_text(selected_model)
        if not selected_model and provider == "openai":
            self._set_model_state(
                ModelLoadState.LOADED,
                message="Список моделей пуст. Введите название модели вручную.",
            )
        else:
            self._set_model_state(ModelLoadState.LOADED)
        self._sync_image_support_after_model_load(selected_model)
        self._on_form_changed()

    def _start_fetch(self) -> None:
        self._fetch_debounce.setInterval(600)
        request = self._current_fetch_inputs()
        if request is None:
            self._set_model_state(ModelLoadState.IDLE)
            return

        provider, api_key, base_url, fetcher, cache_key = request
        cached_entries = self._model_cache.get(cache_key)
        if cached_entries is not None:
            self._apply_loaded_models(cached_entries)
            return

        self._fetch_request_id += 1
        request_id = self._fetch_request_id
        self._set_combo_placeholder("Загрузка моделей…")
        self._set_model_state(ModelLoadState.LOADING, message="Загрузка…")

        worker = ModelFetchWorker(request_id, fetcher, api_key, base_url)
        self._model_workers.append(worker)
        worker.fetched.connect(self._on_models_fetched)
        worker.failed.connect(self._on_models_failed)
        worker.finished.connect(lambda: self._cleanup_model_worker(worker))
        worker.start()

    def _on_models_fetched(self, request_id: int, payload: object) -> None:
        if request_id != self._fetch_request_id:
            return
        entries = [entry for entry in list(payload or []) if isinstance(entry, ModelEntry)]
        request = self._current_fetch_inputs()
        if request is None:
            return
        cache_key = request[-1]
        self._model_cache[cache_key] = entries
        self._apply_loaded_models(entries)

    def _on_models_failed(self, request_id: int, message: str) -> None:
        if request_id != self._fetch_request_id:
            return
        current_value = self._get_current_model_value()
        self._set_current_model_widgets_text(current_value)
        if self._normalized_provider() == "openai":
            self._set_model_state(ModelLoadState.FALLBACK, message=message)
            return
        self._set_combo_placeholder(current_value or "Модели недоступны")
        self._set_model_state(ModelLoadState.ERROR, message=message)

    def _reload_models(self) -> None:
        if self._loading_form:
            return
        request = self._current_fetch_inputs()
        if request is not None:
            self._model_cache.pop(request[-1], None)
        self._invalidate_pending_fetches()
        if request is None:
            self._set_model_state(ModelLoadState.IDLE)
            return
        self._schedule_fetch(0)

    def _on_api_key_changed(self, _text: str) -> None:
        if self._loading_form:
            return
        self._sync_rotation_editor_from_active_key()
        self._invalidate_pending_fetches()
        self._clear_model_options()
        self._set_model_state(ModelLoadState.IDLE)
        if self._current_fetch_inputs() is not None:
            self._schedule_fetch(600)
        self._on_form_changed()

    def _on_api_key_rotation_text_changed(self) -> None:
        if self._loading_form:
            return
        self._sync_api_key_field_from_rotation_editor()
        self._invalidate_pending_fetches()
        self._clear_model_options()
        self._set_model_state(ModelLoadState.IDLE)
        if self._current_fetch_inputs() is not None:
            self._schedule_fetch(600)
        self._on_form_changed()

    def _on_base_url_changed(self, _text: str) -> None:
        if self._loading_form:
            return
        if self._normalized_provider() == "openai":
            self._invalidate_pending_fetches()
            self._clear_model_options()
            self._set_model_state(ModelLoadState.IDLE)
            if self._current_fetch_inputs() is not None:
                self._schedule_fetch(600)
        self._on_form_changed()

    def _profile_api_key_index(self, row: int) -> int:
        if row < 0 or row >= len(self._profiles):
            return 0
        api_keys = self._profile_api_keys(row)
        if not api_keys:
            return 0
        try:
            index = int(self._profiles[row].get("api_key_index") or 0)
        except (TypeError, ValueError):
            index = 0
        return max(0, min(index, len(api_keys) - 1))

    def _current_rotation_api_keys(self) -> list[str]:
        return normalize_api_key_list(self.api_key_rotation_editor.toPlainText())

    def _set_api_key_rotation_editor_text(self, api_keys: list[str]) -> None:
        normalized_keys = normalize_api_key_list(api_keys)
        text = "\n".join(normalized_keys)
        if self.api_key_rotation_editor.toPlainText() == text:
            self._update_api_key_rotation_summary(api_keys=normalized_keys)
            return
        self.api_key_rotation_editor.blockSignals(True)
        self.api_key_rotation_editor.setPlainText(text)
        self.api_key_rotation_editor.blockSignals(False)
        self._update_api_key_rotation_summary(api_keys=normalized_keys)

    def _update_api_key_rotation_summary(self, *, api_keys: list[str] | None = None, active_key: str | None = None) -> None:
        keys = normalize_api_key_list(api_keys if api_keys is not None else self.api_key_rotation_editor.toPlainText())
        active = str(self.api_key_edit.text() if active_key is None else active_key or "").strip()
        self.api_key_rotation_count_chip.setText(f"{len(keys)} key(s)")
        if not keys:
            self.api_key_rotation_status_label.setText("If the pool is empty, the API key field is used on its own.")
            return
        if active and active in keys:
            self.api_key_rotation_status_label.setText(f"Active key: entry {keys.index(active) + 1} of {len(keys)}.")
            return
        self.api_key_rotation_status_label.setText("Pool ready. The first valid key will be used.")

    def _set_save_state(self, message: str) -> None:
        text = str(message or "").strip()
        self.save_state_label.setText(text)
        self.save_state_label.setVisible(bool(text))

    def _profile_id_for_row(self, row: int) -> str:
        if row < 0 or row >= len(self._profiles):
            return ""
        return str(self._profiles[row].get("id") or "").strip()

    def _row_for_profile_id(self, profile_id: str) -> int:
        target_id = str(profile_id or "").strip()
        if not target_id:
            return -1
        for idx, profile in enumerate(self._profiles):
            if str(profile.get("id") or "").strip() == target_id:
                return idx
        return -1

    def _refresh_profile_counts(self) -> None:
        total = len(self._profiles)
        enabled = sum(1 for profile in self._profiles if bool(profile.get("enabled", True)))
        self.profile_count_chip.setText(f"{total} total")
        self.left_meta_chip.setText(f"{enabled} enabled")
        active_name = str(self._active_profile or "").strip() or "none"
        self.active_profile_label.setText(
            f"Active now: <span style='color:#D5D9DF;font-weight:700'>{active_name}</span>"
        )

    def _apply_profile_filter(self, value: str) -> None:
        self._filter_text = str(value or "").strip().lower()
        visible_rows: list[int] = []
        for row in range(self.profile_list.count()):
            item = self.profile_list.item(row)
            if item is None:
                continue
            haystack = " ".join((str(item.data(Qt.UserRole) or ""), str(item.toolTip() or ""))).lower()
            should_hide = bool(self._filter_text) and self._filter_text not in haystack
            item.setHidden(should_hide)
            if not should_hide:
                visible_rows.append(row)

        current_row = self.profile_list.currentRow()
        if current_row < 0:
            if visible_rows:
                self.profile_list.setCurrentRow(visible_rows[0])
            return

        current_item = self.profile_list.item(current_row)
        if current_item is not None and current_item.isHidden():
            if visible_rows:
                self.profile_list.setCurrentRow(visible_rows[0])
            else:
                self.profile_list.clearSelection()
                self._sync_form_to_profile(-1)

    def _update_base_url_field_state(self, provider: str) -> None:
        provider_normalized = str(provider or "").strip().lower()
        enabled = provider_normalized == "openai"
        self.base_url_edit.setEnabled(enabled)
        if enabled:
            self.base_url_edit.setPlaceholderText("https://api.openai.com/v1")
            self.base_url_edit.setToolTip("")
        else:
            self.base_url_edit.setPlaceholderText("Not used for gemini")
            self.base_url_edit.setToolTip("Base URL is only used for openai profiles.")

    def _display_name(self, profile: dict[str, str]) -> str:
        profile_id = str(profile.get("id") or "").strip()
        provider = str(profile.get("provider") or "").strip()
        model_name = str(profile.get("model") or "").strip()
        marker = ""
        if profile_id and profile_id == self._active_profile:
            marker = " • active"
        elif not bool(profile.get("enabled", True)):
            marker = " • disabled"
        title = profile_id if profile_id else "(unnamed)"
        if marker:
            title = f"{title}{marker}"
        details = " · ".join(part for part in (provider, model_name) if part)
        return f"{title}\n{details}" if details else title

    def _refresh_profile_item_states(self) -> None:
        current_row = self._current_row()
        for row in range(self.profile_list.count()):
            item = self.profile_list.item(row)
            widget = self.profile_list.itemWidget(item) if item is not None else None
            if widget is None:
                continue
            profile = self._profiles[row] if 0 <= row < len(self._profiles) else {}
            is_selected = row == current_row and not bool(item.isHidden()) if item is not None else False
            is_active = bool(str(profile.get("id") or "").strip() and str(profile.get("id") or "").strip() == self._active_profile)
            is_enabled = bool(profile.get("enabled", True))
            widget.setProperty("selectedProfile", is_selected)
            widget.setProperty("activeProfile", is_active)
            widget.setProperty("disabledProfile", not is_enabled)
            widget.style().unpolish(widget)
            widget.style().polish(widget)

    def _toggle_api_key_visibility(self) -> None:
        reveal = self.api_key_edit.echoMode() != QLineEdit.Normal
        self.api_key_edit.setEchoMode(QLineEdit.Normal if reveal else QLineEdit.Password)
        icon_name = "fa5s.eye-slash" if reveal else "fa5s.eye"
        self.api_key_reveal_button.setIcon(_fa_icon(icon_name, color=TEXT_MUTED, size=12))

    def _copy_api_key(self) -> None:
        QApplication.clipboard().setText(self.api_key_edit.text())
        self._set_save_state("API key copied to clipboard.")

    def _profile_api_keys(self, row: int) -> list[str]:
        if row < 0 or row >= len(self._profiles):
            return []
        profile = self._profiles[row]
        return normalize_api_key_list(profile.get("api_keys"), fallback=profile.get("api_key"))

    def _resolve_api_key_selection(
        self,
        api_keys: list[str],
        *,
        current_key: str,
        preferred_index: int,
    ) -> tuple[int, str]:
        if not api_keys:
            return 0, ""
        clamped_index = max(0, min(int(preferred_index), len(api_keys) - 1))
        if current_key and current_key in api_keys:
            selected_index = api_keys.index(current_key)
        else:
            selected_index = clamped_index
        return selected_index, api_keys[selected_index]

    def _apply_api_key_rotation_to_profile(self, row: int, api_keys: list[str]) -> None:
        if row < 0 or row >= len(self._profiles):
            return
        cleaned_keys = normalize_api_key_list(api_keys)
        profile = dict(self._profiles[row])
        current_key = str(profile.get("api_key") or "").strip()
        api_key_index, active_api_key = self._resolve_api_key_selection(
            cleaned_keys,
            current_key=current_key,
            preferred_index=0,
        )
        profile["api_keys"] = cleaned_keys
        profile["api_key_index"] = api_key_index
        profile["invalid_api_keys"] = []
        profile["key_error_timestamps"] = {}
        profile["api_key"] = active_api_key
        self._profiles[row] = profile
        if row == self._current_row():
            self._loading_form = True
            self.api_key_edit.setText(active_api_key)
            self._set_api_key_rotation_editor_text(cleaned_keys)
            self._loading_form = False
            self._update_api_key_rotation_summary(api_keys=cleaned_keys, active_key=active_api_key)

    def _edit_api_key_rotation(self) -> None:
        row = self._current_row()
        if row < 0 or row >= len(self._profiles):
            return
        self.api_key_rotation_section.set_expanded(True)
        self.api_key_rotation_editor.setFocus(Qt.OtherFocusReason)
        self.api_key_rotation_editor.ensureCursorVisible()
        self._set_save_state("Edit the rotation pool inline and press Save to persist.")

    def _build_profile_item_widget(self, profile: dict[str, Any], row: int) -> QWidget:
        container = QWidget()
        container.setObjectName("ModelProfileRowCard")
        is_enabled = bool(profile.get("enabled", True))
        container.setProperty("disabledProfile", not is_enabled)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(14, 10, 12, 10)
        layout.setSpacing(12)

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(2)

        first_row = QHBoxLayout()
        first_row.setContentsMargins(0, 0, 0, 0)
        first_row.setSpacing(4)

        profile_id = str(profile.get("id") or "").strip() or "(unnamed)"
        is_active = bool(profile_id and profile_id != "(unnamed)" and profile_id == self._active_profile)
        title_label = QLabel(profile_id)
        title_label.setObjectName("ModelProfileItemTitle")
        title_label.setEnabled(is_enabled)
        title_label.setMargin(0)
        title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title_label.setMinimumWidth(0)
        title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        first_row.addWidget(title_label, 0, Qt.AlignLeft | Qt.AlignVCenter)

        provider = str(profile.get("provider") or "").strip()
        if provider:
            provider_label = QLabel(provider)
            provider_label.setObjectName("ModelProfileItemBadge")
            provider_label.setProperty("badgeVariant", "provider")
            first_row.addWidget(provider_label, 0, Qt.AlignLeft | Qt.AlignVCenter)

        if is_active:
            active_label = QLabel("Active")
            active_label.setObjectName("ModelProfileItemBadge")
            active_label.setProperty("badgeVariant", "active")
            first_row.addWidget(active_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
        elif not is_enabled:
            disabled_label = QLabel("Disabled")
            disabled_label.setObjectName("ModelProfileItemBadge")
            disabled_label.setProperty("badgeVariant", "muted")
            first_row.addWidget(disabled_label, 0, Qt.AlignLeft | Qt.AlignVCenter)

        first_row.addStretch(1)
        text_column.addLayout(first_row)

        model_name = str(profile.get("model") or "").strip()
        details = " · ".join(part for part in (provider, model_name) if part)
        details_label = QLabel(details)
        details_label.setObjectName("ModelProfileItemMeta")
        details_label.setEnabled(is_enabled)
        details_label.setMargin(0)
        details_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        details_label.setWordWrap(False)
        details_label.setMinimumWidth(0)
        details_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        text_column.addWidget(details_label, 0, Qt.AlignLeft | Qt.AlignVCenter)

        layout.addLayout(text_column, 1)

        enabled_switch = QCheckBox()
        enabled_switch.setObjectName("ModelProfileEnabledSwitch")
        enabled_switch.setChecked(is_enabled)
        enabled_switch.setCursor(Qt.PointingHandCursor)
        enabled_switch.setToolTip("Temporarily enable or disable this model")
        enabled_switch.setFocusPolicy(Qt.NoFocus)
        enabled_switch.setFixedSize(QSize(34, 20))
        enabled_switch.pressed.connect(lambda target_row=row: self.profile_list.setCurrentRow(target_row))
        enabled_switch.toggled.connect(lambda checked, target_row=row: self._toggle_profile_enabled(target_row, checked))
        layout.addWidget(enabled_switch, 0, Qt.AlignRight | Qt.AlignVCenter)
        container.ensurePolished()
        title_label.ensurePolished()
        details_label.ensurePolished()
        title_label.setMinimumHeight(title_label.fontMetrics().height() + 6)
        details_label.setMinimumHeight(details_label.fontMetrics().height() + 4)
        container.adjustSize()
        return container

    def _refresh_profile_list(
        self,
        preferred_row: int | None = None,
        *,
        preferred_profile_id: str = "",
        restore_scroll_value: int | None = None,
    ) -> None:
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        for row, profile in enumerate(self._profiles):
            item_widget = self._build_profile_item_widget(profile, row)
            item = QListWidgetItem("")
            item.setData(Qt.UserRole, self._display_name(profile))
            item_widget.ensurePolished()
            widget_hint = item_widget.sizeHint()
            minimum_hint = item_widget.minimumSizeHint()
            item_height = max(72, widget_hint.height(), minimum_hint.height()) + 6
            item.setSizeHint(QSize(widget_hint.width(), item_height))
            provider = str(profile.get("provider") or "").strip()
            model_name = str(profile.get("model") or "").strip()
            enabled = "yes" if bool(profile.get("enabled", True)) else "no"
            item.setToolTip(f"Provider: {provider}\nModel: {model_name}\nEnabled: {enabled}".strip())
            self.profile_list.addItem(item)
            self.profile_list.setItemWidget(item, item_widget)
        self.profile_list.blockSignals(False)
        if self.save_button is not None:
            self.save_button.setEnabled(bool(self._profiles))

        if not self._profiles:
            self._selected_row = -1
            self._set_form_enabled(False)
            self.form_hint.setText("Add a profile to start configuring models.")
            self.profile_state_chip.setText("No profile selected")
            self.summary_provider.setText("Provider: —")
            self.summary_model.setText("Model: —")
            self.summary_images.setText("Image input: off")
            self.duplicate_button.setEnabled(False)
            return

        row = preferred_row
        if preferred_profile_id:
            resolved_row = self._row_for_profile_id(preferred_profile_id)
            if resolved_row >= 0:
                row = resolved_row
        if row is None:
            row = self._preferred_row_for_open()
        row = max(0, min(row, len(self._profiles) - 1))
        self.profile_list.setCurrentRow(row)
        self._apply_profile_filter(self.search_edit.text())
        self._refresh_profile_item_states()
        if restore_scroll_value is not None:
            self.profile_list.verticalScrollBar().setValue(restore_scroll_value)

    def _preferred_row_for_open(self) -> int:
        active_id = str(self._active_profile or "").strip()
        if active_id:
            for index, profile in enumerate(self._profiles):
                if str(profile.get("id") or "").strip() == active_id:
                    return index
        return self._current_row() if self._current_row() >= 0 else 0

    def _sync_form_to_profile(self, row: int) -> None:
        if row < 0 or row >= len(self._profiles):
            self._invalidate_pending_fetches()
            self._clear_model_options()
            self._set_current_model_widgets_text("")
            self._selected_row = -1
            self._set_form_enabled(False)
            self._set_model_state(ModelLoadState.IDLE)
            self.form_hint.setText("Select a profile and edit keys, model settings, and image support on the right.")
            self.profile_state_chip.setText("No profile selected")
            self.summary_provider.setText("Provider: —")
            self.summary_model.setText("Model: —")
            self.summary_images.setText("Image input: off")
            self.duplicate_button.setEnabled(False)
            self._refresh_profile_item_states()
            return
        profile = self._profiles[row]
        self._loading_form = True
        self._invalidate_pending_fetches()
        self._clear_model_options()
        self._set_form_enabled(True)
        self.name_edit.setText(str(profile.get("id", "")))
        provider = str(profile.get("provider", "openai")).strip().lower()
        if provider not in ALLOWED_PROVIDERS:
            provider = "openai"
        self.provider_combo.setCurrentText(provider)
        self._set_current_model_widgets_text(str(profile.get("model", "")))
        self.api_key_edit.setText(str(profile.get("api_key", "")))
        self._set_api_key_rotation_editor_text(self._profile_api_keys(row))
        self.base_url_edit.setText(str(profile.get("base_url", "")))
        self.supports_images_checkbox.setChecked(bool(profile.get("supports_image_input")))
        self._update_base_url_field_state(provider)
        self._update_api_key_rotation_summary(
            api_keys=self._profile_api_keys(row),
            active_key=str(profile.get("api_key", "")),
        )
        self._set_model_state(ModelLoadState.IDLE)
        self._loading_form = False
        self._selected_row = row
        profile_id = str(profile.get("id") or "").strip() or "(unnamed)"
        status = "enabled" if bool(profile.get("enabled", True)) else "disabled"
        self.form_hint.setText(f"Editing profile: {profile_id} ({status})")
        is_active = bool(profile_id != "(unnamed)" and profile_id == self._active_profile)
        self.profile_state_chip.setText("Active profile" if is_active else status.title())
        self.duplicate_button.setEnabled(True)
        provider_text = str(profile.get("provider") or "—").strip() or "—"
        model_name = str(profile.get("model") or "—").strip() or "—"
        self.summary_provider.setText(f"Provider: {provider_text}")
        self.summary_model.setText(f"Model: {model_name}")
        self.summary_images.setText(
            "Image input: on" if bool(profile.get("supports_image_input")) else "Image input: off"
        )
        self._refresh_profile_counts()
        self._refresh_profile_item_states()
        if self._current_fetch_inputs() is not None:
            self._schedule_fetch(100)

    def _sync_current_profile_from_form(self, row: int | None = None) -> None:
        if self._loading_form:
            return
        target_row = self._current_row() if row is None else row
        if target_row < 0 or target_row >= len(self._profiles):
            return
        provider = self._normalized_provider()
        base_url = str(self.base_url_edit.text() or "").strip() if provider == "openai" else ""
        existing_profile = dict(self._profiles[target_row])
        current_api_key = str(self.api_key_edit.text() or "").strip()
        api_keys = self._current_rotation_api_keys()
        existing_index = self._profile_api_key_index(target_row)
        if api_keys:
            existing_index = max(0, min(existing_index, len(api_keys) - 1))
        if current_api_key:
            if api_keys:
                api_keys[existing_index] = current_api_key
            else:
                api_keys = [current_api_key]
        elif api_keys:
            api_keys.pop(existing_index)
            existing_index = min(existing_index, max(0, len(api_keys) - 1))
        api_keys = normalize_api_key_list(api_keys)
        api_key_index, current_api_key = self._resolve_api_key_selection(
            api_keys,
            current_key=current_api_key,
            preferred_index=existing_index,
        )
        self._profiles[target_row] = {
            "id": str(self.name_edit.text() or "").strip(),
            "provider": provider,
            "model": self._get_current_model_value(),
            "api_key": current_api_key,
            "api_keys": api_keys,
            "api_key_index": api_key_index,
            "invalid_api_keys": [],
            "key_error_timestamps": {},
            "base_url": base_url,
            "supports_image_input": self.supports_images_checkbox.isChecked(),
            "enabled": bool(existing_profile.get("enabled", True)),
        }
        item = self.profile_list.item(target_row)
        if item is not None:
            item.setData(Qt.UserRole, self._display_name(self._profiles[target_row]))
            provider_text = str(self._profiles[target_row].get("provider") or "").strip()
            model_name = str(self._profiles[target_row].get("model") or "").strip()
            enabled = "yes" if bool(self._profiles[target_row].get("enabled", True)) else "no"
            item.setToolTip(f"Provider: {provider_text}\nModel: {model_name}\nEnabled: {enabled}".strip())
        if target_row == self._selected_row:
            self.summary_provider.setText(f"Provider: {provider or '—'}")
            self.summary_model.setText(f"Model: {self._profiles[target_row].get('model') or '—'}")
            self.summary_images.setText(
                "Image input: on" if bool(self._profiles[target_row].get("supports_image_input")) else "Image input: off"
            )
        self._refresh_profile_counts()

    def _reconcile_active_profile(self) -> None:
        enabled_ids = [
            str(profile.get("id") or "").strip()
            for profile in self._profiles
            if bool(profile.get("enabled", True)) and str(profile.get("id") or "").strip()
        ]
        active_id = str(self._active_profile or "").strip()
        if active_id in enabled_ids:
            return
        self._active_profile = enabled_ids[0] if enabled_ids else ""

    def _toggle_profile_enabled(self, row: int, state: bool | int) -> None:
        if row < 0 or row >= len(self._profiles):
            return
        self._sync_current_profile_from_form(self._selected_row)
        preferred_profile_id = self._profile_id_for_row(row)
        scroll_value = self.profile_list.verticalScrollBar().value()
        if isinstance(state, bool):
            is_enabled = bool(state)
        elif isinstance(state, int):
            is_enabled = state == Qt.Checked
        else:
            is_enabled = bool(state)
        self._profiles[row]["enabled"] = is_enabled
        self._reconcile_active_profile()
        self._set_save_state("")
        self._refresh_profile_list(
            preferred_row=row,
            preferred_profile_id=preferred_profile_id,
            restore_scroll_value=scroll_value,
        )
        self._refresh_profile_counts()

    def _suggest_unique_id(self, model_text: str, *, row: int) -> str:
        used = {
            str(profile.get("id") or "").strip()
            for idx, profile in enumerate(self._profiles)
            if idx != row and str(profile.get("id") or "").strip()
        }
        return generate_profile_id(model_text, used)

    def _compute_initial_name_manual_flags(self) -> list[bool]:
        flags: list[bool] = []
        for idx, profile in enumerate(self._profiles):
            profile_id = str(profile.get("id") or "").strip()
            if not profile_id:
                flags.append(False)
                continue
            expected_auto_id = self._suggest_unique_id(str(profile.get("model") or ""), row=idx)
            flags.append(profile_id != expected_auto_id)
        return flags

    def _on_selection_changed(self, row: int) -> None:
        previous_row = self._selected_row
        if previous_row != row:
            self._sync_current_profile_from_form(previous_row)
        self._sync_form_to_profile(row)

    def _on_name_edited(self, text: str) -> None:
        row = self._current_row()
        if 0 <= row < len(self._name_manual_flags):
            self._name_manual_flags[row] = bool(str(text or "").strip())
        self._sync_current_profile_from_form()

    def _on_model_changed(self, _text: str) -> None:
        if self._loading_form:
            return
        row = self._current_row()
        if row < 0 or row >= len(self._profiles):
            return
        current_model = self._get_current_model_value()
        if self._model_state == ModelLoadState.LOADED:
            self._apply_entry_image_support(current_model)
        self._loading_form = True
        self.name_edit.setText(self._suggest_unique_id(current_model, row=row))
        self._loading_form = False
        if 0 <= row < len(self._name_manual_flags):
            self._name_manual_flags[row] = False
        self._on_form_changed()

    def _on_form_changed(self) -> None:
        self._set_save_state("")
        self._sync_current_profile_from_form()

    def _on_provider_changed(self, provider: str) -> None:
        if self._loading_form:
            return
        self._update_base_url_field_state(provider)
        self._invalidate_pending_fetches()
        self._clear_model_options()
        self._loading_form = True
        self._set_current_model_widgets_text("")
        if str(provider or "").strip().lower() != "openai":
            self.base_url_edit.clear()
        self._loading_form = False
        self._set_model_state(ModelLoadState.IDLE)
        if self._current_fetch_inputs() is not None:
            self._schedule_fetch(600)
        self._on_form_changed()

    def _add_profile(self) -> None:
        self._sync_current_profile_from_form(self._selected_row)
        self._profiles.append(
            {
                "id": "",
                "provider": "openai",
                "model": "",
                "api_key": "",
                "api_keys": [],
                "api_key_index": 0,
                "invalid_api_keys": [],
                "key_error_timestamps": {},
                "base_url": "",
                "supports_image_input": False,
                "enabled": True,
            }
        )
        self._name_manual_flags.append(False)
        self._set_save_state("")
        self._refresh_profile_list(preferred_row=len(self._profiles) - 1)
        self.name_edit.setFocus()

    def _duplicate_selected_profile(self) -> None:
        row = self._current_row()
        if row < 0 or row >= len(self._profiles):
            return
        self._sync_current_profile_from_form(self._selected_row)
        source = dict(self._profiles[row])
        duplicated = dict(source)
        duplicated["id"] = ""
        duplicated["enabled"] = bool(source.get("enabled", True))
        self._profiles.insert(row + 1, duplicated)
        self._name_manual_flags.insert(row + 1, False)
        self._set_save_state("Duplicated profile. Rename it before saving if needed.")
        self._refresh_profile_list(preferred_row=row + 1)
        self.name_edit.setFocus()

    def _delete_selected_profile(self) -> None:
        row = self._current_row()
        if row < 0 or row >= len(self._profiles):
            return
        self._sync_current_profile_from_form(self._selected_row)
        removed_id = str(self._profiles[row].get("id") or "").strip()
        self._profiles.pop(row)
        self._name_manual_flags.pop(row)

        if removed_id and removed_id == self._active_profile:
            self._active_profile = ""
        self._reconcile_active_profile()

        self._set_save_state("")
        self._refresh_profile_list(preferred_row=row)
        self._refresh_profile_counts()

    def _validated_payload(self) -> dict[str, Any] | None:
        self._sync_current_profile_from_form(self._selected_row)
        profiles: list[dict[str, str]] = []
        used_ids: set[str] = set()

        for idx, profile in enumerate(self._profiles):
            provider = str(profile.get("provider") or "").strip().lower()
            model_name = str(profile.get("model") or "").strip()
            if provider not in ALLOWED_PROVIDERS:
                self.profile_list.setCurrentRow(idx)
                QMessageBox.warning(self, "Validation", "Provider must be openai or gemini.")
                return None
            if not model_name:
                self.profile_list.setCurrentRow(idx)
                QMessageBox.warning(self, "Validation", "Model cannot be empty.")
                return None

            requested_id = sanitize_profile_id(profile.get("id") or "")
            if not requested_id:
                requested_id = model_name
            profile_id = generate_profile_id(requested_id, used_ids)
            profiles.append(
                {
                    "id": profile_id,
                    "provider": provider,
                    "model": model_name,
                    "api_key": str(profile.get("api_key") or "").strip(),
                    "api_keys": list(profile.get("api_keys") or []),
                    "api_key_index": int(profile.get("api_key_index") or 0),
                    "invalid_api_keys": list(profile.get("invalid_api_keys") or []),
                    "key_error_timestamps": dict(profile.get("key_error_timestamps") or {}),
                    "base_url": str(profile.get("base_url") or "").strip(),
                    "supports_image_input": bool(profile.get("supports_image_input")),
                    "enabled": bool(profile.get("enabled", True)),
                }
            )

        active = str(self._active_profile or "").strip()
        enabled_ids = [item["id"] for item in profiles if bool(item.get("enabled", True))]
        if active not in enabled_ids:
            active = enabled_ids[0] if enabled_ids else ""
        return {"active_profile": active or None, "profiles": profiles}

    def _persist_profiles(self, message: str) -> bool:
        validated = self._validated_payload()
        if validated is None:
            return False
        self._result_payload = normalize_profiles_payload(validated)
        self._profiles = [dict(item) for item in self._result_payload.get("profiles", [])]
        self._active_profile = str(self._result_payload.get("active_profile") or "").strip()
        self._name_manual_flags = self._compute_initial_name_manual_flags()
        current_row = self._current_row()
        preferred_row = current_row if current_row >= 0 else self._preferred_row_for_open()
        self._refresh_profile_list(preferred_row=preferred_row)
        self._refresh_profile_counts()
        self._set_save_state(str(message or "").strip())
        self.profiles_saved.emit(dict(self._result_payload))
        return True

    def _save_and_accept(self) -> None:
        self._persist_profiles("Saved. You can keep this window open and continue editing.")


class ApprovalDialog(QDialog):
    def __init__(self, payload: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.choice: tuple[bool, bool] = (False, False)
        self.setObjectName("ApprovalDialog")
        self.setWindowTitle("Approval required")
        self.setModal(True)
        self.resize(640, 420)
        self.setMinimumSize(520, 340)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        risk_level = str(summary.get("risk_level", "unknown") or "unknown")
        impacts = [str(item).strip() for item in list(summary.get("impacts", []) or []) if str(item).strip()]
        tools = list(payload.get("tools", []) or [])

        hero_card = QFrame()
        hero_card.setObjectName("ApprovalRequestCard")
        hero_layout = QVBoxLayout(hero_card)
        hero_layout.setContentsMargins(16, 14, 16, 14)
        hero_layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)

        title = QLabel("Protected action review")
        title.setObjectName("ApprovalCardTitle")
        title_row.addWidget(title)

        self.dialog_risk_badge = QLabel(risk_level.title())
        self.dialog_risk_badge.setObjectName("ApprovalRiskBadge")
        self.dialog_risk_badge.setProperty("riskLevel", risk_level)
        style = self.dialog_risk_badge.style()
        if style is not None:
            style.unpolish(self.dialog_risk_badge)
            style.polish(self.dialog_risk_badge)
        title_row.addWidget(self.dialog_risk_badge, 0, Qt.AlignVCenter)
        title_row.addStretch(1)
        hero_layout.addLayout(title_row)

        noun = "action" if len(tools) == 1 else "actions"
        summary_label = QLabel(f"The agent is paused. Review {len(tools)} protected {noun}.")
        summary_label.setObjectName("ApprovalCardSummary")
        summary_label.setWordWrap(True)
        hero_layout.addWidget(summary_label)

        impacts_label = QLabel(f"Will affect: {', '.join(impacts)}")
        impacts_label.setObjectName("ApprovalCardImpacts")
        impacts_label.setWordWrap(True)
        impacts_label.setVisible(bool(impacts))
        hero_layout.addWidget(impacts_label)
        layout.addWidget(hero_card)

        scroll = QScrollArea()
        scroll.setObjectName("ApprovalDialogScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(8)
        for tool in tools:
            card = QFrame()
            card.setObjectName("ApprovalToolCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 12, 12, 12)
            card_layout.setSpacing(4)

            tool_name = str(tool.get("name") or tool.get("display") or "tool").strip() or "tool"
            tool_args = dict(tool.get("args") or {})
            labels = build_tool_ui_labels(tool_name, tool_args, phase="finished")

            name_label = QLabel(labels.get("title") or str(tool.get("display") or tool_name))
            name_label.setObjectName("ApprovalToolTitle")
            card_layout.addWidget(name_label)

            subtitle = str(labels.get("subtitle", "") or "").strip()
            if subtitle:
                subtitle_label = QLabel(subtitle)
                subtitle_label.setObjectName("ApprovalToolSubtitle")
                subtitle_label.setWordWrap(True)
                card_layout.addWidget(subtitle_label)

            args_view = CopySafePlainTextEdit()
            args_view.setObjectName("ApprovalDetailView")
            args_view.setReadOnly(True)
            args_view.setPlainText(format_approval_detail_text(tool_args))
            _sync_plain_text_height(args_view, min_lines=6, max_lines=12, extra_padding=18)
            card_layout.addWidget(CollapsibleSection("Details", args_view, expanded=len(tools) == 1))
            container_layout.addWidget(card)
        container_layout.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox()
        approve_button = QPushButton("Approve")
        approve_button.setObjectName("PrimaryButton")
        always_button = QPushButton("Always allow")
        always_button.setObjectName("SecondaryButton")
        deny_button = QPushButton("Deny")
        deny_button.setObjectName("DangerButton")
        buttons.addButton(approve_button, QDialogButtonBox.AcceptRole)
        buttons.addButton(always_button, QDialogButtonBox.ActionRole)
        buttons.addButton(deny_button, QDialogButtonBox.RejectRole)
        layout.addWidget(buttons)

        approve_button.setDefault(True)
        approve_button.clicked.connect(self._approve)
        always_button.clicked.connect(self._always)
        deny_button.clicked.connect(self._deny)

    def _approve(self) -> None:
        self.choice = (True, False)
        self.accept()

    def _always(self) -> None:
        self.choice = (True, True)
        self.accept()

    def _deny(self) -> None:
        self.choice = (False, False)
        self.reject()
