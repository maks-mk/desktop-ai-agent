from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
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
    QSizePolicy,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.model_profiles import ALLOWED_PROVIDERS, generate_profile_id, normalize_profiles_payload, sanitize_profile_id
from ui.theme import TEXT_MUTED, TEXT_PRIMARY
from .foundation import CollapsibleSection, _fa_icon, _make_mono_font


class ModelSettingsDialog(QDialog):
    def __init__(self, payload: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ModelSettingsDialog")
        self.setWindowTitle("Model Settings")
        self.setModal(True)
        self.resize(900, 520)
        self.setMinimumSize(820, 450)

        normalized = normalize_profiles_payload(payload or {})
        self._profiles: list[dict[str, Any]] = [dict(item) for item in normalized.get("profiles", [])]
        self._active_profile = str(normalized.get("active_profile") or "").strip()
        self._name_manual_flags: list[bool] = []
        self._selected_row = -1
        self._loading_form = False
        self._result_payload = normalized
        self._name_manual_flags = self._compute_initial_name_manual_flags()

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        header_title = QLabel("Model Profiles")
        header_title.setObjectName("ModelSettingsTitle")
        root.addWidget(header_title)

        header_hint = QLabel("Manage provider/model/API credentials. Active profile is used for new runs.")
        header_hint.setObjectName("ModelSettingsSubtitle")
        header_hint.setWordWrap(True)
        root.addWidget(header_hint)

        active_name = str(self._active_profile or "").strip() or "none"
        self.active_profile_label = QLabel(f"Active now: {active_name}")
        self.active_profile_label.setObjectName("ModelSettingsMeta")
        self.active_profile_label.setWordWrap(True)
        root.addWidget(self.active_profile_label)

        body = QHBoxLayout()
        body.setSpacing(12)

        left_container = QFrame()
        left_container.setObjectName("ModelSettingsPane")
        left = QVBoxLayout(left_container)
        left.setContentsMargins(10, 10, 10, 10)
        left.setSpacing(6)
        left_label = QLabel("Profiles")
        left_label.setObjectName("SectionTitle")
        left.addWidget(left_label)

        self.profile_list = QListWidget()
        self.profile_list.setObjectName("ModelProfileList")
        self.profile_list.setMinimumWidth(280)
        self.profile_list.currentRowChanged.connect(self._on_selection_changed)
        left.addWidget(self.profile_list, 1)

        left_buttons = QHBoxLayout()
        left_buttons.setSpacing(6)
        self.add_button = QPushButton("Add")
        self.add_button.setObjectName("SettingsAddButton")
        self.add_button.setIcon(_fa_icon("fa5s.plus", color=TEXT_PRIMARY, size=11))
        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("SettingsDeleteButton")
        self.delete_button.setIcon(_fa_icon("fa5s.trash", color=TEXT_PRIMARY, size=11))
        left_buttons.addWidget(self.add_button)
        left_buttons.addWidget(self.delete_button)
        left.addLayout(left_buttons)

        left_hint = QLabel("Tip: leave Name empty to auto-generate it from Model.")
        left_hint.setObjectName("ModelSettingsMeta")
        left_hint.setWordWrap(True)
        left.addWidget(left_hint)

        right_container = QFrame()
        right_container.setObjectName("ModelSettingsPane")
        right = QVBoxLayout(right_container)
        right.setContentsMargins(10, 10, 10, 10)
        right.setSpacing(8)
        right_label = QLabel("Profile")
        right_label.setObjectName("SectionTitle")
        right.addWidget(right_label)

        self.form_hint = QLabel("Select a profile and edit fields on the right.")
        self.form_hint.setObjectName("ModelSettingsMeta")
        self.form_hint.setWordWrap(True)
        right.addWidget(self.form_hint)

        form_frame = QFrame()
        form_frame.setObjectName("ModelSettingsFormCard")
        form_layout = QFormLayout(form_frame)
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setHorizontalSpacing(10)
        form_layout.setVerticalSpacing(8)
        form_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form_layout.setFormAlignment(Qt.AlignTop)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("profile-id")
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["openai", "gemini"])
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("e.g. openai/gpt-oss-120b or gemini-1.5-flash")
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("API key")
        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("https://api.openai.com/v1")
        self.supports_images_checkbox = QCheckBox("Поддержка изображений")
        self.supports_images_checkbox.setObjectName("ModelSupportsImagesCheckbox")
        self.supports_images_checkbox.setToolTip("Разрешает прикреплять изображения для этого профиля.")

        label_width = 76
        name_label = QLabel("Name")
        provider_label = QLabel("Provider")
        model_label = QLabel("Model")
        api_key_label = QLabel("API Key")
        base_url_label = QLabel("Base URL")
        images_label = QLabel("Images")
        for label in (name_label, provider_label, model_label, api_key_label, base_url_label, images_label):
            label.setObjectName("ModelSettingsFieldLabel")
            label.setFixedWidth(label_width)

        form_layout.addRow(name_label, self.name_edit)
        form_layout.addRow(provider_label, self.provider_combo)
        form_layout.addRow(model_label, self.model_edit)
        form_layout.addRow(api_key_label, self.api_key_edit)
        form_layout.addRow(base_url_label, self.base_url_edit)
        form_layout.addRow(images_label, self.supports_images_checkbox)
        right.addWidget(form_frame, 1)

        left_container.setMinimumWidth(320)
        body.addWidget(left_container, 5)
        body.addWidget(right_container, 9)
        root.addLayout(body, 1)

        actions = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        actions.setObjectName("ModelSettingsActions")
        self.save_button = actions.button(QDialogButtonBox.StandardButton.Save)
        if self.save_button is not None:
            self.save_button.setObjectName("PrimaryButton")
            self.save_button.setIcon(_fa_icon("fa5s.save", color="#FFFFFF", size=11))
            self.save_button.setMinimumHeight(30)
        self.cancel_button = actions.button(QDialogButtonBox.StandardButton.Cancel)
        if self.cancel_button is not None:
            self.cancel_button.setMinimumHeight(30)
        root.addWidget(actions)

        self.add_button.clicked.connect(self._add_profile)
        self.delete_button.clicked.connect(self._delete_selected_profile)
        actions.accepted.connect(self._save_and_accept)
        actions.rejected.connect(self.reject)

        self.name_edit.textEdited.connect(self._on_name_edited)
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        self.model_edit.textChanged.connect(self._on_model_changed)
        self.api_key_edit.textChanged.connect(self._on_form_changed)
        self.base_url_edit.textChanged.connect(self._on_form_changed)
        self.supports_images_checkbox.checkStateChanged.connect(self._on_form_changed)

        self._refresh_profile_list()
        if self.profile_list.count() > 0:
            self.profile_list.setCurrentRow(self._preferred_row_for_open())
        else:
            self._set_form_enabled(False)

    def result_payload(self) -> dict[str, Any]:
        return dict(self._result_payload)

    def _current_row(self) -> int:
        return self.profile_list.currentRow()

    def _set_form_enabled(self, enabled: bool) -> None:
        for widget in (
            self.name_edit,
            self.provider_combo,
            self.model_edit,
            self.api_key_edit,
            self.base_url_edit,
            self.supports_images_checkbox,
        ):
            widget.setEnabled(enabled)
        self.delete_button.setEnabled(enabled)
        if enabled:
            self._update_base_url_field_state(self.provider_combo.currentText())

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

    def _build_profile_item_widget(self, profile: dict[str, Any], row: int) -> QWidget:
        container = QWidget()
        container.setObjectName("ModelProfileItemWidget")
        is_enabled = bool(profile.get("enabled", True))
        container.setProperty("disabledProfile", not is_enabled)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(4, 3, 2, 3)
        layout.setSpacing(10)

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
        title_label.setMinimumWidth(0)
        title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        first_row.addWidget(title_label, 0, Qt.AlignLeft | Qt.AlignVCenter)

        if is_active:
            active_label = QLabel("• active")
            active_label.setObjectName("ModelProfileItemActive")
            first_row.addWidget(active_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
        elif not is_enabled:
            disabled_label = QLabel("• disabled")
            disabled_label.setObjectName("ModelProfileItemMeta")
            first_row.addWidget(disabled_label, 0, Qt.AlignLeft | Qt.AlignVCenter)

        first_row.addStretch(1)
        text_column.addLayout(first_row)

        provider = str(profile.get("provider") or "").strip()
        model_name = str(profile.get("model") or "").strip()
        details = " · ".join(part for part in (provider, model_name) if part)
        details_label = QLabel(details)
        details_label.setObjectName("ModelProfileItemMeta")
        details_label.setEnabled(is_enabled)
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
        enabled_switch.setFixedSize(QSize(30, 18))
        enabled_switch.pressed.connect(lambda target_row=row: self.profile_list.setCurrentRow(target_row))
        enabled_switch.toggled.connect(lambda checked, target_row=row: self._toggle_profile_enabled(target_row, checked))
        layout.addWidget(enabled_switch, 0, Qt.AlignRight | Qt.AlignVCenter)
        container.adjustSize()
        return container

    def _refresh_profile_list(self, preferred_row: int | None = None) -> None:
        self.profile_list.blockSignals(True)
        self.profile_list.clear()
        for row, profile in enumerate(self._profiles):
            item_widget = self._build_profile_item_widget(profile, row)
            item = QListWidgetItem("")
            item.setData(Qt.UserRole, self._display_name(profile))
            widget_hint = item_widget.sizeHint()
            item_height = max(48, widget_hint.height() + 4)
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
            return

        row = preferred_row if preferred_row is not None else self._preferred_row_for_open()
        row = max(0, min(row, len(self._profiles) - 1))
        self.profile_list.setCurrentRow(row)

    def _preferred_row_for_open(self) -> int:
        active_id = str(self._active_profile or "").strip()
        if active_id:
            for index, profile in enumerate(self._profiles):
                if str(profile.get("id") or "").strip() == active_id:
                    return index
        return self._current_row() if self._current_row() >= 0 else 0

    def _sync_form_to_profile(self, row: int) -> None:
        if row < 0 or row >= len(self._profiles):
            self._selected_row = -1
            self._set_form_enabled(False)
            self.form_hint.setText("Select a profile and edit fields on the right.")
            return
        profile = self._profiles[row]
        self._loading_form = True
        self._set_form_enabled(True)
        self.name_edit.setText(str(profile.get("id", "")))
        provider = str(profile.get("provider", "openai")).strip().lower()
        if provider not in ALLOWED_PROVIDERS:
            provider = "openai"
        self.provider_combo.setCurrentText(provider)
        self.model_edit.setText(str(profile.get("model", "")))
        self.api_key_edit.setText(str(profile.get("api_key", "")))
        self.base_url_edit.setText(str(profile.get("base_url", "")))
        self.supports_images_checkbox.setChecked(bool(profile.get("supports_image_input")))
        self._update_base_url_field_state(provider)
        self._loading_form = False
        self._selected_row = row
        profile_id = str(profile.get("id") or "").strip() or "(unnamed)"
        status = "enabled" if bool(profile.get("enabled", True)) else "disabled"
        self.form_hint.setText(f"Editing profile: {profile_id} ({status})")
        active_name = str(self._active_profile or "").strip() or "none"
        self.active_profile_label.setText(f"Active now: {active_name}")

    def _sync_current_profile_from_form(self, row: int | None = None) -> None:
        if self._loading_form:
            return
        target_row = self._current_row() if row is None else row
        if target_row < 0 or target_row >= len(self._profiles):
            return
        provider = str(self.provider_combo.currentText() or "").strip().lower()
        if provider not in ALLOWED_PROVIDERS:
            provider = "openai"
        base_url = str(self.base_url_edit.text() or "").strip() if provider == "openai" else ""
        self._profiles[target_row] = {
            "id": str(self.name_edit.text() or "").strip(),
            "provider": provider,
            "model": str(self.model_edit.text() or "").strip(),
            "api_key": str(self.api_key_edit.text() or "").strip(),
            "base_url": base_url,
            "supports_image_input": self.supports_images_checkbox.isChecked(),
            "enabled": bool(self._profiles[target_row].get("enabled", True)),
        }
        item = self.profile_list.item(target_row)
        if item is not None:
            item.setText(self._display_name(self._profiles[target_row]))

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
        if isinstance(state, bool):
            is_enabled = bool(state)
        elif isinstance(state, int):
            is_enabled = state == Qt.Checked
        else:
            is_enabled = bool(state)
        self._profiles[row]["enabled"] = is_enabled
        self._reconcile_active_profile()
        self._refresh_profile_list(preferred_row=row)

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
        current_name = str(self.name_edit.text() or "").strip()
        if (not self._name_manual_flags[row]) or not current_name:
            self._loading_form = True
            self.name_edit.setText(self._suggest_unique_id(self.model_edit.text(), row=row))
            self._loading_form = False
        self._on_form_changed()

    def _on_form_changed(self) -> None:
        self._sync_current_profile_from_form()

    def _on_provider_changed(self, provider: str) -> None:
        if self._loading_form:
            return
        self._update_base_url_field_state(provider)
        if str(provider or "").strip().lower() != "openai":
            self._loading_form = True
            self.base_url_edit.clear()
            self._loading_form = False
        self._on_form_changed()

    def _add_profile(self) -> None:
        self._sync_current_profile_from_form(self._selected_row)
        self._profiles.append(
            {
                "id": "",
                "provider": "openai",
                "model": "",
                "api_key": "",
                "base_url": "",
                "supports_image_input": False,
                "enabled": True,
            }
        )
        self._name_manual_flags.append(False)
        self._refresh_profile_list(preferred_row=len(self._profiles) - 1)

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

        self._refresh_profile_list(preferred_row=row)

    def _validated_payload(self) -> dict[str, Any] | None:
        self._sync_current_profile_from_form(self._selected_row)
        profiles: list[dict[str, str]] = []
        used_ids: set[str] = set()

        for idx, profile in enumerate(self._profiles):
            provider = str(profile.get("provider") or "").strip().lower()
            model_name = str(profile.get("model") or "").strip()
            if provider not in ALLOWED_PROVIDERS:
                self.profile_list.setCurrentRow(idx)
                QMessageBox.warning(self, "Validation", "Provider должен быть openai или gemini.")
                return None
            if not model_name:
                self.profile_list.setCurrentRow(idx)
                QMessageBox.warning(self, "Validation", "Model не может быть пустым.")
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

    def _save_and_accept(self) -> None:
        validated = self._validated_payload()
        if validated is None:
            return
        self._result_payload = normalize_profiles_payload(validated)
        self.accept()


class ApprovalDialog(QDialog):
    def __init__(self, payload: dict[str, Any], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.choice: tuple[bool, bool] = (False, False)
        self.setWindowTitle("Confirmation Needed")
        self.setModal(True)
        self.resize(470, 320)
        self.setMinimumSize(420, 280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Protected action requires confirmation")
        title.setStyleSheet("font-weight: 600; font-size: 13pt;")
        layout.addWidget(title)

        summary = payload.get("summary", {})
        impacts = ", ".join(summary.get("impacts", [])) or "local state"
        summary_label = QLabel(
            f"Risk: {summary.get('risk_level', 'unknown')} • Impacts: {impacts} • "
            f"Default: {'approve' if summary.get('default_approve') else 'deny'}"
        )
        summary_label.setStyleSheet(f"color: {TEXT_MUTED};")
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(10)
        for tool in payload.get("tools", []):
            card = QFrame()
            card.setObjectName("ApprovalCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(10, 10, 10, 10)
            card_layout.setSpacing(6)
            name_label = QLabel(tool.get("name", "tool"))
            name_label.setStyleSheet("font-weight: 600;")
            card_layout.addWidget(name_label)
            args_view = QPlainTextEdit()
            args_view.setObjectName("CodeView")
            args_view.setReadOnly(True)
            args_view.setFont(_make_mono_font())
            args_view.setPlainText(json.dumps(tool.get("args", {}), ensure_ascii=False, indent=2))
            args_view.setFixedHeight(78)
            card_layout.addWidget(CollapsibleSection("Arguments", args_view, expanded=False))
            container_layout.addWidget(card)
        container_layout.addStretch(1)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox()
        approve_button = QPushButton(_fa_icon("fa5s.check", color="white", size=12), "Approve")
        approve_button.setObjectName("PrimaryButton")
        deny_button = QPushButton(_fa_icon("fa5s.times", color="white", size=12), "Deny")
        deny_button.setObjectName("DangerButton")
        always_button = QPushButton("Always for this session")
        buttons.addButton(approve_button, QDialogButtonBox.AcceptRole)
        buttons.addButton(always_button, QDialogButtonBox.ActionRole)
        buttons.addButton(deny_button, QDialogButtonBox.RejectRole)
        layout.addWidget(buttons)

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

