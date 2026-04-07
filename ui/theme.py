from __future__ import annotations

# --- Claude.ai Neutral-Warm Dark Theme ---
ACCENT_BLUE = "#A8A49E"          # Нейтральный тёплый серый акцент
ACCENT_BLUE_SOFT = "#2A2825"     # Очень слабый тёплый оттенок для hover

TEXT_PRIMARY = "#ECEAE6"         # Почти белый, чуть тёплый
TEXT_SECONDARY = "#9E9A94"       # Нейтральный серо-тёплый
TEXT_MUTED = "#72706A"           # Приглушённый, едва тёплый
TEXT_DIM = "#524F4A"             # Dim, нейтральный

# Основные фоны — нейтральные с минимальным тёплым подтоном
SURFACE_BG = "#1E1D1B"          # Главный фон (почти нейтральный, чуть теплее чёрного)
SURFACE_CARD = "#282623"        # Карточки — +2 ед. тепла от фона
SURFACE_ALT = "#38352F"         # Кнопки и выделения
BORDER = "#38352F"
SEPARATOR = "#282623"

AMBER_WARNING = "#D97706"
ERROR_RED = "#EF4444"
SUCCESS_GREEN = "#10B981"
CODE_BG = "#161514"             # Фон кода
CODE_TEXT = "#ECEAE6"

_COMPOSER_BG = "#282623"
_COMPOSER_BORDER = "#38352F"
_SEND_BTN_BG = "#ECEAE6"
_SEND_BTN_HOVER = "#FFFFFF"
_SEND_BTN_DISABLED = "#38352F"

APP_FONT_FAMILY = "Segoe UI"
MONO_FONT_FAMILY = "Cascadia Mono"

SOFT_RADIUS_XS = 2
SOFT_RADIUS_SM = 4
SOFT_RADIUS_MD = 6
SOFT_RADIUS_LG = 15

def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def blend_hex(start_hex: str, end_hex: str, factor: float) -> str:
    factor = max(0.0, min(1.0, factor))
    start = _hex_to_rgb(start_hex)
    end = _hex_to_rgb(end_hex)
    blended = tuple(
        round(start[index] + (end[index] - start[index]) * factor)
        for index in range(3)
    )
    return "#{:02X}{:02X}{:02X}".format(*blended)


def build_stylesheet() -> str:
    transcript_panel_bg = blend_hex(SURFACE_CARD, SURFACE_ALT, 0.18)
    transcript_panel_border = blend_hex(BORDER, "#FFFFFF", 0.08)
    transcript_panel_hover = blend_hex(SURFACE_CARD, SURFACE_ALT, 0.32)
    tool_panel_bg = blend_hex(SURFACE_CARD, "#FFFFFF", 0.04)
    tool_panel_border = blend_hex(tool_panel_bg, "#FFFFFF", 0.04)
    tool_panel_hover = blend_hex(tool_panel_bg, "#FFFFFF", 0.05)
    tool_code_bg = blend_hex(SURFACE_CARD, "#FFFFFF", 0.07)
    tool_toggle_text = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.26)
    tool_toggle_hover_text = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.52)
    tool_call_idle = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.18)
    tool_call_hover = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.42)
    return f"""
    QWidget {{
        background: {SURFACE_BG};
        color: {TEXT_PRIMARY};
        font-family: "{APP_FONT_FAMILY}";
        font-size: 10pt;
    }}

    QMainWindow {{
        background: {SURFACE_BG};
    }}

    QFrame#SidebarCard,
    QFrame#StatusCard,
    QFrame#NoticeCard {{
        background: {SURFACE_CARD};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QFrame#TranscriptMetaChip {{
    background: transparent;
    border: none;
    }}
    
    QFrame#ApprovalCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.02)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QFrame#UserBubble {{
        background: {blend_hex(SURFACE_CARD, ACCENT_BLUE_SOFT, 0.32)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QFrame#ImageAttachmentCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.05)};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QLabel#ImageAttachmentThumb {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.03)};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QToolButton#ImageAttachmentRemoveButton {{
        background: {blend_hex(SURFACE_BG, "#FFFFFF", 0.08)};
        border: none;
        border-radius: {SOFT_RADIUS_XS + 6}px;
        padding: 2px;
    }}

    QToolButton#ImageAttachmentRemoveButton:hover {{
        background: {blend_hex(SURFACE_BG, "#FFFFFF", 0.16)};
    }}

    QDialog#InfoPopup {{
        background: {SURFACE_CARD};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QDialog#ModelSettingsDialog {{
        background: {SURFACE_BG};
        border: none;
    }}

    QLabel#ModelSettingsTitle {{
        color: {TEXT_PRIMARY};
        font-weight: 700;
        font-size: 12.5pt;
        padding-left: 1px;
    }}

    QLabel#ModelSettingsSubtitle {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.28)};
        font-size: 9.3pt;
        padding-left: 1px;
        padding-bottom: 2px;
    }}

    QLabel#ModelSettingsMeta {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.26)};
        font-size: 8.9pt;
        padding-left: 1px;
    }}

    QFrame#ModelSettingsPane {{
        background: {SURFACE_CARD};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QFrame#ModelSettingsFormCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.04)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QListWidget#ModelProfileList {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.02)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 2px;
        outline: none;
    }}

    QListWidget#ModelProfileList::item {{
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 6px 8px;
        margin: 1px 0px;
        color: {TEXT_PRIMARY};
    }}

    QListWidget#ModelProfileList::item:selected {{
        background: {blend_hex(ACCENT_BLUE_SOFT, SURFACE_ALT, 0.42)};
        color: {TEXT_PRIMARY};
    }}

    QWidget#ModelProfileItemWidget {{
        background: transparent;
    }}

    QLabel#ModelProfileItemTitle {{
        color: {TEXT_PRIMARY};
        font-size: 10pt;
        font-weight: 500;
        background: transparent;
    }}

    QLabel#ModelProfileItemActive {{
        color: {SUCCESS_GREEN};
        font-size: 9.1pt;
        font-weight: 600;
        background: transparent;
    }}

    QLabel#ModelProfileItemMeta {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.24)};
        font-size: 8.8pt;
        background: transparent;
    }}

    QCheckBox#ModelProfileEnabledSwitch {{
        background: transparent;
        spacing: 0px;
        padding: 0px;
    }}

    QCheckBox#ModelProfileEnabledSwitch::indicator {{
        width: 26px;
        height: 14px;
        border-radius: 7px;
        background: {blend_hex(SURFACE_ALT, "#000000", 0.08)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.08)};
    }}

    QCheckBox#ModelProfileEnabledSwitch::indicator:hover {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.04)};
    }}

    QCheckBox#ModelProfileEnabledSwitch::indicator:checked {{
        background: {blend_hex(SUCCESS_GREEN, SURFACE_ALT, 0.28)};
        border: 1px solid {blend_hex(SUCCESS_GREEN, "#FFFFFF", 0.18)};
    }}

    QCheckBox#ModelProfileEnabledSwitch::indicator:checked:hover {{
        background: {blend_hex(SUCCESS_GREEN, "#FFFFFF", 0.16)};
    }}

    QDialog#ModelSettingsDialog QLineEdit,
    QDialog#ModelSettingsDialog QComboBox {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.055)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        min-height: 28px;
        padding: 2px 8px;
        color: {TEXT_PRIMARY};
        selection-background-color: {blend_hex(ACCENT_BLUE_SOFT, ACCENT_BLUE, 0.28)};
    }}

    QDialog#ModelSettingsDialog QLineEdit:disabled,
    QDialog#ModelSettingsDialog QComboBox:disabled {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.02)};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
    }}

    QDialog#ModelSettingsDialog QComboBox::drop-down {{
        border: none;
        width: 18px;
        padding-right: 2px;
    }}

    QDialog#ModelSettingsDialog QComboBox QAbstractItemView {{
        background: {SURFACE_CARD};
        border: none;
        selection-background-color: {blend_hex(ACCENT_BLUE_SOFT, ACCENT_BLUE, 0.28)};
        color: {TEXT_PRIMARY};
    }}

    QLabel#ModelSettingsFieldLabel {{
        background: {blend_hex(SURFACE_CARD, "#000000", 0.22)};
        color: {TEXT_PRIMARY};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        min-height: 26px;
        padding: 2px 8px;
        font-size: 9.4pt;
        font-weight: 500;
    }}

    QPushButton#SettingsAddButton,
    QPushButton#SettingsDeleteButton {{
        min-height: 28px;
        font-size: 9.2pt;
    }}

    QDialog#ModelSettingsDialog QDialogButtonBox {{
        border-top: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.06)};
        padding-top: 8px;
    }}

    QWidget#TranscriptContainer {{
        background: transparent;
    }}

    QFrame#UserChoiceCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.04)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QFrame#TranscriptRow,
    QFrame#ToolRow,
    QFrame#FlatNoticeRow,
    QFrame#InlineStatusRow {{
        background: transparent;
        border: none;
    }}

    QLabel#SectionTitle {{
        color: {TEXT_PRIMARY};
        font-weight: 600;
        font-size: 12pt;
    }}

    QLabel#SidebarSectionTitle {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
        font-weight: 600;
        font-size: 11pt;
        padding-left: 2px;
    }}

    QLabel#TranscriptRole {{
        color: {TEXT_MUTED};
        font-weight: normal;
        font-size: 9.5pt;
    }}

    QLabel#TranscriptBody {{
        color: {TEXT_PRIMARY};
        font-size: 11pt;
        background: transparent;
    }}

    QLabel#TranscriptMeta {{
        color: {TEXT_MUTED};
        font-size: 8.5pt;
        background: transparent;
    }}

    QLabel#ComposerCapabilityBadge {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.18)};
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.04)};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 4px 8px;
        font-size: 8.6pt;
        font-weight: 600;
    }}

    QLabel#ComposerNoticeLabel {{
        color: {TEXT_MUTED};
        background: transparent;
        font-size: 8.8pt;
        padding-left: 2px;
    }}

    QLabel#ComposerNoticeLabel[severity="warning"] {{
        color: {AMBER_WARNING};
    }}

    QLabel#ComposerNoticeLabel[severity="error"] {{
        color: {ERROR_RED};
    }}

    QLabel#ComposerNoticeLabel[severity="success"] {{
        color: {SUCCESS_GREEN};
    }}

    QLabel#UserChoiceCardTitle {{
        color: {TEXT_PRIMARY};
        font-size: 10.2pt;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#UserChoiceCardQuestion {{
        color: {TEXT_PRIMARY};
        font-size: 10pt;
        background: transparent;
    }}

    QLabel#UserChoiceCardHint {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.34)};
        font-size: 8.8pt;
        background: transparent;
    }}

    QLabel#MutedText,
    QLabel#MetaText {{
        color: {TEXT_MUTED};
    }}

    QLabel#MetaText[severity="error"] {{
        color: {ERROR_RED};
    }}

    QLabel#WarningText {{
        color: {AMBER_WARNING};
    }}

    QLabel#StatusLabel {{
        color: {TEXT_PRIMARY};
        font-weight: 600;
    }}

    QToolButton#InlineStatusSpinner {{
        background: transparent;
        border: none;
        padding: 0px;
    }}

    QTabWidget::pane {{
        border: none;
        background: transparent;
    }}

    QTabBar::tab {{
        background: transparent;
        border: none;
        padding: 3px 6px;
        margin-right: 2px;
        color: {TEXT_MUTED};
        border-radius: {SOFT_RADIUS_SM}px;
        font-size: 8.5pt;
    }}

    QTabBar::tab:selected {{
        background: {SURFACE_ALT};
        color: {TEXT_PRIMARY};
        border: none;
    }}

    QToolBar {{
        background: {SURFACE_CARD};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        spacing: 3px;
        padding: 1px 3px;
    }}

    QToolButton,
    QPushButton {{
        background: {SURFACE_ALT};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 3px 6px;
        color: {TEXT_PRIMARY};
    }}

    QToolButton:hover,
    QPushButton:hover {{
        background: {blend_hex(SURFACE_ALT, ACCENT_BLUE_SOFT, 0.55)};
        border: none;
    }}

    QToolButton:pressed,
    QPushButton:pressed {{
        background: {blend_hex(SURFACE_ALT, ACCENT_BLUE_SOFT, 0.8)};
    }}

    QPushButton#PrimaryButton {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.08)};
        color: white;
        border: none;
        font-weight: 600;
    }}

    QPushButton#DangerButton {{
        background: {ERROR_RED};
        color: white;
        border: none;
        font-weight: 600;
    }}

    QPushButton:disabled,
    QToolButton:disabled {{
        color: {TEXT_DIM};
        background: {SURFACE_ALT};
        border: none;
    }}

    QToolButton#DisclosureButton,
    QPushButton#DisclosureButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 0px 2px;
        min-height: 18px;
        font-size: 9pt;
        color: {TEXT_MUTED};
        text-align: left;
    }}

    QToolButton#DisclosureButton:hover,
    QPushButton#DisclosureButton:hover {{
        background: transparent;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.45)};
    }}

    QFrame#ToolExpandablePanel {{
        background: {tool_panel_bg};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 5px 8px;
    }}

    QFrame#CliExecPanel {{
        background: {tool_panel_bg};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QLabel#CliExecHeader {{
        background: transparent;
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.1)};
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9.3pt;
        font-weight: 500;
    }}

    QPlainTextEdit#CliExecOutput {{
        background: {tool_code_bg};
        color: {CODE_TEXT};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 7px 10px 8px 10px;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
    }}

    QFrame#DiffPanel {{
        background: {tool_code_bg};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QLabel#DiffHeaderPath {{
        background: transparent;
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.08)};
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9.2pt;
        font-weight: 500;
    }}

    QLabel#DiffStatAdded {{
        background: transparent;
        color: #7EDB8B;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
        font-weight: 600;
    }}

    QLabel#DiffStatRemoved {{
        background: transparent;
        color: #F08F8F;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
        font-weight: 600;
    }}

    QPlainTextEdit#DiffCodeView {{
        background: transparent;
        color: {CODE_TEXT};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 6px 10px 8px 10px;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
    }}

    QWidget#ToolExpandableContent {{
        background: transparent;
        border: none;
    }}

    QPushButton#ToolExpandableToggle {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 5px 8px;
        min-height: 20px;
        font-size: 9.2pt;
        font-weight: 400;
        color: {tool_toggle_text};
        text-align: left;
    }}

    QPushButton#ToolExpandableToggle:hover {{
        background: {tool_panel_hover};
        color: {tool_toggle_hover_text};
    }}

    QPushButton#ToolExpandableToggle:checked {{
        background: {tool_panel_hover};
        color: {TEXT_PRIMARY};
    }}

    QToolButton#CodeCopyButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 2px 6px;
        color: {tool_toggle_text};
        font-size: 8.8pt;
    }}

    QToolButton#CodeCopyButton:hover {{
        background: {tool_panel_hover};
        color: {TEXT_PRIMARY};
    }}

    QPlainTextEdit,
    QTextBrowser,
    QListView,
    QListWidget,
    QTreeWidget {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.01)};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        selection-background-color: {blend_hex(ACCENT_BLUE_SOFT, ACCENT_BLUE, 0.25)};
        selection-color: {TEXT_PRIMARY};
    }}

    QListView#SessionListView {{
        background: transparent;
        border: none;
        outline: none;
        padding: 0px;
    }}

    /* ---- Composer pill ---- */
    QFrame#ComposerPill {{
        background: {_COMPOSER_BG};
        border: none;
        border-radius: 18px;
    }}

    QPushButton#UserChoiceOptionButton,
    QPushButton#UserChoiceCustomButton {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.03)};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 10px 12px;
        color: {TEXT_PRIMARY};
        text-align: left;
        font-size: 9.6pt;
        font-weight: 500;
    }}

    QPushButton#UserChoiceOptionButton:hover,
    QPushButton#UserChoiceCustomButton:hover {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.09)};
    }}

    QPushButton#UserChoiceOptionButton:pressed,
    QPushButton#UserChoiceCustomButton:pressed {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.14)};
    }}

    QPushButton#UserChoiceOptionButton[recommended="true"] {{
        background: {blend_hex(SURFACE_ALT, ACCENT_BLUE, 0.18)};
        color: {TEXT_PRIMARY};
    }}

    QPushButton#UserChoiceOptionButton:disabled,
    QPushButton#UserChoiceCustomButton:disabled {{
        background: {blend_hex(SURFACE_ALT, SURFACE_CARD, 0.35)};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.18)};
    }}

    QPlainTextEdit#ComposerEdit {{
        background: transparent;
        border: none;
        border-radius: 10px;
        padding: 4px 2px 2px 2px;
        font-size: 10.5pt;
    }}

    QPlainTextEdit#ComposerEdit QScrollBar:vertical {{
        background: transparent;
        border: none;
        width: 6px;
        margin: 2px 0 2px 0;
    }}

    QPlainTextEdit#ComposerEdit QScrollBar::handle:vertical {{
        background: {blend_hex(TEXT_MUTED, "#FFFFFF", 0.06)};
        border-radius: {SOFT_RADIUS_XS}px;
        min-height: 16px;
    }}

    QPlainTextEdit#ComposerEdit QScrollBar::add-line:vertical,
    QPlainTextEdit#ComposerEdit QScrollBar::sub-line:vertical,
    QPlainTextEdit#ComposerEdit QScrollBar::add-page:vertical,
    QPlainTextEdit#ComposerEdit QScrollBar::sub-page:vertical {{
        background: transparent;
        border: none;
        height: 0px;
    }}

    QPushButton#ComposerSendButton {{
        background: {_SEND_BTN_BG};
        color: #08090B;
        border: 1px solid {blend_hex(_SEND_BTN_BG, "#000000", 0.18)};
        border-radius: 15px;
        min-width: 32px;
        max-width: 32px;
        min-height: 32px;
        max-height: 32px;
        padding: 0px;
        font-weight: 700;
    }}

    QPushButton#ComposerSendButton:hover {{
        background: {_SEND_BTN_HOVER};
        border: 1px solid {blend_hex(_SEND_BTN_HOVER, "#000000", 0.20)};
    }}

    QPushButton#ComposerSendButton:pressed {{
        background: #C8CACF;
        border: 1px solid {blend_hex("#C8CACF", "#000000", 0.22)};
    }}

    QPushButton#ComposerSendButton:disabled {{
        background: {_SEND_BTN_DISABLED};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.2)};
        border: 1px solid {blend_hex(_SEND_BTN_DISABLED, "#FFFFFF", 0.08)};
    }}

    QToolButton#ComposerAttachButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 0px;
    }}

    QToolButton#ComposerAttachButton:hover {{
        background: {blend_hex(_COMPOSER_BG, SURFACE_ALT, 0.32)};
        border: none;
    }}

    QPushButton#ComposerMetaChip {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 2px 8px;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
        font-size: 8.8pt;
        font-weight: 500;
    }}

    QPushButton#ComposerMetaChip:disabled {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
        background: transparent;
        border: 1px solid transparent;
    }}

    QToolButton#ComposerMetaChipButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 2px 8px;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
        font-size: 8.8pt;
        font-weight: 500;
    }}

    QToolButton#ComposerMetaChipButton:hover {{
        background: {blend_hex(_COMPOSER_BG, SURFACE_ALT, 0.24)};
    }}

    QToolButton#ComposerMetaChipButton:disabled {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
        background: transparent;
        border: 1px solid transparent;
    }}

    QLabel#ComposerNoModelText {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.32)};
        font-size: 8.8pt;
    }}

    QPushButton#ComposerOpenSettingsButton {{
        background: transparent;
        border: 1px solid {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.15)};
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 2px 8px;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.45)};
        font-size: 8.8pt;
    }}

    QPushButton#ComposerOpenSettingsButton:hover {{
        background: {blend_hex(_COMPOSER_BG, SURFACE_ALT, 0.24)};
        border: 1px solid {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.3)};
    }}

    QToolButton#ComposerGhostButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 0px;
    }}

    QToolButton#ComposerGhostButton:disabled {{
        background: transparent;
        border: 1px solid transparent;
    }}

    QFrame#ComposerMentionPopup {{
        background: {_COMPOSER_BG};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QListWidget#ComposerMentionList {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 2px;
        outline: none;
        font-size: 11.2pt;
    }}

    QListWidget#ComposerMentionList::item {{
        background: transparent;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 8px 10px;
        margin: 1px 0px;
        color: {TEXT_PRIMARY};
    }}

    QListWidget#ComposerMentionList::item:selected {{
        background: {blend_hex(SURFACE_ALT, ACCENT_BLUE_SOFT, 0.35)};
        color: {TEXT_PRIMARY};
    }}

    QTextBrowser#AssistantBody {{
        background: transparent;
        border: none;
        border-radius: 0px;
        padding: 0px;
        font-family: "{APP_FONT_FAMILY}";
        font-size: 11.5pt;
    }}

    QPushButton#ToolCallButton {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 0px;
        color: {tool_call_idle};
        font-size: 11.5pt;
        font-weight: 400;
        text-align: left;
    }}

    QPushButton#ToolCallButton:hover {{
        background: transparent;
        color: {tool_call_hover};
        border: none;
    }}

    QPushButton#ToolCallButton:checked {{
        background: transparent;
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.08)};
        border: none;
    }}

    QPlainTextEdit#CodeView,
    QPlainTextEdit#InlineCodeView {{
        background: {tool_code_bg};
        color: {CODE_TEXT};
        border: none;
        border-radius: {SOFT_RADIUS_SM}px;
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
        padding: 8px 10px;
    }}

    QScrollArea {{
        border: none;
        background: transparent;
    }}

    QScrollBar:vertical {{
        background: transparent;
        width: 6px;
        margin: 4px 0 4px 0;
    }}

    QScrollBar:horizontal {{
        background: transparent;
        height: 6px;
        margin: 0 4px 0 4px;
    }}

    QScrollBar::handle:vertical {{
        background: {blend_hex(TEXT_MUTED, "#FFFFFF", 0.06)};
        border-radius: {SOFT_RADIUS_XS}px;
        min-height: 16px;
    }}

    QScrollBar::handle:horizontal {{
        background: {blend_hex(TEXT_MUTED, "#FFFFFF", 0.06)};
        border-radius: {SOFT_RADIUS_XS}px;
        min-width: 16px;
    }}

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{
        background: transparent;
        border: none;
        height: 0px;
    }}

    QScrollBar::add-line:horizontal,
    QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal,
    QScrollBar::sub-page:horizontal {{
        background: transparent;
        border: none;
        width: 0px;
    }}

    QMenuBar {{
        background: {SURFACE_BG};
        color: {TEXT_SECONDARY};
    }}

    QMenuBar::item {{
        background: transparent;
        padding: 4px 8px;
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QMenu {{
        background: {SURFACE_CARD};
        color: {TEXT_PRIMARY};
        border: none;
        padding: 6px;
    }}

    QMenuBar::item:selected,
    QMenu::item:selected {{
        background: {ACCENT_BLUE_SOFT};
        border-radius: {SOFT_RADIUS_SM}px;
    }}

    QStatusBar {{
        background: {blend_hex(SURFACE_BG, SURFACE_CARD, 0.28)};
        color: {TEXT_MUTED};
        border-top: 1px solid {BORDER};
        min-height: 22px;
        padding-left: 6px;
    }}

    QStatusBar::item {{
        border: none;
    }}

    QLabel#StatusBarMeta {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.16)};
        font-size: 9pt;
        padding-right: 4px;
    }}

    QLabel#StatusBarState {{
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.12)};
        font-size: 9.6pt;
        font-weight: 600;
        padding-left: 2px;
    }}
    """

