from __future__ import annotations

from functools import lru_cache

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


def _build_theme_palette() -> dict[str, str]:
    transcript_panel_bg = blend_hex(SURFACE_CARD, SURFACE_ALT, 0.18)
    tool_panel_bg = blend_hex(SURFACE_CARD, "#FFFFFF", 0.04)
    tool_toggle_text = blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.26)
    return {
        "transcript_panel_bg": transcript_panel_bg,
        "transcript_panel_border": blend_hex(BORDER, "#FFFFFF", 0.08),
        "transcript_panel_hover": blend_hex(SURFACE_CARD, SURFACE_ALT, 0.32),
        "tool_panel_bg": tool_panel_bg,
        "tool_panel_border": blend_hex(tool_panel_bg, "#FFFFFF", 0.04),
        "tool_panel_hover": blend_hex(tool_panel_bg, "#FFFFFF", 0.05),
        "tool_code_bg": blend_hex(SURFACE_CARD, "#FFFFFF", 0.07),
        "tool_toggle_text": tool_toggle_text,
        "tool_toggle_hover_text": blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.52),
        "tool_call_idle": blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.18),
        "tool_call_hover": blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.42),
    }


@lru_cache(maxsize=1)
def build_stylesheet() -> str:
    palette = _build_theme_palette()
    transcript_panel_bg = palette["transcript_panel_bg"]
    transcript_panel_border = palette["transcript_panel_border"]
    transcript_panel_hover = palette["transcript_panel_hover"]
    tool_panel_bg = palette["tool_panel_bg"]
    tool_panel_border = palette["tool_panel_border"]
    tool_panel_hover = palette["tool_panel_hover"]
    tool_code_bg = palette["tool_code_bg"]
    tool_toggle_text = palette["tool_toggle_text"]
    tool_toggle_hover_text = palette["tool_toggle_hover_text"]
    tool_call_idle = palette["tool_call_idle"]
    tool_call_hover = palette["tool_call_hover"]
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
    
    QFrame#ApprovalCard,
    QFrame#ApprovalRequestCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.02)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QLabel#TopStatusChip {{
        color: {TEXT_PRIMARY};
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.06)};
        border: none;
        border-radius: {SOFT_RADIUS_MD + 6}px;
        padding: 4px 10px;
        font-size: 8.8pt;
        font-weight: 600;
    }}

    QLabel#TopStatusChip[statusState="busy"] {{
        background: {blend_hex(ACCENT_BLUE_SOFT, ACCENT_BLUE, 0.24)};
    }}

    QLabel#TopStatusChip[statusState="success"] {{
        background: {blend_hex(SUCCESS_GREEN, SURFACE_ALT, 0.82)};
        color: {blend_hex(SUCCESS_GREEN, TEXT_PRIMARY, 0.18)};
    }}

    QLabel#TopStatusChip[statusState="error"] {{
        background: {blend_hex(ERROR_RED, SURFACE_ALT, 0.82)};
        color: {blend_hex(ERROR_RED, TEXT_PRIMARY, 0.18)};
    }}

    QLabel#TopStatusChip[statusState="idle"] {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.22)};
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
        font-weight: 750;
        font-size: 15pt;
        padding-left: 1px;
    }}

    QLabel#ModelSettingsSubtitle {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.28)};
        font-size: 10.1pt;
        padding-left: 1px;
        padding-bottom: 2px;
    }}

    QLabel#ModelSettingsMeta {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.26)};
        font-size: 9.2pt;
        padding-left: 1px;
    }}

    QLabel#ModelSettingsChip {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.42)};
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.05)};
        border: none;
        border-radius: {SOFT_RADIUS_MD + 6}px;
        padding: 5px 11px;
        font-size: 8.8pt;
        font-weight: 600;
    }}

    QFrame#ModelSettingsPane {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.015)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.08)};
        border-radius: {SOFT_RADIUS_MD + 4}px;
    }}

    QFrame#ModelSettingsFormCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.04)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.08)};
        border-radius: {SOFT_RADIUS_MD + 4}px;
    }}

    QFrame#ModelSettingsSummaryCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.055)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.08)};
        border-radius: {SOFT_RADIUS_MD + 2}px;
    }}

    QLabel#ModelSettingsSummaryLabel {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.4)};
        background: transparent;
        font-size: 9pt;
        font-weight: 500;
    }}

    QListWidget#ModelProfileList {{
        background: transparent;
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 2px;
        outline: none;
    }}

    QLineEdit#ModelSettingsSearchField {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.06)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.08)};
        border-radius: {SOFT_RADIUS_MD + 4}px;
        min-height: 34px;
        padding: 4px 12px;
        color: {TEXT_PRIMARY};
        selection-background-color: {blend_hex(ACCENT_BLUE_SOFT, ACCENT_BLUE, 0.28)};
    }}

    QLineEdit#ModelSettingsSearchField:focus {{
        border: 1px solid {blend_hex(SUCCESS_GREEN, "#FFFFFF", 0.18)};
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.075)};
    }}

    QListWidget#ModelProfileList::item {{
        border: none;
        background: transparent;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 7px 4px;
        margin: 3px 0px;
        color: {TEXT_PRIMARY};
    }}

    QListWidget#ModelProfileList::item:selected {{
        background: transparent;
        color: {TEXT_PRIMARY};
    }}

    QWidget#ModelProfileRowCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.045)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.08)};
        border-radius: {SOFT_RADIUS_MD + 4}px;
    }}

    QWidget#ModelProfileRowCard[selectedProfile="true"] {{
        background: {blend_hex(SURFACE_CARD, SUCCESS_GREEN, 0.09)};
        border: 1px solid {blend_hex(SUCCESS_GREEN, "#FFFFFF", 0.12)};
    }}

    QWidget#ModelProfileRowCard[disabledProfile="true"] {{
        background: {blend_hex(SURFACE_CARD, "#000000", 0.08)};
        border: 1px solid {blend_hex(BORDER, "#000000", 0.08)};
    }}

    QLabel#ModelProfileItemTitle {{
        color: {TEXT_PRIMARY};
        font-size: 10.8pt;
        font-weight: 650;
        background: transparent;
        padding: 0px;
    }}

    QLabel#ModelProfileItemActive {{
        color: {SUCCESS_GREEN};
        font-size: 9.1pt;
        font-weight: 600;
        background: transparent;
    }}

    QLabel#ModelProfileItemMeta {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.24)};
        font-size: 9.2pt;
        background: transparent;
        padding: 0px;
    }}

    QLabel#ModelProfileItemBadge {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.05)};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.34)};
        border: none;
        border-radius: {SOFT_RADIUS_MD + 6}px;
        padding: 2px 8px;
        font-size: 8.3pt;
        font-weight: 700;
    }}

    QLabel#ModelProfileItemBadge[badgeVariant="active"] {{
        background: {blend_hex(SUCCESS_GREEN, SURFACE_ALT, 0.76)};
        color: {blend_hex(SUCCESS_GREEN, TEXT_PRIMARY, 0.18)};
    }}

    QCheckBox#ModelProfileEnabledSwitch {{
        background: #D4D4D4;
        border-radius: 9px;
        padding: 0px;
        spacing: 0px;
    }}

    QCheckBox#ModelProfileEnabledSwitch:hover {{
        background: #C5C5C5;
    }}

    QCheckBox#ModelProfileEnabledSwitch::indicator {{
        width: 14px;
        height: 14px;
        border-radius: 7px;
        background: #1A1A1A;
        border: none;
    }}

    /* Выключенное состояние (сдвигаем кружок влево) */
    QCheckBox#ModelProfileEnabledSwitch::indicator:unchecked {{
        subcontrol-origin: margin;
        subcontrol-position: left center;
        margin-left: 2px;
    }}

    /* Включенное состояние (сдвигаем кружок вправо) */
    QCheckBox#ModelProfileEnabledSwitch::indicator:checked {{
        subcontrol-origin: margin;
        subcontrol-position: right center;
        margin-right: 2px;
    }}
    
    QCheckBox#ModelSupportsImagesCheckbox {{
        background: transparent;
        border: none;
        padding: 0px;
        spacing: 6px;
        color: {TEXT_PRIMARY};
    }}

    QDialog#ModelSettingsDialog QLineEdit,
    QDialog#ModelSettingsDialog QComboBox {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.05)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.07)};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        min-height: 32px;
        padding: 4px 10px;
        color: {TEXT_PRIMARY};
        selection-background-color: {blend_hex(ACCENT_BLUE_SOFT, ACCENT_BLUE, 0.28)};
    }}

    QDialog#ModelSettingsDialog QLineEdit:focus,
    QDialog#ModelSettingsDialog QComboBox:focus {{
        border: 1px solid {blend_hex(SUCCESS_GREEN, "#FFFFFF", 0.18)};
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
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.06)};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        min-height: 34px;
        padding: 4px 10px;
        font-size: 9.8pt;
        font-weight: 600;
    }}

    QPushButton#SettingsAddButton,
    QPushButton#SettingsDeleteButton {{
        min-height: 38px;
        font-size: 9.8pt;
        font-weight: 600;
    }}

    QPushButton#SettingsAddButton {{
        background: {blend_hex(SUCCESS_GREEN, SURFACE_ALT, 0.32)};
        color: {TEXT_PRIMARY};
    }}

    QPushButton#SettingsAddButton:hover {{
        background: {blend_hex(SUCCESS_GREEN, SURFACE_ALT, 0.24)};
    }}

    QPushButton#SettingsDeleteButton {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.04)};
    }}

    QPushButton#ModelSettingsInlineButton,
    QToolButton#ModelSettingsInlineToolButton {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.04)};
        border: 1px solid {blend_hex(BORDER, "#FFFFFF", 0.07)};
        border-radius: {SOFT_RADIUS_MD + 2}px;
        color: {TEXT_PRIMARY};
    }}

    QPushButton#ModelSettingsInlineButton {{
        min-height: 34px;
        padding: 4px 12px;
        font-size: 9.4pt;
        font-weight: 600;
    }}

    QToolButton#ModelSettingsInlineToolButton {{
        min-width: 32px;
        max-width: 32px;
        min-height: 32px;
        max-height: 32px;
        padding: 0px;
    }}

    QLabel#ModelSettingsHintText {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.24)};
        background: transparent;
        font-size: 8.8pt;
        padding-left: 2px;
    }}

    QSplitter::handle {{
        background: transparent;
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

    QLineEdit#SidebarSearchField {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.03)};
        border: 1px solid transparent;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 7px 10px;
        color: {TEXT_PRIMARY};
        selection-background-color: {blend_hex(ACCENT_BLUE_SOFT, ACCENT_BLUE, 0.25)};
    }}

    QLineEdit#SidebarSearchField:focus {{
        border: 1px solid {blend_hex(ACCENT_BLUE, "#FFFFFF", 0.08)};
    }}

    QLabel#SidebarEmptyState {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.28)};
        font-size: 9.2pt;
        padding: 18px 10px;
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

    QLabel#ApprovalCardTitle {{
        color: {TEXT_PRIMARY};
        font-size: 10.4pt;
        font-weight: 700;
        background: transparent;
    }}

    QLabel#ApprovalCardSummary {{
        color: {TEXT_PRIMARY};
        font-size: 9.7pt;
        background: transparent;
    }}

    QLabel#ApprovalCardImpacts {{
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.36)};
        font-size: 8.9pt;
        background: transparent;
    }}

    QLabel#ApprovalRiskBadge {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.06)};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.34)};
        border: none;
        border-radius: {SOFT_RADIUS_XS + 6}px;
        padding: 2px 8px;
        font-size: 8.1pt;
        font-weight: 700;
    }}

    QLabel#ApprovalRiskBadge[riskLevel="low"] {{
        background: {blend_hex(SUCCESS_GREEN, SURFACE_ALT, 0.82)};
        color: {blend_hex(SUCCESS_GREEN, TEXT_PRIMARY, 0.18)};
    }}

    QLabel#ApprovalRiskBadge[riskLevel="medium"] {{
        background: {blend_hex(AMBER_WARNING, SURFACE_ALT, 0.82)};
        color: {blend_hex(AMBER_WARNING, TEXT_PRIMARY, 0.14)};
    }}

    QLabel#ApprovalRiskBadge[riskLevel="high"],
    QLabel#ApprovalRiskBadge[riskLevel="critical"] {{
        background: {blend_hex(ERROR_RED, SURFACE_ALT, 0.82)};
        color: {blend_hex(ERROR_RED, TEXT_PRIMARY, 0.18)};
    }}

    QFrame#ApprovalToolCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.04)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QLabel#ApprovalToolTitle {{
        color: {TEXT_PRIMARY};
        font-family: "{MONO_FONT_FAMILY}";
        font-size: 9pt;
        font-weight: 600;
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

    QFrame#InlineStatusRow {{
        border: none;
    }}

    QFrame#InlineStatusRow[phase="active"] QLabel#TranscriptMeta,
    QFrame#InlineStatusRow[phase="reviewing"] QLabel#TranscriptMeta {{
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.08)};
    }}

    QFrame#InlineStatusRow[phase="waiting"] QLabel#TranscriptMeta {{
        color: {blend_hex(AMBER_WARNING, TEXT_PRIMARY, 0.24)};
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

    QPushButton#SecondaryButton {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.04)};
        color: {TEXT_PRIMARY};
        border: none;
        font-weight: 600;
    }}

    QPushButton#DangerButton {{
        background: {ERROR_RED};
        color: white;
        border: none;
        font-weight: 600;
    }}

    QToolButton#SidebarGhostButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: {SOFT_RADIUS_SM}px;
        padding: 4px;
    }}

    QToolButton#SidebarGhostButton:hover {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.06)};
    }}

    QToolButton#SidebarGhostButton:disabled {{
        background: transparent;
        border: 1px solid transparent;
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

    QLabel#ToolSubtitle {{
        background: transparent;
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.30)};
        font-size: 8.9pt;
        padding-left: 13px;
        padding-bottom: 1px;
    }}

    QLabel#ToolPhaseBadge {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.04)};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.34)};
        border: none;
        border-radius: {SOFT_RADIUS_XS + 5}px;
        padding: 2px 7px;
        font-size: 8.1pt;
        font-weight: 600;
    }}

    QLabel#ToolPhaseBadge[variant="pending"] {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.05)};
        color: {blend_hex(TEXT_MUTED, TEXT_PRIMARY, 0.26)};
    }}

    QLabel#ToolPhaseBadge[variant="active"] {{
        background: {blend_hex(ACCENT_BLUE_SOFT, ACCENT_BLUE, 0.18)};
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.04)};
    }}

    QLabel#ToolPhaseBadge[variant="success"] {{
        background: transparent;
        color: {SUCCESS_GREEN};
        padding: 0px;
        border-radius: 0px;
        font-size: 9.2pt;
        font-weight: 700;
    }}

    QLabel#ToolPhaseBadge[variant="error"] {{
        background: {blend_hex(ERROR_RED, SURFACE_ALT, 0.18)};
        color: {TEXT_PRIMARY};
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

    QPushButton#ComposerStopButton {{
        background: {blend_hex(ERROR_RED, "#000000", 0.08)};
        color: white;
        border: none;
        border-radius: 15px;
        min-width: 32px;
        max-width: 32px;
        min-height: 32px;
        max-height: 32px;
        padding: 0px;
        font-weight: 700;
    }}

    QPushButton#ComposerStopButton:hover {{
        background: {blend_hex(ERROR_RED, "#FFFFFF", 0.08)};
    }}

    QPushButton#ComposerStopButton:pressed {{
        background: {blend_hex(ERROR_RED, "#000000", 0.18)};
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

    QPushButton#TranscriptJumpButton {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.08)};
        color: {TEXT_PRIMARY};
        border: none;
        border-radius: {SOFT_RADIUS_MD + 8}px;
        padding: 7px 12px;
        font-size: 8.9pt;
        font-weight: 600;
    }}

    QPushButton#TranscriptJumpButton:hover {{
        background: {blend_hex(SURFACE_ALT, "#FFFFFF", 0.08)};
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
        color: {blend_hex(TEXT_PRIMARY, TEXT_MUTED, 0.06)};
        font-size: 10.7pt;
        font-weight: 500;
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

    QWidget#ToolsContainer,
    QWidget#InspectorPanel {{
        background: transparent;
    }}

    QTextBrowser#InspectorHelpText {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.02)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
        padding: 10px 12px;
    }}

    QFrame#ToolCard {{
        background: {blend_hex(SURFACE_CARD, "#FFFFFF", 0.04)};
        border: none;
        border-radius: {SOFT_RADIUS_MD}px;
    }}

    QLabel#ToolGroupHeader {{
        color: {TEXT_MUTED};
        font-size: 7.2pt;
        font-weight: 700;
        letter-spacing: 0.8px;
        padding: 8px 4px 3px 4px;
    }}

    QLabel#ToolGroupHeader[toolGroup="protected"] {{
        color: {AMBER_WARNING};
    }}

    QLabel#ToolGroupHeader[toolGroup="mcp"] {{
        color: {ACCENT_BLUE};
    }}

    QLabel#ToolCardTitle {{
        color: {TEXT_PRIMARY};
        font-weight: 600;
        font-size: 8.8pt;
        font-family: "{MONO_FONT_FAMILY}";
        background: transparent;
    }}

    QLabel#ToolCardDescription {{
        color: {TEXT_MUTED};
        font-size: 8pt;
        background: transparent;
    }}

    QLabel#ToolFlagChip {{
        color: {TEXT_MUTED};
        font-size: 7pt;
        border: 1px solid {blend_hex(TEXT_MUTED, "#FFFFFF", 0.14)};
        border-radius: 3px;
        padding: 0px 4px;
        background: transparent;
    }}

    QLabel#ToolFlagChip[flagVariant="warning"] {{
        color: {AMBER_WARNING};
        border: 1px solid {blend_hex(AMBER_WARNING, SURFACE_CARD, 0.48)};
    }}

    QLabel#ToolFlagChip[flagVariant="accent"] {{
        color: {ACCENT_BLUE};
        border: 1px solid {blend_hex(ACCENT_BLUE, SURFACE_CARD, 0.56)};
    }}

    QFrame#ToolGroupSeparator {{
        background: {BORDER};
        margin: 4px 0px;
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

