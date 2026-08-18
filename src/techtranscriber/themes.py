from __future__ import annotations

from string import Template


THEME_OPTIONS = (
    ("minimal_light", "1. Минимализм / Light"),
    ("modern_dark", "2. Тёмный / Modern"),
    ("glass_fluent", "3. Glassmorphism / Fluent"),
    ("bold_brutal", "4. Необрутализм / Bold"),
    ("sleek_light", "5. Max-стиль / Sleek"),
)
THEME_IDS = frozenset(theme_id for theme_id, _ in THEME_OPTIONS)
DEFAULT_THEME = "modern_dark"


_BASE = {
    "font": "Segoe UI",
    "root": "#080a12",
    "workspace": "#0a0c15",
    "sidebar": "#0d0f19",
    "card": "#10131e",
    "input": "#161925",
    "table": "#0e111b",
    "table_alt": "#121521",
    "header": "#171a27",
    "border": "#272b3c",
    "input_border": "#2b3042",
    "line": "#202433",
    "text": "#f5f3fb",
    "input_text": "#edf0f7",
    "muted": "#858a9c",
    "subtle": "#73788b",
    "nav_text": "#aeb2c2",
    "nav_hover": "#171a28",
    "nav_disabled": "#555a6a",
    "active_bg": "#39236b",
    "active_text": "#ffffff",
    "accent": "#9b72f2",
    "focus": "#8058d9",
    "primary": "#7042cf",
    "primary_hover": "#8051dd",
    "primary_pressed": "#6136b8",
    "primary_text": "#ffffff",
    "secondary": "#1b1e2c",
    "secondary_hover": "#25293a",
    "secondary_text": "#cbd0dc",
    "disabled": "#171925",
    "disabled_text": "#5e6373",
    "danger": "#c44461",
    "danger_hover": "#d34b69",
    "danger_border": "#e15b76",
    "privacy": "#131622",
    "privacy_text": "#8c91a3",
    "timer": "#19132a",
    "timer_text": "#c3a7ff",
    "timer_border": "#35255b",
    "chip": "#211834",
    "chip_text": "#c6b4ec",
    "chip_border": "#3a285e",
    "hint": "#151825",
    "selection": "#2d2150",
    "selection_input": "#6f45ce",
    "scrollbar": "#35394b",
    "scrollbar_hover": "#494e64",
    "tooltip": "#1c2030",
    "success": "#55d6a7",
    "error": "#ef6a7f",
    "mic": "#a78bfa",
    "system": "#55d6be",
    "radius_card": "14px",
    "radius_control": "8px",
    "radius_button": "9px",
    "border_width": "1px",
    "title_weight": "700",
    "button_weight": "600",
}


def _theme(**overrides: str) -> dict[str, str]:
    return {**_BASE, **overrides}


THEMES: dict[str, dict[str, str]] = {
    "modern_dark": _theme(),
    "minimal_light": _theme(
        root="#f4f6f9",
        workspace="#ffffff",
        sidebar="#f7f9fc",
        card="#ffffff",
        input="#fbfcfe",
        table="#ffffff",
        table_alt="#f9fbfd",
        header="#f1f5f9",
        border="#dfe5ec",
        input_border="#d9e0e8",
        line="#e8edf2",
        text="#17202d",
        input_text="#1d2733",
        muted="#697586",
        subtle="#8490a0",
        nav_text="#4b5868",
        nav_hover="#edf3fb",
        nav_disabled="#a8b0bb",
        active_bg="#e5f0ff",
        active_text="#075fc7",
        accent="#1677e8",
        focus="#1677e8",
        primary="#0f6fdf",
        primary_hover="#0b60c7",
        primary_pressed="#094f9f",
        secondary="#f2f5f8",
        secondary_hover="#e8edf3",
        secondary_text="#344054",
        disabled="#edf0f3",
        disabled_text="#98a2b3",
        danger="#d6455d",
        danger_hover="#bd364e",
        danger_border="#d6455d",
        privacy="#eef4fb",
        privacy_text="#5f6f82",
        timer="#edf5ff",
        timer_text="#0b63ce",
        timer_border="#cfe2fb",
        chip="#eaf3ff",
        chip_text="#0a60c4",
        chip_border="#cfe2fb",
        hint="#f3f6fa",
        selection="#dceaff",
        selection_input="#1677e8",
        scrollbar="#c4ccd6",
        scrollbar_hover="#9da8b5",
        tooltip="#ffffff",
        success="#149766",
        error="#d6455d",
        mic="#6b3fc5",
        system="#087c72",
        radius_card="8px",
        radius_control="5px",
        radius_button="6px",
    ),
    "glass_fluent": _theme(
        root="qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #273d61, stop:0.5 #58688f, stop:1 #897079)",
        workspace="rgba(32, 45, 73, 175)",
        sidebar="rgba(31, 46, 76, 185)",
        card="rgba(57, 71, 108, 155)",
        input="rgba(48, 61, 94, 170)",
        table="rgba(37, 49, 78, 175)",
        table_alt="rgba(60, 72, 105, 150)",
        header="rgba(42, 55, 87, 190)",
        border="rgba(222, 232, 255, 90)",
        input_border="rgba(229, 235, 255, 100)",
        line="rgba(231, 237, 255, 45)",
        text="#ffffff",
        input_text="#ffffff",
        muted="#d4dcf0",
        subtle="#bcc7df",
        nav_text="#e4e9f5",
        nav_hover="rgba(255, 255, 255, 32)",
        nav_disabled="rgba(225, 232, 247, 100)",
        active_bg="rgba(102, 180, 255, 95)",
        active_text="#ffffff",
        accent="#71c8ff",
        focus="#a9ddff",
        primary="qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3f7cf3, stop:1 #7552ed)",
        primary_hover="#5d80ef",
        primary_pressed="#4c63cc",
        secondary="rgba(255, 255, 255, 26)",
        secondary_hover="rgba(255, 255, 255, 48)",
        secondary_text="#ffffff",
        disabled="rgba(40, 51, 78, 130)",
        disabled_text="rgba(235, 239, 250, 120)",
        danger="#d85872",
        danger_hover="#e46780",
        danger_border="rgba(255, 210, 220, 120)",
        privacy="rgba(255, 255, 255, 24)",
        privacy_text="#e1e7f4",
        timer="rgba(79, 57, 132, 125)",
        timer_text="#eee5ff",
        timer_border="rgba(230, 218, 255, 90)",
        chip="rgba(102, 76, 169, 120)",
        chip_text="#f4edff",
        chip_border="rgba(235, 224, 255, 90)",
        hint="rgba(33, 44, 71, 135)",
        selection="rgba(100, 86, 191, 145)",
        selection_input="#6f58d9",
        scrollbar="rgba(240, 244, 255, 100)",
        scrollbar_hover="rgba(255, 255, 255, 155)",
        tooltip="#354665",
        success="#73f0bf",
        error="#ff91a5",
        mic="#d3b6ff",
        system="#7ff5e0",
        radius_card="16px",
        radius_control="9px",
        radius_button="9px",
    ),
    "bold_brutal": _theme(
        font="Arial",
        root="#f1eee5",
        workspace="#f6f3ea",
        sidebar="#f0ede4",
        card="#fffdf7",
        input="#ffffff",
        table="#ffffff",
        table_alt="#fffdf7",
        header="#f4f0e6",
        border="#111111",
        input_border="#111111",
        line="#111111",
        text="#111111",
        input_text="#111111",
        muted="#373737",
        subtle="#4b4b4b",
        nav_text="#111111",
        nav_hover="#fff07a",
        nav_disabled="#929292",
        active_bg="#6654ef",
        active_text="#ffffff",
        accent="#111111",
        focus="#6654ef",
        primary="#ffe500",
        primary_hover="#ffef55",
        primary_pressed="#e4cb00",
        primary_text="#111111",
        secondary="#ffffff",
        secondary_hover="#eee9df",
        secondary_text="#111111",
        disabled="#dedbd3",
        disabled_text="#777777",
        danger="#ff5a70",
        danger_hover="#ff7688",
        danger_border="#111111",
        privacy="#fff07a",
        privacy_text="#111111",
        timer="#ffffff",
        timer_text="#111111",
        timer_border="#111111",
        chip="#6654ef",
        chip_text="#ffffff",
        chip_border="#111111",
        hint="#f0ede4",
        selection="#dcd5ff",
        selection_input="#6654ef",
        scrollbar="#111111",
        scrollbar_hover="#6654ef",
        tooltip="#ffffff",
        success="#008a51",
        error="#d92846",
        mic="#5a43d6",
        system="#007a72",
        radius_card="0px",
        radius_control="0px",
        radius_button="0px",
        border_width="2px",
        title_weight="900",
        button_weight="800",
    ),
    "sleek_light": _theme(
        root="#edf0f4",
        workspace="#fafbfd",
        sidebar="#f1f3f6",
        card="#ffffff",
        input="#fbfcfd",
        table="#ffffff",
        table_alt="#fbfcfe",
        header="#f6f7f9",
        border="#e3e7ec",
        input_border="#e0e5eb",
        line="#edf0f3",
        text="#24282f",
        input_text="#252a31",
        muted="#79818d",
        subtle="#9299a4",
        nav_text="#59616d",
        nav_hover="#e9edf2",
        nav_disabled="#abb1ba",
        active_bg="#e8f2ff",
        active_text="#0865c7",
        accent="#0a73dd",
        focus="#0a73dd",
        primary="#0874df",
        primary_hover="#0668ca",
        primary_pressed="#075bb0",
        secondary="#f5f6f8",
        secondary_hover="#e9edf1",
        secondary_text="#3e4651",
        disabled="#eef0f2",
        disabled_text="#a2a8b1",
        danger="#d84d63",
        danger_hover="#c23d53",
        danger_border="#d84d63",
        privacy="#ffffff",
        privacy_text="#707986",
        timer="#ffffff",
        timer_text="#252a31",
        timer_border="#e0e4e9",
        chip="#edf5ff",
        chip_text="#0b65c7",
        chip_border="#d6e8fb",
        hint="#f7f8fa",
        selection="#e5f0ff",
        selection_input="#0a73dd",
        scrollbar="#c9cfd7",
        scrollbar_hover="#a8b0ba",
        tooltip="#ffffff",
        success="#168a62",
        error="#d84d63",
        mic="#6346bc",
        system="#087c72",
        radius_card="18px",
        radius_control="11px",
        radius_button="10px",
    ),
}


_QSS = Template(
    r"""
    QMainWindow, QDialog, QMessageBox, QWidget#appRoot {
        background: $root;
        color: $text;
        font-family: '$font';
        font-size: 13px;
    }
    QLabel { color: $text; background: transparent; }
    QWidget#workspace { background: $workspace; }
    QFrame#sidebar {
        background: $sidebar;
        border-right: $border_width solid $border;
    }
    QLabel#brand { color: $text; font-size: 19px; font-weight: $title_weight; }
    QLabel#sidebarCaption, QLabel#version { color: $subtle; font-size: 11px; }
    QLabel#themeLabel {
        color: $subtle;
        font-size: 10px;
        font-weight: 700;
        padding-top: 4px;
    }
    QLabel#privacy {
        color: $privacy_text;
        background: $privacy;
        border: $border_width solid $border;
        border-radius: $radius_control;
        padding: 11px;
    }
    QPushButton#nav, QPushButton#navActive {
        min-height: 44px;
        padding: 0 14px;
        border: 0;
        border-radius: $radius_button;
        text-align: left;
        font-weight: $button_weight;
        color: $nav_text;
        background: transparent;
    }
    QPushButton#nav:hover { color: $text; background: $nav_hover; }
    QPushButton#navActive {
        color: $active_text;
        background: $active_bg;
        border-left: 3px solid $accent;
    }
    QPushButton#nav:disabled { color: $nav_disabled; background: transparent; }
    QLabel#eyebrow {
        color: $accent;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
    }
    QLabel#appTitle { font-size: 27px; font-weight: $title_weight; color: $text; }
    QLabel#subtitle, QLabel#hint { color: $muted; }
    QLabel#timerCaption {
        color: $subtle;
        font-size: 10px;
        font-weight: 700;
        padding-right: 2px;
    }
    QLabel#timer {
        font-size: 22px;
        font-weight: 700;
        color: $timer_text;
        padding: 9px 14px;
        background: $timer;
        border: $border_width solid $timer_border;
        border-radius: $radius_control;
    }
    QFrame#card {
        background: $card;
        border: $border_width solid $border;
        border-radius: $radius_card;
    }
    QLabel#section { font-size: 16px; font-weight: $title_weight; color: $text; }
    QLabel#chip {
        color: $chip_text;
        background: $chip;
        border: $border_width solid $chip_border;
        border-radius: $radius_control;
        padding: 6px 10px;
        font-size: 11px;
        font-weight: 600;
    }
    QLabel#editorHint {
        color: $muted;
        background: $hint;
        border-radius: $radius_control;
        padding: 9px 11px;
    }
    QFrame#divider { color: $border; background: $border; max-height: $border_width; }
    QScrollArea, QScrollArea#settingsScroll, QScrollArea > QWidget > QWidget {
        background: transparent;
        border: 0;
    }
    QLineEdit, QComboBox, QPlainTextEdit, QListWidget {
        background: $input;
        color: $input_text;
        border: $border_width solid $input_border;
        border-radius: $radius_control;
        padding: 9px;
        selection-background-color: $selection_input;
        selection-color: #ffffff;
    }
    QComboBox { min-height: 22px; padding-right: 26px; }
    QComboBox#themeSelector { min-height: 20px; font-size: 11px; }
    QComboBox::drop-down { border: 0; width: 28px; }
    QComboBox QAbstractItemView {
        background: $input;
        color: $input_text;
        border: $border_width solid $input_border;
        selection-background-color: $selection_input;
        selection-color: #ffffff;
        outline: 0;
    }
    QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QListWidget:focus {
        border: $border_width solid $focus;
    }
    QListWidget::item { padding: 7px 5px; border-radius: $radius_control; }
    QListWidget::item:hover { background: $secondary_hover; }
    QListWidget::item:selected { background: $selection; color: $text; }
    QCheckBox { color: $text; spacing: 8px; }
    QCheckBox::indicator { width: 17px; height: 17px; }
    QCheckBox::indicator:unchecked {
        background: $input;
        border: $border_width solid $input_border;
        border-radius: 4px;
    }
    QCheckBox::indicator:checked {
        background: $primary;
        border: $border_width solid $accent;
        border-radius: 4px;
    }
    QPushButton {
        background: $secondary;
        color: $secondary_text;
        border: $border_width solid $input_border;
        border-radius: $radius_button;
        padding: 10px 15px;
        font-weight: $button_weight;
    }
    QPushButton:hover { background: $secondary_hover; color: $text; }
    QPushButton#primary {
        background: $primary;
        color: $primary_text;
        border: $border_width solid $accent;
    }
    QPushButton#primary:hover { background: $primary_hover; }
    QPushButton#primary:pressed { background: $primary_pressed; }
    QPushButton#danger {
        background: $danger;
        color: #ffffff;
        border: $border_width solid $danger_border;
    }
    QPushButton#danger:hover { background: $danger_hover; }
    QPushButton#secondary {
        background: $secondary;
        color: $secondary_text;
        border: $border_width solid $input_border;
    }
    QPushButton#secondary:hover { background: $secondary_hover; color: $text; }
    QPushButton:disabled {
        background: $disabled;
        color: $disabled_text;
        border-color: $border;
    }
    QFrame#commandBar {
        background: $card;
        border: $border_width solid $border;
        border-radius: $radius_card;
    }
    QLabel#statusDot { color: $success; font-size: 10px; }
    QLabel#status { color: $nav_text; }
    QTableWidget#transcriptTable {
        background: $table;
        alternate-background-color: $table_alt;
        color: $input_text;
        border: $border_width solid $border;
        border-radius: $radius_control;
        outline: 0;
        selection-background-color: $selection;
        selection-color: $text;
    }
    QHeaderView::section {
        background: $header;
        color: $muted;
        border: 0;
        border-bottom: $border_width solid $border;
        padding: 11px;
        font-size: 10px;
        font-weight: 700;
    }
    QTableWidget::item { padding: 10px; border-bottom: $border_width solid $line; }
    QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
    QScrollBar::handle:vertical {
        background: $scrollbar;
        min-height: 28px;
        border-radius: 4px;
        border: none;
    }
    QScrollBar::handle:vertical:hover { background: $scrollbar_hover; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0;
        border: none;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: transparent;
        border: none;
    }
    QToolTip {
        background: $tooltip;
        color: $text;
        border: $border_width solid $border;
        padding: 5px;
    }
    """
)


def theme_stylesheet(theme_id: str) -> str:
    return _QSS.substitute(THEMES.get(theme_id, THEMES[DEFAULT_THEME]))


def theme_colors(theme_id: str) -> dict[str, str]:
    palette = THEMES.get(theme_id, THEMES[DEFAULT_THEME])
    return {
        "success": palette["success"],
        "error": palette["error"],
        "mic": palette["mic"],
        "system": palette["system"],
    }
