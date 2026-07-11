from __future__ import annotations


class InspectorController:
    def __init__(self, window) -> None:
        self.window = window

    def toggle_info_popup(self) -> None:
        self.set_inspector_collapsed(not self.window.inspector_collapsed)

    def set_inspector_collapsed(self, collapsed: bool) -> None:
        if self.window.inspector_collapsed == collapsed:
            return
        self.window.inspector_collapsed = collapsed
        if collapsed:
            self.window._inspector_width = max(320, self.window.inspector_container.width())
            self.window.inspector_container.hide()
            self.window.info_action.setToolTip("Show inspector (Ctrl+I)")
            self.window.info_button.setToolTip(self.window.info_action.toolTip())
            self.window.splitter.setSizes([0 if self.window.sidebar_collapsed else self.window._sidebar_width, 1000, 0])
            return
        self.window.inspector_container.show()
        self.window.info_action.setToolTip("Hide inspector (Ctrl+I)")
        self.window.info_button.setToolTip(self.window.info_action.toolTip())
        self.window.splitter.setSizes([0 if self.window.sidebar_collapsed else self.window._sidebar_width, 1000, self.window._inspector_width])
