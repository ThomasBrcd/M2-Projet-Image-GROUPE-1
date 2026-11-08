import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from src.gui.main_window import MainWindow


@pytest.mark.gui
def test_window_opens_without_checkpoints_and_viewer_accepts_images():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()
    assert window.tabs.count() == 3
    assert not window.anon_button.isEnabled()
    image = np.zeros((100, 150, 3), np.uint8)
    window.reference_view.set_image(image)
    window.target_view.set_image(image)
    window.after_view.set_image(image)
    window.after_view.reset_zoom()
    app.processEvents()
    window.close()
