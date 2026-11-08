from PySide6.QtCore import QThread, Signal


class Worker(QThread):
    """Un seul travail GPU à la fois ; résultats transmis par signaux au thread Qt."""
    progress = Signal(int, str)
    result = Signal(object)
    error = Signal(str)

    def __init__(self, function, parent=None):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            result = self.function(lambda p, message: self.progress.emit(int(p), str(message)))
            self.result.emit(result)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__} : {exc}")
