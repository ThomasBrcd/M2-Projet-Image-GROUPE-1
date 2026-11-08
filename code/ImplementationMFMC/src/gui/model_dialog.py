import json
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox,
                              QPlainTextEdit, QProgressBar, QFileDialog, QMessageBox, QLabel)
from .workers import Worker
from ..utils.model_manager import ModelManager, SPECS, diagnostics


class ModelDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gestionnaire des modèles — FaceAnon")
        self.resize(840, 650)
        self.manager = ModelManager(settings)
        self.worker = None
        layout = QVBoxLayout(self)
        notice = QLabel("Installation explicite depuis les sources officielles. Les photos ne sont jamais envoyées.\n"
                        "La présence d'un fichier ne prouve pas sa compatibilité : un essai réel reste nécessaire.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        row = QHBoxLayout()
        self.model = QComboBox()
        self.model.addItems(list(SPECS))
        row.addWidget(self.model)
        self.download_button = QPushButton("Télécharger officiellement")
        self.import_button = QPushButton("Importer un checkpoint…")
        self.check_button = QPushButton("État et diagnostic CUDA")
        for button in (self.download_button, self.import_button, self.check_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.download_button.clicked.connect(self.download)
        self.import_button.clicked.connect(self.import_checkpoint)
        self.check_button.clicked.connect(self.refresh)
        self.refresh()

    def _busy(self):
        return self.worker is not None and self.worker.isRunning()

    def run_job(self, function, callback):
        if self._busy():
            return
        self.progress.setValue(0)
        for button in (self.download_button, self.import_button, self.check_button):
            button.setEnabled(False)
        self.worker = Worker(function, self)
        self.worker.progress.connect(self._progress)
        self.worker.result.connect(callback)
        self.worker.error.connect(self._error)
        self.worker.finished.connect(self._finished)
        self.worker.start()

    def _progress(self, percent, message):
        self.progress.setValue(percent)
        self.progress.setFormat(message + " (%p%)")

    def _error(self, message):
        self.text.appendPlainText(message)

    def _finished(self):
        finished = self.worker
        self.worker = None
        if finished is not None:
            finished.deleteLater()
        for button in (self.download_button, self.import_button, self.check_button):
            button.setEnabled(True)

    def refresh(self):
        self.run_job(lambda progress: {"environnement": diagnostics(), "modèles": self.manager.status()},
                     lambda report: self.text.setPlainText(json.dumps(report, indent=2, ensure_ascii=False)))

    def _pickle_consent(self, key):
        if not SPECS[key].pickle:
            return True
        return QMessageBox.question(self, "Checkpoint sérialisé officiel",
                                    "Ce modèle utilise pickle, un format pouvant exécuter du code.\n"
                                    "Accepter uniquement un checkpoint obtenu depuis la source officielle indiquée :\n"
                                    + SPECS[key].source + "\n\nAutoriser ce chargement ?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                    QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes

    def download(self):
        key = self.model.currentText()
        if not self._pickle_consent(key):
            return
        self.run_job(lambda progress: str(self.manager.download(key, progress, trusted_pickle=SPECS[key].pickle)),
                     lambda path: self.text.appendPlainText("Installé : " + path + "\nRelancer la détection pour utiliser les nouveaux poids."))

    def import_checkpoint(self):
        key = self.model.currentText()
        file, _ = QFileDialog.getOpenFileName(self, "Importer " + key, str(self.manager.settings.root), "Checkpoint (*)")
        if not file or not self._pickle_consent(key):
            return
        self.run_job(lambda progress: str(self.manager.import_file(key, file, trusted_pickle=SPECS[key].pickle)),
                     lambda path: self.text.appendPlainText("Importé : " + path + "\nRelancer l'application si un modèle était déjà chargé."))

    def reject(self):
        if not self._busy():
            super().reject()

    def closeEvent(self, event):
        if self._busy():
            event.ignore()
        else:
            event.accept()
