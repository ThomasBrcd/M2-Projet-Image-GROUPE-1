"""Interface de bureau française : sélection explicite, workers et export contrôlé."""
import json
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
                              QPushButton, QLabel, QFileDialog, QTabWidget, QGroupBox,
                              QSplitter, QListWidget, QListWidgetItem, QDoubleSpinBox,
                              QSpinBox, QComboBox, QCheckBox, QProgressBar, QPlainTextEdit,
                              QMessageBox, QInputDialog)
from .image_viewer import ImageViewer, pixmap
from .workers import Worker
from .model_dialog import ModelDialog
from ..utils.config import Settings
from ..utils.image_utils import load_image, save_image, face_thumbnail
from ..core.pipeline import Analysis
from ..core.recognition import cosine
from ..core.service import ProcessingService
from ..core.blending import emergency_mask

STYLE = """
QMainWindow, QDialog { background: #101723; color: #e7edf7; }
QWidget { color: #e7edf7; font-family: 'Segoe UI'; font-size: 10pt; }
QGroupBox { border: 1px solid #334157; border-radius: 8px; margin-top: 12px; padding: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; }
QPushButton { background: #26364c; padding: 9px 15px; border: 1px solid #415571; border-radius: 6px; }
QPushButton:hover { background: #354e6c; }
QPushButton:disabled { background: #1b2534; color: #778397; }
QPushButton#primary { background: #147c61; font-weight: bold; }
QTabWidget::pane { border: 1px solid #334157; }
QTabBar::tab { background: #1b283c; padding: 12px 20px; }
QTabBar::tab:selected { background: #305071; }
QListWidget, QPlainTextEdit, QComboBox, QDoubleSpinBox, QSpinBox { background: #192437; padding: 5px; border: 1px solid #334157; }
QProgressBar { border: 1px solid #334157; text-align: center; min-height: 22px; }
QProgressBar::chunk { background: #147c61; }
QCheckBox { spacing: 8px; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceAnon — Anonymisation photographique locale")
        self.resize(1320, 920)
        self.setStyleSheet(STYLE)
        self.service = ProcessingService()
        self.worker = None
        self.reference = None
        self.image = None
        self.reference_path = None
        self.image_path = None
        self.reference_faces = []
        self.target_faces = []
        self.analysis = None
        self.result = None
        self.export_image = None
        self._changing = False
        self._auto_run = False
        self._pipeline = None
        self._identity_keys = []
        self._build()
        self._load_bank_choices()

    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        header = QHBoxLayout()
        title = QLabel("FaceAnon")
        title.setStyleSheet("font-size: 25pt; font-weight: bold; color: #65e1b4;")
        header.addWidget(title)
        header.addWidget(QLabel("Photographies · Traitement local · SimSwap"))
        header.addStretch()
        self.models_button = QPushButton("Modèles et CUDA")
        self.bank_import = QPushButton("Importer une banque…")
        self.bank_generate = QPushButton("Générer avec StyleGAN2…")
        for button in (self.models_button, self.bank_import, self.bank_generate):
            header.addWidget(button)
        layout.addLayout(header)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_inputs()
        self._build_options()
        self._build_results()
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(115)
        self.log.setPlaceholderText("Avertissements et état du traitement")
        layout.addWidget(self.log)
        self.models_button.clicked.connect(self._models)
        self.bank_import.clicked.connect(self._import_bank)
        self.bank_generate.clicked.connect(self._generate_bank)
        self._busy_widgets = [self.models_button, self.bank_import, self.bank_generate, self.reference_button,
                              self.image_button, self.detect_button, self.anon_button, self.options,
                              self.save_button, self.retry_button, self.fallback_button]
        self._update_actions()
        self.log.appendPlainText("Seuils expérimentaux à calibrer : les scores ne garantissent pas l'anonymat.")

    def _build_inputs(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        split = QSplitter()
        left = QGroupBox("1 · Personne de référence")
        ll = QVBoxLayout(left)
        self.reference_button = QPushButton("Choisir la photo de référence…")
        self.reference_label = QLabel("Aucune référence")
        self.reference_view = ImageViewer("Photographie de référence")
        self.reference_list = QListWidget()
        self.reference_list.setMaximumHeight(110)
        self.reference_list.setIconSize(QSize(60, 60))
        ll.addWidget(self.reference_button)
        ll.addWidget(self.reference_label)
        ll.addWidget(self.reference_view, 1)
        ll.addWidget(QLabel("Si plusieurs visages sont détectés, cliquer sur la personne recherchée."))
        ll.addWidget(self.reference_list)
        right = QGroupBox("2 · Photographie à anonymiser")
        rl = QVBoxLayout(right)
        self.image_button = QPushButton("Choisir la photographie…")
        self.image_label = QLabel("Aucune photographie")
        self.target_view = ImageViewer("Photographie à traiter")
        self.target_list = QListWidget()
        self.target_list.setMaximumHeight(140)
        self.target_list.setIconSize(QSize(54, 54))
        rl.addWidget(self.image_button)
        rl.addWidget(self.image_label)
        rl.addWidget(self.target_view, 1)
        rl.addWidget(QLabel("Vert : sélectionné · Orange : conservé · Cliquer pour corriger la sélection."))
        rl.addWidget(self.target_list)
        split.addWidget(left)
        split.addWidget(right)
        layout.addWidget(split, 1)
        self.detect_button = QPushButton("Détecter les visages")
        layout.addWidget(self.detect_button)
        self.tabs.addTab(page, "Entrées et correspondances")
        self.reference_button.clicked.connect(lambda: self._choose_image(True))
        self.image_button.clicked.connect(lambda: self._choose_image(False))
        self.detect_button.clicked.connect(self._detect)
        self.reference_list.currentRowChanged.connect(self._reference_selected)
        self.reference_view.faceClicked.connect(self._select_reference)
        self.target_view.faceClicked.connect(self._toggle_target)
        self.target_list.itemChanged.connect(self._selection_changed)

    @staticmethod
    def _threshold(default):
        spin = QDoubleSpinBox()
        spin.setRange(-1, 1)
        spin.setDecimals(2)
        spin.setSingleStep(.01)
        spin.setValue(default)
        return spin

    def _build_options(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.options = QGroupBox("Réglages d'anonymisation")
        form = QFormLayout(self.options)
        self.mode = QComboBox()
        for label, value in [("Identité fictive aléatoire", "random"), ("Identité stable pendant la session", "session"),
                              ("Choisir une identité dans la banque", "manual")]:
            self.mode.addItem(label, value)
        self.identity = QComboBox()
        self.match = self._threshold(.45)
        self.validation = self._threshold(.30)
        self.candidate = self._threshold(.30)
        self.segmentation = QCheckBox("Utiliser BiSeNet (repli géométrique signalé en cas d'échec)")
        self.segmentation.setChecked(True)
        self.attributes = QCheckBox("Favoriser une luminosité voisine parmi les identités admissibles")
        self.resolution = QComboBox()
        self.resolution.addItem("SimSwap 224", 224)
        self.resolution.addItem("SimSwap 512 (checkpoint optionnel)", 512)
        self.resolution.model().item(1).setEnabled((Settings().models/"simswap/512/550000_net_G.pth").is_file())
        self.device = QComboBox()
        for label, value in [("Automatique — CUDA si disponible", "auto"), ("CPU", "cpu"), ("CUDA obligatoire", "cuda")]:
            self.device.addItem(label, value)
        self.attempts = QSpinBox()
        self.attempts.setRange(1, 10)
        self.attempts.setValue(3)
        form.addRow("Identité de remplacement", self.mode)
        form.addRow("Banque locale", self.identity)
        form.addRow("Correspondance : cosinus ≥", self.match)
        form.addRow("Candidat synthétique : cosinus ≤", self.candidate)
        form.addRow("Validation après remplacement : cosinus ≤", self.validation)
        form.addRow("Nombre maximal d'identités essayées", self.attempts)
        form.addRow("Modèle", self.resolution)
        form.addRow("Calcul", self.device)
        form.addRow(self.segmentation)
        form.addRow(self.attributes)
        layout.addWidget(self.options)
        legend = QLabel("Le modèle de reconnaissance est commun à la référence, à la cible et au contrôle final.\n"
                        "Son embedding est distinct de l'ArcFace historique qui conditionne SimSwap.\n\n"
                        "Ces seuils sont des réglages expérimentaux, pas une garantie d'anonymat.\n"
                        "Vérifier les faux positifs et le résultat avant de partager une photographie.")
        legend.setWordWrap(True)
        layout.addWidget(legend)
        layout.addStretch()
        self.anon_button = QPushButton("Anonymiser")
        self.anon_button.setObjectName("primary")
        self.anon_button.setMinimumHeight(52)
        layout.addWidget(self.anon_button)
        self.tabs.addTab(page, "Anonymisation")
        self.anon_button.clicked.connect(self._anonymize)
        self.match.valueChanged.connect(self._threshold_changed)
        self.mode.currentIndexChanged.connect(lambda: self.identity.setEnabled(self.mode.currentData() == "manual"))
        self.identity.setEnabled(False)

    def _build_results(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.summary = QLabel("Aucun résultat")
        layout.addWidget(self.summary)
        split = QSplitter()
        self.before_view = ImageViewer("Avant")
        self.after_view = ImageViewer("Après")
        split.addWidget(self.before_view)
        split.addWidget(self.after_view)
        layout.addWidget(split, 1)
        controls = QHBoxLayout()
        self.boxes = QCheckBox("Afficher les cadres")
        self.boxes.setChecked(True)
        self.masks = QCheckBox("Afficher les masques de fusion")
        fit = QPushButton("Ajuster les vues")
        controls.addWidget(self.boxes)
        controls.addWidget(self.masks)
        controls.addWidget(fit)
        controls.addStretch()
        layout.addLayout(controls)
        self.scores = QPlainTextEdit()
        self.scores.setReadOnly(True)
        self.scores.setMaximumHeight(120)
        layout.addWidget(self.scores)
        row = QHBoxLayout()
        self.retry_button = QPushButton("Nouvelle identité et nouvel essai")
        self.fallback_button = QPushButton("Masquage opaque de secours…")
        self.save_button = QPushButton("Enregistrer sous…")
        for button in (self.retry_button, self.fallback_button, self.save_button):
            row.addWidget(button)
        layout.addLayout(row)
        self.tabs.addTab(page, "Résultats · Avant / Après")
        self.save_button.clicked.connect(self._save)
        self.retry_button.clicked.connect(self._retry)
        self.fallback_button.clicked.connect(self._fallback)
        fit.clicked.connect(lambda: [self.before_view.reset_zoom(), self.after_view.reset_zoom()])
        self.boxes.toggled.connect(self._overlays)
        self.masks.toggled.connect(self._overlays)

    def settings(self):
        return Settings(device=self.device.currentData(), crop_size=self.resolution.currentData(),
                        match_threshold=self.match.value(), candidate_max_similarity=self.candidate.value(),
                        validation_max_similarity=self.validation.value(), max_attempts=self.attempts.value(),
                        segmentation=self.segmentation.isChecked(), identity_mode=self.mode.currentData(),
                        manual_identity=self.identity.currentData(), attribute_matching=self.attributes.isChecked())

    def _is_busy(self):
        return self.worker is not None and self.worker.isRunning()

    def _job(self, function, callback):
        if self._is_busy():
            return
        self.progress.setValue(0)
        for widget in self._busy_widgets:
            widget.setEnabled(False)
        self.reference_list.setEnabled(False)
        self.target_list.setEnabled(False)
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
        self._auto_run = False
        self.log.appendPlainText(message)
        QMessageBox.warning(self, "Traitement impossible", message)

    def _finished(self):
        finished = self.worker
        self.worker = None
        if finished is not None:
            finished.deleteLater()
        for widget in self._busy_widgets:
            widget.setEnabled(True)
        self.reference_list.setEnabled(True)
        self.target_list.setEnabled(True)
        self.identity.setEnabled(self.mode.currentData() == "manual")
        self._update_actions()
        if self._auto_run:
            self._auto_run = False
            if self.analysis is not None:
                self._anonymize()

    def _update_actions(self):
        self.detect_button.setEnabled(self.reference is not None and self.image is not None)
        self.anon_button.setEnabled(self.reference is not None and self.image is not None)
        self.save_button.setEnabled(self.export_image is not None)
        self.retry_button.setEnabled(self.analysis is not None and bool(self._selected()))
        self.fallback_button.setEnabled(self.analysis is not None and bool(self._selected()))

    def _choose_image(self, is_reference):
        file, _ = QFileDialog.getOpenFileName(self, "Choisir une photographie", str(Settings().root),
                                               "Images (*.png *.jpg *.jpeg *.webp)")
        if not file:
            return
        # Lecture et orientation EXIF aussi exécutées hors du thread graphique.
        def done(image):
            if is_reference:
                self.reference, self.reference_path = image, file
                self.reference_view.set_image(image)
                self.reference_label.setText(Path(file).name)
            else:
                self.image, self.image_path = image, file
                self.target_view.set_image(image)
                self.image_label.setText(f"{Path(file).name} · {image.shape[1]} × {image.shape[0]}")
            self._invalidate()
        self._job(lambda progress: load_image(file), done)

    def _invalidate(self):
        self.analysis = None
        self.result = None
        self.export_image = None
        self.reference_faces = []
        self.target_faces = []
        self._changing = True
        self.reference_list.clear()
        self.target_list.clear()
        self._changing = False
        self.reference_view.set_faces([])
        self.target_view.set_faces([])
        self.before_view.clear_image("Avant")
        self.after_view.clear_image("Après — lancer un nouveau traitement")
        self.scores.clear()
        self.summary.setText("Détection à effectuer")

    def _detect(self):
        settings = self.settings()
        def function(progress):
            progress(5, "Chargement local du détecteur et du reconnaisseur")
            pipeline = self.service.pipeline(settings)
            progress(25, "Détection des visages de référence")
            refs = pipeline.recognizer.detect(self.reference)
            progress(60, "Détection de tous les visages de la photographie")
            targets = pipeline.recognizer.detect(self.image)
            progress(100, "Détection terminée")
            return pipeline, refs, targets
        self._job(function, self._detected)

    def _detected(self, data):
        self._pipeline, self.reference_faces, self.target_faces = data
        self.analysis = None
        self.result = None
        self.export_image = None
        self._changing = True
        self.reference_list.clear()
        self.target_list.clear()
        for i, face in enumerate(self.reference_faces):
            item = QListWidgetItem(QIcon(pixmap(face_thumbnail(self.reference, face.bbox))), f"Personne {i+1}")
            self.reference_list.addItem(item)
        self._changing = False
        self.reference_view.set_faces(self.reference_faces)
        self.target_view.set_faces(self.target_faces)
        for notice in getattr(self._pipeline.recognizer, "notices", []):
            self.log.appendPlainText(notice)
        if len(self.reference_faces) == 1:
            self.reference_list.setCurrentRow(0)
        elif not self.reference_faces:
            self.log.appendPlainText("Aucun visage dans la référence. Choisir une autre photographie.")
        else:
            self.log.appendPlainText("Plusieurs visages dans la référence : sélectionner la personne recherchée.")
            self.tabs.setCurrentIndex(0)
        if not self.target_faces:
            self.log.appendPlainText("Aucun visage détecté dans la photographie. Aucun remplacement possible.")

    def _reference_selected(self, index):
        if self._changing or index < 0 or index >= len(self.reference_faces):
            return
        ref = self.reference_faces[index]
        self.export_image = None
        self.result = None
        self.after_view.clear_image("Sélection modifiée — lancer un nouveau traitement")
        self.summary.setText("Sélection modifiée — résultat à recalculer")
        self.scores.clear()
        similarities = [cosine(ref.embedding, face.embedding) for face in self.target_faces]
        selected = [i for i, score in enumerate(similarities) if score >= self.match.value()]
        self.analysis = Analysis(self.image.copy(), ref, self.target_faces, similarities, selected,
                                 self._pipeline.recognizer.model_id)
        self.reference_view.set_faces(self.reference_faces, [index])
        self._changing = True
        self.target_list.clear()
        for i, face in enumerate(self.target_faces):
            item = QListWidgetItem(QIcon(pixmap(face_thumbnail(self.image, face.bbox))),
                                   f"Visage {i+1} · similarité {similarities[i]:.3f}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if i in selected else Qt.CheckState.Unchecked)
            self.target_list.addItem(item)
        self._changing = False
        self.target_view.set_faces(self.target_faces, selected, similarities)
        self.log.appendPlainText(f"{len(self.target_faces)} visage(s), {len(selected)} correspondance(s). Vérifier la sélection.")
        if not self._is_busy():
            self._update_actions()

    def _selected(self):
        return [i for i in range(self.target_list.count())
                if self.target_list.item(i).checkState() == Qt.CheckState.Checked]

    def _select_reference(self, index):
        if not self._is_busy():
            self.reference_list.setCurrentRow(index)

    def _toggle_target(self, index):
        if self._is_busy() or index >= self.target_list.count():
            return
        item = self.target_list.item(index)
        item.setCheckState(Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked)

    def _selection_changed(self, item):
        if self._changing or self.analysis is None:
            return
        self.export_image = None
        self.result = None
        self.after_view.clear_image("Sélection modifiée — lancer un nouveau traitement")
        self.summary.setText("Sélection modifiée — résultat à recalculer")
        self.scores.clear()
        self.target_view.set_faces(self.target_faces, self._selected(), self.analysis.similarities)
        self._update_actions()

    def _threshold_changed(self):
        if self.analysis is not None:
            self._reference_selected(self.reference_list.currentRow())

    def _anonymize(self):
        if self._is_busy():
            return
        if self.analysis is None:
            self._auto_run = True
            self._detect()
            return
        selected = self._selected()
        if not selected:
            self.log.appendPlainText("Aucune correspondance ou aucun visage sélectionné : aucun remplacement.")
            return
        settings = self.settings()
        analysis = self.analysis
        self.export_image = None
        def function(progress):
            pipeline = self.service.pipeline(settings)
            return pipeline.run(analysis, settings, selected, progress)
        self._job(function, self._show_result)

    def _show_result(self, result):
        self.result = result
        self.export_image = result.image if result.success else None
        self.before_view.set_image(self.image)
        self.after_view.set_image(result.image)
        self.summary.setText(f"{'Résultat validé selon le seuil expérimental' if result.success else 'Échec — original conservé'} · "
                             f"{result.detected} détecté(s) · {result.recognized} reconnu(s) · {result.replaced} remplacé(s) · {result.attempts} essai(s)")
        lines = [f"Visage {r.index+1} · avant {r.before:.3f} · après {r.after if r.after is not None else 'non mesurable'} · {r.message}"
                 for r in result.reports]
        lines.append("Similarité cosinus : un score plus bas ne constitue pas une garantie d'anonymat.")
        self.scores.setPlainText("\n".join(lines))
        for warning in dict.fromkeys(result.warnings):
            self.log.appendPlainText(warning)
        self._overlays()
        self.tabs.setCurrentIndex(2)

    def _overlays(self):
        if self.analysis is None or self.result is None:
            return
        self.before_view.set_faces(self.target_faces, self._selected(), self.analysis.similarities, self.boxes.isChecked())
        self.after_view.set_faces(self.target_faces, self._selected(), visible=self.boxes.isChecked())
        self.after_view.set_masks(self.result.masks, self.masks.isChecked())

    def _retry(self):
        self.service.clear_session()
        self.mode.setCurrentIndex(0)
        self._anonymize()

    def _fallback(self):
        if self.analysis is None or not self._selected():
            return
        if QMessageBox.question(self, "Masquage de secours", "Appliquer un rectangle noir opaque aux visages sélectionnés ?\n"
                                "Cette action produit un masquage explicite, distinct du remplacement SimSwap.") != QMessageBox.StandardButton.Yes:
            return
        selected = self._selected()
        protected = [f for i, f in enumerate(self.analysis.faces) if i not in selected]
        self.export_image = emergency_mask(self.image, [self.analysis.faces[i] for i in selected], protected)
        self.before_view.set_image(self.image)
        self.after_view.set_image(self.export_image)
        self.summary.setText(f"Masquage opaque de secours · {len(selected)} visage(s) sélectionné(s)")
        self.scores.setPlainText("Masquage de secours appliqué aux visages sélectionnés.")
        self.tabs.setCurrentIndex(2)
        self._update_actions()

    def _save(self):
        if self.export_image is None:
            return
        stem = Path(self.image_path).stem + "_anonymise.png"
        file, _ = QFileDialog.getSaveFileName(self, "Enregistrer le résultat", str(Path(self.image_path).parent/stem),
                                             "PNG (*.png);;JPEG (*.jpg);;WebP (*.webp)")
        if not file:
            return
        image = self.export_image
        self._job(lambda progress: save_image(file, image, [self.image_path, self.reference_path]),
                  lambda _: self.log.appendPlainText("Résultat enregistré : " + file))

    def _models(self):
        ModelDialog(self.settings(), self).exec()
        # Les poids peuvent avoir changé ; abandonner les instances chargées et la sélection.
        self.service = ProcessingService()
        self._pipeline = None
        self._invalidate()
        available512 = (Settings().models/"simswap/512/550000_net_G.pth").is_file()
        self.resolution.model().item(1).setEnabled(available512)
        if not available512:
            self.resolution.setCurrentIndex(0)
        self._update_actions()

    def _load_bank_choices(self):
        current = self.identity.currentData()
        self.identity.clear()
        path = Settings().bank / "bank.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"identities": []}
            for entry in data["identities"]:
                portrait = (Settings().bank/entry["file"]).resolve()
                if portrait.is_relative_to(Settings().bank.resolve()):
                    self.identity.addItem(QIcon(str(portrait)), entry["key"], entry["key"])
            self.identity.setIconSize(QSize(48, 48))
            index = self.identity.findData(current)
            if index >= 0:
                self.identity.setCurrentIndex(index)
        except (ValueError, OSError, KeyError) as exc:
            self.log.appendPlainText(f"Banque illisible : {exc}")

    def _import_bank(self):
        folder = QFileDialog.getExistingDirectory(self, "Dossier de visages synthétiques", str(Settings().root))
        if not folder:
            return
        if QMessageBox.question(self, "Origine de la banque", "Confirmez-vous que tous les portraits de ce dossier sont des visages fictifs générés ?\n"
                                "Les photographies de personnes réelles ne doivent pas servir d'identités de remplacement.",
                                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        settings = self.settings()
        self._job(lambda progress: self.service.pipeline(settings).bank.import_folder(folder, synthetic_confirmed=True, progress=progress),
                  self._bank_done)

    def _generate_bank(self):
        count, ok = QInputDialog.getInt(self, "StyleGAN2-ADA FFHQ", "Nombre de visages synthétiques", 20, 1, 1000)
        if not ok:
            return
        start, ok = QInputDialog.getInt(self, "StyleGAN2-ADA FFHQ", "Première seed (reproductible)", 0, 0, 2**31-1001)
        if not ok:
            return
        settings = self.settings()
        def function(progress):
            from ..backends.stylegan_backend import StyleGANBackend
            pipeline = self.service.pipeline(settings)
            self.service.release_swapper()  # libérer SimSwap pendant la génération FFHQ
            generator = StyleGANBackend(settings)
            return pipeline.bank.generate(generator, range(start, start+count), progress)
        self._job(function, self._bank_done)

    def _bank_done(self, result):
        count, rejected = result
        self._load_bank_choices()
        self.log.appendPlainText(f"{count} identité(s) synthétique(s) ajoutée(s).")
        for message in rejected:
            self.log.appendPlainText(message)

    def closeEvent(self, event):
        if self._is_busy():
            self.log.appendPlainText("Attendre la fin du travail en cours avant de fermer l'application.")
            event.ignore()
        else:
            event.accept()
