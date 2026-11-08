from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from .detection import Face
from .recognition import cosine
from .errors import FaceAnonError, SelectionError
from .blending import compose
from .validation import validate_face

Progress = Callable[[int, str], None]


@dataclass
class Analysis:
    image: np.ndarray
    reference_face: Face
    faces: list[Face]
    similarities: list[float]
    selected: list[int]
    model_id: str


@dataclass
class FaceReport:
    index: int
    before: float
    after: float | None
    accepted: bool
    message: str
    independent_after: float | None = None


@dataclass
class Result:
    image: np.ndarray
    success: bool
    detected: int
    recognized: int
    replaced: int
    reports: list[FaceReport] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    masks: list = field(default_factory=list)  # ROI (x, y, alpha), évite N masques pleine résolution
    identity: str | None = None
    attempts: int = 0


class Pipeline:
    def __init__(self, recognizer, swapper, bank, independent_recognizer=None):
        self.recognizer = recognizer
        self.swapper = swapper
        self.bank = bank
        self.independent_recognizer = independent_recognizer
        self._session_identities = []  # embeddings réels uniquement en RAM

    def clear_session(self):
        self._session_identities.clear()

    def analyze(self, reference, image, settings, reference_index=None, progress=lambda p, m: None):
        progress(5, "Détection de la référence")
        refs = self.recognizer.detect(reference)
        if not refs:
            raise SelectionError("Aucun visage dans la photographie de référence")
        if reference_index is None:
            if len(refs) > 1:
                raise SelectionError("Plusieurs visages dans la référence : sélectionner explicitement la personne")
            reference_index = 0
        if reference_index < 0 or reference_index >= len(refs):
            raise SelectionError("Indice de référence invalide")
        progress(35, "Détection de tous les visages de la photographie")
        faces = self.recognizer.detect(image)
        ref = refs[reference_index]
        similarities = [cosine(ref.embedding, face.embedding) for face in faces]
        selected = [i for i, score in enumerate(similarities) if score >= settings.match_threshold]
        progress(100, f"{len(faces)} visage(s), {len(selected)} correspondance(s)")
        return Analysis(image.copy(), ref, faces, similarities, selected, self.recognizer.model_id)

    def run(self, analysis, settings, selected=None, progress=lambda p, m: None):
        if analysis.model_id != self.recognizer.model_id:
            raise SelectionError("Le modèle de reconnaissance a changé ; relancer la détection")
        indices = list(analysis.selected if selected is None else selected)
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(analysis.faces) for i in indices):
            raise SelectionError("Sélection de visages invalide")
        recognized = sum(s >= settings.match_threshold for s in analysis.similarities)
        result = Result(analysis.image.copy(), False, len(analysis.faces), recognized, 0)
        result.warnings.append("Les scores cosinus et les seuils expérimentaux ne garantissent pas l'anonymat.")
        if not indices:
            result.warnings.append("Aucun visage sélectionné ou aucune correspondance : aucun remplacement effectué.")
            progress(100, "Aucun remplacement")
            return result
        originals = [analysis.reference_face.embedding] + [analysis.faces[i].embedding for i in indices]
        protected = [f for i, f in enumerate(analysis.faces) if i not in indices]
        best = None
        best_score = float("inf")
        last_reports = []
        runtime_warnings = []
        try:
            progress(5, "Choix d'une identité synthétique admissible")
            candidates = self.bank.candidates(originals, settings.candidate_max_similarity,
                                              settings.manual_identity if settings.identity_mode == "manual" else None,
                                              float(analysis.image.mean()) if settings.attribute_matching else None)
            if settings.identity_mode == "manual" and not settings.manual_identity:
                raise SelectionError("Sélectionner une identité de la banque")
            if settings.identity_mode == "session":
                for embedding, key in self._session_identities:
                    if cosine(embedding, analysis.reference_face.embedding) >= settings.match_threshold:
                        candidates.sort(key=lambda e: e.key != key)
                        break
            attempts = candidates[:1 if settings.identity_mode == "manual" else settings.max_attempts]
            for attempt, identity in enumerate(attempts):
                result.attempts += 1
                current = analysis.image.copy()
                masks = []
                warnings = []
                donor = self.bank.image(identity)
                for n, index in enumerate(indices):
                    progress(10+int(65*(attempt+(n/len(indices)))/len(attempts)),
                             f"SimSwap : visage {n+1}/{len(indices)}, essai {attempt+1}/{len(attempts)}")
                    # Toujours préserver la géométrie du cliché original, même pour plusieurs occurrences.
                    patch = self.swapper.swap(analysis.image, analysis.faces[index], donor, identity.face,
                                              segmentation=settings.segmentation)
                    current, mask = compose(current, patch, protected, settings.photometric)
                    masks.append(mask)
                    warnings.extend(patch.warnings)
                    runtime_warnings.extend(patch.warnings)
                progress(10+int(65*(attempt+1)/len(attempts)), "Redétection et validation des visages remplacés")
                reports = []
                for index in indices:
                    independent_score = None
                    check = validate_face(self.recognizer, current, analysis.faces[index],
                                          analysis.reference_face.embedding, settings.validation_max_similarity)
                    if check.accepted and self.independent_recognizer is not None:
                        # Le second modèle recalcule aussi l'original ; aucun mélange d'espaces d'embeddings.
                        original_candidates = self.independent_recognizer.detect(analysis.image)
                        from .detection import intersection_over_union
                        aligned = [f for f in original_candidates if intersection_over_union(f.bbox, analysis.faces[index].bbox) > .5]
                        if len(aligned) != 1:
                            check.accepted = False
                            check.message = "Second modèle : référence locale absente ou ambiguë"
                        else:
                            independent = validate_face(self.independent_recognizer, current, aligned[0],
                                                        aligned[0].embedding, settings.independent_validation_max_similarity)
                            independent_score = independent.similarity
                            check.accepted = independent.accepted
                            check.message += f" ; second modèle : {independent.message} (cos={independent.similarity})"
                    reports.append(FaceReport(index, analysis.similarities[index], check.similarity,
                                               check.accepted, check.message, independent_score))
                last_reports = reports
                if all(r.accepted for r in reports):
                    score = max(r.after for r in reports)
                    if score < best_score:
                        best_score = score
                        best = (current, masks, reports, warnings, identity.key)
                    if settings.identity_mode == "session":
                        break
        except FaceAnonError as exc:
            result.warnings.append(str(exc))
        if best is None:
            result.reports = last_reports
            result.warnings.append("Aucun essai entièrement validé. Original conservé ; masquage de secours disponible sur demande explicite.")
        else:
            result.image, result.masks, result.reports, warnings, result.identity = best
            result.success = True
            result.replaced = len(indices)
            result.warnings.extend(warnings)
            if settings.identity_mode == "session":
                self._session_identities = [(e, k) for e, k in self._session_identities
                                            if cosine(e, analysis.reference_face.embedding) < settings.match_threshold]
                self._session_identities.append((analysis.reference_face.embedding.copy(), result.identity))
        result.warnings.extend(runtime_warnings)
        result.warnings.extend(getattr(self.swapper, "notices", []))
        result.warnings = list(dict.fromkeys(result.warnings))
        progress(100, f"{result.replaced} visage(s) remplacé(s)" if result.success else "Échec ou absence de remplacement")
        return result
