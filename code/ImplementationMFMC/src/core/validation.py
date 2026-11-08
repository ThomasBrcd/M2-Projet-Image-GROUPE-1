from dataclasses import dataclass
from .detection import intersection_over_union
from .recognition import cosine


@dataclass
class Validation:
    accepted: bool
    similarity: float | None
    message: str


def validate_face(recognizer, image, original_face, reference_embedding, maximum):
    """Nouvelle détection réelle. Une détection absente/ambiguë n'est jamais un succès."""
    candidates = sorted(((intersection_over_union(original_face.bbox, f.bbox), f)
                         for f in recognizer.detect(image)), key=lambda x: x[0], reverse=True)
    if not candidates or candidates[0][0] < .25:
        return Validation(False, None, "Visage remplacé non redétecté ; confidentialité non évaluée")
    if len(candidates) > 1 and candidates[1][0] > .25:
        return Validation(False, None, "Redétection ambiguë dans la région remplacée")
    embedding = candidates[0][1].embedding
    similarity = max(cosine(embedding, original_face.embedding), cosine(embedding, reference_embedding))
    accepted = similarity <= maximum
    return Validation(accepted, similarity, "Seuil expérimental satisfait" if accepted else "Identité originale encore trop similaire")
