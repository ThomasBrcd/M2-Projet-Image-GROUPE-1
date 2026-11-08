from pathlib import Path
import cv2
import numpy as np
from ..core.detection import Face
from ..core.recognition import normalized
from ..core.errors import ModelError
from ..utils.model_manager import ModelManager, sha256


class InsightFaceBackend:
    """Détection SCRFD + ArcFace buffalo_l 0.7.3 ; chemins locaux obligatoires.

    Les embeddings de ce modèle ne sont JAMAIS injectés dans SimSwap.
    get() de l'ArcFace ONNX officiel assure alignement 112 et normalisation propres au modèle.
    """
    def __init__(self, settings, recognition_path=None):
        manager = ModelManager(settings)
        detector_path = manager.require("detector")
        recognition_path = Path(recognition_path) if recognition_path else manager.require("recognizer")
        if not recognition_path.is_file():
            raise ModelError(f"Modèle de reconnaissance absent : {recognition_path}")
        try:
            import onnxruntime as ort
            ort.disable_telemetry_events()
            from insightface.model_zoo import get_model
            from insightface.app.common import Face as InsightFace
            self._face_type = InsightFace
            available = ort.get_available_providers()
            self.notices = []
            # Le profil de base privilégie ONNX CPU, tout en accélérant SimSwap avec PyTorch CUDA.
            providers = ["CPUExecutionProvider"]
            if settings.device != "cpu" and "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")
            self.detector = get_model(str(detector_path), providers=providers)
            self.recognizer = get_model(str(recognition_path), providers=providers)
            if getattr(self.detector, "taskname", None) != "detection":
                raise ModelError("Le checkpoint de détection n'est pas un modèle InsightFace de détection")
            if getattr(self.recognizer, "taskname", None) != "recognition":
                raise ModelError("Le checkpoint ArcFace ONNX n'est pas compatible")
            ctx = -1 if providers == ["CPUExecutionProvider"] else 0
            self.detector.prepare(ctx_id=ctx, input_size=(640, 640), det_thresh=.6)
            self.recognizer.prepare(ctx_id=ctx)
            self._model_id = "arcface-onnx:" + sha256(recognition_path)
            self.notices.append("Reconnaissance ONNX : " + ", ".join(self.recognizer.session.get_providers()))
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Initialisation InsightFace impossible : {exc}") from exc

    @property
    def model_id(self):
        return self._model_id

    def detect(self, bgr):
        try:
            boxes, landmarks = self._detect_boxes(bgr)
            if len(boxes) and landmarks is None:
                raise ModelError("Le détecteur doit fournir cinq landmarks")
            result = []
            for box, points in zip(boxes, landmarks if landmarks is not None else []):
                face = self._face_type(bbox=box[:4], kps=points, det_score=box[4])
                embedding = self.recognizer.get(bgr, face)
                result.append(Face(box[:4].astype(np.float32), points.astype(np.float32),
                                   normalized(embedding), float(box[4])))
            return result
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"Détection/reconnaissance impossible (modèle ou mémoire) : {exc}") from exc

    def _detect_boxes(self, bgr):
        boxes, landmarks = self.detector.detect(bgr, max_num=0, metric="default")
        if len(boxes):
            return boxes, landmarks
        # Un portrait très serré peut dépasser les tailles de visage reconnues à 640.
        # Ajouter une marge réduit sa taille apparente sans changer le modèle ni son seuil.
        height, width = bgr.shape[:2]
        scale = min(1., 640/max(height, width))
        preview = cv2.resize(bgr, (max(1, round(width*scale)), max(1, round(height*scale))))
        pad_y, pad_x = preview.shape[0]//2, preview.shape[1]//2
        canvas = cv2.copyMakeBorder(preview, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT)
        boxes, landmarks = self.detector.detect(canvas, max_num=0, metric="default")
        if len(boxes):
            # Utiliser les rapports réels après arrondi pour revenir aux coordonnées d'entrée.
            factors = np.float32([width/preview.shape[1], height/preview.shape[0]])
            boxes = boxes.copy()
            boxes[:, :4] = (boxes[:, :4]-np.float32([pad_x, pad_y, pad_x, pad_y]))*np.tile(factors, 2)
            if landmarks is not None:
                landmarks = (landmarks-np.float32([pad_x, pad_y]))*factors
        return boxes, landmarks
