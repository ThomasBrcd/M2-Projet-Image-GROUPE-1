import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap, QPen, QColor, QBrush, QPainter
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem


def pixmap(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy())


class FaceRect(QGraphicsRectItem):
    def __init__(self, bbox, index, selected, callback):
        x1, y1, x2, y2 = map(float, bbox)
        super().__init__(x1, y1, x2-x1, y2-y1)
        self.index = index
        self.callback = callback
        pen = QPen(QColor("#35df9b" if selected else "#f1b74c"), 2)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(53, 223, 155, 24) if selected else QColor(241, 183, 76, 16)))
        self.setZValue(2)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        self.callback(self.index)
        event.accept()


class ImageViewer(QGraphicsView):
    faceClicked = Signal(int)

    def __init__(self, placeholder="Choisir une photographie", parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#141b26"))
        self.setMinimumSize(220, 200)
        self._image = None
        self._boxes = []
        self._masks = []
        self.scene().addText(placeholder).setDefaultTextColor(QColor("#9baac0"))

    def set_image(self, image):
        self.scene().clear()
        self._boxes.clear()
        self._masks.clear()
        self._image = self.scene().addPixmap(pixmap(image))
        self.scene().setSceneRect(self._image.boundingRect())
        self.reset_zoom()

    def clear_image(self, placeholder="Aucun résultat"):
        self.scene().clear()
        self._image = None
        self._boxes.clear()
        self._masks.clear()
        text = self.scene().addText(placeholder)
        text.setDefaultTextColor(QColor("#9baac0"))
        self.scene().setSceneRect(text.boundingRect())
        self.resetTransform()

    def reset_zoom(self):
        if self._image:
            self.fitInView(self._image, Qt.AspectRatioMode.KeepAspectRatio)

    def set_faces(self, faces, selected=(), scores=None, visible=True):
        for item in self._boxes:
            self.scene().removeItem(item)
        self._boxes.clear()
        for index, face in enumerate(faces):
            box = FaceRect(face.bbox, index, index in selected, self.faceClicked.emit)
            box.setVisible(visible)
            self.scene().addItem(box)
            text = self.scene().addText(f"{index+1}" + (f" · {scores[index]:.2f}" if scores is not None else ""))
            text.setDefaultTextColor(QColor("#35df9b" if index in selected else "#f1b74c"))
            text.setPos(float(face.bbox[0]), float(face.bbox[1]))
            text.setScale(max(1, self.sceneRect().width()/1000))
            text.setZValue(3)
            text.setVisible(visible)
            self._boxes.extend([box, text])

    def set_masks(self, masks, visible):
        for item in self._masks:
            self.scene().removeItem(item)
        self._masks.clear()
        if not visible:
            return
        for x, y, alpha in masks:
            rgba = np.zeros((*alpha.shape, 4), np.uint8)
            rgba[..., 1] = 220
            rgba[..., 2] = 180
            rgba[..., 3] = np.rint(alpha*100).astype(np.uint8)
            h, w = alpha.shape
            qimage = QImage(rgba.data, w, h, rgba.strides[0], QImage.Format.Format_RGBA8888).copy()
            item = self.scene().addPixmap(QPixmap.fromImage(qimage))
            item.setPos(x, y)
            item.setZValue(1)
            self._masks.append(item)

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        if .01 < self.transform().m11()*factor < 30:
            self.scale(factor, factor)
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image and self.transform().isIdentity():
            self.reset_zoom()
