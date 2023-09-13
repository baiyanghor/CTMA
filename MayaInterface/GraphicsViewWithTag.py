from PySide2.QtCore import *
from PySide2.QtWidgets import *


class GraphicsViewWithTag(QGraphicsView):
    GV_STATUS_VIEWING = 1
    GV_STATUS_TAGGING = 2
    
    def __init__(self, parent = None):
        super(GraphicsViewWithTag, self).__init__(parent)
        self.zoomInScale = 1.15
        self.zoomOutScale = 1 / self.zoomInScale


    def wheelEvent(self, event):
        if(event.angleDelta().y() > 0):
            zoomFactor = self.zoomInScale
        else:
            zoomFactor = self.zoomOutScale
        self.scale(zoomFactor, zoomFactor)
    
    def mousePressEvent(self, event):
        if event.modifiers() == Qt.AltModifier:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        elif event.modifiers() != Qt.ShiftModifier:
            self.setDragMode(QGraphicsView.RubberBandDrag)
            
        QGraphicsView.mousePressEvent(self, event)        
    
    def mouseReleaseEvent(self, event):        
        QGraphicsView.mouseReleaseEvent(self, event)
        self.setDragMode(QGraphicsView.NoDrag)


            