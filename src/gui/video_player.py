import cv2
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np

class VideoPlayerWidget(QtWidgets.QLabel):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.analysis_mode = False
        self.frame_lock = threading.Lock()
        self.current_frame = None

    def open_video(self, path_or_url):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(path_or_url)
        self.timer.start(30)  # ~30 FPS

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def set_analysis_mode(self, enabled):
        self.analysis_mode = enabled

    def next_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self.timer.stop()
            return
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
        with self.frame_lock:
            self.current_frame = frame.copy()
        # Если анализ — сигнал наверх, иначе отображаем
        if self.analysis_mode:
            self.frame_ready.emit(frame)
        else:
            self.show_frame(frame)

    def show_frame(self, frame):
        # Преобразовать BGR (cv2) -> RGB (Qt)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(pixmap.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def sizeHint(self):
        return QtCore.QSize(960, 540)

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(event)