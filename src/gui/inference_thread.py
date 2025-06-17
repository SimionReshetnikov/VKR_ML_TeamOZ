from PyQt5 import QtCore
import queue
import time
import numpy as np

from inference.engine import InferenceEngine
from utils.config import Config

class InferenceThread(QtCore.QThread):
    result_ready = QtCore.pyqtSignal(np.ndarray)
    result_full_ready = QtCore.pyqtSignal(np.ndarray, dict)

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.frame_queue = queue.Queue(maxsize=2)
        self._running = True
        self.config = config or Config()
        self.engine = InferenceEngine(self.config)
        self.daemon = True
        self.frame_idx = 0

    def put_frame(self, frame):
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def update_config(self, config):
        self.config = config
        self.engine.reload_config(self.config)

    def run(self):
        self.frame_idx = 0
        while self._running:
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            results = self.engine.infer(frame)
            vis = self.engine.draw_results(frame, results, draw_heatmap=True)
            self.result_ready.emit(vis)
            frame_result = {
                "frame_idx": self.frame_idx,
                "objects": results
            }
            self.result_full_ready.emit(vis, frame_result)
            self.frame_idx += 1
            time.sleep(0.01)
        self.quit()

    def stop(self):
        self._running = False
        self.wait()