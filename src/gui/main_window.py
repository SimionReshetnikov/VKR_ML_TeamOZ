import os
from utils.report_generator import ReportGenerator
from PyQt5 import QtWidgets
from gui.video_player import VideoPlayerWidget
from gui.inference_thread import InferenceThread
from utils.config import Config

class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки")
        self.config = config
        layout = QtWidgets.QFormLayout(self)

        self.det_ckpt = QtWidgets.QLineEdit(self.config.get("detector_ckpt"))
        self.cls_ckpt = QtWidgets.QLineEdit(self.config.get("classifier_ckpt"))
        self.device = QtWidgets.QComboBox()
        self.device.addItems(["cuda", "cpu"])
        self.device.setCurrentText(self.config.get("device"))
        self.defect_thr = QtWidgets.QDoubleSpinBox()
        self.defect_thr.setRange(0, 1)
        self.defect_thr.setSingleStep(0.01)
        self.defect_thr.setValue(float(self.config.get("defect_threshold", 0.7)))

        self.good_thr = QtWidgets.QDoubleSpinBox()
        self.good_thr.setRange(0, 1)
        self.good_thr.setSingleStep(0.01)
        self.good_thr.setValue(float(self.config.get("good_threshold", 0.5)))

        layout.addRow("Путь к детектору", self.det_ckpt)
        layout.addRow("Путь к классификатору", self.cls_ckpt)
        layout.addRow("Устройство (cuda/cpu)", self.device)
        layout.addRow("Порог дефекта (0-1)", self.defect_thr)
        layout.addRow("Порог good-класса (0-1)", self.good_thr)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addRow(btns)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def get_config(self):
        return {
            "detector_ckpt": self.det_ckpt.text(),
            "classifier_ckpt": self.cls_ckpt.text(),
            "device": self.device.currentText(),
            "defect_threshold": self.defect_thr.value(),
            "good_threshold": self.good_thr.value()
        }

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InsPLAD Defect Detection")
        self.resize(1200, 800)

        self.config = Config()
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        self.layout = QtWidgets.QVBoxLayout(central_widget)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_open_file = QtWidgets.QPushButton("Открыть видеофайл", self)
        self.btn_open_rtsp = QtWidgets.QPushButton("RTSP-поток", self)
        self.btn_start = QtWidgets.QPushButton("Старт анализа", self)
        self.btn_stop = QtWidgets.QPushButton("Стоп", self)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_open_file)
        btn_layout.addWidget(self.btn_open_rtsp)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)

        self.btn_config = QtWidgets.QPushButton("Настройки", self)
        btn_layout.addWidget(self.btn_config)
        self.btn_config.clicked.connect(self.open_config_dialog)

        self.btn_save_report = QtWidgets.QPushButton("Сохранить отчёт", self)
        self.btn_save_report.setEnabled(False)
        btn_layout.addWidget(self.btn_save_report)
        self.btn_save_report.clicked.connect(self.save_report)
        self.analysis_results = []
        self.analysis_frames = []

        self.layout.addLayout(btn_layout)
        self.video_player = VideoPlayerWidget(self)
        self.layout.addWidget(self.video_player)
        self.statusBar().showMessage("Готово")

        self.inference_thread = None
        self.video_path = None
        self.rtsp_url = None

        self.btn_open_file.clicked.connect(self.open_file)
        self.btn_open_rtsp.clicked.connect(self.open_rtsp)
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.video_player.frame_ready.connect(self.on_frame_ready)

    def open_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выбрать видео", "", "Видео (*.mp4 *.avi *.mov *.mkv)")
        if file_path:
            self.video_path = file_path
            self.rtsp_url = None
            self.video_player.open_video(file_path)
            self.statusBar().showMessage(f"Открыт файл: {file_path}")

    def open_rtsp(self):
        url, ok = QtWidgets.QInputDialog.getText(self, "RTSP поток", "Введите RTSP URL:")
        if ok and url:
            self.rtsp_url = url
            self.video_path = None
            self.video_player.open_video(url)
            self.statusBar().showMessage(f"Открыт RTSP: {url}")

    def start_analysis(self):
        if not self.video_player.is_opened():
            QtWidgets.QMessageBox.warning(self, "Нет видео", "Сначала выберите файл или RTSP.")
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.statusBar().showMessage("Анализ запущен...")
        self.inference_thread = InferenceThread(self.config)
        self.inference_thread.result_ready.connect(self.on_result_ready)
        self.inference_thread.result_full_ready.connect(self.on_result_full_ready)
        self.inference_thread.finished.connect(self.on_analysis_finished)
        self.video_player.set_analysis_mode(True)
        self.inference_thread.start()

    def stop_analysis(self):
        if self.inference_thread is not None:
            self.inference_thread.stop()
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.statusBar().showMessage("Анализ остановлен")
        self.video_player.set_analysis_mode(False)

    def on_frame_ready(self, frame):
        if self.inference_thread is not None and self.inference_thread.isRunning():
            self.inference_thread.put_frame(frame)
        else:
            self.video_player.show_frame(frame)

    def on_result_ready(self, frame_vis):
        self.video_player.show_frame(frame_vis)

    def on_result_full_ready(self, frame_vis, frame_result):
        self.video_player.show_frame(frame_vis)
        self.analysis_frames.append(frame_vis.copy())
        self.analysis_results.append(frame_result)

    def on_analysis_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Анализ завершён")
        self.video_player.set_analysis_mode(False)
        self.btn_save_report.setEnabled(True)

    def save_report(self):
        if not self.analysis_results or not self.analysis_frames:
            QtWidgets.QMessageBox.warning(self, "Нет данных", "Нет результатов для отчёта.")
            return
        video_name = os.path.splitext(os.path.basename(self.video_path or "rtsp_stream"))[0]
        gen = ReportGenerator()
        total_frames = len(self.analysis_results)
        defect_thr = float(self.config.get("defect_threshold", 0.7))
        defect_frames = sum(
            any(
                "good" not in obj["defect_class"] and obj["defect_conf"] >= defect_thr
                for obj in frame["objects"]
            )
            for frame in self.analysis_results
        )
        extra_metrics = {
            "Всего кадров": total_frames,
            f"Кадров с дефектами (>={int(defect_thr*100)}%)": defect_frames,
            "Доля кадров с дефектами": f"{defect_frames / (total_frames or 1):.2%}",
        }
        path = gen.save_report(
            self.analysis_results,
            video_name=video_name,
            frame_images=self.analysis_frames,
            extra_metrics=extra_metrics,
        )
        QtWidgets.QMessageBox.information(self, "Отчёт", f"Отчёт сохранён:\n{path}")
        self.btn_save_report.setEnabled(False)
        self.analysis_results.clear()
        self.analysis_frames.clear()

    def open_config_dialog(self):
        dlg = ConfigDialog(self.config, self)
        if dlg.exec_():
            new_cfg = dlg.get_config()
            for k, v in new_cfg.items():
                self.config.set(k, v)
            if self.inference_thread is not None and self.inference_thread.isRunning():
                self.inference_thread.update_config(self.config)
            QtWidgets.QMessageBox.information(self, "Настройки", "Конфигурация обновлена!")