# integrated_gui.py
"""
包含：实时摄像头检测 / 视频文件处理
布局：左侧视频区+控制面板，右侧三个监测曲线图+日志
修改：三张实时监测图的Y轴对齐，X轴长度一致（通过统一subplots_adjust实现）
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import cv2
import numpy as np
import queue
from collections import deque

from PyQt5.QtWidgets import (

    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QSpinBox, QGroupBox, QFormLayout,
    QRadioButton, QButtonGroup, QFileDialog, QMessageBox, QLineEdit,
    QSlider, QGraphicsView, QGraphicsScene, QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QWaitCondition, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont, QBrush, QColor

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

from pose_detector import PoseDetector
from state_analyzer import RiskAdaptiveStateAnalyzer
from visualizer import Visualizer
import config


# ==================== 检测工作线程 ====================
class DetectionWorker(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    plot_signal = pyqtSignal(object, float, object, object, float)
    status_signal = pyqtSignal(str, float)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    total_frames_signal = pyqtSignal(int)
    current_frame_signal = pyqtSignal(int)
    video_size_signal = pyqtSignal(int, int)
    paused_state_changed = pyqtSignal(bool)

    def __init__(self, source=0, is_video_file=False, inference_skip=2, model_mode='full'):
        super().__init__()
        self.source = source
        self.is_video_file = is_video_file
        self.inference_skip = inference_skip
        self.model_mode = model_mode
        self.running = False
        self.paused = False

        self.detector = PoseDetector()
        self.analyzer = RiskAdaptiveStateAnalyzer()
        self.visualizer = Visualizer()

        self.frame_count = 0
        self.status = "Standing"
        self.total_frames = 0
        self.detected_frames = 0
        self.fall_frames = 0

        self.cap = None
        self.seek_requested = False
        self.seek_target_frame = 0

        self.pause_condition = QWaitCondition()
        self.pause_mutex = QMutex()

        self.plot_update_counter = 0
        self.plot_update_interval = 3

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.log_signal.emit(f"错误：无法打开视频源 {self.source}")
            self.finished_signal.emit()
            return

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_size_signal.emit(w, h)

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_video_file else 0
        if self.is_video_file:
            self.total_frames_signal.emit(total_frames)
        mode_names = {'raw': '纯YOLO', 'kf': 'YOLO+卡尔曼', 'spectral': 'YOLO+插值', 'full': '完整系统'}
        self.log_signal.emit(f"开始处理，视频源: {self.source}，模型: {mode_names[self.model_mode]}")

        while self.running:
            self.pause_mutex.lock()
            if self.paused:
                self.paused_state_changed.emit(True)
                self.pause_condition.wait(self.pause_mutex)
                self.paused_state_changed.emit(False)
            self.pause_mutex.unlock()

            if self.seek_requested and self.is_video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_target_frame)
                self.frame_count = self.seek_target_frame
                self.seek_requested = False
                self.log_signal.emit(f"跳转至第 {self.seek_target_frame} 帧，曲线已重置")
                self.detector.kf_initialized = False
                self.detector.enhancer.prev_kpts = None
                self.detector.enhancer.prev_confs = None

            ret, frame = self.cap.read()
            if not ret:
                if self.is_video_file:
                    self.log_signal.emit("视频播放完毕")
                break

            do_inference = (self.frame_count % self.inference_skip == 0)

            if do_inference:
                kpts, confs = self._detect_pose_with_mode(frame)
            else:
                kpts = self.detector.last_keypoints
                confs = self.detector.last_confidences

            if kpts is None or np.mean(confs) < 0.1:
                kpts = [[-1, -1]] * 17
                confs = [0.0] * 17
                self.status = "Not detected"
                details = None
                torso_angle = None
                left_ankle = None
                right_ankle = None
                raw_displacement = 0.0
            else:
                self.detected_frames += 1
                h, w = frame.shape[:2]
                is_fall, angle, vertical_cond, aspect = self.analyzer.detect_falling(kpts, w, h)
                is_confirmed = self.analyzer.update_fall_status(is_fall)

                if is_confirmed:
                    self.status = "Falling"
                    self.fall_frames += 1
                    details = (angle, vertical_cond, aspect)
                else:
                    walking = self.analyzer.detect_walking(kpts, w, h)
                    self.status = "Walking" if walking else "Standing"
                    details = None

                torso_angle = self.analyzer.compute_joint_angles(kpts, confs)
                if self.detector.enhancer.reconstruction_error_history:
                    raw_displacement = self.detector.enhancer.reconstruction_error_history[-1]
                else:
                    raw_displacement = 0.0
                left_ankle = kpts[15] if confs[15] > 0.2 else None
                right_ankle = kpts[16] if confs[16] > 0.2 else None

            kf_conv = np.mean(self.detector.convergence_results_history) if self.detector.convergence_results_history else 0.0
            frame = self.visualizer.draw(frame, kpts, confs, self.status, details, kf_conv, None)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.frame_signal.emit(rgb_frame)

            self.plot_update_counter += 1
            if self.plot_update_counter >= self.plot_update_interval:
                self.plot_signal.emit(torso_angle, raw_displacement, left_ankle, right_ankle, kf_conv)
                self.plot_update_counter = 0

            self.status_signal.emit(self.status, kf_conv)

            self.frame_count += 1
            if self.is_video_file:
                self.current_frame_signal.emit(self.frame_count)
            else:
                self.current_frame_signal.emit(0)

            if self.frame_count % 30 == 0:
                progress = f"已处理 {self.frame_count} 帧" if not self.is_video_file else f"进度: {self.frame_count}/{total_frames}"
                self.log_signal.emit(progress)

        self.cap.release()
        self.log_signal.emit(f"处理结束，总帧数: {self.frame_count}，检测率: {self.detected_frames/max(1,self.frame_count)*100:.1f}%，跌倒帧: {self.fall_frames}")
        self.finished_signal.emit()

    def _detect_pose_with_mode(self, frame):
        if self.model_mode == 'raw':
            results = self.detector.model(frame, imgsz=640, conf=0.15, iou=0.45, device=self.detector.device, verbose=False)
            kpts, confs = self.detector._process_yolo_results(results, frame.shape)
            return kpts, confs
        elif self.model_mode == 'kf':
            kpts, confs = self.detector.detect_pose(frame, motion_state=self.status, return_raw=True)[:2]
            if kpts is not None and self.detector.kf_initialized:
                kpts, confs = self.detector.temporal_smoothing_with_distributed_kf(kpts, confs, frame.shape, self.status)
            return kpts, confs
        elif self.model_mode == 'spectral':
            results = self.detector.model(frame, imgsz=640, conf=0.15, iou=0.45, device=self.detector.device, verbose=False)
            kpts, confs = self.detector._process_yolo_results(results, frame.shape)
            if kpts is not None:
                kpts, confs = self.detector.enhancer.adaptive_spectral_interpolation(kpts, confs, frame.shape)
            return kpts, confs
        else:  # full
            return self.detector.detect_pose(frame, motion_state=self.status)

    def stop(self):
        self.running = False
        self.pause_mutex.lock()
        self.paused = False
        self.pause_condition.wakeAll()
        self.pause_mutex.unlock()
        self.wait()

    def pause_resume(self):
        self.pause_mutex.lock()
        self.paused = not self.paused
        if not self.paused:
            self.pause_condition.wakeAll()
        self.pause_mutex.unlock()

    def seek_to_frame(self, frame_num):
        if self.is_video_file:
            self.seek_target_frame = frame_num
            self.seek_requested = True


# ==================== Matplotlib 画布 ====================
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(200)


# ==================== 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("跌倒检测集成系统")
        self.setGeometry(50, 50, 1600, 850)

        self.torso_angle_data = deque(maxlen=70)
        self.raw_disp_data = deque(maxlen=70)
        self.smooth_disp_data = deque(maxlen=70)
        self.kf_conv_data = deque(maxlen=70)
        self.left_ankle_traj = deque(maxlen=70)
        self.right_ankle_traj = deque(maxlen=70)
        self.ewma = None
        self.alpha = 0.3

        self.worker = None
        self.slider_dragging = False
        self.video_width = 1920
        self.video_height = 1080
        self.last_rgb_frame = None

        self.init_ui()

        self.log_queue = queue.Queue()
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.poll_log_queue)
        self.log_timer.start(100)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # ========== 左侧 ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)

        self.video_view = QGraphicsView()
        self.video_view.setMinimumSize(640, 480)
        self.video_view.setStyleSheet("border: 2px solid #aaa; background-color: #222;")
        self.video_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.video_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.video_scene = QGraphicsScene()
        self.video_view.setScene(self.video_scene)
        self.video_pixmap_item = None
        left_layout.addWidget(self.video_view)

        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setEnabled(False)
        self.progress_slider.sliderPressed.connect(self.on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.on_slider_released)
        self.progress_slider.valueChanged.connect(self.on_slider_value_changed)
        left_layout.addWidget(self.progress_slider)

        self.frame_info_label = QLabel("帧: 0 / 0")
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.frame_info_label)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("选择模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["纯YOLO", "YOLO+卡尔曼", "YOLO+谱图插值", "完整系统"])
        model_layout.addWidget(self.model_combo)
        left_layout.addLayout(model_layout)

        ctrl_group = QGroupBox("控制面板")
        ctrl_layout = QFormLayout()

        mode_layout = QHBoxLayout()
        self.radio_camera = QRadioButton("摄像头")
        self.radio_camera.setChecked(True)
        self.radio_video = QRadioButton("视频文件")
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.radio_camera, 0)
        mode_group.addButton(self.radio_video, 1)
        mode_layout.addWidget(self.radio_camera)
        mode_layout.addWidget(self.radio_video)
        ctrl_layout.addRow("功能模式:", mode_layout)

        self.cam_spin = QSpinBox()
        self.cam_spin.setRange(0, 5)
        self.cam_spin.setValue(0)
        ctrl_layout.addRow("摄像头ID:", self.cam_spin)

        file_layout = QHBoxLayout()
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setPlaceholderText("选择视频文件...")
        self.browse_btn = QPushButton("浏览")
        self.browse_btn.clicked.connect(self.browse_video)
        file_layout.addWidget(self.video_path_edit)
        file_layout.addWidget(self.browse_btn)
        ctrl_layout.addRow("视频路径:", file_layout)

        self.skip_spin = QSpinBox()
        self.skip_spin.setRange(1, 5)
        self.skip_spin.setValue(2)
        ctrl_layout.addRow("推理跳帧:", self.skip_spin)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self.start_processing)
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.pause_processing)
        self.pause_btn.setEnabled(False)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.stop_btn)
        ctrl_layout.addRow(btn_layout)

        ctrl_group.setLayout(ctrl_layout)
        left_layout.addWidget(ctrl_group)

        main_layout.addWidget(left_widget, 2)

        # ========== 右侧：三个监测图 ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(5)

        self.canvas1 = PlotCanvas(self, width=6, height=2.5)
        self.canvas2 = PlotCanvas(self, width=6, height=2.5)
        self.canvas3 = PlotCanvas(self, width=6, height=2.5)
        self.ax1 = self.canvas1.fig.add_subplot(111)
        self.ax2 = self.canvas2.fig.add_subplot(111)
        self.ax3 = self.canvas3.fig.add_subplot(111)
        self.init_plots()
        right_layout.addWidget(self.canvas1)
        right_layout.addWidget(self.canvas2)
        right_layout.addWidget(self.canvas3)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        right_layout.addWidget(QLabel("日志输出:"))
        right_layout.addWidget(self.log_text)

        main_layout.addWidget(right_widget, 3)

        self.radio_camera.toggled.connect(self.on_mode_changed)
        self.radio_video.toggled.connect(self.on_mode_changed)
        self.on_mode_changed()

    def init_plots(self):
        """初始化三个子图，并统一边距使坐标轴对齐"""
        # 统一的边距设置，使三个图的绘图区域宽度一致，Y轴标签对齐
        left_margin = 0.15   # 左侧留白
        right_margin = 0.95  # 右侧边界
        top_margin = 0.9
        bottom_margin = 0.2

        # 图1：躯干倾斜角
        self.ax1.set_title('躯干倾斜角 (°)')
        self.ax1.set_ylim(0, 100)
        self.ax1.set_xlabel('水平角度(°)', color='black')
        self.ax1.xaxis.set_label_coords(1.0, -0.12)
        self.ax1.xaxis.label.set_horizontalalignment('right')
        self.ax1.set_ylabel('垂直角度(°)', color='black')
        self.ax1.grid(True, alpha=0.3)
        self.line_angle, = self.ax1.plot([], [], 'm-', linewidth=2)
        self.canvas1.fig.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin)

        # 图2：帧间位移 & KF收敛度
        self.ax2.set_title('帧间位移 & KF收敛度')
        self.ax2.set_ylabel('位移 (px)', color='black')
        self.line_raw, = self.ax2.plot([], [], 'r--', alpha=0.7, label='原始位移')
        self.line_smooth, = self.ax2.plot([], [], 'g-', linewidth=2, label=f'EWMA平滑(α={self.alpha})')
        self.ax2.set_ylim(0, 100)
        self.ax2.tick_params(axis='y', labelcolor='black')
        self.ax2_twin = self.ax2.twinx()
        self.ax2_twin.set_ylabel('KF收敛度', color='blue')
        self.line_conv, = self.ax2_twin.plot([], [], 'b-', linewidth=2, label='KF收敛度')
        self.ax2_twin.set_ylim(0, 1.05)
        self.ax2_twin.tick_params(axis='y', labelcolor='blue')
        self.ax2.legend(loc='upper left')
        self.ax2.grid(True, alpha=0.3)
        self.canvas2.fig.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin)

        # 图3：脚踝轨迹
        self.ax3.set_title('脚踝轨迹')
        self.ax3.set_xlabel('X (px)', color='black')
        self.ax3.xaxis.set_label_coords(1.0, -0.12)
        self.ax3.xaxis.label.set_horizontalalignment('right')
        self.ax3.set_ylabel('Y (px)', color='black')
        self.ax3.invert_yaxis()
        self.ax3.set_xlim(0, self.video_width)
        self.ax3.set_ylim(self.video_height, 0)
        self.scat_left = self.ax3.scatter([], [], c='cyan', s=10, label='左脚踝')
        self.scat_right = self.ax3.scatter([], [], c='orange', s=10, label='右脚踝')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        self.canvas3.fig.subplots_adjust(left=left_margin, right=right_margin, top=top_margin, bottom=bottom_margin)

        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

    def on_mode_changed(self):
        is_video = self.radio_video.isChecked()
        is_camera = self.radio_camera.isChecked()
        self.cam_spin.setEnabled(is_camera)
        self.video_path_edit.setEnabled(is_video)
        self.browse_btn.setEnabled(is_video)
        self.progress_slider.setEnabled(is_video)

    def browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.video_path_edit.setText(path)

    def start_processing(self):
        if self.radio_video.isChecked():
            source = self.video_path_edit.text()
            if not source or not os.path.exists(source):
                QMessageBox.warning(self, "错误", "请选择有效的视频文件")
                return
            is_video_file = True
        else:
            source = self.cam_spin.value()
            is_video_file = False

        mode_map = {0: 'raw', 1: 'kf', 2: 'spectral', 3: 'full'}
        model_mode = mode_map[self.model_combo.currentIndex()]

        self.worker = DetectionWorker(
            source=source,
            is_video_file=is_video_file,
            inference_skip=self.skip_spin.value(),
            model_mode=model_mode
        )
        self.worker.frame_signal.connect(self.update_frame)
        self.worker.plot_signal.connect(self.update_plots)
        self.worker.status_signal.connect(self.update_status)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.processing_finished)
        self.worker.video_size_signal.connect(self.set_video_size)
        self.worker.paused_state_changed.connect(self.on_paused_state_changed)
        if is_video_file:
            self.worker.total_frames_signal.connect(self.set_total_frames)
            self.worker.current_frame_signal.connect(self.update_progress)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        self.append_log("启动处理...")
        self.clear_plot_data()
        if not is_video_file:
            self.frame_info_label.setText("帧: 实时")
            self.progress_slider.setEnabled(False)

    def set_video_size(self, w, h):
        self.video_width = w
        self.video_height = h
        self.ax3.set_xlim(0, w)
        self.ax3.set_ylim(h, 0)
        self.canvas3.draw_idle()

    def pause_processing(self):
        if self.worker:
            self.worker.pause_resume()

    def on_paused_state_changed(self, is_paused):
        if is_paused:
            self.pause_btn.setText("继续")
        else:
            self.pause_btn.setText("暂停")

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.processing_finished()
        self.append_log("已手动停止")

    def processing_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("暂停")
        self.stop_btn.setEnabled(False)
        self.worker = None

    def update_frame(self, rgb_frame):
        self.last_rgb_frame = rgb_frame
        self._redraw_frame(rgb_frame)

    def _redraw_frame(self, rgb_frame):
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        view_size = self.video_view.size()
        scaled_pixmap = pixmap.scaled(view_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_scene.clear()
        self.video_scene.setBackgroundBrush(QBrush(QColor(34, 34, 34)))
        self.video_pixmap_item = self.video_scene.addPixmap(scaled_pixmap)
        self.video_scene.setSceneRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.last_rgb_frame is not None:
            self._redraw_frame(self.last_rgb_frame)

    def update_status(self, status, kf_conv):
        pass

    def update_plots(self, torso_angle, raw_disp, left_ankle, right_ankle, kf_conv):
        self.torso_angle_data.append(torso_angle if torso_angle is not None else np.nan)
        self.raw_disp_data.append(raw_disp)
        if self.ewma is None:
            self.ewma = raw_disp
        else:
            self.ewma = self.alpha * raw_disp + (1 - self.alpha) * self.ewma
        self.smooth_disp_data.append(self.ewma)
        self.kf_conv_data.append(kf_conv)
        if left_ankle is not None and left_ankle[0] > 0:
            self.left_ankle_traj.append(left_ankle)
        if right_ankle is not None and right_ankle[0] > 0:
            self.right_ankle_traj.append(right_ankle)

        x_vals = list(range(len(self.torso_angle_data)))
        self.line_angle.set_data(x_vals, list(self.torso_angle_data))
        self.ax1.set_xlim(0, max(10, len(self.torso_angle_data)))
        angles_clean = [a for a in self.torso_angle_data if not np.isnan(a)]
        if angles_clean:
            self.ax1.set_ylim(0, max(100, max(angles_clean) + 10))
        self.canvas1.draw_idle()

        self.line_raw.set_data(x_vals, list(self.raw_disp_data))
        self.line_smooth.set_data(x_vals, list(self.smooth_disp_data))
        self.line_conv.set_data(x_vals, list(self.kf_conv_data))
        self.ax2.set_xlim(0, max(10, len(self.raw_disp_data)))
        if self.raw_disp_data:
            max_disp = max(self.raw_disp_data)
            self.ax2.set_ylim(0, max(100, max_disp * 1.1))
        self.canvas2.draw_idle()

        if self.left_ankle_traj:
            self.scat_left.set_offsets(np.array(self.left_ankle_traj))
        if self.right_ankle_traj:
            self.scat_right.set_offsets(np.array(self.right_ankle_traj))
        self.canvas3.draw_idle()

    def clear_plot_data(self):
        self.torso_angle_data.clear()
        self.raw_disp_data.clear()
        self.smooth_disp_data.clear()
        self.kf_conv_data.clear()
        self.left_ankle_traj.clear()
        self.right_ankle_traj.clear()
        self.ewma = None

    def set_total_frames(self, total):
        self.progress_slider.setMaximum(total)
        self.frame_info_label.setText(f"帧: 0 / {total}")

    def update_progress(self, current_frame):
        if not self.slider_dragging:
            self.progress_slider.blockSignals(True)
            self.progress_slider.setValue(current_frame)
            self.progress_slider.blockSignals(False)
            total = self.progress_slider.maximum()
            self.frame_info_label.setText(f"帧: {current_frame} / {total}")

    def on_slider_pressed(self):
        self.slider_dragging = True

    def on_slider_released(self):
        self.slider_dragging = False
        if self.worker and self.worker.is_video_file:
            target = self.progress_slider.value()
            self.clear_plot_data()
            self.worker.seek_to_frame(target)
            self.append_log(f"跳转至第 {target} 帧")

    def on_slider_value_changed(self, value):
        if self.slider_dragging:
            total = self.progress_slider.maximum()
            self.frame_info_label.setText(f"跳转至: {value} / {total}")

    def append_log(self, msg):
        self.log_queue.put(msg)

    def poll_log_queue(self):
        while True:
            try:
                msg = self.log_queue.get_nowait()
                self.log_text.append(msg)
                self.log_text.ensureCursorVisible()
            except queue.Empty:
                break

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())