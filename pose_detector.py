# pose_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pose_enhancer import SpectralPoseEnhancer
from kalman_filter import DistributedKalmanFilter
from config import ADJACENCY_MATRIX


class PoseDetector:
    def __init__(self, model_path='yolov8m-pose.pt', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(f"使用设备: {device}")
        self.model = YOLO(model_path).to(device)

        self.enhancer = SpectralPoseEnhancer()
        self.kf_network = [DistributedKalmanFilter(i) for i in range(17)]
        self.pose_history = []
        self.max_history = 8
        self.last_keypoints = None
        self.last_confidences = None
        self.frame_counter = 0

        # 收敛性统计
        self.convergence_results_history = []
        self.filter_mse_history = []

        # 卡尔曼滤波器初始化标志
        self.kf_initialized = False

    def get_kf_neighbors(self, kf_idx):
        neighbor_map = {
            0: [1, 2, 5, 6], 1: [0, 2], 2: [0, 1],
            3: [0, 1], 4: [0, 2], 5: [0, 6, 7, 11],
            6: [0, 5, 8, 12], 7: [5, 9], 8: [6, 10],
            9: [7], 10: [8], 11: [5, 12, 13],
            12: [6, 11, 14], 13: [11, 15], 14: [12, 16],
            15: [13], 16: [14]
        }
        return [self.kf_network[i] for i in neighbor_map.get(kf_idx, [5, 6])]

    def temporal_smoothing_with_distributed_kf(self, kpts, confs, image_shape, motion_state):
        smoothed_kpts = kpts.copy()
        smoothed_conf = confs.copy()

        # 更新自适应噪声
        for kf in self.kf_network:
            kf.update_noise(motion_state)

        mse_list = []
        for i in range(17):
            kf = self.kf_network[i]
            measurement = kpts[i] if confs[i] > 0.15 else np.array([np.nan, np.nan])

            kf.predict()
            updated = kf.update(measurement)
            neighbors = self.get_kf_neighbors(i)
            consensus_updated, _, mse = kf.consensus_update(neighbors)

            mse_list.append(mse)

            if confs[i] > 0.15:
                smoothed_kpts[i] = 0.7 * updated + 0.3 * measurement
            else:
                smoothed_kpts[i] = consensus_updated
            smoothed_conf[i] = confs[i] if confs[i] > 0.15 else 0.25

        # 真实帧级收敛判定
        frame_avg_mse = np.mean(mse_list)
        threshold = self.kf_network[0].convergence_threshold
        frame_convergence = 1.0 if frame_avg_mse < threshold else 0.0
        self.convergence_results_history.append(frame_convergence)
        self.filter_mse_history.append(frame_avg_mse)

        return smoothed_kpts, smoothed_conf

    def detect_pose(self, image, motion_state="Standing", return_raw=False):
        """
        姿态检测主函数
        """
        self.frame_counter += 1
        h, w = image.shape[:2]

        # 自适应帧率推理
        if self.frame_counter % 2 == 0 or self.last_keypoints is None:
            results = self.model(image, imgsz=640, conf=0.15, iou=0.45, device=self.device, verbose=False)
            kpts, confs = self._process_yolo_results(results, image.shape)
            if kpts is not None:
                self.last_keypoints = kpts.copy()
                self.last_confidences = confs.copy()
        else:
            kpts = self.last_keypoints.copy() if self.last_keypoints is not None else None
            confs = self.last_confidences.copy() if self.last_confidences is not None else None

        if kpts is None:
            if self.pose_history:
                kpts, confs = self.pose_history[-1]
                confs *= 0.7
            else:
                if return_raw:
                    return None, None, None, None
                return None, None

        # 保存 YOLO 原始输出
        raw_kpts = kpts.copy()
        raw_confs = confs.copy()

        # 谱图增强
        kpts, confs = self.enhancer.adaptive_spectral_interpolation(kpts, confs, image.shape)

        # 首帧强制对齐卡尔曼滤波器状态
        if not self.kf_initialized and kpts is not None and np.mean(confs) > 0.15:
            for i in range(17):
                if confs[i] > 0.2:
                    # 用当前测量位置初始化状态向量 [x, y, vx, vy]，速度置零
                    self.kf_network[i].state = np.array([[kpts[i][0]], [kpts[i][1]], [0.0], [0.0]], dtype=np.float32)
                    # 适当降低初始协方差，让滤波器更信任初始状态，但仍保留一定的调整空间
                    self.kf_network[i].P = 0.1 * np.eye(4, dtype=np.float32)
            self.kf_initialized = True
            # print(f"✅ 第 {self.frame_counter} 帧：卡尔曼滤波器已用测量值初始化")
        # --------------------------------------------------------

        # 分布式卡尔曼滤波平滑
        kpts, confs = self.temporal_smoothing_with_distributed_kf(kpts, confs, image.shape, motion_state)

        if np.mean(confs) > 0.15:
            self.pose_history.append((kpts.copy(), confs.copy()))
            if len(self.pose_history) > self.max_history:
                self.pose_history.pop(0)

        if return_raw:
            return kpts, confs, raw_kpts, raw_confs
        else:
            return kpts, confs

    def _process_yolo_results(self, results, image_shape):
        """处理YOLO输出，安全处理空检测情况"""
        if not results or results[0].keypoints is None:
            return None, None

        boxes = results[0].boxes
        keypoints_data = results[0].keypoints

        if boxes is None or len(boxes) == 0:
            return None, None

        max_area, best_idx = 0, 0
        for i, box in enumerate(boxes):
            area = box.xywh[0][2].item() * box.xywh[0][3].item()
            if area > max_area:
                max_area = area
                best_idx = i

        if keypoints_data.xy is None or len(keypoints_data.xy) == 0 or best_idx >= len(keypoints_data.xy):
            return None, None

        xy = keypoints_data.xy[best_idx].cpu().numpy()
        conf = keypoints_data.conf[best_idx].cpu().numpy() if keypoints_data.has_visible else np.ones(17) * 0.3

        h, w = image_shape[:2]
        kpts = np.full((17, 2), -1.0, dtype=np.float32)
        confs = np.zeros(17, dtype=np.float32)

        for i in range(17):
            if i < len(xy):
                x, y = xy[i]
                if 0 < x < w and 0 < y < h:
                    kpts[i] = [float(x), float(y)]
                    confs[i] = float(conf[i] if i < len(conf) else 0.0)

        return kpts, confs

    def get_convergence_analysis(self):
        if not self.convergence_results_history:
            return "无收敛数据"

        avg_convergence = np.mean(self.convergence_results_history)
        avg_mse = np.mean(self.filter_mse_history) if self.filter_mse_history else 0
        convergence_rate = sum(1 for c in self.convergence_results_history if c == 1.0) / len(
            self.convergence_results_history)

        analysis = f"""
        分布式卡尔曼滤波收敛性分析：
        - 平均收敛度: {avg_convergence:.3f}
        - 平均滤波MSE: {avg_mse:.3f}
        - 收敛率: {convergence_rate * 100:.1f}%
        - 收敛阈值: {self.kf_network[0].convergence_threshold}
        """
        return analysis