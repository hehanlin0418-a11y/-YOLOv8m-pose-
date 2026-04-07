# # pose_detector.py 姿态检测主类
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import torch
# from kalman_filter import KalmanFilter
# from pose_enhancer import PoseEnhancer
#
# class PoseDetector:
#     def __init__(self, model_path='yolov8m-pose.pt', device=None):
#         if device is None:
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.device = device
#         print(f"使用设备: {self.device}")
#         self.model = YOLO(model_path).to(self.device)
#
#         self.enhancer = PoseEnhancer()
#         self.kalman_filters = [KalmanFilter() for _ in range(17)]
#         self.pose_history = []  # 存储历史 (keypoints, confidences)
#         self.max_history = 8
#         self.last_keypoints = None
#         self.last_confidences = None
#         self.frame_counter = 0  # 用于自适应推理频率
#
#     def detect_pose(self, image):
#         """检测姿态，返回 (keypoints, confidences)"""
#         self.frame_counter += 1
#         h, w = image.shape[:2]
#
#         # 自适应推理频率：每隔一帧完整推理，否则复用上一帧结果
#         if self.frame_counter % 2 == 0 or self.last_keypoints is None:
#             # 完整推理
#             results = self.model(image, imgsz=640, conf=0.15, iou=0.45, device=self.device, verbose=False)
#             kpts, confs = self._process_yolo_results(results, image.shape)
#
#             if kpts is not None:
#                 self.last_keypoints = kpts.copy()
#                 self.last_confidences = confs.copy()
#         else:
#             # 复用上一帧结果
#             kpts = self.last_keypoints.copy() if self.last_keypoints is not None else None
#             confs = self.last_confidences.copy() if self.last_confidences is not None else None
#
#         # 如果没有检测到姿态，使用历史数据
#         if kpts is None:
#             if len(self.pose_history) > 0:
#                 kpts, confs = self.pose_history[-1]
#                 confs = confs * 0.7  # 降低历史数据置信度
#             else:
#                 return None, None
#
#         # 应用链式插值（传入图像尺寸）
#         kpts, confs = self.enhancer.chain_interpolation(kpts, confs, image.shape)
#
#         # 时序平滑（传入图像尺寸）
#         kpts, confs = self._temporal_smoothing(kpts, confs, image.shape)
#
#         return kpts, confs
#
#     def _process_yolo_results(self, results, image_shape):
#         """处理YOLO输出"""
#         if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
#             return None, None
#
#         boxes = results[0].boxes
#         keypoints_data = results[0].keypoints
#
#         # 选择最大的人体检测框
#         max_area, best_idx = 0, 0
#         for i, box in enumerate(boxes):
#             area = box.xywh[0][2].item() * box.xywh[0][3].item()
#             if area > max_area:
#                 max_area = area
#                 best_idx = i
#
#         # 检查是否有效检测和关键点
#         if best_idx >= len(keypoints_data.xy) or len(keypoints_data.xy[best_idx]) == 0:
#             return None, None
#
#         # 获取关键点
#         xy = keypoints_data.xy[best_idx].cpu().numpy()
#         conf = keypoints_data.conf[best_idx].cpu().numpy() if keypoints_data.has_visible else np.ones(17) * 0.3
#
#         h, w = image_shape[:2]
#         kpts = np.full((17, 2), -1.0, dtype=np.float32)
#         confs = np.zeros(17, dtype=np.float32)
#
#         for i in range(17):
#             if i < len(xy):  # 额外检查以防关键点数量不足
#                 x, y = xy[i]
#                 if 0 < x < w and 0 < y < h:
#                     kpts[i] = [float(x), float(y)]
#                     confs[i] = float(conf[i] if i < len(conf) else 0.0)
#                 else:
#                     # 坐标超出图像范围，标记为无效
#                     kpts[i] = [-1.0, -1.0]
#                     confs[i] = 0.0
#             else:
#                 kpts[i] = [-1.0, -1.0]
#                 confs[i] = 0.0
#
#         return kpts, confs
#
#     def _temporal_smoothing(self, current_keypoints, current_confidences, image_shape=None):
#         """时序平滑滤波，使用Kalman滤波器，添加运动约束"""
#         smoothed_kpts = current_keypoints.copy()
#         smoothed_conf = current_confidences.copy()
#
#         if image_shape is not None:
#             h, w = image_shape[:2]
#         else:
#             h, w = 1000, 1000
#
#         # 计算运动幅度（使用历史，如果可用）
#         movement = 0
#         valid_count = 0
#         if len(self.pose_history) > 0:
#             prev_kpts, prev_conf = self.pose_history[-1]
#             for i in range(17):
#                 if current_confidences[i] > 0.2 and prev_conf[i] > 0.2:
#                     movement += np.linalg.norm(current_keypoints[i] - prev_kpts[i])
#                     valid_count += 1
#
#         avg_movement = movement / max(valid_count, 1)
#
#         # 自适应Kalman参数
#         if avg_movement > 50:  # 剧烈运动
#             process_noise = 0.05
#         elif avg_movement > 20:  # 中等运动
#             process_noise = 0.01
#         else:  # 轻微运动或静止
#             process_noise = 0.0001
#
#         for kf in self.kalman_filters:
#             kf.Q = process_noise * np.eye(4, dtype=np.float32)
#
#         # 应用Kalman滤波
#         for i in range(17):
#             measurement = current_keypoints[i]
#             conf = current_confidences[i]
#
#             # 检查测量是否有效
#             is_valid_measurement = (conf > 0.15 and
#                                     measurement[0] > 0 and measurement[1] > 0 and
#                                     measurement[0] < w and measurement[1] < h)
#
#             # 预测
#             predicted = self.kalman_filters[i].predict()
#
#             if is_valid_measurement:
#                 # 有有效测量，进行更新
#                 smoothed_kpts[i] = self.kalman_filters[i].update(measurement)
#                 smoothed_conf[i] = conf
#             else:
#                 # 无测量，使用预测，并降低置信度
#                 smoothed_kpts[i] = predicted
#                 smoothed_conf[i] = 0.25  # 默认低置信度
#
#                 # 如果历史可用且当前置信低，使用历史增强
#                 if len(self.pose_history) > 0 and prev_conf[i] > 0.2 and avg_movement < 40:
#                     # 增加历史数据的权重
#                     smoothed_kpts[i] = 0.1 * smoothed_kpts[i] + 0.9 * prev_kpts[i]
#                     smoothed_conf[i] = prev_conf[i] * 0.8
#
#                     # 更新Kalman滤波器的状态
#                     self.kalman_filters[i].state[:2] = smoothed_kpts[i].reshape(2, 1)
#
#         # 添加历史
#         if np.mean(current_confidences) > 0.15:
#             self.pose_history.append((smoothed_kpts.copy(), smoothed_conf.copy()))
#             if len(self.pose_history) > self.max_history:
#                 self.pose_history.pop(0)
#
#         return smoothed_kpts, smoothed_conf

# pose_detector.py 姿态检测+分布式KF网络
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

    def detect_pose(self, image, motion_state="Standing"):
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
                return None, None

        # 谱图增强 + 分布式KF平滑
        kpts, confs = self.enhancer.adaptive_spectral_interpolation(kpts, confs, image.shape)
        kpts, confs = self.temporal_smoothing_with_distributed_kf(kpts, confs, image.shape, motion_state)

        if np.mean(confs) > 0.15:
            self.pose_history.append((kpts.copy(), confs.copy()))
            if len(self.pose_history) > self.max_history:
                self.pose_history.pop(0)

        return kpts, confs

    def _process_yolo_results(self, results, image_shape):
        if not results or results[0].keypoints is None:
            return None, None

        boxes = results[0].boxes
        keypoints_data = results[0].keypoints
        max_area, best_idx = 0, 0
        for i, box in enumerate(boxes):
            area = box.xywh[0][2].item() * box.xywh[0][3].item()
            if area > max_area:
                max_area, best_idx = area, i

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