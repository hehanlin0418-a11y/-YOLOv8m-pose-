# # kalman_filter.py 卡尔曼滤波器
# import numpy as np
#
# class KalmanFilter:
#     def __init__(self, process_noise=0.0001, measurement_noise=0.05, error_cov_post=0.5):
#         # 状态向量 [x, y, vx, vy]
#         self.state = np.zeros((4, 1), dtype=np.float32)
#         # 状态转移矩阵 F (恒速模型)
#         self.F = np.array([
#             [1, 0, 1, 0],
#             [0, 1, 0, 1],
#             [0, 0, 1, 0],
#             [0, 0, 0, 1]
#         ], dtype=np.float32)
#         # 测量矩阵 H (只测量位置)
#         self.H = np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, 0]
#         ], dtype=np.float32)
#         # 过程噪声协方差 Q
#         self.Q = process_noise * np.eye(4, dtype=np.float32)
#         # 测量噪声协方差 R
#         self.R = measurement_noise * np.eye(2, dtype=np.float32)
#         # 后验误差协方差 P
#         self.P = error_cov_post * np.eye(4, dtype=np.float32)
#
#     def predict(self):
#         self.state = self.F @ self.state
#         self.P = self.F @ self.P @ self.F.T + self.Q
#         return self.state[:2].flatten()  # 返回预测位置 [x, y]
#
#     def update(self, measurement):
#         if np.any(np.isnan(measurement)):
#             return self.state[:2].flatten()  # 如果测量无效，返回预测
#         z = measurement.reshape(2, 1)
#         y = z - self.H @ self.state
#         S = self.H @ self.P @ self.H.T + self.R
#         K = self.P @ self.H.T @ np.linalg.inv(S)
#         self.state = self.state + K @ y
#         self.P = (np.eye(4) - K @ self.H) @ self.P
#         return self.state[:2].flatten()  # 返回更新位置 [x, y]

# kalman_filter.py 分布式一致性卡尔曼滤波
import numpy as np
from config import KF_CONFIG


class DistributedKalmanFilter:
    def __init__(self, keypoint_idx, total_keypoints=17):
        self.keypoint_idx = keypoint_idx
        self.total_keypoints = total_keypoints

        self.state = np.zeros((4, 1), dtype=np.float32)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        self.process_noise_base = KF_CONFIG["process_noise_base"]
        self.measurement_noise_base = KF_CONFIG["measurement_noise_base"]
        self.Q = self.process_noise_base * np.eye(4, dtype=np.float32)
        self.R = self.measurement_noise_base * np.eye(2, dtype=np.float32)

        self.P = 0.5 * np.eye(4, dtype=np.float32)

        self.consensus_gain = KF_CONFIG["consensus_gain"]
        self.neighbor_states = []
        self.convergence_error = []
        self.convergence_threshold = KF_CONFIG["convergence_threshold"]

    def update_noise(self, motion_state):
        if motion_state == "Falling":
            self.Q = 0.1 * np.eye(4, dtype=np.float32)
            self.R = 0.1 * np.eye(2, dtype=np.float32)
        elif motion_state == "Walking":
            self.Q = 0.01 * np.eye(4, dtype=np.float32)
            self.R = 0.05 * np.eye(2, dtype=np.float32)
        else:
            self.Q = self.process_noise_base * np.eye(4, dtype=np.float32)
            self.R = self.measurement_noise_base * np.eye(2, dtype=np.float32)

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state[:2].flatten()

    def update(self, measurement):
        if np.any(np.isnan(measurement)) or np.any(measurement < 0):
            return self.state[:2].flatten()

        z = measurement.reshape(2, 1)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return self.state[:2].flatten()

    def consensus_update(self, neighbor_kf_list):
        self.neighbor_states = [kf.state for kf in neighbor_kf_list]
        if len(self.neighbor_states) == 0:
            return self.state[:2].flatten(), False, 999.0

        # 真实一致性融合
        neighbor_coords = [ns[:2] for ns in self.neighbor_states]
        consensus_coord = np.mean(neighbor_coords, axis=0)
        self.state[:2] = self.state[:2] + self.consensus_gain * (consensus_coord - self.state[:2])

        # 真实坐标误差计算
        mse = np.mean([np.linalg.norm(self.state[:2] - ns[:2]) for ns in self.neighbor_states])
        self.convergence_error.append(mse)

        is_converged = mse < self.convergence_threshold

        return self.state[:2].flatten(), is_converged, mse