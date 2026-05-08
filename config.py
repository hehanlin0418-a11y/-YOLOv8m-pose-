# config.py
import numpy as np

# COCO关键点名称
KEYPOINT_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}

# 人体关键点链式结构+ 谱图拓扑邻接关系
BODY_CHAINS = {
    "torso_center": [0, 1, 2, 5, 6, 11, 12],
    "left_arm": [5, 7, 9],
    "right_arm": [6, 8, 10],
    "left_leg": [11, 13, 15],
    "right_leg": [12, 14, 16],
    "head": [0, 1, 2, 3, 4],
    "shoulder_hip": [5, 11, 6, 12]
}

# 谱图拓扑邻接矩阵
ADJACENCY_MATRIX = np.array([
    # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 nose
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 left_eye
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2 right_eye
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 3 left_ear
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4 right_ear
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5 left_shoulder
    [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # 6 right_shoulder
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 7 left_elbow
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 8 right_elbow
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 9 left_wrist
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 10 right_wrist
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 11 left_hip
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],  # 12 right_hip
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # 13 left_knee
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # 14 right_knee
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 15 left_ankle
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]   # 16 right_ankle
], dtype=np.float32)

# 骨骼连接关系
SKELETON = [
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 手臂
    (5, 11), (6, 12), (11, 12),                # 躯干
    (11, 13), (13, 15),                        # 左腿
    (12, 14), (14, 16)                         # 右腿
]

# 关键点索引常量
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# 分布式卡尔曼滤波配置
KF_CONFIG = {
    "process_noise_base": 0.0001,
    "measurement_noise_base": 0.05,
    "consensus_gain": 0.8,
    "convergence_threshold": 25.0,
}


# 谱图重建误差下界配置
SPECTRAL_CONFIG = {
    "lambda_min": 0.01,  # 图拉普拉斯最小特征值
    "reconstruction_error_weight": 0.8,  # 误差权重
}