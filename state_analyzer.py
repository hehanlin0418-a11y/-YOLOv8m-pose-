# # state_analyzer.py 状态分析模块
# import numpy as np
#
# class StateAnalyzer:
#     def __init__(self, max_history=10):
#         self.max_history = max_history
#         self.ankle_positions = []  # 存储 (left_ankle, right_ankle)
#         self.fall_flags = []       # 存储 0/1 跌倒标志
#         self.fall_confirmed_frames = 0
#
#     def add_ankle_positions(self, left_ankle, right_ankle):
#         self.ankle_positions.append((left_ankle, right_ankle))
#         if len(self.ankle_positions) > self.max_history:
#             self.ankle_positions.pop(0)
#
#     def get_movement_magnitude(self):
#         """计算运动幅度"""
#         if len(self.ankle_positions) < 2:
#             return 0
#
#         total = 0
#         count = 0
#
#         for i in range(1, len(self.ankle_positions)):
#             prev_l, prev_r = self.ankle_positions[i - 1]
#             curr_l, curr_r = self.ankle_positions[i]
#
#             if prev_l and curr_l:
#                 total += np.linalg.norm(np.array(curr_l) - np.array(prev_l))
#                 count += 1
#             if prev_r and curr_r:
#                 total += np.linalg.norm(np.array(curr_r) - np.array(prev_r))
#                 count += 1
#
#         return total / count if count > 0 else 0
#
#     def detect_walking(self, keypoints, width, height):
#         """检测行走状态"""
#         if len(keypoints) < 17:
#             return False
#
#         # 检查腿部关键点是否可见
#         leg_indices = [13, 14, 15, 16]
#         valid_leg_points = sum(1 for i in leg_indices if keypoints[i][0] > 0 and keypoints[i][1] > 0)
#
#         if valid_leg_points < 2:
#             return False
#
#         # 获取脚踝位置
#         left_ankle = (keypoints[15][0], keypoints[15][1]) if keypoints[15][0] > 0 else None
#         right_ankle = (keypoints[16][0], keypoints[16][1]) if keypoints[16][0] > 0 else None
#
#         self.add_ankle_positions(left_ankle, right_ankle)
#
#         if len(self.ankle_positions) < 3:
#             return False
#
#         movement = self.get_movement_magnitude()
#         return movement > 3.0 and movement < 50.0  # 过滤异常大的运动
#
#     def detect_falling(self, keypoints, width, height):
#         """检测跌倒状态，返回 (is_fall_candidate, angle, vertical_cond, aspect)"""
#         try:
#             # 检查必要的关键点
#             required_indices = [5, 6, 11, 12]  # 双肩和双髋
#             for i in required_indices:
#                 if not (0 < keypoints[i][0] < width and 0 < keypoints[i][1] < height):
#                     return False, 0, False, 0.0
#
#             # 计算躯干向量
#             chest_x = (keypoints[5][0] + keypoints[6][0]) / 2
#             chest_y = (keypoints[5][1] + keypoints[6][1]) / 2
#             waist_x = (keypoints[11][0] + keypoints[12][0]) / 2
#             waist_y = (keypoints[11][1] + keypoints[12][1]) / 2
#
#             # 计算身体倾斜角度
#             v_up = np.array([0, -1])  # 垂直向上向量
#             v_body = np.array([waist_x - chest_x, waist_y - chest_y])
#
#             if np.linalg.norm(v_body) > 0:
#                 v_body = v_body / np.linalg.norm(v_body)
#
#             angle = np.degrees(np.arccos(np.clip(np.dot(v_up, v_body), -1.0, 1.0)))
#
#             # 计算身体宽高比
#             torso_points = [keypoints[i] for i in [5, 6, 11, 12] if 0 < keypoints[i][0] < width]
#             if len(torso_points) >= 2:
#                 xs = [p[0] for p in torso_points]
#                 ys = [p[1] for p in torso_points]
#                 width_body = max(xs) - min(xs)
#                 height_body = max(ys) - min(ys)
#                 aspect = width_body / height_body if height_body > 0 else 0
#             else:
#                 aspect = 0
#
#             # 判断条件
#             angle_condition = angle > 45  # 身体倾斜超过45度
#             vertical_condition = chest_y > waist_y + 10  # 胸部低于腰部
#             aspect_condition = aspect > 0.8  # 身体宽高比大（水平姿态）
#
#             # 综合判断：满足两个条件视为跌倒候选
#             is_fall_candidate = sum([angle_condition, vertical_condition, aspect_condition]) >= 2
#
#             return is_fall_candidate, angle, vertical_condition, aspect
#
#         except Exception as e:
#             print(f"跌倒检测错误: {e}")
#             return False, 0, False, 0.0
#
#     def update_fall_status(self, is_fall_candidate):
#         """更新跌倒历史，返回是否确认跌倒"""
#         self.fall_flags.append(1 if is_fall_candidate else 0)
#         if len(self.fall_flags) > 5:
#             self.fall_flags.pop(0)
#
#         if is_fall_candidate:
#             self.fall_confirmed_frames = min(self.fall_confirmed_frames + 1, 10)
#         else:
#             self.fall_confirmed_frames = max(self.fall_confirmed_frames - 1, 0)
#
#         # 最近3帧中有2帧检测到跌倒，或累计确认帧数>=3
#         if len(self.fall_flags) >= 3:
#             return sum(self.fall_flags[-3:]) >= 2 or self.fall_confirmed_frames >= 3
#         else:
#             return self.fall_confirmed_frames >= 3

# state_analyzer.py 风险自适应状态分析
import numpy as np


class RiskAdaptiveStateAnalyzer:
    def __init__(self):
        # 跌倒检测基础参数
        self.aspect_ratio_threshold = 0.75
        self.vertical_angle_threshold = 60
        self.confidence_threshold = 0.3

        self.fall_candidate_history = []  # 跌倒候选帧历史
        self.window_size = 5  # 滑动窗口帧数（连续5帧判定跌倒）

        # Neyman-Pearson风险决策参数
        self.alpha = 0.05  # 虚警率约束 ≤5%
        self.beta = 0.1  # 漏检率约束 ≤10%
        self.current_fall_threshold = 0.6  # 初始跌倒确认阈值

        # 风险统计
        self.false_alarm_count = 0
        self.miss_detection_count = 0
        self.total_decisions = 0

    def detect_falling(self, kpts, img_w, img_h):
        """跌倒候选检测（几何特征：宽高比+垂直角度）"""
        valid_kpts = []
        for kpt in kpts:
            if 0 <= kpt[0] < img_w and 0 <= kpt[1] < img_h:
                valid_kpts.append(kpt)
        if len(valid_kpts) < 5:
            return False, 0, False, 1.0

        x_coords = [p[0] for p in valid_kpts]
        y_coords = [p[1] for p in valid_kpts]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        w = x_max - x_min
        h = y_max - y_min
        aspect = w / h if h != 0 else 1.0

        center_y = np.mean(y_coords)
        vertical_pos = center_y / img_h
        vertical_cond = vertical_pos > 0.5

        angle = 0
        if h > 0:
            angle = np.degrees(np.arctan(w / h))

        is_candidate = (aspect > self.aspect_ratio_threshold) and vertical_cond
        return is_candidate, angle, vertical_cond, aspect

    def update_fall_status(self, is_fall_candidate, ground_truth=None):
        # 滑动窗口统计
        self.fall_candidate_history.append(is_fall_candidate)
        if len(self.fall_candidate_history) > self.window_size:
            self.fall_candidate_history.pop(0)

        self.total_decisions += 1
        current_ratio = sum(self.fall_candidate_history) / len(self.fall_candidate_history)
        is_confirmed = current_ratio >= self.current_fall_threshold

        # 有真值时执行风险自适应调整
        if ground_truth is not None:
            if is_confirmed and not ground_truth:
                self.false_alarm_count += 1
            if not is_confirmed and ground_truth:
                self.miss_detection_count += 1
            self.adjust_threshold()

        return is_confirmed

    def adjust_threshold(self):
        """Neyman-Pearson 自适应调整阈值"""
        far = self.false_alarm_count / self.total_decisions if self.total_decisions > 0 else 0
        mdr = self.miss_detection_count / self.total_decisions if self.total_decisions > 0 else 0

        if far > self.alpha:
            self.current_fall_threshold += 0.05
        if mdr > self.beta:
            self.current_fall_threshold -= 0.05

        self.current_fall_threshold = np.clip(self.current_fall_threshold, 0.1, 0.9)

    def detect_walking(self, kpts, img_w, img_h):
        """行走检测"""
        try:
            left_ankle = kpts[15]
            right_ankle = kpts[16]
            left_knee = kpts[13]
            right_knee = kpts[14]

            if all(0 <= p[0] < img_w and 0 <= p[1] < img_h for p in [left_ankle, right_ankle, left_knee, right_knee]):
                ankle_dist = np.linalg.norm(left_ankle - right_ankle)
                knee_dist = np.linalg.norm(left_knee - right_knee)
                return ankle_dist > 20 and knee_dist > 10
        except:
            pass
        return False