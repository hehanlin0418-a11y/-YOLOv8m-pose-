# state_analyzer.py
import numpy as np

class RiskAdaptiveStateAnalyzer:
    def __init__(self):
        self.aspect_ratio_threshold = 0.75
        self.vertical_angle_threshold = 60
        self.confidence_threshold = 0.3

        self.fall_candidate_history = []
        self.window_size = 5

        self.alpha = 0.05
        self.beta = 0.1
        self.current_fall_threshold = 0.6

        self.false_alarm_count = 0
        self.miss_detection_count = 0
        self.total_decisions = 0

    def detect_falling(self, kpts, img_w, img_h):
        """跌倒候选检测（宽高比+垂直位置）"""
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
        self.fall_candidate_history.append(is_fall_candidate)
        if len(self.fall_candidate_history) > self.window_size:
            self.fall_candidate_history.pop(0)

        self.total_decisions += 1
        current_ratio = sum(self.fall_candidate_history) / len(self.fall_candidate_history)
        is_confirmed = current_ratio >= self.current_fall_threshold

        if ground_truth is not None:
            if is_confirmed and not ground_truth:
                self.false_alarm_count += 1
            if not is_confirmed and ground_truth:
                self.miss_detection_count += 1
            self.adjust_threshold()

        return is_confirmed

    def adjust_threshold(self):
        far = self.false_alarm_count / self.total_decisions if self.total_decisions > 0 else 0
        mdr = self.miss_detection_count / self.total_decisions if self.total_decisions > 0 else 0

        if far > self.alpha:
            self.current_fall_threshold += 0.05
        if mdr > self.beta:
            self.current_fall_threshold -= 0.05
        self.current_fall_threshold = np.clip(self.current_fall_threshold, 0.1, 0.9)

    def detect_walking(self, kpts, img_w, img_h):
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

    def compute_joint_angles(self, kpts, confs):
        """计算躯干倾斜角（度）"""
        try:
            if (confs[5] > 0.3 and confs[6] > 0.3 and
                confs[11] > 0.3 and confs[12] > 0.3):
                chest = (kpts[5] + kpts[6]) / 2
                hip = (kpts[11] + kpts[12]) / 2
                v_body = hip - chest
                v_up = np.array([0, -1])
                if np.linalg.norm(v_body) > 0:
                    v_body = v_body / np.linalg.norm(v_body)
                    angle = np.degrees(np.arccos(np.clip(np.dot(v_up, v_body), -1, 1)))
                    return angle
        except:
            pass
        return None