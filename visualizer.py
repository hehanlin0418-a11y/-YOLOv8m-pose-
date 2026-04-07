# # visualizer.py 可视化
# import cv2
# import numpy as np
# from config import SKELETON
#
# class Visualizer:
#     @staticmethod
#     def draw(frame, keypoints, confidences, status, fall_details=None):
#         """绘制骨骼、关键点和状态信息，返回绘制后的图像"""
#         overlay = frame.copy()
#         h, w = frame.shape[:2]
#
#         # 绘制骨骼
#         for (i, j) in SKELETON:
#             if keypoints[i][0] > 0 and keypoints[j][0] > 0:
#                 conf = (confidences[i] + confidences[j]) / 2
#                 color_intensity = int(255 * conf)
#                 cv2.line(overlay,
#                          (int(keypoints[i][0]), int(keypoints[i][1])),
#                          (int(keypoints[j][0]), int(keypoints[j][1])),
#                          (0, color_intensity, 0), 2)
#
#         # 绘制关键点
#         for i, (x, y) in enumerate(keypoints):
#             if x > 0 and y > 0:
#                 conf = confidences[i]
#                 if conf > 0.7:
#                     color = (0, 0, 255)  # 高置信度：红色
#                 elif conf > 0.4:
#                     color = (0, 165, 255)  # 中置信度：橙色
#                 else:
#                     color = (0, 255, 255)  # 低置信度：黄色
#
#                 cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
#
#                 # 显示关键点编号
#                 cv2.putText(overlay, str(i), (int(x) + 5, int(y) - 5),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
#
#         # 绘制状态
#         status_colors = {
#             "Falling": (0, 0, 255),      # 红色
#             "Walking": (255, 0, 0),      # 蓝色
#             "Standing": (0, 255, 0),     # 绿色
#             "Not detected": (128, 128, 128)  # 灰色
#         }
#         color = status_colors.get(status, (255, 255, 255))
#         cv2.putText(overlay, f"Status:{status}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#
#         # 显示详细信息
#         y_offset = 60
#         if status == "Falling" and fall_details:
#             angle, vertical_cond, aspect = fall_details
#             cv2.putText(overlay, f"Angle: {angle:.1f}deg", (10, y_offset),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#             y_offset += 20
#
#             cv2.putText(overlay, f"Chest>Waist: {'True' if vertical_cond else 'False'}",
#                         (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#             y_offset += 20
#
#             cv2.putText(overlay, f"Aspect: {aspect:.2f}", (10, y_offset),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#
#         # 显示可见关键点数量
#         visible_count = sum(1 for x, y in keypoints if x > 0 and y > 0)
#         cv2.putText(overlay, f"Keypoints: {visible_count}/17",
#                     (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#         return overlay

# visualizer.py 增强可视化（模块协同/收敛性/风险）
import cv2
import numpy as np
from config import SKELETON

class Visualizer:
    @staticmethod
    def draw(frame, keypoints, confidences, status, fall_details=None, kf_convergence=0, risk_metrics=None):
        """绘制骨骼、状态、收敛性、风险指标"""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # 绘制骨骼
        for (i, j) in SKELETON:
            if keypoints[i][0] > 0 and keypoints[j][0] > 0:
                conf = (confidences[i] + confidences[j]) / 2
                color_intensity = int(255 * conf)
                cv2.line(overlay,
                         (int(keypoints[i][0]), int(keypoints[i][1])),
                         (int(keypoints[j][0]), int(keypoints[j][1])),
                         (0, color_intensity, 0), 2)

        # 绘制关键点
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:
                conf = confidences[i]
                if conf > 0.7:
                    color = (0, 0, 255)
                elif conf > 0.4:
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 255)

                cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
                cv2.putText(overlay, str(i), (int(x) + 5, int(y) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 绘制状态
        status_colors = {
            "Falling": (0, 0, 255),
            "Walking": (255, 0, 0),
            "Standing": (0, 255, 0),
            "Not detected": (128, 128, 128)
        }
        color = status_colors.get(status, (255, 255, 255))
        cv2.putText(overlay, f"Status:{status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 绘制跌倒详情
        y_offset = 60
        if status == "Falling" and fall_details:
            angle, vertical_cond, aspect = fall_details
            cv2.putText(overlay, f"Angle: {angle:.1f}deg", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
            cv2.putText(overlay, f"Chest>Waist: {'True' if vertical_cond else 'False'}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
            cv2.putText(overlay, f"Aspect: {aspect:.2f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20

        # 绘制KF收敛性
        cv2.putText(overlay, f"KF Convergence: {kf_convergence:.2f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20

        # 绘制可见关键点数量
        visible_count = sum(1 for x, y in keypoints if x > 0 and y > 0)
        cv2.putText(overlay, f"Keypoints: {visible_count}/17",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return overlay