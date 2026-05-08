# camera_realtime.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import argparse
from pose_detector import PoseDetector
from state_analyzer import RiskAdaptiveStateAnalyzer
from visualizer import Visualizer
from multi_plotter import MultiPlotter

class CameraRealtimeDetector:
    def __init__(self, camera_id=0, enable_plot=True, inference_skip=2):
        """
        摄像头实时跌倒检测器
        :param camera_id: 摄像头索引
        :param enable_plot: 是否启用异步曲线绘图
        :param inference_skip: 隔帧推理间隔（1=每帧，2=隔一帧，推荐2~3）
        """
        self.camera_id = camera_id
        self.enable_plot = enable_plot
        self.inference_skip = inference_skip

        # 初始化核心模块
        self.detector = PoseDetector()
        self.analyzer = RiskAdaptiveStateAnalyzer()
        self.visualizer = Visualizer()

        self.cap = None
        self.frame_w = 0
        self.frame_h = 0
        self.plotter = None

        self.frame_count = 0
        self.status = "Standing"
        self.kf_convergence = 0.0

    def initialize(self):
        """打开摄像头并初始化绘图器"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.camera_id}")

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头已打开: {self.frame_w}x{self.frame_h}")

        if self.enable_plot:
            self.plotter = MultiPlotter(
                max_len=70,
                alpha=0.3,
                video_w=self.frame_w,
                video_h=self.frame_h,
                update_interval=0.4
            )

    def run(self):
        """主循环"""
        self.initialize()

        print("实时检测已启动，按 'q' 退出...")
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("摄像头读取失败")
                    break

                # 控制推理频率
                do_inference = (self.frame_count % self.inference_skip == 0)

                if do_inference:
                    kpts, confs = self.detector.detect_pose(frame, motion_state=self.status)
                else:
                    # 使用上一帧缓存的关键点
                    kpts = self.detector.last_keypoints
                    confs = self.detector.last_confidences

                # 处理未检测到人体的情况
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
                    # 跌倒检测
                    is_fall, angle, vertical_cond, aspect = self.analyzer.detect_falling(kpts, self.frame_w, self.frame_h)
                    is_confirmed = self.analyzer.update_fall_status(is_fall)

                    if is_confirmed:
                        self.status = "Falling"
                        details = (angle, vertical_cond, aspect)
                    else:
                        walking = self.analyzer.detect_walking(kpts, self.frame_w, self.frame_h)
                        self.status = "Walking" if walking else "Standing"
                        details = None

                    # 获取可视化数据
                    torso_angle = self.analyzer.compute_joint_angles(kpts, confs)
                    if self.detector.enhancer.reconstruction_error_history:
                        raw_displacement = self.detector.enhancer.reconstruction_error_history[-1]
                    else:
                        raw_displacement = 0.0
                    left_ankle = kpts[15] if confs[15] > 0.2 else None
                    right_ankle = kpts[16] if confs[16] > 0.2 else None

                # KF 收敛度
                if self.detector.convergence_results_history:
                    self.kf_convergence = np.mean(self.detector.convergence_results_history)

                # 异步绘图更新
                if self.plotter:
                    self.plotter.update(torso_angle, raw_displacement, left_ankle, right_ankle, self.kf_convergence)

                # 绘制骨架
                frame = self.visualizer.draw(frame, kpts, confs, self.status, details, self.kf_convergence, None)

                # 显示实时画面
                cv2.imshow('老年人跌倒检测 - 实时摄像头', frame)

                self.frame_count += 1

                # 打印状态（每 30 帧）
                if self.frame_count % 30 == 0:
                    print(f"Frame {self.frame_count} | Status: {self.status} | KF Conv: {self.kf_convergence:.2f}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户退出")
                    break

        finally:
            self.cleanup()

    def cleanup(self):
        """释放资源"""

        if self.plotter:
            self.plotter.close()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("资源已释放")


def main():
    parser = argparse.ArgumentParser(description="摄像头实时跌倒检测")
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引（默认 0）')
    parser.add_argument('--no-plot', action='store_true', help='禁用实时曲线绘图（可提升性能）')
    parser.add_argument('--skip', type=int, default=2, help='推理跳帧间隔（1=每帧，2=隔一帧，推荐2~3）')
    args = parser.parse_args()

    detector = CameraRealtimeDetector(
        camera_id=args.camera,
        enable_plot=not args.no_plot,
        inference_skip=args.skip
    )
    detector.run()


if __name__ == "__main__":
    main()