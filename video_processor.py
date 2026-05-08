# video_processor.py
import cv2
import os
import numpy as np
from pose_detector import PoseDetector
from state_analyzer import RiskAdaptiveStateAnalyzer
from visualizer import Visualizer
from multi_plotter import MultiPlotter

def process_video(input_path, output_path='results/output.mp4', ground_truth_fall_frames=None):
    """模块闭环协同 + 可视化"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    detector = PoseDetector()
    analyzer = RiskAdaptiveStateAnalyzer()
    visualizer = Visualizer()
    # multi_plotter = MultiPlotter(max_len=100, alpha=0.3, video_w=w, video_h=h)
    multi_plotter = MultiPlotter(
        max_len=70,  # 缩短历史长度，减轻渲染压力
        alpha=0.3,
        video_w=w,
        video_h=h,
        update_interval=0.4  # 降低刷新频率
    )
    total_frames = 0
    detected_frames = 0
    fall_detected_frames = 0
    module_coordination_log = []

    print("开始处理视频...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 获取上一帧状态
            prev_status = "Standing" if total_frames == 0 else status

            # 姿态检测
            kpts, confs = detector.detect_pose(frame, motion_state=prev_status)

            if kpts is None:
                kpts = [[-1, -1] for _ in range(17)]
                confs = [0.0] * 17
                status = "Not detected"
                details = None
                torso_angle = None
                left_ankle = None
                right_ankle = None
                raw_displacement = 0.0
            else:
                detected_frames += 1

                # 地面真值
                ground_truth = None
                if ground_truth_fall_frames and total_frames in ground_truth_fall_frames:
                    ground_truth = True
                elif ground_truth_fall_frames:
                    ground_truth = False

                # 跌倒检测
                is_fall_candidate, angle, vertical_cond, aspect = analyzer.detect_falling(kpts, w, h)
                is_fall_confirmed = analyzer.update_fall_status(is_fall_candidate, ground_truth)

                if is_fall_confirmed:
                    status = "Falling"
                    fall_detected_frames += 1
                    details = (angle, vertical_cond, aspect)
                else:
                    walking = analyzer.detect_walking(kpts, w, h)
                    status = "Walking" if walking else "Standing"
                    details = None

                # 获取可视化所需数据
                torso_angle = analyzer.compute_joint_angles(kpts, confs)
                if detector.enhancer.reconstruction_error_history:
                    raw_displacement = detector.enhancer.reconstruction_error_history[-1]
                else:
                    raw_displacement = 0.0
                left_ankle = kpts[15] if confs[15] > 0.2 else None
                right_ankle = kpts[16] if confs[16] > 0.2 else None

            # 统一计算 KF 收敛度
            kf_convergence = np.mean(detector.convergence_results_history) if detector.convergence_results_history else 0.0

            # 更新多子图
            multi_plotter.update(torso_angle, raw_displacement, left_ankle, right_ankle, kf_convergence)

            # 绘制视频帧
            frame = visualizer.draw(frame, kpts, confs, status, details, kf_convergence, None)
            out.write(frame)
            cv2.imshow('老年人跌倒检测', frame)

            # 记录日志
            module_coordination_log.append({
                "frame": total_frames,
                "status": status,
                "kf_convergence": kf_convergence,
                "reconstruction_error": raw_displacement,
                "fall_threshold": analyzer.current_fall_threshold
            })

            total_frames += 1
            if total_frames % 30 == 0:
                print(f"已处理 {total_frames} 帧 | 收敛度: {kf_convergence:.2f} | 跌倒阈值: {analyzer.current_fall_threshold:.2f}")
                if detector.enhancer.debug_info:
                    latest = detector.enhancer.debug_info[-1]
                    if latest["improvement"] > 0:
                        print(f"  谱图补全: {latest['invalid_before']} → {latest['invalid_after']} 无效点 | 帧间位移: {raw_displacement:.2f}px")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        multi_plotter.close()
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # 输出统计信息
    print(f"\n===== 处理完成！=====")
    print(f"总帧数: {total_frames}")
    print(f"检测到姿态的帧数: {detected_frames}")
    print(f"检测率: {detected_frames / total_frames * 100:.1f}%")
    print(f"检测到跌倒的帧数: {fall_detected_frames}")

    if detector.enhancer.debug_info:
        improvements = [info["improvement"] for info in detector.enhancer.debug_info]
        avg_improvement = sum(improvements) / len(improvements)
        print(f"\n平均每帧补全 {avg_improvement:.1f} 个关键点")

    print(detector.get_convergence_analysis())
    print(f"\n输出保存至: {output_path}")

    log_path = os.path.join(os.path.dirname(output_path), "module_coordination_log.npy")
    np.save(log_path, module_coordination_log)
    print(f"模块协同日志保存至: {log_path}")