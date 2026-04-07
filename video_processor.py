# # video_processor.py 视频处理主流程
# import cv2
# import os
# from pose_detector import PoseDetector
# from state_analyzer import StateAnalyzer
# from visualizer import Visualizer
#
# def process_video(input_path, output_path='results/output.mp4'):
#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         print(f"无法打开视频: {input_path}")
#         return
#
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#
#     detector = PoseDetector()
#     analyzer = StateAnalyzer()
#     visualizer = Visualizer()
#
#     total_frames = 0
#     detected_frames = 0
#
#     print("开始处理视频...")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 检测姿态
#         kpts, confs = detector.detect_pose(frame)
#
#         if kpts is None:
#             # 未检测到人体
#             kpts = [[-1, -1] for _ in range(17)]
#             confs = [0.0] * 17
#             status = "Not detected"
#             details = None
#         else:
#             detected_frames += 1
#             # 检测跌倒候选
#             is_fall_candidate, angle, vertical_cond, aspect = analyzer.detect_falling(kpts, w, h)
#             is_fall_confirmed = analyzer.update_fall_status(is_fall_candidate)
#
#             if is_fall_confirmed:
#                 status = "Falling"
#                 details = (angle, vertical_cond, aspect)
#             else:
#                 walking = analyzer.detect_walking(kpts, w, h)
#                 status = "Walking" if walking else "Standing"
#                 details = None
#
#         # 绘制结果
#         frame = visualizer.draw(frame, kpts, confs, status, details)
#         out.write(frame)
#
#         cv2.imshow('老年人跌倒检测', frame)
#
#         total_frames += 1
#         if total_frames % 30 == 0:
#             print(f"已处理 {total_frames} 帧")
#             # 输出补全统计
#             if detector.enhancer.debug_info:
#                 latest = detector.enhancer.debug_info[-1]
#                 if latest["improvement"] > 0:
#                     print(f"  补全效果: {latest['invalid_before']} -> {latest['invalid_after']} 无效点")
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # 输出统计信息
#     if detector.enhancer.debug_info:
#         print(f"\n补全效果统计:")
#         improvements = [info["improvement"] for info in detector.enhancer.debug_info]
#         avg_improvement = sum(improvements) / len(improvements)
#         print(f"平均每帧补全 {avg_improvement:.1f} 个关键点")
#
#     print(f"\n处理完成！")
#     print(f"总帧数: {total_frames}")
#     print(f"检测到姿态的帧数: {detected_frames}")
#     print(f"检测率: {detected_frames / total_frames * 100:.1f}%")
#     print(f"输出保存至: {output_path}")
#
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

# video_processor.py 视频处理主流程（模块闭环协同）
import cv2
import os
import numpy as np
from pose_detector import PoseDetector
from state_analyzer import RiskAdaptiveStateAnalyzer
from visualizer import Visualizer


def process_video(input_path, output_path='results/output.mp4', ground_truth_fall_frames=None):
    """模块闭环协同（插值→滤波→决策→反馈插值/滤波）"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # 初始化模块
    detector = PoseDetector()
    analyzer = RiskAdaptiveStateAnalyzer()
    visualizer = Visualizer()

    total_frames = 0
    detected_frames = 0
    fall_detected_frames = 0
    module_coordination_log = []  # 模块协同日志

    print("开始处理视频...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 步骤1：获取上一帧状态（模块协同：决策反馈给检测）
        prev_status = "Standing" if total_frames == 0 else status

        # 步骤2：姿态检测（传入上一帧状态，实现闭环）
        kpts, confs = detector.detect_pose(frame, motion_state=prev_status)

        if kpts is None:
            kpts = [[-1, -1] for _ in range(17)]
            confs = [0.0] * 17
            status = "Not detected"
            details = None
        else:
            detected_frames += 1
            # 步骤3：跌倒检测（风险自适应决策）
            ground_truth = None
            if ground_truth_fall_frames and total_frames in ground_truth_fall_frames:
                ground_truth = True
            elif ground_truth_fall_frames:
                ground_truth = False

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

        # 步骤4：模块协同日志记录
        kf_convergence = np.mean(detector.convergence_results_history) if detector.convergence_results_history else 0.0
        risk_metrics = None
        if analyzer.total_decisions > 0:
            far = (analyzer.false_alarm_count / analyzer.total_decisions) * 100
            mdr = (analyzer.miss_detection_count / analyzer.total_decisions) * 100
            risk_metrics = (far, mdr, analyzer.current_fall_threshold)

        module_coordination_log.append({
            "frame": total_frames,
            "status": status,
            "kf_convergence": kf_convergence,
            "reconstruction_error": detector.enhancer.reconstruction_error_history[-1] if len(
                detector.enhancer.reconstruction_error_history) > 0 else 0,
            "fall_threshold": analyzer.current_fall_threshold
        })

        # 步骤5：绘制结果（包含协同/收敛/风险信息）
        frame = visualizer.draw(frame, kpts, confs, status, details, kf_convergence, risk_metrics)
        out.write(frame)

        cv2.imshow('老年人跌倒检测（模块协同版）', frame)

        total_frames += 1
        if total_frames % 30 == 0:
            print(
                f"已处理 {total_frames} 帧 | 收敛度: {kf_convergence:.2f} | 跌倒阈值: {analyzer.current_fall_threshold:.2f}")
            if detector.enhancer.debug_info:
                latest = detector.enhancer.debug_info[-1]
                if latest["improvement"] > 0:
                    print(
                        f"  谱图补全: {latest['invalid_before']} → {latest['invalid_after']} 无效点 | 重建误差下界: {latest['reconstruction_error_bound']:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 输出统计信息（增强：理论+收敛+风险）
    print(f"\n===== 处理完成！=====")
    print(f"总帧数: {total_frames}")
    print(f"检测到姿态的帧数: {detected_frames}")
    print(f"检测率: {detected_frames / total_frames * 100:.1f}%")
    print(f"检测到跌倒的帧数: {fall_detected_frames}")

    # 谱图增强统计
    if detector.enhancer.debug_info:
        print(f"\n===== 谱图增强统计 =====")
        improvements = [info["improvement"] for info in detector.enhancer.debug_info]
        avg_improvement = sum(improvements) / len(improvements)
        avg_reconstruction_error = np.mean(detector.enhancer.reconstruction_error_history) if len(
            detector.enhancer.reconstruction_error_history) > 0 else 0
        print(f"平均每帧补全 {avg_improvement:.1f} 个关键点")
        print(f"平均重建误差下界: {avg_reconstruction_error:.2f}")

    # 分布式KF收敛性统计
    print(f"\n===== 分布式KF收敛性分析 =====")
    print(detector.get_convergence_analysis())

    print(f"\n输出保存至: {output_path}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 保存模块协同日志（便于后续分析模块协同机制）
    log_path = os.path.join(os.path.dirname(output_path), "module_coordination_log.npy")
    np.save(log_path, module_coordination_log)
    print(f"模块协同日志保存至: {log_path}")