# collect_kpts_data.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import argparse
from pose_detector import PoseDetector


def collect_keypoints_sequence(video_path, output_npy_path, mode='full'):
    """
    提取视频每一帧的关键点，保存为 .npy 文件
    :param video_path: 输入视频路径
    :param output_npy_path: 输出 .npy 文件路径
    :param mode: 采集模式，可选 'raw', 'kf', 'spectral', 'full'
                 raw      -> 纯 YOLO 原始输出
                 kf       -> YOLO + 分布式卡尔曼滤波
                 spectral -> YOLO + 谱图插值
                 full     -> 完整系统
    """
    detector = PoseDetector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return

    all_kpts = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == 'raw':
            # 纯 YOLO 原始输出
            results = detector.model(frame, imgsz=640, conf=0.15, iou=0.45, device=detector.device, verbose=False)
            kpts, confs = detector._process_yolo_results(results, frame.shape)

        elif mode == 'kf':
            # YOLO + 卡尔曼滤波
            # 先获取原始 YOLO 结果
            results = detector.model(frame, imgsz=640, conf=0.15, iou=0.45, device=detector.device, verbose=False)
            kpts, confs = detector._process_yolo_results(results, frame.shape)
            if kpts is not None:
                # 确保卡尔曼滤波器已初始化
                if not detector.kf_initialized:
                    for i in range(17):
                        if confs[i] > 0.2:
                            detector.kf_network[i].state = np.array(
                                [[kpts[i][0]], [kpts[i][1]], [0.0], [0.0]], dtype=np.float32)
                            detector.kf_network[i].P = 0.1 * np.eye(4, dtype=np.float32)
                    detector.kf_initialized = True
                # 应用分布式卡尔曼滤波
                kpts, confs = detector.temporal_smoothing_with_distributed_kf(
                    kpts, confs, frame.shape, "Standing")

        elif mode == 'spectral':
            # YOLO + 谱图插值
            results = detector.model(frame, imgsz=640, conf=0.15, iou=0.45, device=detector.device, verbose=False)
            kpts, confs = detector._process_yolo_results(results, frame.shape)
            if kpts is not None:
                kpts, confs = detector.enhancer.adaptive_spectral_interpolation(kpts, confs, frame.shape)

        else:  # 'full' 完整系统
            kpts, confs = detector.detect_pose(frame, motion_state="Standing")

        # 将无效点设为 nan
        if kpts is not None:
            kpts_clean = kpts.copy()
            kpts_clean[confs < 0.2] = np.nan
            all_kpts.append(kpts_clean)
        else:
            all_kpts.append(np.full((17, 2), np.nan, dtype=np.float32))

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"已处理 {frame_idx} 帧")

    cap.release()

    # 保存为 .npy 文件
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    np.save(output_npy_path, np.array(all_kpts))
    print(f"关键点序列已保存至: {output_npy_path} (形状: {np.array(all_kpts).shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="采集跌倒检测关键点序列（四种模式）")
    parser.add_argument('--video', type=str, default='video/fall_hhl.mp4', help='视频文件路径')
    parser.add_argument('--mode', type=str, choices=['raw', 'kf', 'spectral', 'full', 'all'],
                        default='all', help='采集模式：raw(纯YOLO), kf(卡尔曼), spectral(插值), full(完整系统), all(全部四种)')
    parser.add_argument('--output_dir', type=str, default='eval_data', help='输出目录')
    args = parser.parse_args()

    video_path = args.video
    output_dir = args.output_dir

    mode_map = {
        'raw': ('raw_yolo_kpts.npy', '纯YOLO'),
        'kf': ('kf_yolo_kpts.npy', 'YOLO+卡尔曼'),
        'spectral': ('spectral_yolo_kpts.npy', 'YOLO+插值'),
        'full': ('enhanced_kpts.npy', '完整系统')   # 保持与原命名一致
    }

    if args.mode == 'all':
        for key, (filename, desc) in mode_map.items():
            print(f"\n正在采集 {desc} 关键点序列...")
            output_path = os.path.join(output_dir, filename)
            collect_keypoints_sequence(video_path, output_path, mode=key)
        print("\n全部四种模式采集完成！")
    else:
        filename, desc = mode_map[args.mode]
        print(f"\n正在采集 {desc} 关键点序列...")
        output_path = os.path.join(output_dir, filename)
        collect_keypoints_sequence(video_path, output_path, mode=args.mode)