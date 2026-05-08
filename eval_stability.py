# eval_stability.py
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False


def moving_average(data, window_size=5):
    """一维滑动平均"""
    if window_size <= 1 or len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def compute_stability_metrics(kpts_seq):
    """
    输入: kpts_seq 形状 (N_frames, 17, 2)，无效点用 np.nan 表示
    返回: 指标字典
    """
    N = kpts_seq.shape[0]
    if N < 2:
        return {}

    # 1. 帧间位移
    diffs = np.linalg.norm(kpts_seq[1:] - kpts_seq[:-1], axis=-1)
    valid_diffs = diffs[~np.isnan(diffs)]
    D_avg = np.mean(valid_diffs) if len(valid_diffs) > 0 else np.nan
    D_std = np.std(valid_diffs) if len(valid_diffs) > 0 else np.nan

    # 2. 有效关键点占比
    valid_mask = ~np.isnan(kpts_seq[:, :, 0])
    R_valid = np.mean(valid_mask)

    # 3. 躯干长度变异系数
    left_shoulder = kpts_seq[:, 5, :]
    left_hip = kpts_seq[:, 11, :]
    torso_len = np.linalg.norm(left_shoulder - left_hip, axis=-1)
    valid_torso = ~np.isnan(torso_len)
    if np.sum(valid_torso) > 2:
        mean_len = np.nanmean(torso_len)
        std_len = np.nanstd(torso_len)
        CV_torso = std_len / mean_len if mean_len > 0 else np.nan
    else:
        CV_torso = np.nan

    # 4. 异常跳变帧率
    max_jump_per_frame = np.nanmax(diffs, axis=1)
    J_rate = np.mean(max_jump_per_frame > 50) if len(max_jump_per_frame) > 0 else np.nan

    # 5. 平均每帧有效点数
    avg_valid_per_frame = np.mean(np.sum(valid_mask, axis=1))

    return {
        "D_avg": D_avg,
        "D_std": D_std,
        "R_valid": R_valid,
        "CV_torso": CV_torso,
        "J_rate": J_rate,
        "avg_valid_per_frame": avg_valid_per_frame
    }


def infer_model_name_from_path(path):
    name = os.path.basename(path).lower()
    if "raw" in name:
        return "纯YOLO"
    elif "kf" in name:
        return "YOLO+卡尔曼"
    elif "spectral" in name:
        return "YOLO+插值"
    elif "enhanced" in name or "full" in name:
        return "完整系统"
    else:
        return "模型"


def plot_comparison(raw_kpts, enhanced_kpts, save_path="eval_comparison.png", label_a="模型A", label_b="模型B"):
    N_frames = raw_kpts.shape[0]
    x = np.arange(N_frames)

    # ========== 三个独立的折线图窗口 ==========
    # 窗口1：有效关键点数量
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    raw_valid = np.sum(~np.isnan(raw_kpts[:, :, 0]), axis=1)
    enh_valid = np.sum(~np.isnan(enhanced_kpts[:, :, 0]), axis=1)
    ax1.plot(x, raw_valid, 'r--', alpha=0.7, linewidth=1.5, label=label_a)
    ax1.plot(x, enh_valid, 'g-', alpha=0.8, linewidth=1.5, label=label_b)
    ax1.set_ylabel('有效关键点数')
    ax1.set_title('有效关键点数量对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    plt.show()

    # 窗口2：帧间位移
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    raw_disp = np.linalg.norm(raw_kpts[1:] - raw_kpts[:-1], axis=-1)
    enh_disp = np.linalg.norm(enhanced_kpts[1:] - enhanced_kpts[:-1], axis=-1)
    raw_mean_disp = np.nanmean(raw_disp, axis=1)
    enh_mean_disp = np.nanmean(enh_disp, axis=1)
    ax2.plot(x[1:], raw_mean_disp, 'r--', alpha=0.7, linewidth=1.5, label=label_a)
    ax2.plot(x[1:], enh_mean_disp, 'g-', alpha=0.8, linewidth=1.5, label=label_b)
    ax2.set_ylabel('平均位移 (px)')
    ax2.set_title('帧间位移对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    plt.show()

    # 窗口3：躯干长度变化
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    raw_torso = np.linalg.norm(raw_kpts[:, 5, :] - raw_kpts[:, 11, :], axis=-1)
    enh_torso = np.linalg.norm(enhanced_kpts[:, 5, :] - enhanced_kpts[:, 11, :], axis=-1)
    ax3.plot(x, raw_torso, 'r--', alpha=0.7, linewidth=1.5, label=label_a)
    ax3.plot(x, enh_torso, 'g-', alpha=0.8, linewidth=1.5, label=label_b)
    ax3.set_ylabel('躯干长度 (px)')
    ax3.set_title('躯干长度一致性对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    plt.show()

    # 保存三个曲线图
    if save_path:
        base, ext = os.path.splitext(save_path)
        curves_path1 = f"{base}_curves1{ext}"
        curves_path2 = f"{base}_curves2{ext}"
        curves_path3 = f"{base}_curves3{ext}"
    else:
        curves_path1 = "eval_curves1.png"
        curves_path2 = "eval_curves2.png"
        curves_path3 = "eval_curves3.png"
    fig1.savefig(curves_path1, dpi=150)
    fig2.savefig(curves_path2, dpi=150)
    fig3.savefig(curves_path3, dpi=150)

    # 窗口4：稳定性指标汇总表格
    fig_table, ax_table = plt.subplots(figsize=(8, 4))
    ax_table.axis('off')

    metrics_raw = compute_stability_metrics(raw_kpts)
    metrics_enh = compute_stability_metrics(enhanced_kpts)

    def calc_change(val_a, val_b):
        if np.isnan(val_a) or np.isnan(val_b) or val_a == 0:
            return np.nan
        return (val_b - val_a) / val_a * 100

    metric_names = ['D_avg', 'D_std', 'R_valid', 'CV_torso', 'J_rate']
    metric_labels = ['平均位移(px)', '位移标准差(px)', '有效点比例', '躯干变异系数', '跳变率']

    table_data = [["指标", label_a, label_b, "变化"]]
    for mname, mlabel in zip(metric_names, metric_labels):
        val_a = metrics_raw.get(mname, np.nan)
        val_b = metrics_enh.get(mname, np.nan)
        change = calc_change(val_a, val_b)
        if not np.isnan(change):
            change_str = f"{change:+.1f}%"
        else:
            change_str = "N/A"
        val_a_str = f"{val_a:.3f}" if not np.isnan(val_a) else "N/A"
        val_b_str = f"{val_b:.3f}" if not np.isnan(val_b) else "N/A"
        table_data.append([mlabel, val_a_str, val_b_str, change_str])

    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax_table.set_title('稳定性指标汇总', fontsize=12, pad=20)

    fig_table.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        table_path = f"{base}_table{ext}"
    else:
        table_path = "eval_table.png"
    fig_table.savefig(table_path, dpi=150)
    plt.show()  # 表格窗口也显示

    print(f"对比图已保存至: {curves_path1}, {curves_path2}, {curves_path3} 和 {table_path}")
    print("\n========== 稳定性评估结果 ==========")
    for row in table_data[1:]:
        print(f"{row[0]:12s} | {label_a}: {row[1]:>8s} | {label_b}: {row[2]:>8s} | 变化: {row[3]:>10s}")


def plot_ablation_comparison(kpts_dict, labels_dict, save_path="ablation_comparison.png", smooth_window=5):
    modes = ['raw', 'kf', 'spectral', 'full']
    labels = [labels_dict[m] for m in modes]
    colors = ['blue', 'orange', 'green', 'red']

    min_len = min(kpts_dict[m].shape[0] for m in modes)
    for m in modes:
        kpts_dict[m] = kpts_dict[m][:min_len]

    x = np.arange(min_len)

    # ========== 三个独立的折线图窗口 ==========
    # 窗口1：有效关键点数量
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for mode, label, color in zip(modes, labels, colors):
        valid = np.sum(~np.isnan(kpts_dict[mode][:, :, 0]), axis=1)
        ax1.plot(x, valid, color=color, alpha=0.7, linewidth=1.5, label=label)
    ax1.set_ylabel('有效关键点数')
    ax1.set_title('有效关键点数量对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    plt.show()

    # 窗口2：帧间位移
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for mode, label, color in zip(modes, labels, colors):
        disp = np.linalg.norm(kpts_dict[mode][1:] - kpts_dict[mode][:-1], axis=-1)
        mean_disp = np.nanmean(disp, axis=1)
        if smooth_window > 1:
            mean_disp = moving_average(mean_disp, smooth_window)
        ax2.plot(x[1:], mean_disp, color=color, alpha=0.7, linewidth=1.5, label=label)
    ax2.set_ylabel('平均位移 (px)')
    ax2.set_title(f'帧间位移对比 (平滑窗口={smooth_window})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    plt.show()

    # 窗口3：躯干长度变化
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for mode, label, color in zip(modes, labels, colors):
        torso = np.linalg.norm(kpts_dict[mode][:, 5, :] - kpts_dict[mode][:, 11, :], axis=-1)
        ax3.plot(x, torso, color=color, alpha=0.7, linewidth=1.5, label=label)
    ax3.set_ylabel('躯干长度 (px)')
    ax3.set_title('躯干长度一致性对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    plt.show()

    # 保存三个曲线图
    if save_path:
        base, ext = os.path.splitext(save_path)
        curves_path1 = f"{base}_curves1{ext}"
        curves_path2 = f"{base}_curves2{ext}"
        curves_path3 = f"{base}_curves3{ext}"
    else:
        curves_path1 = "ablation_curves1.png"
        curves_path2 = "ablation_curves2.png"
        curves_path3 = "ablation_curves3.png"
    fig1.savefig(curves_path1, dpi=150)
    fig2.savefig(curves_path2, dpi=150)
    fig3.savefig(curves_path3, dpi=150)

    # 窗口4：稳定性指标汇总表格
    fig_table, ax_table = plt.subplots(figsize=(9, 5))
    ax_table.axis('off')

    metrics_list = [compute_stability_metrics(kpts_dict[m]) for m in modes]
    baseline = metrics_list[0]

    table_data = [["指标", "纯YOLO", "YOLO+卡尔曼", "YOLO+插值", "完整系统"]]
    metric_names = ['D_avg', 'D_std', 'R_valid', 'CV_torso', 'J_rate']
    metric_labels = ['平均位移(px)', '位移标准差(px)', '有效点比例', '躯干变异系数', '跳变率']

    for mname, mlabel in zip(metric_names, metric_labels):
        row = [mlabel]
        base_val = baseline.get(mname, np.nan)
        row.append(f"{base_val:.3f}" if not np.isnan(base_val) else "N/A")
        for m in metrics_list[1:]:
            val = m.get(mname, np.nan)
            if not np.isnan(val) and not np.isnan(base_val) and base_val != 0:
                change = (val - base_val) / base_val * 100
                row.append(f"{val:.3f} ({change:+.1f}%)")
            else:
                row.append(f"{val:.3f}" if not np.isnan(val) else "N/A")
        table_data.append(row)

    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax_table.set_title('稳定性指标汇总（括号内为相对纯YOLO变化）', fontsize=12, pad=20)

    fig_table.tight_layout()
    if save_path:
        base, ext = os.path.splitext(save_path)
        table_path = f"{base}_table{ext}"
    else:
        table_path = "ablation_table.png"
    fig_table.savefig(table_path, dpi=150)
    plt.show()  # 表格窗口也显示

    print(f"消融实验对比图已保存至: {curves_path1}, {curves_path2}, {curves_path3} 和 {table_path}")
    print("\n========== 消融实验结果 ==========")
    for row in table_data[1:]:
        print(f"{row[0]:12s} | 纯YOLO: {row[1]:>15s} | YOLO+卡尔曼: {row[2]:>15s} | YOLO+插值: {row[3]:>15s} | 完整系统: {row[4]:>15s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="稳定性评估与消融实验")
    parser.add_argument('--mode', type=str, choices=['compare', 'ablation'], default='ablation')# 模式切换 对比和消融
    parser.add_argument('--raw', type=str, default='eval_data/raw_yolo_kpts.npy')
    parser.add_argument('--enh', type=str, default='eval_data/enhanced_kpts.npy')
    parser.add_argument('--raw_kpts', type=str, default='eval_data/raw_yolo_kpts.npy')
    parser.add_argument('--kf_kpts', type=str, default='eval_data/kf_yolo_kpts.npy')
    parser.add_argument('--spectral_kpts', type=str, default='eval_data/spectral_yolo_kpts.npy')
    parser.add_argument('--full_kpts', type=str, default='eval_data/enhanced_kpts.npy')
    parser.add_argument('--smooth', type=int, default=5)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.mode == 'compare':
        if not os.path.exists(args.raw) or not os.path.exists(args.enh):
            print("错误：请确保以下文件存在：")
            print(f"  {args.raw}")
            print(f"  {args.enh}")
            exit(1)
        raw_kpts = np.load(args.raw)
        enhanced_kpts = np.load(args.enh)
        min_len = min(raw_kpts.shape[0], enhanced_kpts.shape[0])
        raw_kpts = raw_kpts[:min_len]
        enhanced_kpts = enhanced_kpts[:min_len]
        label_a = infer_model_name_from_path(args.raw)
        label_b = infer_model_name_from_path(args.enh)
        print(f"加载数据完成: 共 {min_len} 帧，对比模型: {label_a} vs {label_b}")
        output_path = args.output if args.output else "eval_comparison.png"
        plot_comparison(raw_kpts, enhanced_kpts, save_path=output_path, label_a=label_a, label_b=label_b)

    elif args.mode == 'ablation':
        files = {
            'raw': args.raw_kpts,
            'kf': args.kf_kpts,
            'spectral': args.spectral_kpts,
            'full': args.full_kpts
        }
        for key, path in files.items():
            if not os.path.exists(path):
                print(f"错误：文件不存在 - {path}")
                exit(1)
        kpts_dict = {}
        labels_dict = {}
        for key, path in files.items():
            kpts_dict[key] = np.load(path)
            labels_dict[key] = infer_model_name_from_path(path)
            print(f"加载 {labels_dict[key]}: {kpts_dict[key].shape}")
        output_path = args.output if args.output else "ablation_comparison.png"
        plot_ablation_comparison(kpts_dict, labels_dict, save_path=output_path, smooth_window=args.smooth)