# pose_enhancer.py
import numpy as np
from config import BODY_CHAINS, ADJACENCY_MATRIX, SPECTRAL_CONFIG

class SpectralPoseEnhancer:
    def __init__(self):
        self.debug_info = []
        self.reconstruction_error_history = []
        self.prev_kpts = None
        self.prev_confs = None
        # 不采用骨骼长度先验，保持简单稳定

    def compute_graph_laplacian(self, confs):
        adj_weighted = ADJACENCY_MATRIX.copy()
        for i in range(17):
            for j in range(17):
                if adj_weighted[i, j] > 0:
                    adj_weighted[i, j] *= (confs[i] + confs[j]) / 2
        D = np.diag(np.sum(adj_weighted, axis=1))
        L = D - adj_weighted
        return L, adj_weighted

    def adaptive_spectral_interpolation(self, kpts, confs, image_shape):
        kpts = kpts.copy()
        confs = confs.copy()
        h, w = image_shape[:2] if image_shape else (1000, 1000)

        # 记录原始无效点数量
        invalid_before = np.sum(confs < 0.2)

        # 步骤1：谱图加权邻域插值
        L, adj_weighted = self.compute_graph_laplacian(confs)
        for i in range(17):
            if confs[i] < 0.2:
                neighbors = np.where(adj_weighted[i] > 0.05)[0]
                # 选择置信度较高的邻居
                valid_neighbors = [n for n in neighbors if confs[n] > 0.15]
                if len(valid_neighbors) >= 2:
                    # 计算邻域加权平均
                    weights = [adj_weighted[i, n] * confs[n] for n in valid_neighbors]
                    weights = np.array(weights) / np.sum(weights)
                    neighbor_kpts = kpts[valid_neighbors]
                    interpolated = np.sum(neighbor_kpts * weights[:, np.newaxis], axis=0)
                    # 如果有时序信息，进行平滑融合
                    if self.prev_kpts is not None and self.prev_confs[i] > 0.2:
                        # 防止跳变，融合上一帧
                        interpolated = 0.6 * interpolated + 0.4 * self.prev_kpts[i]
                    kpts[i] = interpolated
                    # 设置合理置信度
                    avg_conf = np.mean(confs[valid_neighbors])
                    confs[i] = min(0.7, avg_conf * 1.2)
                elif len(valid_neighbors) == 1:
                    # 只有一个邻居，直接使用但降权
                    kpts[i] = kpts[valid_neighbors[0]]
                    confs[i] = confs[valid_neighbors[0]] * 0.5
                # 如果完全没有邻居，则不处理

        # 步骤2：简单的对称补全（仅在躯干可见时）
        torso_visible = confs[5] > 0.25 and confs[6] > 0.25 and confs[11] > 0.25 and confs[12] > 0.25
        if torso_visible:
            center_x = (kpts[5][0] + kpts[6][0] + kpts[11][0] + kpts[12][0]) / 4
            left_right_pairs = [(5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
            for left, right in left_right_pairs:
                if confs[left] > 0.3 and confs[right] < 0.2:
                    kpts[right][0] = 2 * center_x - kpts[left][0]
                    kpts[right][1] = kpts[left][1]
                    confs[right] = confs[left] * 0.7
                elif confs[right] > 0.3 and confs[left] < 0.2:
                    kpts[left][0] = 2 * center_x - kpts[right][0]
                    kpts[left][1] = kpts[right][1]
                    confs[left] = confs[right] * 0.7

        # 步骤3：轻微时序平滑（对高置信度点也做轻微滤波，减少抖动）
        if self.prev_kpts is not None:
            for i in range(17):
                if confs[i] > 0.4 and self.prev_confs[i] > 0.4:
                    # 仅当位移较小时才平滑，避免过度滞后
                    displacement = np.linalg.norm(kpts[i] - self.prev_kpts[i])
                    if displacement < 30:
                        kpts[i] = 0.85 * kpts[i] + 0.15 * self.prev_kpts[i]

        # 记录无效点变化
        invalid_after = np.sum(confs < 0.2)
        improvement = invalid_before - invalid_after

        # 计算帧间平均位移（用于调试）
        relative_error = 0.0
        if self.prev_kpts is not None:
            valid_current = [i for i in range(17) if confs[i] > 0.1 and kpts[i][0] > 0]
            valid_prev = [i for i in range(17) if self.prev_confs[i] > 0.1 and self.prev_kpts[i][0] > 0]
            common = set(valid_current) & set(valid_prev)
            if common:
                displacements = [np.linalg.norm(kpts[i] - self.prev_kpts[i]) for i in common]
                relative_error = np.mean(displacements)

        # 更新历史
        self.prev_kpts = kpts.copy()
        self.prev_confs = confs.copy()
        self.reconstruction_error_history.append(relative_error)

        self.debug_info.append({
            "invalid_before": invalid_before,
            "invalid_after": invalid_after,
            "improvement": improvement,
            "relative_error": relative_error
        })

        return kpts, confs

    # 以下为兼容性占位方法
    def spectral_reconstruction_error(self, kpts, confs):
        return 0, 0

    def mirror_chain_with_spectral_weight(self, kpts, confs):
        return kpts, confs

    def get_spectral_center_x(self, kpts, confs):
        return 0

    def body_model_completion_with_error_constraint(self, kpts, confs, error_bound):
        return kpts, confs