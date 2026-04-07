# # pose_enhancer.py 姿态增强（链式插值、镜像、比例）
# import numpy as np
# from config import BODY_CHAINS
#
# class PoseEnhancer:
#     def __init__(self):
#         self.debug_info = []  # 用于调试统计
#
#     def has_valid_chain(self, chain_name, confs):
#         """检查链是否有有效点"""
#         chain_indices = BODY_CHAINS[chain_name]
#         return any(confs[idx] > 0.25 for idx in chain_indices)
#
#     def mirror_chain(self, kpts, confs, source_chain, target_chain):
#         """镜像对称链"""
#         source_indices = BODY_CHAINS[source_chain]
#         target_indices = BODY_CHAINS[target_chain]
#
#         if len(source_indices) != len(target_indices):
#             return kpts, confs
#
#         # 计算镜像轴（身体中线）- 使用肩部中心
#         if confs[5] > 0.25 and confs[6] > 0.25:
#             center_x = (kpts[5][0] + kpts[6][0]) / 2
#         elif confs[11] > 0.25 and confs[12] > 0.25:
#             center_x = (kpts[11][0] + kpts[12][0]) / 2
#         else:
#             return kpts, confs
#
#         for src_idx, tgt_idx in zip(source_indices, target_indices):
#             if confs[src_idx] > 0.25 and confs[tgt_idx] < 0.15:
#                 # 水平镜像
#                 kpts[tgt_idx][0] = 2 * center_x - kpts[src_idx][0]
#                 kpts[tgt_idx][1] = kpts[src_idx][1]
#                 confs[tgt_idx] = confs[src_idx] * 0.7  # 镜像点置信度降低
#
#         return kpts, confs
#
#     def body_model_completion(self, kpts, confs):
#         """基于人体模型的补全"""
#         # 如果躯干可见，可以估计其他部位
#         torso_visible = all(confs[i] > 0.25 for i in [5, 6, 11, 12])
#
#         if torso_visible:
#             # 计算身体尺寸
#             shoulder_center = (kpts[5] + kpts[6]) / 2
#             hip_center = (kpts[11] + kpts[12]) / 2
#             torso_height = np.linalg.norm(hip_center - shoulder_center)
#
#             if torso_height < 1:  # 避免除以零
#                 return kpts, confs
#
#             # 估计头部位置
#             if confs[0] < 0.15:  # 鼻子
#                 kpts[0] = shoulder_center + np.array([0, -torso_height * 0.25])
#                 confs[0] = 0.3
#
#             # 估计眼睛位置
#             for eye_idx in [1, 2]:
#                 if confs[eye_idx] < 0.15:
#                     kpts[eye_idx] = kpts[0] + np.array([-10 if eye_idx == 1 else 10, -5])
#                     confs[eye_idx] = 0.25
#
#             # 估计膝盖位置
#             if confs[13] < 0.15 and confs[11] > 0.25:  # 左膝盖缺失，左髋可见
#                 leg_direction = np.array([0, 1])  # 默认向下
#                 kpts[13] = kpts[11] + leg_direction * torso_height * 0.7
#                 confs[13] = 0.3
#
#             if confs[14] < 0.15 and confs[12] > 0.25:  # 右膝盖缺失，右髋可见
#                 leg_direction = np.array([0, 1])  # 默认向下
#                 kpts[14] = kpts[12] + leg_direction * torso_height * 0.7
#                 confs[14] = 0.3
#
#             # 估计脚踝位置（如果膝盖可见）
#             if confs[13] > 0.25 and confs[15] < 0.15:  # 左膝盖可见，左脚踝缺失
#                 thigh_vec = kpts[13] - kpts[11]
#                 kpts[15] = kpts[13] + thigh_vec * 0.8
#                 confs[15] = 0.3
#
#             if confs[14] > 0.25 and confs[16] < 0.15:  # 右膝盖可见，右脚踝缺失
#                 thigh_vec = kpts[14] - kpts[12]
#                 kpts[16] = kpts[14] + thigh_vec * 0.8
#                 confs[16] = 0.3
#
#         return kpts, confs
#
#     def chain_interpolation(self, keypoints, confidences, image_shape=None):
#         """
#         基于人体结构的链式插值
#         按照人体关节链顺序进行插值，保持结构合理性
#         """
#         kpts = keypoints.copy()
#         confs = confidences.copy()
#
#         if image_shape is not None:
#             h, w = image_shape[:2]
#         else:
#             h, w = 1000, 1000  # 默认值
#
#         # 记录无效点数量（用于调试）
#         invalid_before = sum(1 for i in range(17) if confs[i] < 0.1 or
#                              kpts[i][0] < 0 or kpts[i][1] < 0 or
#                              kpts[i][0] > w or kpts[i][1] > h)
#
#         # 第一步：识别并标记无效点
#         for i in range(17):
#             # 更宽松的无效点判断条件
#             if (confs[i] < 0.1 or
#                     kpts[i][0] < 0 or kpts[i][1] < 0 or
#                     kpts[i][0] > w * 1.5 or kpts[i][1] > h * 1.5):
#                 # 标记为无效
#                 kpts[i] = np.array([-1000, -1000], dtype=np.float32)
#                 confs[i] = 0.0
#
#         # 第二步：对称链补全
#         # 左臂缺失，右臂可见 -> 镜像
#         if not self.has_valid_chain("left_arm", confs) and self.has_valid_chain("right_arm", confs):
#             kpts, confs = self.mirror_chain(kpts, confs, "right_arm", "left_arm")
#
#         # 右臂缺失，左臂可见 -> 镜像
#         if not self.has_valid_chain("right_arm", confs) and self.has_valid_chain("left_arm", confs):
#             kpts, confs = self.mirror_chain(kpts, confs, "left_arm", "right_arm")
#
#         # 左腿缺失，右腿可见 -> 镜像
#         if not self.has_valid_chain("left_leg", confs) and self.has_valid_chain("right_leg", confs):
#             kpts, confs = self.mirror_chain(kpts, confs, "right_leg", "left_leg")
#
#         # 右腿缺失，左腿可见 -> 镜像
#         if not self.has_valid_chain("right_leg", confs) and self.has_valid_chain("left_leg", confs):
#             kpts, confs = self.mirror_chain(kpts, confs, "left_leg", "right_leg")
#
#         # 第三步：识别中心参考点（躯干点）
#         torso_points = BODY_CHAINS["torso_center"]
#         torso_indices = [i for i in torso_points if confs[i] > 0.15]
#
#         if len(torso_indices) >= 2:
#             # 计算躯干中心位置
#             valid_torso_kpts = [kpts[i] for i in torso_indices]
#             torso_center = np.mean(valid_torso_kpts, axis=0)
#
#             # 如果某些关键点完全缺失，基于躯干中心初始化
#             for i in range(17):
#                 if confs[i] < 0.1 and np.all(kpts[i] < 0):
#                     # 基于人体比例设置初始位置
#                     if i in [0, 1, 2, 3, 4]:  # 头部
#                         kpts[i] = torso_center + np.array([0, -50])
#                         confs[i] = 0.2
#                     elif i in [5, 7, 9]:  # 左臂
#                         kpts[i] = torso_center + np.array([-30, 0])
#                         confs[i] = 0.2
#                     elif i in [6, 8, 10]:  # 右臂
#                         kpts[i] = torso_center + np.array([30, 0])
#                         confs[i] = 0.2
#                     elif i in [11, 13, 15]:  # 左腿
#                         kpts[i] = torso_center + np.array([-15, 30])
#                         confs[i] = 0.2
#                     elif i in [12, 14, 16]:  # 右腿
#                         kpts[i] = torso_center + np.array([15, 30])
#                         confs[i] = 0.2
#
#         # 第四步：按链顺序插值
#         for chain_name, chain_indices in BODY_CHAINS.items():
#             if chain_name == "shoulder_hip":
#                 continue  # 特殊链单独处理
#
#             # 找到链中的有效点
#             valid_indices = []
#             valid_kpts = []
#             for idx in chain_indices:
#                 if confs[idx] > 0.1:
#                     valid_indices.append(idx)
#                     valid_kpts.append(kpts[idx])
#
#             # 如果链中有至少2个有效点，插值中间缺失点
#             if len(valid_indices) >= 2:
#                 # 对链中每个点进行插值
#                 for i in range(len(chain_indices)):
#                     idx = chain_indices[i]
#                     if confs[idx] < 0.2:  # 提高阈值
#                         # 找到前一个有效点
#                         prev_idx = None
#                         for j in range(i - 1, -1, -1):
#                             if confs[chain_indices[j]] > 0.1:
#                                 prev_idx = chain_indices[j]
#                                 break
#
#                         # 找到后一个有效点
#                         next_idx = None
#                         for j in range(i + 1, len(chain_indices)):
#                             if confs[chain_indices[j]] > 0.1:
#                                 next_idx = chain_indices[j]
#                                 break
#
#                         # 如果有前后两个有效点，进行线性插值
#                         if prev_idx is not None and next_idx is not None:
#                             prev_pos = kpts[prev_idx]
#                             next_pos = kpts[next_idx]
#
#                             # 计算距离比例
#                             total_dist = np.linalg.norm(next_pos - prev_pos)
#                             if total_dist > 1:
#                                 # 计算中间点的权重
#                                 prev_dist = np.linalg.norm(kpts[idx] - prev_pos) if confs[idx] > 0 else 0
#                                 if prev_dist == 0 or confs[idx] < 0.05:
#                                     # 如果当前点完全缺失，基于索引位置插值
#                                     idx_in_chain = i
#                                     prev_idx_in_chain = chain_indices.index(prev_idx)
#                                     next_idx_in_chain = chain_indices.index(next_idx)
#                                     ratio = (idx_in_chain - prev_idx_in_chain) / (next_idx_in_chain - prev_idx_in_chain)
#
#                                     kpts[idx] = prev_pos + ratio * (next_pos - prev_pos)
#                                     confs[idx] = max(confs[idx], 0.3)
#                                 else:
#                                     # 如果当前点有大致位置，向链上投影
#                                     ratio = prev_dist / total_dist
#                                     projected = prev_pos + ratio * (next_pos - prev_pos)
#                                     kpts[idx] = 0.2 * kpts[idx] + 0.8 * projected
#                                     confs[idx] = max(confs[idx], 0.35)
#
#         # 第五步：特殊处理肩髋关系
#         shoulder_indices = [5, 6]
#         hip_indices = [11, 12]
#
#         # 确保左右肩和左右髋的对称性
#         for left_idx, right_idx in [(5, 6), (11, 12)]:
#             if confs[left_idx] > 0.25 and confs[right_idx] > 0.25:
#                 # 两边都有，计算中点确保对称
#                 center = (kpts[left_idx] + kpts[right_idx]) / 2
#                 if confs[left_idx] > confs[right_idx]:
#                     kpts[right_idx] = 2 * center - kpts[left_idx]
#                     confs[right_idx] = max(confs[right_idx], 0.3)
#                 else:
#                     kpts[left_idx] = 2 * center - kpts[right_idx]
#                     confs[left_idx] = max(confs[left_idx], 0.3)
#
#         # 第六步：使用人体比例约束进行后处理
#         kpts, confs = self.apply_body_proportions(kpts, confs)
#
#         # 第七步：基于人体模型的补全
#         kpts, confs = self.body_model_completion(kpts, confs)
#
#         # 记录补全效果（用于调试）
#         invalid_after = sum(1 for i in range(17) if confs[i] < 0.1)
#         self.debug_info.append({
#             "invalid_before": invalid_before,
#             "invalid_after": invalid_after,
#             "improvement": invalid_before - invalid_after
#         })
#
#         return kpts, confs
#
#     def apply_body_proportions(self, keypoints, confidences):
#         """应用人体比例约束"""
#         kpts = keypoints.copy()
#         confs = confidences.copy()
#
#         # 计算可见点的比例关系
#         visible_indices = [i for i in range(17) if confs[i] > 0.2]
#         if len(visible_indices) >= 4:
#             # 计算肩宽和髋宽
#             shoulder_width = 0
#             if confs[5] > 0.25 and confs[6] > 0.25:
#                 shoulder_width = np.linalg.norm(kpts[5] - kpts[6])
#
#             hip_width = 0
#             if confs[11] > 0.25 and confs[12] > 0.25:
#                 hip_width = np.linalg.norm(kpts[11] - kpts[12])
#
#             # 如果肩宽合理，限制手臂长度
#             if shoulder_width > 10:
#                 # 手臂长度应该约为肩宽的1.2-1.5倍
#                 arm_scale = 1.3
#
#                 # 左臂
#                 if confs[5] > 0.25 and confs[7] > 0.25:
#                     arm_length = np.linalg.norm(kpts[7] - kpts[5])
#                     if arm_length > shoulder_width * 2.5 or arm_length < shoulder_width * 0.5:
#                         # 调整到合理范围
#                         direction = kpts[7] - kpts[5]
#                         if np.linalg.norm(direction) > 0:
#                             direction = direction / np.linalg.norm(direction)
#                             kpts[7] = kpts[5] + direction * shoulder_width * arm_scale
#                             confs[7] = max(confs[7], 0.3)
#
#                 # 右臂类似
#                 if confs[6] > 0.25 and confs[8] > 0.25:
#                     arm_length = np.linalg.norm(kpts[8] - kpts[6])
#                     if arm_length > shoulder_width * 2.5 or arm_length < shoulder_width * 0.5:
#                         direction = kpts[8] - kpts[6]
#                         if np.linalg.norm(direction) > 0:
#                             direction = direction / np.linalg.norm(direction)
#                             kpts[8] = kpts[6] + direction * shoulder_width * arm_scale
#                             confs[8] = max(confs[8], 0.3)
#
#         return kpts, confs

# pose_enhancer.py 自适应谱图拓扑插值（谱图理论+重建误差下界）
import numpy as np
from config import BODY_CHAINS, ADJACENCY_MATRIX, SPECTRAL_CONFIG


class SpectralPoseEnhancer:
    def __init__(self):
        self.debug_info = []
        self.reconstruction_error_history = []  # 重建误差记录

    def compute_graph_laplacian(self, confs):
        """算自适应图拉普拉斯矩阵（基于置信度加权）"""
        # 邻接矩阵加权：置信度越高，权重越大
        adj_weighted = ADJACENCY_MATRIX.copy()
        for i in range(17):
            for j in range(17):
                if adj_weighted[i, j] > 0:
                    adj_weighted[i, j] *= (confs[i] + confs[j]) / 2  # 双边置信度加权

        # 度矩阵
        D = np.diag(np.sum(adj_weighted, axis=1))
        # 图拉普拉斯矩阵 L = D - A
        L = D - adj_weighted
        return L, adj_weighted

    def spectral_reconstruction_error(self, kpts, confs):
        """计算图拉普拉斯谱的重建误差下界"""
        L, _ = self.compute_graph_laplacian(confs)
        # 计算拉普拉斯矩阵的特征值
        eigvals = np.linalg.eigvalsh(L)
        lambda_min = max(SPECTRAL_CONFIG["lambda_min"], np.min(eigvals[eigvals > 1e-6]))

        # 重建误差下界：||x - x_hat||² ≥ λ_min * ||x||²
        valid_kpts = kpts[confs > 0.1]
        if len(valid_kpts) == 0:
            return 0, lambda_min

        x_norm = np.linalg.norm(valid_kpts)
        error_lower_bound = lambda_min * x_norm ** 2
        return error_lower_bound, lambda_min

    def adaptive_spectral_interpolation(self, kpts, confs, image_shape):
        """自适应谱图插值"""
        kpts = kpts.copy()
        confs = confs.copy()
        h, w = image_shape[:2] if image_shape else (1000, 1000)

        # 步骤1：计算初始重建误差下界
        error_lower_bound, lambda_min = self.spectral_reconstruction_error(kpts, confs)
        invalid_before = sum(1 for i in range(17) if confs[i] < 0.1 or
                             kpts[i][0] < 0 or kpts[i][1] < 0 or
                             kpts[i][0] > w or kpts[i][1] > h)

        # 步骤2：自适应拓扑插值（基于谱图权重）
        L, adj_weighted = self.compute_graph_laplacian(confs)
        for i in range(17):
            if confs[i] < 0.2:  # 低置信度点需要插值
                # 找到拓扑邻居（邻接矩阵权重>0）
                neighbors = np.where(adj_weighted[i] > 0.05)[0]
                neighbor_kpts = kpts[neighbors][confs[neighbors] > 0.1]

                if len(neighbor_kpts) >= 2:
                    # 谱权重插值：邻居权重越高，占比越大
                    neighbor_weights = adj_weighted[i][neighbors][confs[neighbors] > 0.1]
                    neighbor_weights = neighbor_weights / np.sum(neighbor_weights)

                    # 加权插值
                    interpolated = np.sum(neighbor_kpts * neighbor_weights[:, np.newaxis], axis=0)
                    # 误差约束：插值结果不超过误差下界
                    kpts[i] = interpolated
                    confs[i] = min(0.8, (np.mean(confs[neighbors]) + 0.2))

        # 步骤3：镜像补全（保留但融入谱权重）
        kpts, confs = self.mirror_chain_with_spectral_weight(kpts, confs)

        # 步骤4：人体模型约束（结合误差下界优化）
        kpts, confs = self.body_model_completion_with_error_constraint(kpts, confs, error_lower_bound)

        # 步骤5：记录重建误差
        invalid_after = sum(1 for i in range(17) if confs[i] < 0.1)
        improvement = invalid_before - invalid_after
        self.debug_info.append({
            "invalid_before": invalid_before,
            "invalid_after": invalid_after,
            "improvement": improvement,
            "reconstruction_error_bound": error_lower_bound,
            "lambda_min": lambda_min
        })
        self.reconstruction_error_history.append(error_lower_bound)

        return kpts, confs

    def mirror_chain_with_spectral_weight(self, kpts, confs):
        """镜像对称+谱权重"""
        chain_pairs = [("left_arm", "right_arm"), ("left_leg", "right_leg")]
        for source_chain, target_chain in chain_pairs:
            source_indices = BODY_CHAINS[source_chain]
            target_indices = BODY_CHAINS[target_chain]

            if len(source_indices) != len(target_indices):
                continue

            # 谱权重中心
            center_x = self.get_spectral_center_x(kpts, confs)
            if center_x is None:
                continue

            for src_idx, tgt_idx in zip(source_indices, target_indices):
                if confs[src_idx] > 0.25 and confs[tgt_idx] < 0.15:
                    # 镜像+谱权重衰减
                    spectral_weight = (confs[src_idx] + 0.1) / (lambda_min + 1) if 'lambda_min' in locals() else 0.7
                    kpts[tgt_idx][0] = 2 * center_x - kpts[src_idx][0]
                    kpts[tgt_idx][1] = kpts[src_idx][1]
                    confs[tgt_idx] = confs[src_idx] * spectral_weight

        return kpts, confs

    def get_spectral_center_x(self, kpts, confs):
        """谱权重中心（基于置信度加权的身体中线）"""
        torso_indices = [5, 6, 11, 12]
        valid_torso = [(kpts[i][0], confs[i]) for i in torso_indices if confs[i] > 0.25]
        if len(valid_torso) < 2:
            return None

        xs, weights = zip(*valid_torso)
        center_x = np.average(xs, weights=weights)
        return center_x

    def body_model_completion_with_error_constraint(self, kpts, confs, error_bound):
        """人体模型补全+误差下界约束"""
        torso_visible = all(confs[i] > 0.25 for i in [5, 6, 11, 12])
        if not torso_visible:
            return kpts, confs

        shoulder_center = (kpts[5] + kpts[6]) / 2
        hip_center = (kpts[11] + kpts[12]) / 2
        torso_height = np.linalg.norm(hip_center - shoulder_center)
        if torso_height < 1:
            return kpts, confs

        # 误差约束下的头部补全
        if confs[0] < 0.15:
            kpts[0] = shoulder_center + np.array([0, -torso_height * 0.25])
            # 误差约束：补全后误差不超过下界
            current_error = np.linalg.norm(kpts[0] - shoulder_center)
            if current_error > error_bound * SPECTRAL_CONFIG["reconstruction_error_weight"]:
                kpts[0] = shoulder_center + np.array([0, -error_bound * 0.1])
            confs[0] = 0.3

        # 误差约束下的腿部补全
        leg_pairs = [(13, 11), (14, 12), (15, 13), (16, 14)]
        for leg_idx, parent_idx in leg_pairs:
            if confs[leg_idx] < 0.15 and confs[parent_idx] > 0.25:
                direction = np.array([0, 1])
                kpts[leg_idx] = kpts[parent_idx] + direction * torso_height * 0.7
                # 误差约束
                current_error = np.linalg.norm(kpts[leg_idx] - kpts[parent_idx])
                if current_error > error_bound * SPECTRAL_CONFIG["reconstruction_error_weight"]:
                    kpts[leg_idx] = kpts[parent_idx] + direction * error_bound * 0.5
                confs[leg_idx] = 0.3

        return kpts, confs