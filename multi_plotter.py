# multi_plotter.py
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import threading
import queue
import time

class MultiPlotter:
    def __init__(self, max_len=100, alpha=0.3, video_w=1920, video_h=1080, update_interval=0.4):
        """
        异步实时绘图器
        :param max_len: 曲线显示的最大数据点数（建议 50~80 以减轻渲染压力）
        :param alpha: 指数加权移动平均平滑因子
        :param video_w: 视频宽度
        :param video_h: 视频高度
        :param update_interval: 图表刷新间隔（秒），默认 0.4，可调至 0.3~0.5
        """
        self.max_len = max_len
        self.alpha = alpha
        self.video_w = video_w
        self.video_h = video_h
        self.update_interval = update_interval

        # 数据队列（主线程推送，绘图线程消费）
        self.data_queue = queue.Queue()
        self.running = True

        # 初始化 Matplotlib 图表（在主线程创建）
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self._init_plots()
        plt.ion()
        plt.show(block=False)

        # 启动绘图线程
        self.thread = threading.Thread(target=self._plot_loop, daemon=True)
        self.thread.start()

    def _init_plots(self):
        """初始化三个子图的样式"""
        # 子图1：躯干倾斜角
        self.ax1.set_ylabel('Angle (deg)')
        self.ax1.set_title('Torso Angle')
        self.ax1.set_ylim(0, 90)
        self.line_angle, = self.ax1.plot([], [], 'm-', linewidth=2)

        # 子图2：帧间位移 + KF收敛度
        self.ax2.set_ylabel('Displacement (px)', color='black')
        self.ax2.set_title('Frame Displacement & KF Convergence')
        self.line_raw, = self.ax2.plot([], [], 'r--', alpha=0.7, label='Raw Displacement')
        self.line_smooth, = self.ax2.plot([], [], 'g-', linewidth=2, label=f'EWMA Smoothed (α={self.alpha})')
        self.ax2.set_ylim(0, 50)
        self.ax2.tick_params(axis='y', labelcolor='black')

        self.ax2_twin = self.ax2.twinx()
        self.ax2_twin.set_ylabel('KF Convergence', color='blue')
        self.line_convergence, = self.ax2_twin.plot([], [], 'b-', linewidth=2, label='KF Convergence')
        self.ax2_twin.set_ylim(0, 1.05)
        self.ax2_twin.tick_params(axis='y', labelcolor='blue')

        lines = [self.line_raw, self.line_smooth, self.line_convergence]
        labels = [l.get_label() for l in lines]
        self.ax2.legend(lines, labels, loc='upper left')

        # 子图3：脚踝轨迹
        self.ax3.set_xlabel('X (pixels)')
        self.ax3.set_ylabel('Y (pixels)')
        self.ax3.set_title('Ankle Trajectory')
        self.ax3.invert_yaxis()
        self.ax3.set_xlim(0, self.video_w)
        self.ax3.set_ylim(self.video_h, 0)
        self.scat_left = self.ax3.scatter([], [], c='cyan', s=10, label='Left Ankle')
        self.scat_right = self.ax3.scatter([], [], c='orange', s=10, label='Right Ankle')
        self.ax3.legend()

        self.fig.tight_layout()

    def update(self, torso_angle, raw_displacement, left_ankle, right_ankle, kf_convergence):
        """
        主线程调用，将最新数据推入队列
        """
        self.data_queue.put({
            'torso_angle': torso_angle,
            'raw_displacement': raw_displacement,
            'left_ankle': left_ankle,
            'right_ankle': right_ankle,
            'kf_convergence': kf_convergence
        })

    def _plot_loop(self):
        """绘图线程主循环"""
        # 内部数据缓存
        torso_angle_data = deque(maxlen=self.max_len)
        raw_displacement_data = deque(maxlen=self.max_len)
        smoothed_displacement_data = deque(maxlen=self.max_len)
        kf_convergence_data = deque(maxlen=self.max_len)
        left_ankle_traj = deque(maxlen=self.max_len)
        right_ankle_traj = deque(maxlen=self.max_len)
        ewma = None

        last_draw_time = 0
        while self.running:
            latest_data = None
            # 非阻塞获取最新一条数据
            while True:
                try:
                    latest_data = self.data_queue.get_nowait()
                except queue.Empty:
                    break

            if latest_data is not None:
                ta = latest_data['torso_angle']
                rd = latest_data['raw_displacement']
                la = latest_data['left_ankle']
                ra = latest_data['right_ankle']
                kc = latest_data['kf_convergence']

                # 更新缓存
                torso_angle_data.append(ta if ta is not None else np.nan)
                raw_displacement_data.append(rd)
                if ewma is None:
                    ewma = rd
                else:
                    ewma = self.alpha * rd + (1 - self.alpha) * ewma
                smoothed_displacement_data.append(ewma)
                kf_convergence_data.append(kc)
                if la is not None and la[0] > 0:
                    left_ankle_traj.append(la)
                if ra is not None and ra[0] > 0:
                    right_ankle_traj.append(ra)

                # 控制绘图频率
                current_time = time.time()
                if current_time - last_draw_time >= self.update_interval:
                    self._redraw(torso_angle_data, raw_displacement_data, smoothed_displacement_data,
                                 kf_convergence_data, left_ankle_traj, right_ankle_traj)
                    last_draw_time = current_time
            else:
                time.sleep(0.01)  # 队列空时短暂休眠

    def _redraw(self, torso_angle_data, raw_disp, smooth_disp, kf_conv, left_traj, right_traj):
        """执行绘图更新"""
        x_vals = list(range(len(torso_angle_data)))

        # 子图1：躯干角
        self.line_angle.set_data(x_vals, list(torso_angle_data))
        self.ax1.set_xlim(0, max(10, len(torso_angle_data)))
        angles_clean = [a for a in torso_angle_data if not np.isnan(a)]
        if angles_clean:
            self.ax1.set_ylim(0, max(90, max(angles_clean) + 10))

        # 子图2：位移与收敛度
        self.line_raw.set_data(x_vals, list(raw_disp))
        self.line_smooth.set_data(x_vals, list(smooth_disp))
        self.line_convergence.set_data(x_vals, list(kf_conv))
        self.ax2.set_xlim(0, max(10, len(raw_disp)))
        if raw_disp:
            max_disp = max(raw_disp)
            self.ax2.set_ylim(0, max(1, max_disp * 1.1))

        # 子图3：脚踝轨迹
        if left_traj:
            left_x, left_y = zip(*left_traj)
            self.scat_left.set_offsets(np.c_[left_x, left_y])
        if right_traj:
            right_x, right_y = zip(*right_traj)
            self.scat_right.set_offsets(np.c_[right_x, right_y])

        # 关键优化：使用 draw_idle 代替 draw，让 GUI 后端在空闲时渲染
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        """关闭绘图窗口并结束线程"""
        self.running = False
        self.thread.join(timeout=1.0)
        plt.close(self.fig)