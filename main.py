# # main.py
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# from video_processor import process_video
#
# if __name__ == "__main__":
#     input_video = 'video/fall_hhl.mp4'
#     output_video = 'results/results_fall.mp4'
#
#     if not os.path.exists(input_video):
#         print(f"请放入测试视频: {input_video}")
#     else:
#         process_video(input_video, output_video)


# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from video_processor import process_video

if __name__ == "__main__":
    input_video = 'video/fall_hhl.mp4'
    output_video = 'results/results_fall.mp4'

    ground_truth_fall_frames = None

    if not os.path.exists(input_video):
        print(f"请放入测试视频: {input_video}")
    else:
        process_video(input_video, output_video, ground_truth_fall_frames)
