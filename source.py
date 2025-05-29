"""
# @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
# @ Author: Pallab Maji
# @ Create Time: 2025-05-27 14:00:09
# @ Modified time: 2025-05-27 14:00:58
# @ Description: Load the video from the path, and extract driver behaviour using Mediapipe and OpenCV.
       1. Face detection
       2. Landmark detection
       3. Gaze estimation
       4. Head pose estimation
       5. Emotional state estimation

       Dump the processed frames in a video file in the output directory with name of the original video file name appended with _output.mp4

"""

import os
from utils.video_pipeline import VideoPipeline
from utils.facial_analysis import FacialAnalysis


# USER INPUTS
# Flag to concatenate the processed frame & original frame to create a side-by-side view (optional)
# Set to True if you want to concatenate the processed frame & original frame to create a side-by-side view
CONCAT_OUTPUT = True
# Flag to rotate the frame if needed (optional)
ROTATE_FRAME = False  # Set to True if you want to rotate the frame


# Define the path to the video file
video_urls = ["./data/test_clip_day2.mp4", 
            "./data/test_clip_night.mp4", 
            "./data/test_clip_day.mp4"]
output_directory = "./output"
logs_directory = "./logs"


def main():
    for video_url in video_urls:
        # Ensure the video file exists
        if not os.path.exists(video_url):
            print(f"Video file {video_url} does not exist. Please check the path.")
            return
        # Initialize the video pipeline
        video_pipeline = VideoPipeline(
            video_url,
            logs_directory=logs_directory,
            output_directory=output_directory,
            concat_output=CONCAT_OUTPUT,
            rotate_frame=ROTATE_FRAME,
        )

        facial_analysis = FacialAnalysis(
            logs_directory=logs_directory
        )

        try:
            # Process the video frames
            while True:
                frame = video_pipeline.extract_frames()
                if frame is None:
                    break  # No more frames to process

                # Here you can add additional processing on the frame if needed
                # video_pipeline.show_frame(frame, frame_title="Original Frame")

                # Perform facial analysis on the frame
                processed_frame = frame.copy()
                processed_frame = facial_analysis.facial_analysis(processed_frame)

                # Show the processed frame with facial landmarks and emotion
                video_pipeline.show_frame(processed_frame, frame_title="Processed Frame with Facial Analysis")

                # Write the processed frame to the output video
                video_pipeline.write_frame(processed_frame)

            # Release resources
            video_pipeline.release()
            print(f"Processed video saved at: {video_pipeline.output_video_path}")
        except Exception as e:
            video_pipeline.logger.exception(f"An error occurred during video processing: {e}")
            video_pipeline.release()
            print("Video processing interrupted due to an error. Resources released.")


if __name__ == "__main__":
    main()
