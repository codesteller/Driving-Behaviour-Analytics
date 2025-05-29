"""
# @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
# @ Author: Pallab Maji
# @ Create Time: 2025-05-29 12:16:11
# @ Modified time: 2025-05-29 12:16:14
# @ Description: This class script
#          1. loads a video from a specified path
#          2. extracts and process the frame with basic image processing
#          3. returns frames
#          4. dumps the processed frames to a video file in the output directory with the original video file name appended with _output.mp4
"""

import os
import cv2
import logging
from .logger import Logger


class VideoPipeline:
    def __init__(
        self,
        video_url,
        logs_directory="./logs",
        output_directory="./output",
        concat_output=True,
        rotate_frame=True,
        convert_to_rgb=True,
    ):
        self.video_url = video_url
        self.output_directory = output_directory
        self.concat_output = concat_output
        self.rotate_frame = rotate_frame
        self.logger = Logger(
            os.path.join(logs_directory, "pipeline"), log_level=logging.INFO
        )  # Initialize logger
        self.convert_to_rgb = convert_to_rgb

        # Ensure the output directory exists
        os.makedirs(os.path.expanduser(self.output_directory), exist_ok=True)

        # Open the video file
        self.cap = cv2.VideoCapture(self.video_url)
        if not self.cap.isOpened():
            self.logger.error(f"Cannot open video file: {self.video_url}")
            raise IOError(f"Cannot open video file: {self.video_url}")

        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(
            f"Video Properties:\n"
            f"FPS: {self.fps}\n"
            f"Width: {self.width}\n"
            f"Height: {self.height}\n"
            f"Rotate Frame: {'Yes' if self.rotate_frame else 'No'}"
        )

        # Get the original video file name without extension
        video_filename = os.path.splitext(os.path.basename(self.video_url))[0]
        # Define the output video file path
        self.output_video_path = os.path.join(
            self.output_directory, f"{video_filename}_output.mp4"
        )

        self.out_fp = self.initialize_video_writer()

        self.current_frame = None

    def initialize_video_writer(self):
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Define the codec and create VideoWriter object
        if self.rotate_frame:
            if self.concat_output:
                # If concatenating, double the width for side-by-side view
                out = cv2.VideoWriter(
                    self.output_video_path,
                    cv2.CAP_FFMPEG,
                    fourcc,
                    self.fps / 2,
                    (self.height * 2, self.width),
                )
            else:
                out = cv2.VideoWriter(
                    self.output_video_path,
                    cv2.CAP_FFMPEG,
                    fourcc,
                    self.fps,
                    (self.height, self.width),
                )
        else:
            if self.concat_output:
                # If concatenating, double the width for side-by-side view
                out = cv2.VideoWriter(
                    self.output_video_path,
                    cv2.CAP_FFMPEG,
                    fourcc,
                    self.fps,
                    (self.width * 2, self.height),
                )
            else:
                out = cv2.VideoWriter(
                    self.output_video_path,
                    cv2.CAP_FFMPEG,
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                )

        self.logger.info(f"Output video will be saved at: {self.output_video_path}")

        return out

    def process_frame(self, frame):
        """
        Process the frame with basic image processing.
        This method can be extended to include more complex processing.
        """
        # Example: Convert to grayscale (can be replaced with any other processing)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return processed_frame

    def extract_frames(self):
        ret, frame = self.cap.read()
        if not ret:
            assert self.cap.get(cv2.CAP_PROP_POS_FRAMES) > 0, "No frames to process"
        if self.rotate_frame:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if self.convert_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.current_frame = frame

        return frame

    def write_frame(self, frame):

        if self.convert_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.concat_output:
            # Concatenate the processed frame and original frame side by side
            original_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            concatenated_frame = cv2.hconcat([original_frame, frame])
            self.out_fp.write(concatenated_frame)
        else:
            self.out_fp.write(frame)

    def show_frame(self, frame, frame_title="Processed Frame"):
        if frame is None:
            frame = self.current_frame
        if self.convert_to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write on top of Frame to press Q to quit
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 55, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(frame_title, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.logger.info("Video processing interrupted by user.")
            self.release()
            exit(0)

    def release(self):
        self.cap.release()
        self.out_fp.release()
        cv2.destroyAllWindows()
        self.logger.info(f"Processed video saved at: {self.output_video_path}")
        self.logger.info("Video processing completed successfully.")

    def __del__(self):
        self.release()
        self.logger.info("VideoPipeline instance deleted and resources released.")
