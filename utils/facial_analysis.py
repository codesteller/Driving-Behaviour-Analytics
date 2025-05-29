"""
# @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
# @ Author: Pallab Maji
# @ Create Time: 2025-05-29 14:33:19
# @ Modified time: 2025-05-29 14:33:44
# @ Description: Enter description here
"""

import mediapipe as mp
from deepface import DeepFace
import cv2
import os
from .logger import Logger
import logging


class FacialAnalysis:
    def __init__(self, logs_directory="./logs"):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.convert_to_rgb = False  # Flag to convert frame to RGB if needed
        self.logger = Logger(
            os.path.join(logs_directory, "emotion_analysis"), log_level=logging.INFO
        )

        self.initialize_face_analysis()
        self.logger.info(
            "EmotionAnalysis initialized with face detection and face mesh."
        )

    # Initialize face detection and face mesh
    def initialize_face_analysis(self):
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def facial_analysis(self, frame):
        """
        1. Extract facial landmarks using MediaPipe Face Mesh.
        2. Draw the landmarks on the frame.
        3. Return the processed frame with landmarks.
        4. Analyze the age and emotion of a given frame using DeepFace.
        :param frame: Input frame (RGB format)
        :return: Processed frame with landmarks and dominant emotion
        5. If the frame is empty, log an error and return None.
        6. Convert the frame to RGB if needed.
        """

        if frame is None:
            self.logger.error("Received an empty frame for emotion analysis.")
            return None

        # Convert frame to RGB if needed
        if self.convert_to_rgb:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame

        try:
            # Process the frame with MediaPipe Face Mesh
            results = self.face_mesh.process(rgb_frame)

            # Draw face mesh annotations on the frame
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=rgb_frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                        )

                    self.mp_drawing.draw_landmarks(
                        image=rgb_frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )

                    # Find the bounding box of the face
                    h, w, _ = rgb_frame.shape
                    x_min = int(
                        min([landmark.x for landmark in face_landmarks.landmark]) * w
                    )
                    x_max = int(
                        max([landmark.x for landmark in face_landmarks.landmark]) * w
                    )
                    y_min = int(
                        min([landmark.y for landmark in face_landmarks.landmark]) * h
                    )
                    y_max = int(
                        max([landmark.y for landmark in face_landmarks.landmark]) * h
                    )

                    face_roi = rgb_frame[y_min:y_max, x_min:x_max]
                    if face_roi.size > 0:
                        # Analyze the face ROI for emotions using DeepFace
                        analysis = DeepFace.analyze(
                            face_roi, actions=("emotion"), enforce_detection=False
                        )

                        # Draw the bounding box around the facec
                        cv2.rectangle(
                            rgb_frame,
                            (x_min, y_min),
                            (x_max, y_max),
                            (0, 255, 255),
                            2,
                        )

                        dominant_emotion = analysis[0]['dominant_emotion']
                        emotion_score = analysis[0]['emotion'][dominant_emotion]
                        # If emotion detected is sad, neutral, or angry, log the emotion as Neutral
                        if dominant_emotion in ['sad', 'neutral', 'angry', "fear"]:
                            dominant_emotion = 'Neutral'
                            emotion_score = 1.0

                        # Display the dominant emotion and score on the frame
                        cv2.putText(
                            rgb_frame,
                            f"Emotion: {dominant_emotion} ({emotion_score:.2f})",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                        # Log the dominant emotion and score
                        self.logger.info("Dominant emotion detected: {} with score: {}".format(dominant_emotion, emotion_score))

                        # get 5 Dominant emotions and display them in the frame corner with a bar plot style
                        if analysis and isinstance(analysis, list) and len(analysis) > 0:
                            for emotion, score in analysis[0]['emotion'].items():
                                # Display the emotion and score ina bar plot style in top right corner
                                cv2.putText(rgb_frame, f"{emotion}: {score:.2f}", (100, 100 + 20 * list(analysis[0]['emotion'].keys()).index(emotion)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 255, 0), 1)
            else:
                self.logger.warning("No face landmarks detected in the frame.")

            return rgb_frame

        except Exception as e:
            self.logger.exception(f"Error during facial analysis: {e}")
            return None
