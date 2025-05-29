'''
 # @ Copyright: @copyright (c) 2025 Gahan AI Private Limited
 # @ Author: Pallab Maji
 # @ Create Time: 2025-05-27 14:00:09
 # @ Modified time: 2025-05-27 14:00:58
 # @ Description: Load the video from the path, and exract driver behaviour using Mediapipe and OpenCV.
        1. Face detection 
        2. Landmark detection
        3. Gaze estimation
        4. Head pose estimation
        5. Emotional state estimation

        Dump the processed frames in a video file in the output directory with name of the original video file name appended with _output.mp4
 '''

import os
import cv2
import mediapipe as mp
import tqdm
from deepface import DeepFace


# USER INPUTS
# Flag to concatenate the processed frame & original frame to create a side-by-side view (optional)
# Set to True if you want to concatenate the processed frame & original frame to create a side-by-side view
CONCAT_OUTPUT = False  
# Flag to rotate the frame if needed (optional)
ROTATE_FRAME = False  # Set to True if you want to rotate the frame


# Define the path to the video file
# video_url = "./data/video_1637055249499_2204.mp4"
# video_url = "./data/video_1636641427815_88398.mp4"
video_url = "./data/test_clip_day2.mp4"
output_directory = "./output"

# Open the video file
cap = cv2.VideoCapture(video_url, )
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_url}")

# Get the original video file name without extension
video_filename = os.path.splitext(os.path.basename(video_url))[0]
# Define the output video file path
output_video_path = os.path.join(output_directory, f"{video_filename}_output.mp4")
# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Print video properties
print(f"Video Properties:\n"
      f"FPS: {fps}\n"
      f"Width: {width}\n"
      f"Height: {height}\n"
      f"Rotate Frame: {'Yes' if ROTATE_FRAME else 'No'}")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

if ROTATE_FRAME:
    if CONCAT_OUTPUT:
        # If concatenating, double the width for side-by-side view
        out = cv2.VideoWriter(output_video_path, cv2.CAP_FFMPEG, fourcc, fps, (height * 2, width))
    else:
        out = cv2.VideoWriter(output_video_path, cv2.CAP_FFMPEG, fourcc, fps, (height, width))
else:
    if CONCAT_OUTPUT:
        # If concatenating, double the width for side-by-side view
        out = cv2.VideoWriter(output_video_path, cv2.CAP_FFMPEG, fourcc, fps, (width * 2, height))
    else:
        out = cv2.VideoWriter(output_video_path, cv2.CAP_FFMPEG, fourcc, fps, (width, height))

# Check if VideoWriter is opened successfully
if not out.isOpened():
    raise IOError(f"Cannot open video writer for output file: {output_video_path}")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,  # Set to 1 for single face detection
    refine_landmarks=True,  # Enable landmark refinement
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


# Process the video frame by frame  
#find number of frames in the video
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing video: {video_url}")
print(f"Total frames to process: {num_frames}")
for _ in tqdm.tqdm(range(num_frames), desc="Processing frames"):
    # Read a frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break

    if ROTATE_FRAME:
        # Rotate the frame if needed (optional)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    # Draw face mesh annotations on the frame
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            
    # Find ROI for the face mesh
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the bounding box of the face
            h, w, _ = frame.shape
            x_min = int(min([landmark.x * w for landmark in face_landmarks.landmark]))
            x_max = int(max([landmark.x * w for landmark in face_landmarks.landmark]))
            y_min = int(min([landmark.y * h for landmark in face_landmarks.landmark]))
            y_max = int(max([landmark.y * h for landmark in face_landmarks.landmark]))

            # Extract the Gaze Estimation and Head Pose Estimation
            # gaze.gaze(image, results.multi_face_landmarks[0])

            # Detect the face emotions using DeepFace once in every second
            face_roi = frame[y_min:y_max, x_min:x_max]
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if face_roi.size > 0:  # Analyze every 30th frame (approximately every second at 30 FPS)
                try:
                    # Analyze the face ROI for emotions
                    analysis = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
                    
                    # Get 5 dominant emotions
                    if analysis and isinstance(analysis, list) and len(analysis) > 0:
                        for emotion, score in analysis[0]['emotion'].items():
                            # place it on the left side of the frame and show percentage in progress bar style
                            # Display the emotion and score
                            cv2.putText(frame, f"{emotion}: {score:.2f}", (100, 100 + 20 * list(analysis[0]['emotion'].keys()).index(emotion)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    else:
                        cv2.putText(frame, "No emotions or face detected", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Get the age, gender, dominant emotion
                    # age = analysis[0]['age']
                    # gender = analysis[0]['gender']
                    emotion = analysis[0]['dominant_emotion']
                    # Display the age, gender, and dominant emotion on the frame
                    # cv2.putText(frame, f"Age: {age}", (x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    # cv2.putText(frame, f"Gender: {gender}", (x_min, y_min - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Emotion: {emotion}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    

                except Exception as e:
                    print(f"Error analyzing face ROI: {e}")
            else:
                print("Face ROI is empty, skipping emotion analysis.")

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)



        

    # Write the processed frame to the output video file 
    if CONCAT_OUTPUT:
        # concatenate the processed frame & original frame to create a side-by-side view
        conc_frame = cv2.hconcat([bgr_frame, frame]) 
        # Resize the frame to fit the output video dimensions
        conc_frame = cv2.resize(conc_frame, (height*2, width))
        out.write(conc_frame)
        # Display the processed frame
        cv2.imshow('Real-time Facial Emotion Analysis', conc_frame)
    else:
        # Write the processed frame directly to the output video file
        out.write(frame)
        # Display the processed frame
        cv2.imshow('Real-time Facial Emotion Analysis', frame)
    
    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

   


# Release resources
cap.release()
out.release()
print(f"Processed video saved to: {output_video_path}")
# Close MediaPipe resources
face_mesh.close()
# Close OpenCV windows
cv2.destroyAllWindows()
# Print completion message
print("Video processing completed successfully.")

