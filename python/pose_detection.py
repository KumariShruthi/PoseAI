import cv2
import mediapipe as mp

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Pose model
model_path = "pose_landmarker_full.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open camera
cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:

    frame_timestamp = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Camera not found")
            break

        # OpenCV uses BGR, MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Detect pose
        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        # Draw landmarks
        if result.pose_landmarks:

            for pose_landmarks in result.pose_landmarks:

                # Draw points
                for landmark in pose_landmarks:

                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw connections
                connections = [
                    (11, 12),
                    (11, 13),
                    (13, 15),
                    (12, 14),
                    (14, 16),
                    (11, 23),
                    (12, 24),
                    (23, 24),
                    (23, 25),
                    (24, 26),
                    (25, 27),
                    (26, 28)
                ]

                for start, end in connections:

                    x1 = int(pose_landmarks[start].x * frame.shape[1])
                    y1 = int(pose_landmarks[start].y * frame.shape[0])

                    x2 = int(pose_landmarks[end].x * frame.shape[1])
                    y2 = int(pose_landmarks[end].y * frame.shape[0])

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        cv2.imshow("PoseAI - Pose Detection", frame)

        frame_timestamp += 33

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()