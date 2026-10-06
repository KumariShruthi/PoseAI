import cv2
import mediapipe as mp

# -----------------------------
# MediaPipe setup
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

model_path = "pose_landmarker_full.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Open camera
# -----------------------------

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:

    frame_timestamp = 0

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Camera not found")
            break

        # OpenCV: BGR
        # MediaPipe: RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert frame to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Detect pose
        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        # -----------------------------
        # Process landmarks
        # -----------------------------

        if result.pose_landmarks:

            for pose_landmarks in result.pose_landmarks:

                print("\n--- Pose Detected ---")

                # Print 33 landmark coordinates
                for i, landmark in enumerate(pose_landmarks):

                    print(
                        i,
                        "x:", round(landmark.x, 3),
                        "y:", round(landmark.y, 3),
                        "z:", round(landmark.z, 3)
                    )

                # -----------------------------
                # Draw landmark points
                # -----------------------------

                for landmark in pose_landmarks:

                    x = int(
                        landmark.x * frame.shape[1]
                    )

                    y = int(
                        landmark.y * frame.shape[0]
                    )

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # -----------------------------
                # Draw skeleton connections
                # -----------------------------

                connections = [
                    (11, 12),  # shoulders

                    (11, 13),  # left upper arm
                    (13, 15),  # left forearm

                    (12, 14),  # right upper arm
                    (14, 16),  # right forearm

                    (11, 23),  # left torso
                    (12, 24),  # right torso

                    (23, 24),  # hips

                    (23, 25),  # left thigh
                    (25, 27),  # left lower leg

                    (24, 26),  # right thigh
                    (26, 28)   # right lower leg
                ]

                for start, end in connections:

                    x1 = int(
                        pose_landmarks[start].x
                        * frame.shape[1]
                    )

                    y1 = int(
                        pose_landmarks[start].y
                        * frame.shape[0]
                    )

                    x2 = int(
                        pose_landmarks[end].x
                        * frame.shape[1]
                    )

                    y2 = int(
                        pose_landmarks[end].y
                        * frame.shape[0]
                    )

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        # -----------------------------
        # Show camera
        # -----------------------------

        cv2.imshow(
            "PoseAI - Pose Detection",
            frame
        )

        # Increase timestamp
        frame_timestamp += 33

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# -----------------------------
# Release resources
# -----------------------------

cap.release()
cv2.destroyAllWindows()