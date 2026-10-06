import os
import json
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
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=RunningMode.IMAGE,
    num_poses=2,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Paths
# -----------------------------

dataset_path = "../dataset"
output_path = "../pose_data"

os.makedirs(output_path, exist_ok=True)

# -----------------------------
# Only process these folders
# -----------------------------

allowed_folders = [
    "standing\\single",
    "standing\\two_people",
    "sitting\\single",
    "sitting\\two_people"
]

# -----------------------------
# Process dataset
# -----------------------------

with PoseLandmarker.create_from_options(options) as landmarker:

    for root, folders, files in os.walk(dataset_path):

        relative_folder = os.path.relpath(
            root,
            dataset_path
        )

        # Ignore group folders
        if relative_folder not in allowed_folders:
            continue

        for file in files:

            # Supported image formats
            if not file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".avif")
            ):
                continue

            image_path = os.path.join(
                root,
                file
            )

            print("Processing:", image_path)

            # -----------------------------
            # Read image
            # -----------------------------

            frame = cv2.imread(image_path)

            if frame is None:
                print("Could not read:", image_path)
                continue

            # -----------------------------
            # Convert BGR → RGB
            # -----------------------------

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            # -----------------------------
            # Detect poses
            # -----------------------------

            result = landmarker.detect(
                mp_image
            )

            # -----------------------------
            # Create output folder
            # -----------------------------

            save_folder = os.path.join(
                output_path,
                relative_folder
            )

            os.makedirs(
                save_folder,
                exist_ok=True
            )

            # -----------------------------
            # JSON filename
            # -----------------------------

            json_name = (
                os.path.splitext(file)[0]
                + ".json"
            )

            json_path = os.path.join(
                save_folder,
                json_name
            )

            # -----------------------------
            # Prepare pose data
            # -----------------------------

            pose_data = {
                "image": file,
                "category": relative_folder,
                "people_count": len(
                    result.pose_landmarks
                ),
                "landmarks": []
            }

            # -----------------------------
            # Save each person's landmarks
            # -----------------------------

            for person_id, landmarks in enumerate(
                result.pose_landmarks
            ):

                person_data = {
                    "person_id": person_id,
                    "landmarks": []
                }

                for landmark_id, landmark in enumerate(
                    landmarks
                ):

                    person_data["landmarks"].append({
                        "id": landmark_id,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    })

                pose_data["landmarks"].append(
                    person_data
                )

            # -----------------------------
            # Save JSON
            # -----------------------------

            with open(
                json_path,
                "w"
            ) as json_file:

                json.dump(
                    pose_data,
                    json_file,
                    indent=4
                )

print("\nDataset processing completed!")