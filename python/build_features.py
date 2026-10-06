import os
import json
import math
import csv


def calculate_angle(a, b, c):

    angle = math.degrees(
        math.atan2(c["y"] - b["y"], c["x"] - b["x"])
        -
        math.atan2(a["y"] - b["y"], a["x"] - b["x"])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return round(angle, 2)


def calculate_distance(a, b):

    return round(
        math.sqrt(
            (a["x"] - b["x"]) ** 2 +
            (a["y"] - b["y"]) ** 2
        ),
        4
    )


def extract_features(landmarks):

    features = {

        # Arm angles
        "left_elbow_angle": calculate_angle(
            landmarks[11], landmarks[13], landmarks[15]
        ),

        "right_elbow_angle": calculate_angle(
            landmarks[12], landmarks[14], landmarks[16]
        ),

        # Leg angles
        "left_knee_angle": calculate_angle(
            landmarks[23], landmarks[25], landmarks[27]
        ),

        "right_knee_angle": calculate_angle(
            landmarks[24], landmarks[26], landmarks[28]
        ),

        # Shoulder angle
        "shoulder_angle": calculate_angle(
            landmarks[11],
            landmarks[12],
            landmarks[24]
        ),

        # Hip angle
        "hip_angle": calculate_angle(
            landmarks[23],
            landmarks[24],
            landmarks[12]
        ),

        # Wrist positions relative to shoulders
        "left_wrist_shoulder_distance": calculate_distance(
            landmarks[15],
            landmarks[11]
        ),

        "right_wrist_shoulder_distance": calculate_distance(
            landmarks[16],
            landmarks[12]
        ),

        # Ankle positions relative to hips
        "left_ankle_hip_distance": calculate_distance(
            landmarks[27],
            landmarks[23]
        ),

        "right_ankle_hip_distance": calculate_distance(
            landmarks[28],
            landmarks[24]
        )
    }

    return features


pose_data_path = "../pose_data"
output_file = "../pose_features.csv"

rows = []


for root, folders, files in os.walk(pose_data_path):

    for file in files:

        if not file.endswith(".json"):
            continue

        json_path = os.path.join(root, file)

        with open(json_path, "r") as f:
            data = json.load(f)

        if data["people_count"] == 0:
            continue

        for person in data["landmarks"]:

            features = extract_features(
                person["landmarks"]
            )

            row = {
                "image": data["image"],
                "category": data["category"],
                "people_count": data["people_count"],
                "person_id": person["person_id"]
            }

            row.update(features)

            rows.append(row)


fieldnames = [
    "image",
    "category",
    "people_count",
    "person_id",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
    "shoulder_angle",
    "hip_angle",
    "left_wrist_shoulder_distance",
    "right_wrist_shoulder_distance",
    "left_ankle_hip_distance",
    "right_ankle_hip_distance"
]


with open(
    output_file,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


print("Feature extraction completed!")
print("Total feature rows:", len(rows))
print("Features per person:", len(fieldnames) - 4)
print("Saved to:", output_file)