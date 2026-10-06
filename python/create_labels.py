import pandas as pd

input_file = "../pose_features.csv"
output_file = "../pose_labels.csv"

df = pd.read_csv(input_file)

labels = {

    # ---------- SITTING ----------
    "sitting\\single\\10,.jpg": "sitting_relaxed",
    "sitting\\single\\10.jpg": "hand_on_head_face",
    "sitting\\single\\3.jpg": "sitting_relaxed",
    "sitting\\single\\4.jpg": "hand_on_head_face",
    "sitting\\single\\5,.jpg": "hand_on_head_face",
    "sitting\\single\\5.jpg": "sitting_relaxed",
    "sitting\\single\\6,.jpg": "sitting_relaxed",
    "sitting\\single\\6.jpg": "hand_on_head_face",
    "sitting\\single\\8,.jpg": "hand_on_head_face",
    "sitting\\single\\8.jpg": "hand_on_head_face",
    "sitting\\single\\9,.jpg": "sitting_relaxed",
    "sitting\\single\\9.jpg": "sitting_relaxed",
    "sitting\\single\\i2.jpg": "sitting_relaxed",
    "sitting\\single\\i3.jpg": "sitting_relaxed",
    "sitting\\single\\img1.jpg": "hand_on_head_face",

    # ---------- STANDING ----------
    "standing\\single\\images (1).jpg": "dynamic_pose",
    "standing\\single\\images.jpg": "hands_down",
    "standing\\single\\img 11.jpg": "hands_down",
    "standing\\single\\img 5.jpg": "hands_down",
    "standing\\single\\img.jpg": "hand_on_head_face",
    "standing\\single\\img1.jpg": "hands_down",
    "standing\\single\\img10.jpg": "hands_down",
    "standing\\single\\img11.jpg": "hands_down",
    "standing\\single\\img12.jpg": "dynamic_pose",
    "standing\\single\\img13.jpg": "hand_on_head_face",
    "standing\\single\\img14.jpg": "hand_on_head_face",
    "standing\\single\\img15.jpg": "hands_down",
    "standing\\single\\img16.jpg": "dynamic_pose",
    "standing\\single\\img17.jpg": "hand_on_head_face",
    "standing\\single\\img18.jpg": "dynamic_pose",
    "standing\\single\\img19.jpg": "hands_down",
    "standing\\single\\img2.jpg": "hands_down",
    "standing\\single\\img20.jpg": "hand_on_head_face",
    "standing\\single\\img21.jpg": "dynamic_pose",
    "standing\\single\\img3.jpg": "hand_on_head_face",
    "standing\\single\\img5.jpg": "hand_on_head_face",
    "standing\\single\\img8.jpg": "hands_down",
    "standing\\single\\img9.jpg": "hands_down",

    "standing\\single\\WhatsApp Image 2026-10-05 at 7.24.30 PM (1).jpeg": "hand_on_head_face",
    "standing\\single\\WhatsApp Image 2026-10-05 at 7.24.31 PM.jpeg": "hands_down",
    "standing\\single\\WhatsApp Image 2026-10-05 at 7.28.11 PM.jpeg": "hands_down",
    "standing\\single\\WhatsApp Image 2026-10-05 at 7.28.19 PM.jpeg": "hand_on_head_face",
    "standing\\single\\WhatsApp Image 2026-10-05 at 7.28.20 PM.jpeg": "hands_down",
    "standing\\single\\WhatsApp Image 2026-10-05 at 7.28.21 PM.jpeg": "dynamic_pose",
}

df["key"] = df["category"] + "\\" + df["image"]

df["pose_label"] = df["key"].map(labels)

df = df.dropna(subset=["pose_label"])

df[["image", "category", "pose_label"]].to_csv(
    output_file,
    index=False
)

print("\nLabels created successfully!")
print("Total labeled images:", len(df))
print("\nClass distribution:")
print(df["pose_label"].value_counts())

print("\nSaved to:", output_file)