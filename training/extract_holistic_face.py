"""
Extends Holistic extraction to add a curated subset of face landmarks
relevant to ASL grammar (not the full 468-point mesh, which would be
mostly noise for a data-starved model):

  - Eyebrows (raised/furrowed = question markers, intensity)
  - Eyes (openness, a component of several grammatical markers)
  - Mouth (shape/openness = mouth morphemes, negation, adjectives)
  - Head-pose reference points (nose/chin/forehead = head tilt, a real
    ASL grammatical signal for topic marking and negation)

Re-extracts from the same already-downloaded clips (clips_v2), so this
is additive to the hand+pose extraction we already did, not a redo of
the network fetch.
"""

import os
import json
import cv2
import mediapipe as mp

CLIPS_DIR = "/home/claude/wlasl_data_v2/clips_v2"
OUT_DIR = "/home/claude/wlasl_data_v2/landmarks_face"

mp_holistic = mp.solutions.holistic

POSE_INDICES = {
    "nose": 0,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_hip": 23, "right_hip": 24,
}

# Curated face subset — MediaPipe Face Mesh indices
FACE_INDICES = {
    "left_eyebrow_inner": 105, "left_eyebrow_outer": 70,
    "right_eyebrow_inner": 334, "right_eyebrow_outer": 300,
    "left_eye_top": 159, "left_eye_bottom": 145,
    "right_eye_top": 386, "right_eye_bottom": 374,
    "mouth_left": 61, "mouth_right": 291,
    "mouth_top": 13, "mouth_bottom": 14,
    "nose_tip": 1, "chin": 152, "forehead": 10,
}


def extract_pose_subset(pose_landmarks):
    if pose_landmarks is None:
        return None
    return {name: {"x": pose_landmarks.landmark[idx].x,
                    "y": pose_landmarks.landmark[idx].y,
                    "z": pose_landmarks.landmark[idx].z}
            for name, idx in POSE_INDICES.items()}


def extract_face_subset(face_landmarks):
    if face_landmarks is None:
        return None
    return {name: {"x": face_landmarks.landmark[idx].x,
                    "y": face_landmarks.landmark[idx].y,
                    "z": face_landmarks.landmark[idx].z}
            for name, idx in FACE_INDICES.items()}


def extract_hand(hand_landmarks):
    if hand_landmarks is None:
        return None
    return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]


def extract_from_video(path, holistic):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        frames.append({
            "left_hand": extract_hand(results.left_hand_landmarks),
            "right_hand": extract_hand(results.right_hand_landmarks),
            "pose": extract_pose_subset(results.pose_landmarks),
            "face": extract_face_subset(results.face_landmarks),
        })
    cap.release()
    return frames


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    words = sorted(os.listdir(CLIPS_DIR))

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:

        for word in words:
            word_dir = os.path.join(CLIPS_DIR, word)
            if not os.path.isdir(word_dir):
                continue

            out_path = os.path.join(OUT_DIR, f"{word}.json")
            if os.path.exists(out_path):
                print(f"Skipping {word} (already extracted)")
                continue

            clip_files = sorted(f for f in os.listdir(word_dir) if f.endswith(".mp4"))
            word_sequences = []

            for clip_file in clip_files:
                clip_path = os.path.join(word_dir, clip_file)
                print(f"Processing {word}/{clip_file}...")
                frames = extract_from_video(clip_path, holistic)

                any_hand = sum(1 for f in frames if f["left_hand"] or f["right_hand"])
                any_face = sum(1 for f in frames if f["face"])
                total = len(frames)

                word_sequences.append({
                    "clip_file": clip_file,
                    "total_frames": total,
                    "hand_detection_rate": round(any_hand / total, 3) if total else 0,
                    "face_detection_rate": round(any_face / total, 3) if total else 0,
                    "frames": frames,
                })
                print(f"  {total} frames, hand {any_hand}/{total}, face {any_face}/{total}")

            with open(out_path, "w") as f:
                json.dump({"gloss": word, "sequences": word_sequences}, f)

    print("\nDone.")


if __name__ == "__main__":
    main()
