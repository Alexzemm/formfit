import os
import cv2
import numpy as np
import mediapipe as mp

# Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)

def extract_keypoints_from_video(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    frame_idx = 0
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # Convert to RGB for mediapipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                keypoints = []
                for lm in results.pose_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
                frames.append(keypoints)
            else:
                # If no landmarks detected, append zeros
                frames.append([0.0] * (33 * 3))

        frame_idx += 1

    cap.release()

    # Pad if less than num_frames
    while len(frames) < num_frames:
        frames.append([0.0] * (33 * 3))

    return np.array(frames, dtype=np.float32)

def process_dataset(input_dir, output_dir, num_frames=30):
    os.makedirs(output_dir, exist_ok=True)
    classes = os.listdir(input_dir)

    for cls in classes:
        cls_input = os.path.join(input_dir, cls)
        cls_output = os.path.join(output_dir, cls)
        os.makedirs(cls_output, exist_ok=True)

        for video_file in os.listdir(cls_input):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue
            video_path = os.path.join(cls_input, video_file)
            keypoints = extract_keypoints_from_video(video_path, num_frames=num_frames)

            output_path = os.path.join(cls_output, video_file.replace('.mp4', '.npy'))
            np.save(output_path, keypoints)
            print(f"Processed: {video_path} -> {output_path}")

if __name__ == "__main__":
    input_dir = "../dataset/lunges"       # Raw videos
    output_dir = "/lunges/keypoints"       # Save keypoints here
    process_dataset(input_dir, output_dir, num_frames=30)
