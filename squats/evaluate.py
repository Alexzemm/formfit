import cv2
import torch
import numpy as np
from trainer import PoseTransformer
import mediapipe as mp
import os
from threading import Thread
import time  # Add time module for delay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model
# -----------------------------
num_classes = 2
model = PoseTransformer(num_classes=num_classes)
model.load_state_dict(torch.load("squat_transformer.pth", map_location=device))
model.to(device)
model.eval()

classes = ['correct', 'knees_in']  # Ensure matches training

# -----------------------------
# Feedback messages for each class
# -----------------------------
feedback_messages = {
    'correct': "Perfect Form! Keep it up!",
    'knees_in': "Keep your knees aligned with your toes"
}

# Function to speak feedback using Windows PowerShell
def speak_feedback(message):
    # Only speak for incorrect forms
    if message != feedback_messages['correct']:
        command = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{message}\')"'
        # Run as a separate thread so it doesn't block the video feed
        Thread(target=lambda: os.system(command)).start()

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# -----------------------------
# Function to extract keypoints
# -----------------------------
def extract_keypoints(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.array(keypoints, dtype=np.float32)
    else:
        return np.zeros(33*3, dtype=np.float32)

# -----------------------------
# Real-time webcam loop
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
seq_len = 30
sequence = []

# Variables to control feedback frequency
last_feedback_time = 0
feedback_cooldown = 3  # seconds between spoken feedback
last_pred_class = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = extract_keypoints(frame)
    sequence.append(keypoints)
    

    if len(sequence) > seq_len:
        sequence.pop(0)  # Keep last 30 frames

    # Predict if we have enough frames
    if len(sequence) == seq_len:
        input_seq = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_seq)
            pred_class = classes[torch.argmax(output, 1).item()]

        class_colors = {
            'correct': (0, 255, 0),  # Green
            'knees_in': (0, 0, 255),     # Red
        }
        color = class_colors.get(pred_class, (0, 0, 255))

        # Display feedback
        cv2.putText(frame, feedback_messages[pred_class], (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Speak feedback for incorrect forms with cooldown
        current_time = time.time()
        if pred_class != 'correct' and (current_time - last_feedback_time > feedback_cooldown or pred_class != last_pred_class):
            speak_feedback(feedback_messages[pred_class])
            last_feedback_time = current_time
            last_pred_class = pred_class

    cv2.namedWindow("Squat Pose Classifier", cv2.WINDOW_NORMAL)

    cv2.setWindowProperty("Squat Pose Classifier", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Squat Pose Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
