import cv2
import torch
import torch.nn.functional as F
import numpy as np
from trainer import PoseTransformer
import mediapipe as mp
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load model
# -----------------------------
num_classes = 3
model = PoseTransformer(num_classes=num_classes)
model.load_state_dict(torch.load("lunge_transformer.pth", map_location=device))
model.to(device)
model.eval()

classes = ['back_straight', 'correct', 'legs_far']  # Ensure matches training

# -----------------------------
# Feedback messages for each class
# -----------------------------
feedback_messages = {
    'back_straight': "Keep your back straight",
    'correct': "Perfect Form! Keep it up!",
    'legs_far': "Keep your legs farther"
}

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

def normalize_pose(keypoints):
    """
    Normalize pose to be invariant to camera position/distance.
    Centers on hip midpoint and scales by torso length.
    """
    if np.all(keypoints == 0):
        return keypoints
    
    # Reshape to (33, 3)
    kp = keypoints.reshape(33, 3)
    
    # MediaPipe landmark indices
    # 23: Left Hip, 24: Right Hip, 11: Left Shoulder, 12: Right Shoulder
    left_hip = kp[23]
    right_hip = kp[24]
    left_shoulder = kp[11]
    right_shoulder = kp[12]
    
    # Calculate hip center as reference point
    hip_center = (left_hip + right_hip) / 2
    
    # Calculate torso length for scaling
    shoulder_center = (left_shoulder + right_shoulder) / 2
    torso_length = np.linalg.norm(shoulder_center - hip_center)
    
    # Avoid division by zero
    if torso_length < 0.01:
        torso_length = 1.0
    
    # Center and scale
    kp_normalized = (kp - hip_center) / torso_length
    
    return kp_normalized.flatten()

# -----------------------------
# Accuracy improvement parameters
# -----------------------------
CONFIDENCE_THRESHOLD = 0.60  # Lowered to 50% for debugging (was 0.65)
SMOOTHING_WINDOW = 5  # Reduced to 5 for faster response (was 7)
SHOW_PROBABILITIES = False  # Display confidence percentages on screen
DEBUG_MODE = False  # Show additional debug info

# -----------------------------
# Real-time webcam loop
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
seq_len = 30
sequence = []
prediction_history = deque(maxlen=SMOOTHING_WINDOW)  # For temporal smoothing
last_confident_prediction = None  # Fallback when confidence is low

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Fix mirrored camera: flip frame horizontally so processing and display
    # use the corrected (non-mirrored) view.
    frame = cv2.flip(frame, 1)

    keypoints = extract_keypoints(frame)
    keypoints_normalized = normalize_pose(keypoints)  # Normalize for camera invariance
    sequence.append(keypoints_normalized)
    

    if len(sequence) > seq_len:
        sequence.pop(0)  # Keep last 30 frames

    # Predict if we have enough frames
    if len(sequence) == seq_len:
        input_seq = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_seq)
            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probabilities)
            pred_class = classes[pred_idx]
            confidence = probabilities[pred_idx]
            
            # Add to prediction history for temporal smoothing
            prediction_history.append(pred_class)
            
            # Use majority vote from prediction history
            if len(prediction_history) == SMOOTHING_WINDOW:
                # Count votes for each class
                votes = {}
                for pred in prediction_history:
                    votes[pred] = votes.get(pred, 0) + 1
                smoothed_class = max(votes, key=votes.get)
                smoothed_confidence = votes[smoothed_class] / SMOOTHING_WINDOW
            else:
                smoothed_class = pred_class
                smoothed_confidence = confidence
            
            # Only update prediction if confidence is high enough
            if confidence >= CONFIDENCE_THRESHOLD:
                final_class = smoothed_class
                last_confident_prediction = smoothed_class
            elif last_confident_prediction is not None:
                # Use last confident prediction when uncertain
                final_class = last_confident_prediction
            else:
                final_class = 'correct'  # Default assumption
            
            # Set display colors
            class_colors = {
                'back_straight': (0, 0, 255),     # Red
                'correct': (0, 255, 0),           # Green
                'legs_far': (255, 0, 0)           # Blue
            }
            color = class_colors.get(final_class, (0, 255, 0))
            
            # Display main feedback message (simple format like russian twists)
            feedback_text = f"Prediction: {final_class}"
            
            cv2.putText(frame, feedback_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.namedWindow("Lunges Pose Classifier", cv2.WINDOW_NORMAL)

    cv2.setWindowProperty("Lunges Pose Classifier", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Lunges Pose Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()