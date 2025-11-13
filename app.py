import customtkinter as ctk
import cv2
import torch
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import threading
import time
from collections import Counter
import sys
import os
from threading import Thread
import subprocess

# Add the folders to Python path
sys.path.append(os.path.join(os.getcwd(), 'squats'))
sys.path.append(os.path.join(os.getcwd(), 'pushups'))
sys.path.append(os.path.join(os.getcwd(), 'plank'))
sys.path.append(os.path.join(os.getcwd(), 'russian_twists'))
sys.path.append(os.path.join(os.getcwd(), 'lunges'))

# Import the trainer module from all folders
try:
    from squats.trainer import PoseTransformer as SquatTransformer
    from pushups.trainer import PoseTransformer as PushupTransformer
    from plank.trainer import PoseTransformer as PlankTransformer
    from russian_twists.trainer import PoseTransformer as RussianTransformer
    from lunges.trainer import PoseTransformer as LungeTransformer
except ImportError:
    # Fallback if trainer is in the same directory structure
    from trainer import PoseTransformer as SquatTransformer
    from trainer import PoseTransformer as PushupTransformer
    from trainer import PoseTransformer as PlankTransformer
    from trainer import PoseTransformer as RussianTransformer
    from trainer import PoseTransformer as LungeTransformer

# Configure customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class FitnessTrainerApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("AI Fitness Trainer")
        self.root.geometry("1200x800")
        
        # Exercise configurations
        self.exercises = {
            "pushups": {
                "name": "Push-ups",
                "model_path": "pushups/pushup_transformer.pth",
                "classes": ['correct', 'pike', 'snake'],
                "colors": {
                    'correct': (0, 255, 0),
                    'pike': (0, 0, 255),
                    'snake': (255, 0, 0)
                },
                "feedback_messages": {
                    'correct': "Perfect Form! Keep it up!",
                    'pike': "Bring your hips down",
                    'snake': "Bring your hips up, back straight"
                }
            },
            "squats": {
                "name": "Squats",
                "model_path": "squats/squat_transformer.pth", 
                "classes": ['correct', 'knees_in'],
                "colors": {
                    'correct': (0, 255, 0),
                    'knees_in': (0, 0, 255)
                },
                "feedback_messages": {
                    'correct': "Perfect Form! Keep it up!",
                    'knees_in': "Keep your knees aligned with your toes"
                }
            },
            "plank": {
                "name": "Plank",
                "model_path": "plank/plank_transformer.pth", 
                "classes": ['correct', 'hips_down', 'hips_up'],
                "colors": {
                    'correct': (0, 255, 0),
                    'hips_down': (0, 0, 255),
                    'hips_up': (255, 0, 0)
                },
                "feedback_messages": {
                    'correct': "Perfect Form! Keep it up!",
                    'hips_down': "Bring your hips up",
                    'hips_up': "Bring your hips down, back straight"
                }
            },
            "russian_twists": {
                "name": "Russian Twists",
                "model_path": "russian_twists/russian_transformer.pth", 
                "classes": ['correct', 'legs_bent'],
                "colors": {
                    'correct': (0, 255, 0),
                    'legs_bent': (0, 0, 255)
                },
                "feedback_messages": {
                    'correct': "Perfect Form! Keep it up!",
                    'legs_bent': "Straighten your legs more"
                }
            },
            "lunges": {
                "name": "Lunges",
                "model_path": "lunges/lunge_transformer.pth", 
                "classes": ['back_straight', 'correct', 'legs_far'],
                "colors": {
                    'back_straight': (0, 0, 255),
                    'correct': (0, 255, 0),
                    'legs_far': (255, 0, 0)
                },
                "feedback_messages": {
                    'back_straight': "Keep your back straight",
                    'correct': "Perfect Form! Keep it up!",
                    'legs_far': "Keep your legs farther"
                }
            }
        }
        
        # Initialize variables
        self.selected_exercise = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pose = None
        self.cap = None
        self.is_evaluating = False
        self.sequence = []
        self.seq_len = 30
        self.predictions = []
        self.start_time = None
        self.frame_count = 0
        
        # Voice feedback variables
        self.last_feedback_time = 0
        self.feedback_cooldown = 3  # seconds between spoken feedback
        self.last_pred_class = None
        
        # Lunges-specific variables for smoothing and normalization
        self.prediction_history = None
        self.last_confident_prediction = None
        self.CONFIDENCE_THRESHOLD = 0.60
        self.SMOOTHING_WINDOW = 5
        
        # Setup MediaPipe
        self.mp_pose = mp.solutions.pose
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame, 
            text="AI Fitness Trainer", 
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title_label.pack(pady=20)
        
        # Exercise selection frame
        selection_frame = ctk.CTkFrame(main_frame)
        selection_frame.pack(pady=20, padx=50, fill="x")
        
        selection_label = ctk.CTkLabel(
            selection_frame,
            text="Select Exercise:",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        selection_label.pack(pady=10)
        
        # Exercise buttons
        button_frame = ctk.CTkFrame(selection_frame)
        button_frame.pack(pady=10)
        
        self.pushup_btn = ctk.CTkButton(
            button_frame,
            text="Push-ups",
            width=120,
            height=60,
            font=ctk.CTkFont(size=16),
            command=lambda: self.select_exercise("pushups")
        )
        self.pushup_btn.pack(side="left", padx=10)
        
        self.squat_btn = ctk.CTkButton(
            button_frame,
            text="Squats", 
            width=120,
            height=60,
            font=ctk.CTkFont(size=16),
            command=lambda: self.select_exercise("squats")
        )
        self.squat_btn.pack(side="left", padx=10)
        
        self.plank_btn = ctk.CTkButton(
            button_frame,
            text="Plank", 
            width=120,
            height=60,
            font=ctk.CTkFont(size=16),
            command=lambda: self.select_exercise("plank")
        )
        self.plank_btn.pack(side="left", padx=10)
        
        self.russian_btn = ctk.CTkButton(
            button_frame,
            text="Russian Twists", 
            width=120,
            height=60,
            font=ctk.CTkFont(size=16),
            command=lambda: self.select_exercise("russian_twists")
        )
        self.russian_btn.pack(side="left", padx=10)
        
        self.lunge_btn = ctk.CTkButton(
            button_frame,
            text="Lunges", 
            width=120,
            height=60,
            font=ctk.CTkFont(size=16),
            command=lambda: self.select_exercise("lunges")
        )
        self.lunge_btn.pack(side="left", padx=10)
        
        # Selected exercise display
        self.selected_label = ctk.CTkLabel(
            selection_frame,
            text="No exercise selected",
            font=ctk.CTkFont(size=14)
        )
        self.selected_label.pack(pady=10)
        
        # Control buttons frame
        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(pady=20, fill="x", padx=50)
        
        self.start_btn = ctk.CTkButton(
            control_frame,
            text="Start Evaluation",
            width=200,
            height=50,
            font=ctk.CTkFont(size=16),
            command=self.start_evaluation,
            state="disabled"
        )
        self.start_btn.pack(side="left", padx=20)
        
        self.stop_btn = ctk.CTkButton(
            control_frame,
            text="Stop Evaluation",
            width=200,
            height=50,
            font=ctk.CTkFont(size=16),
            command=self.stop_evaluation,
            state="disabled"
        )
        self.stop_btn.pack(side="right", padx=20)
        
        # Status frame
        self.status_frame = ctk.CTkFrame(main_frame)
        self.status_frame.pack(pady=10, fill="x", padx=50)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Status: Ready",
            font=ctk.CTkFont(size=14)
        )
        self.status_label.pack(pady=10)
        
        # Workout Generation Button - New addition
        workout_frame = ctk.CTkFrame(main_frame)
        workout_frame.pack(pady=10, fill="x", padx=50)
        
        workout_label = ctk.CTkLabel(
            workout_frame,
            text="Need a personalized workout plan?",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        workout_label.pack(pady=5)
        
        self.workout_btn = ctk.CTkButton(
            workout_frame,
            text="Generate Workout & Diet Plan",
            width=250,
            height=40,
            font=ctk.CTkFont(size=16),
            fg_color="#4CAF50",  # Green color
            hover_color="#45a049",  # Darker green for hover
            command=self.launch_workout_generator
        )
        self.workout_btn.pack(pady=10)
        
        # Benchmarks frame
        self.benchmark_frame = ctk.CTkFrame(main_frame)
        self.benchmark_frame.pack(pady=10, fill="both", expand=True, padx=50)
        
        benchmark_title = ctk.CTkLabel(
            self.benchmark_frame,
            text="Session Benchmarks",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        benchmark_title.pack(pady=10)
        
        self.benchmark_text = ctk.CTkTextbox(
            self.benchmark_frame,
            height=200,
            font=ctk.CTkFont(size=12)
        )
        self.benchmark_text.pack(fill="both", expand=True, padx=20, pady=10)
    
    def launch_workout_generator(self):
        """Launch the workout generation web app in a separate process"""
        try:
            # Get the path to the workout generator app
            workout_app_path = os.path.join(os.getcwd(), 'workout_generation', 'app.py')
            
            # Check if file exists
            if not os.path.exists(workout_app_path):
                self.status_label.configure(text="Error: Workout generation app not found.")
                return
            
            # Create a message to show the user
            self.status_label.configure(text="Launching Workout Generator App...")
            
            # Launch the Flask app in a separate process
            workout_thread = Thread(
                target=lambda: subprocess.Popen([
                    sys.executable, 
                    workout_app_path
                ], cwd=os.path.dirname(workout_app_path))
            )
            workout_thread.daemon = True
            workout_thread.start()
            
            # Update status after a short delay
            self.root.after(2000, lambda: self.status_label.configure(
                text="Workout Generator launched! Check your browser."
            ))
            
            # Open the browser after a short delay
            self.root.after(3000, lambda: os.system("start http://127.0.0.1:5000"))
            
        except Exception as e:
            self.status_label.configure(text=f"Error launching workout generator: {str(e)}")
        
    def select_exercise(self, exercise_key):
        self.selected_exercise = exercise_key
        exercise_name = self.exercises[exercise_key]["name"]
        self.selected_label.configure(text=f"Selected: {exercise_name}")
        self.start_btn.configure(state="normal")
        
        # Update button colors to show selection
        default_color = ["#3B8ED0", "#1F6AA5"]
        self.pushup_btn.configure(fg_color=default_color)
        self.squat_btn.configure(fg_color=default_color)
        self.plank_btn.configure(fg_color=default_color)
        self.russian_btn.configure(fg_color=default_color)
        self.lunge_btn.configure(fg_color=default_color)
        
        # Highlight the selected exercise button
        if exercise_key == "pushups":
            self.pushup_btn.configure(fg_color="green")
        elif exercise_key == "squats":
            self.squat_btn.configure(fg_color="green")
        elif exercise_key == "plank":
            self.plank_btn.configure(fg_color="green")
        elif exercise_key == "russian_twists":
            self.russian_btn.configure(fg_color="green")
        elif exercise_key == "lunges":
            self.lunge_btn.configure(fg_color="green")
    
    # Function to speak feedback using Windows PowerShell
    def speak_feedback(self, message, feedback_messages):
        # Only speak for incorrect forms
        if message != feedback_messages['correct']:
            command = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{message}\')"'
            # Run as a separate thread so it doesn't block the video feed
            Thread(target=lambda: os.system(command)).start()
    
    def load_model(self):
        try:
            exercise_config = self.exercises[self.selected_exercise]
            num_classes = len(exercise_config["classes"])
            
            # Use appropriate transformer class based on exercise
            if self.selected_exercise == "pushups":
                self.model = PushupTransformer(num_classes=num_classes)
            elif self.selected_exercise == "squats":
                self.model = SquatTransformer(num_classes=num_classes)
            elif self.selected_exercise == "plank":
                self.model = PlankTransformer(num_classes=num_classes)
            elif self.selected_exercise == "russian_twists":
                self.model = RussianTransformer(num_classes=num_classes)
            elif self.selected_exercise == "lunges":
                self.model = LungeTransformer(num_classes=num_classes)
            
            self.model.load_state_dict(torch.load(
                exercise_config["model_path"], 
                map_location=self.device
            ))
            self.model.to(self.device)
            self.model.eval()
            
            return True
            
        except Exception as e:
            self.status_label.configure(text=f"Error loading model: {str(e)}")
            return False
    
    def extract_keypoints(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        if results.pose_landmarks:
            keypoints = []
            for lm in results.pose_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            return np.array(keypoints, dtype=np.float32)
        else:
            return np.zeros(33*3, dtype=np.float32)
    
    def normalize_pose(self, keypoints):
        """
        Normalize pose to be invariant to camera position/distance.
        Centers on hip midpoint and scales by torso length.
        Used only for lunges exercise.
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
    
    def start_evaluation(self):
        if not self.selected_exercise:
            return
        
        # Load model
        if not self.load_model():
            return
        
        # Initialize MediaPipe pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
        
        # Reset tracking variables
        self.sequence = []
        self.predictions = []
        self.start_time = time.time()
        self.frame_count = 0
        self.is_evaluating = True
        
        # Reset voice feedback variables
        self.last_feedback_time = 0
        self.last_pred_class = None
        
        # Reset lunges-specific variables
        if self.selected_exercise == "lunges":
            from collections import deque
            self.prediction_history = deque(maxlen=self.SMOOTHING_WINDOW)
            self.last_confident_prediction = None
        else:
            self.prediction_history = None
            self.last_confident_prediction = None
        
        # Update UI
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Status: Evaluating...")
        self.benchmark_text.delete("1.0", "end")
        
        # Start evaluation thread
        self.eval_thread = threading.Thread(target=self.evaluation_loop)
        self.eval_thread.daemon = True
        self.eval_thread.start()
    
    def evaluation_loop(self):
        exercise_config = self.exercises[self.selected_exercise]
        classes = exercise_config["classes"]
        colors = exercise_config["colors"]
        feedback_messages = exercise_config["feedback_messages"]
        
        while self.is_evaluating:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally to fix mirror effect
            frame = cv2.flip(frame, 1)
            
            self.frame_count += 1
            keypoints = self.extract_keypoints(frame)
            
            # Apply normalization only for lunges
            if self.selected_exercise == "lunges":
                keypoints = self.normalize_pose(keypoints)
            
            self.sequence.append(keypoints)
            
            if len(self.sequence) > self.seq_len:
                self.sequence.pop(0)
            
            # Predict if we have enough frames
            if len(self.sequence) == self.seq_len:
                input_seq = torch.tensor(
                    np.array(self.sequence), 
                    dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(input_seq)
                    
                    # Special handling for lunges with confidence-based smoothing
                    if self.selected_exercise == "lunges":
                        import torch.nn.functional as F
                        probabilities = F.softmax(output, dim=1).cpu().numpy()[0]
                        pred_idx = np.argmax(probabilities)
                        pred_class = classes[pred_idx]
                        confidence = probabilities[pred_idx]
                        
                        # Add to prediction history for temporal smoothing
                        self.prediction_history.append(pred_class)
                        
                        # Use majority vote from prediction history
                        if len(self.prediction_history) == self.SMOOTHING_WINDOW:
                            votes = {}
                            for pred in self.prediction_history:
                                votes[pred] = votes.get(pred, 0) + 1
                            smoothed_class = max(votes, key=votes.get)
                        else:
                            smoothed_class = pred_class
                        
                        # Only update prediction if confidence is high enough
                        if confidence >= self.CONFIDENCE_THRESHOLD:
                            pred_class = smoothed_class
                            self.last_confident_prediction = smoothed_class
                        elif self.last_confident_prediction is not None:
                            pred_class = self.last_confident_prediction
                        else:
                            pred_class = 'correct'
                    else:
                        pred_class = classes[torch.argmax(output, 1).item()]
                
                self.predictions.append(pred_class)
                color = colors.get(pred_class, (0, 0, 255))
                
                # Get feedback message
                feedback = feedback_messages.get(pred_class, f"Prediction: {pred_class}")
                
                # Draw prediction and feedback on frame
                cv2.putText(
                    frame, 
                    feedback,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    color, 
                    2
                )
                
                # Speak feedback for incorrect forms with cooldown
                current_time = time.time()
                if pred_class != 'correct' and (current_time - self.last_feedback_time > self.feedback_cooldown or pred_class != self.last_pred_class):
                    self.speak_feedback(feedback_messages[pred_class], feedback_messages)
                    self.last_feedback_time = current_time
                    self.last_pred_class = pred_class
                
                # Draw frame count
                cv2.putText(
                    frame,
                    f"Frames: {self.frame_count}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            
            cv2.namedWindow(f"{exercise_config['name']} Pose Classifier", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                f"{exercise_config['name']} Pose Classifier", 
                cv2.WND_PROP_FULLSCREEN, 
                cv2.WINDOW_FULLSCREEN
            )
            
            cv2.imshow(f"{exercise_config['name']} Pose Classifier", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_evaluation()
                break
    
    def stop_evaluation(self):
        self.is_evaluating = False
        
        if self.cap:
            self.cap.release()
        if self.pose:
            self.pose.close()
        
        cv2.destroyAllWindows()
        
        # Update UI
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.status_label.configure(text="Status: Evaluation Complete")
        
        # Show benchmarks
        self.show_benchmarks()
    
    def show_benchmarks(self):
        if not self.predictions:
            self.benchmark_text.insert("end", "No predictions recorded.\n")
            return
        
        end_time = time.time()
        total_time = end_time - self.start_time
        
        # Calculate statistics
        prediction_counts = Counter(self.predictions)
        total_predictions = len(self.predictions)
        fps = self.frame_count / total_time if total_time > 0 else 0
        
        # Generate benchmark report
        report = f"SESSION BENCHMARK REPORT\n"
        report += f"=" * 50 + "\n\n"
        report += f"Exercise: {self.exercises[self.selected_exercise]['name']}\n"
        report += f"Total Duration: {total_time:.2f} seconds\n"
        report += f"Total Frames Processed: {self.frame_count}\n"
        report += f"Average FPS: {fps:.2f}\n"
        report += f"Total Predictions Made: {total_predictions}\n\n"
        
        report += "FORM ANALYSIS:\n"
        report += "-" * 30 + "\n"
        
        for class_name, count in prediction_counts.items():
            percentage = (count / total_predictions) * 100
            report += f"{class_name.title()}: {count} predictions ({percentage:.1f}%)\n"
        
        # Performance rating
        correct_percentage = (prediction_counts.get('correct', 0) / total_predictions) * 100
        report += f"\nOVERALL FORM SCORE: {correct_percentage:.1f}%\n"
        
        if correct_percentage >= 80:
            rating = "Excellent! 🏆"
        elif correct_percentage >= 60:
            rating = "Good! Keep improving! 👍"
        elif correct_percentage >= 40:
            rating = "Fair. Focus on form! ⚠️"
        else:
            rating = "Needs improvement. Practice more! 📈"
        
        report += f"Rating: {rating}\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS:\n"
        report += "-" * 30 + "\n"
        
        if self.selected_exercise == "pushups":
            if prediction_counts.get('pike', 0) > prediction_counts.get('correct', 0):
                report += "• Keep your body straight - avoid piking up\n"
            if prediction_counts.get('snake', 0) > prediction_counts.get('correct', 0):
                report += "• Engage your core to prevent sagging\n"
        elif self.selected_exercise == "squats":
            if prediction_counts.get('knees_in', 0) > prediction_counts.get('correct', 0):
                report += "• Keep your knees aligned with your toes\n"
                report += "• Focus on pushing knees outward\n"
        elif self.selected_exercise == "plank":
            if prediction_counts.get('hips_down', 0) > prediction_counts.get('correct', 0):
                report += "• Engage your core to raise your hips\n"
                report += "• Keep your body in a straight line\n"
            if prediction_counts.get('hips_up', 0) > prediction_counts.get('correct', 0):
                report += "• Lower your hips to be aligned with your shoulders and ankles\n"
                report += "• Keep your back straight\n"
        elif self.selected_exercise == "russian_twists":
            if prediction_counts.get('legs_bent', 0) > prediction_counts.get('correct', 0):
                report += "• Keep your legs straighter for proper form\n"
                report += "• Focus on core rotation, not just moving your arms\n"
        elif self.selected_exercise == "lunges":
            if prediction_counts.get('back_straight', 0) > prediction_counts.get('correct', 0):
                report += "• Keep your back straight and upright\n"
                report += "• Engage your core for better posture\n"
            if prediction_counts.get('legs_far', 0) > prediction_counts.get('correct', 0):
                report += "• Increase the distance between your front and back leg\n"
                report += "• Take a larger step forward for proper form\n"
        
        if correct_percentage < 80:
            report += "• Practice the movement slowly to build muscle memory\n"
            report += "• Consider working with a trainer for personalized guidance\n"
        
        self.benchmark_text.delete("1.0", "end")
        self.benchmark_text.insert("end", report)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = FitnessTrainerApp()
    app.run()