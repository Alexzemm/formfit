import customtkinter as ctk
import cv2
import torch
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import threading
import time
import multiprocessing
from collections import Counter
import sys
import os

# Add the folders to Python path
sys.path.append(os.path.join(os.getcwd(), 'squats'))
sys.path.append(os.path.join(os.getcwd(), 'pushups'))
sys.path.append(os.path.join(os.getcwd(), 'plank'))

# Import evaluate modules from each exercise folder
try:
    # Import as modules to preserve their namespaces
    import pushups.evaluate as pushups_evaluate
    import squats.evaluate as squats_evaluate
    import plank.evaluate as plank_evaluate
except ImportError:
    # Fallback if modules are in the same directory structure
    print("Error: Could not import evaluate modules from exercise directories.")

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
                }
            },
            "squats": {
                "name": "Squats",
                "model_path": "squats/squat_transformer.pth", 
                "classes": ['correct', 'knees_in'],
                "colors": {
                    'correct': (0, 255, 0),
                    'knees_in': (0, 0, 255)
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
        self.status_frame.pack(pady=20, fill="x", padx=50)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Status: Ready",
            font=ctk.CTkFont(size=14)
        )
        self.status_label.pack(pady=10)
        
        # Benchmarks frame
        self.benchmark_frame = ctk.CTkFrame(main_frame)
        self.benchmark_frame.pack(pady=20, fill="both", expand=True, padx=50)
        
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
        
        # Highlight the selected exercise button
        if exercise_key == "pushups":
            self.pushup_btn.configure(fg_color="green")
        elif exercise_key == "squats":
            self.squat_btn.configure(fg_color="green")
        elif exercise_key == "plank":
            self.plank_btn.configure(fg_color="green")
    
    def load_model(self):
        # This function is no longer needed as we'll use the evaluate modules directly
        return True
    
    def extract_keypoints(self, frame):
        # This function is no longer needed as we'll use the evaluate modules directly
        pass
    
    def start_evaluation(self):
        if not self.selected_exercise:
            return
        
        # Stop any running evaluation
        self.stop_evaluation()
        
        # Update UI
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_label.configure(text="Status: Evaluating...")
        self.benchmark_text.delete("1.0", "end")
        
        # Start the appropriate evaluation module in a separate thread
        self.is_evaluating = True
        self.start_time = time.time()
        self.predictions = []
        self.frame_count = 0
        
        self.eval_thread = threading.Thread(target=self.evaluation_loop)
        self.eval_thread.daemon = True
        self.eval_thread.start()
    
    def evaluation_loop(self):
        # Launch the appropriate evaluate module based on selected exercise
        try:
            # Set up event to stop evaluation
            self.stop_event = threading.Event()
            
            # Store the original stdout to restore it later
            original_stdout = sys.stdout
            
            # Create a custom stdout to capture predictions
            class CustomStdout:
                def __init__(self, app_instance):
                    self.app_instance = app_instance
                    self.buffer = ""
                
                def write(self, text):
                    # Still write to original stdout for debugging
                    original_stdout.write(text)
                    
                    # Process predictions if they appear in the output
                    self.buffer += text
                    if "Prediction:" in self.buffer:
                        for exercise_class in self.app_instance.exercises[self.app_instance.selected_exercise]["classes"]:
                            if exercise_class in self.buffer:
                                self.app_instance.predictions.append(exercise_class)
                                self.app_instance.frame_count += 1
                                break
                        self.buffer = ""
                
                def flush(self):
                    original_stdout.flush()
            
            # Redirect stdout
            sys.stdout = CustomStdout(self)
            
            # Create a way to stop the evaluate modules (they use infinite while loops)
            def run_evaluate_with_timeout(module_name):
                # Import in a separate process that we can terminate
                process = multiprocessing.Process(
                    target=self._import_and_run_module, 
                    args=(module_name,)
                )
                process.start()
                
                # Wait for the stop event or process completion
                while not self.stop_event.is_set() and process.is_alive():
                    time.sleep(0.1)
                
                # Terminate the process if it's still running
                if process.is_alive():
                    process.terminate()
                    process.join()
            
            # Launch the appropriate evaluate module
            if self.selected_exercise == "pushups":
                run_evaluate_with_timeout("pushups.evaluate")
            elif self.selected_exercise == "squats":
                run_evaluate_with_timeout("squats.evaluate")
            elif self.selected_exercise == "plank":
                run_evaluate_with_timeout("plank.evaluate")
            
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}")
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            
            # Update UI
            self.stop_evaluation()
    
    def _import_and_run_module(self, module_name):
        """Helper function to import and run a module in a separate process."""
        try:
            __import__(module_name)
        except Exception as e:
            print(f"Error importing {module_name}: {str(e)}")
    
    def stop_evaluation(self):
        self.is_evaluating = False
        
        # Set stop event if it exists
        if hasattr(self, 'stop_event'):
            self.stop_event.set()
        
        # Close all OpenCV windows that might have been opened by the evaluate modules
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