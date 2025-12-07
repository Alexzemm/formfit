# AI-Based Fitness Form Correction & Workout/Diet Plan Generator

This project is an AI-powered fitness assistant that helps users perform exercises with correct posture and generates personalized workout and diet plans. The system uses PoseFormer for exercise form classification and Gemini API for weekly fitness guidance.

---

## 🚀 Features

### 🔹 Real-Time Form Correction
- **Supports 5 exercises:**
  - Push-ups
  - Squats
  - Lunges
  - Russian Twists
  - Plank
- Detects correct vs incorrect form using **PoseFormer**
- Provides on-screen feedback **and** voice feedback (TTS)
- Helps users maintain proper alignment during workouts

### 🔹 AI-Generated Workout & Diet Plans
- Uses **Gemini API** to generate:
  - Weekly workout plans
  - Daily meal suggestions
  - Motivation & progress recommendations

### 🔹 User-Friendly Web Interface
- Built using **Flask + HTML + CSS + JavaScript**
- Webcam integration for real-time exercise monitoring
- Smooth UI for selecting workouts and viewing results

---

## 🧠 Tech Stack

**Machine Learning & Processing:**
- Python
- PyTorch
- PoseFormer (Transformer-based architecture)
- MediaPipe (keypoint extraction)
- OpenCV

**Backend:**
- Flask

**Frontend:**
- HTML, CSS, JavaScript

**AI API:**
- Gemini API (Workout & Diet Plan Generation)

**Development:**
- Jupyter Notebook / Google Colab

---

## 📊 How It Works

1. **User selects an exercise** on the web interface
2. **Webcam feed** is processed using MediaPipe
3. **Extracted keypoints** are passed into PoseFormer
4. **Model predicts** Correct / Wrong form
5. System displays:
   - Visual cues
   - Text instructions
   - Optional voice output (TTS)
6. **Gemini API** generates weekly workout & diet plans

---