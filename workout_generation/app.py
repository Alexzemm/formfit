from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from datetime import datetime
from typing import Dict, Any
import os
import re
import hashlib

app = Flask(__name__)

# Configure Gemini API
genai.configure(api_key="AIzaSyBkd2kSANEaOUI-mWmoyYHkZz6KcAu5SnU")

class FitnessPlanGenerator:
    def __init__(self):
        """
        Initialize the Fitness Plan Generator with Gemini API (API key hardcoded)
        """
        # Use the newer Gemini model
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_workout_plan(self, user_info: Dict[str, Any]) -> str:
        """
        Generate a personalized workout plan using Gemini API
        """
        prompt = f"""
        Create a detailed personalized workout plan for:

        User Profile:
        - Name: {user_info['name']}
        - Age: {user_info['age']}, Gender: {user_info['gender']}
        - Height: {user_info['height']}cm, Weight: {user_info['weight']}kg
        - Fitness Goal: {user_info['fitness_goal']}
        - Activity Level: {user_info['activity_level']}
        - Available workout days: {user_info['workout_days']} days/week
        - Preferred duration: {user_info['workout_duration']} minutes
        - Workout location: {user_info['workout_location']}
        - Medical conditions: {user_info.get('medical_conditions', 'None')}

        Please provide:
        1. A weekly workout schedule with specific days
        2. Detailed exercises for each workout day
        3. Sets, reps, and rest periods
        4. Progression plan for 4 weeks
        5. Warm-up and cool-down routines
        6. Safety tips and modifications
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating workout plan: {str(e)}"

    def generate_diet_plan(self, user_info: Dict[str, Any]) -> str:
        """
        Generate a personalized diet plan using Gemini API
        """
        # Calculate BMR and daily calories
        if user_info['gender'].lower() == 'male':
            bmr = 88.362 + (13.397 * user_info['weight']) + (4.799 * user_info['height']) - (5.677 * user_info['age'])
        else:
            bmr = 447.593 + (9.247 * user_info['weight']) + (3.098 * user_info['height']) - (4.330 * user_info['age'])

        activity_multipliers = {
            "Sedentary": 1.2,
            "Lightly Active": 1.375,
            "Moderately Active": 1.55,
            "Very Active": 1.725,
            "Extremely Active": 1.9
        }

        daily_calories = bmr * activity_multipliers.get(user_info['activity_level'], 1.55)

        # Adjust calories based on goal
        if user_info['fitness_goal'] == "Weight Loss":
            daily_calories *= 0.85
        elif user_info['fitness_goal'] == "Muscle Gain":
            daily_calories *= 1.15

        prompt = f"""
        Create a detailed personalized diet plan for:

        User Profile:
        - Name: {user_info['name']}
        - Age: {user_info['age']}, Gender: {user_info['gender']}
        - Height: {user_info['height']}cm, Weight: {user_info['weight']}kg
        - Fitness Goal: {user_info['fitness_goal']}
        - Activity Level: {user_info['activity_level']}
        - Estimated daily calories needed: {daily_calories:.0f} calories
        - Dietary Preference: {user_info['diet_preference']}
        - Allergies/Intolerances: {user_info.get('allergies', 'None')}
        - Medical conditions: {user_info.get('medical_conditions', 'None')}

        Please provide:
        1. Daily calorie and macronutrient breakdown
        2. 7-day meal plan with breakfast, lunch, dinner, and 2 snacks
        3. Portion sizes and food quantities
        4. Meal prep tips
        5. Hydration recommendations
        6. Supplement recommendations (if any)
        7. Foods to avoid and prioritize
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating diet plan: {str(e)}"

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def save_user_credentials(username, password):
    """Save user credentials to file"""
    if not os.path.exists('users'):
        os.makedirs('users')
    
    hashed_password = hash_password(password)
    with open('users/credentials.txt', 'a', encoding='utf-8') as f:
        f.write(f"{username}:{hashed_password}\n")

def verify_user_credentials(username, password):
    """Verify user credentials"""
    if not os.path.exists('users/credentials.txt'):
        return False
    
    hashed_password = hash_password(password)
    with open('users/credentials.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stored_username, stored_password = line.strip().split(':', 1)
            if stored_username == username and stored_password == hashed_password:
                return True
    return False

def user_exists(username):
    """Check if user already exists"""
    if not os.path.exists('users/credentials.txt'):
        return False
    
    with open('users/credentials.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stored_username, _ = line.strip().split(':', 1)
            if stored_username == username:
                return True
    return False

def check_user_plans(username):
    """Check if user already has plans in SQL files"""
    workout_plan = None
    diet_plan = None
    
    # Check workout plans
    if os.path.exists('plans/workouts.sql'):
        try:
            with open('plans/workouts.sql', 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for the most recent plan for this user
                pattern = rf'-- Workout Plan for {re.escape(username)}.*?(?=-- Workout Plan for|$)'
                matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                if matches:
                    workout_plan = matches[-1].replace('-- ', '').replace('\n-- ', '\n').strip()
                    # Clean up formatting for the retrieved plan
                    workout_plan = clean_plan_text(workout_plan)
        except Exception:
            pass
    
    # Check diet plans
    if os.path.exists('plans/diets.sql'):
        try:
            with open('plans/diets.sql', 'r', encoding='utf-8') as f:
                content = f.read()
                # Look for the most recent plan for this user
                pattern = rf'-- Diet Plan for {re.escape(username)}.*?(?=-- Diet Plan for|$)'
                matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
                if matches:
                    diet_plan = matches[-1].replace('-- ', '').replace('\n-- ', '\n').strip()
                    # Clean up formatting for the retrieved plan
                    diet_plan = clean_plan_text(diet_plan)
        except Exception:
            pass
    
    return workout_plan, diet_plan

# Initialize the generator
generator = FitnessPlanGenerator()

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '').strip()
    
    # Handle the old single-field login (for backward compatibility)
    if not request.form.get('password') and not request.form.get('action'):
        if not username:
            return jsonify({'error': 'Please enter your name'}), 400
        
        # Check if user already has plans (old behavior)
        workout_plan, diet_plan = check_user_plans(username)
        
        if workout_plan and diet_plan:
            # User has existing plans, show them
            user_info = {'name': username}  # Minimal user info for existing plans
            return render_template('results.html', 
                                 user_info=user_info,
                                 workout_plan=workout_plan,
                                 diet_plan=diet_plan,
                                 timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        else:
            # New user, redirect to form
            return render_template('index.html')
    
    # Handle new password-based login
    password = request.form.get('password', '').strip()
    action = request.form.get('action', 'login')
    
    if not username or not password:
        return jsonify({'error': 'Please enter both name and password'}), 400
    
    if action == 'register':
        # Register new user
        if user_exists(username):
            return jsonify({'error': 'User already exists. Please login instead.'}), 400
        
        save_user_credentials(username, password)
        # New user, redirect to form
        return render_template('index.html')
    
    else:  # action == 'login'
        # Login existing user
        if not verify_user_credentials(username, password):
            return jsonify({'error': 'Invalid username or password'}), 400
        
        # Check if user already has plans
        workout_plan, diet_plan = check_user_plans(username)
        
        if workout_plan and diet_plan:
            # User has existing plans, show them
            user_info = {'name': username}  # Minimal user info for existing plans
            return render_template('results.html', 
                                 user_info=user_info,
                                 workout_plan=workout_plan,
                                 diet_plan=diet_plan,
                                 timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        else:
            # User doesn't have plans, redirect to form
            return render_template('index.html')

@app.route('/new')
def new_user():
    return render_template('index.html')

def clean_plan_text(text):
    """
    Clean up text formatting from AI-generated plans by:
    1. Removing excess asterisks used for Markdown formatting
    2. Maintaining clear headings and structure
    3. Preserving important formatting like numbering and lists
    """
    if not text:
        return ""
    
    # Remove markdown formatting for bold/italic text
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove ** bold **
    cleaned = re.sub(r'\*([^*]+)\*', r'\1', cleaned)   # Remove * italic *
    
    # Clean up heading markers
    cleaned = re.sub(r'#{1,3} ', '', cleaned)
    
    # Convert markdown lists to cleaner bullet points
    cleaned = re.sub(r'^\* ', '• ', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^- ', '• ', cleaned, flags=re.MULTILINE)
    
    # Keep indentation and line breaks consistent
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned

@app.route('/generate', methods=['POST'])
def generate_plans():
    try:
        # Collect user data from form
        user_info = {
            'name': request.form.get('name'),
            'age': int(request.form.get('age')),
            'gender': request.form.get('gender'),
            'height': float(request.form.get('height')),
            'weight': float(request.form.get('weight')),
            'fitness_goal': request.form.get('fitness_goal'),
            'activity_level': request.form.get('activity_level'),
            'workout_days': int(request.form.get('workout_days')),
            'workout_duration': int(request.form.get('workout_duration')),
            'workout_location': request.form.get('workout_location'),
            'diet_preference': request.form.get('diet_preference'),
            'allergies': request.form.get('allergies', ''),
            'medical_conditions': request.form.get('medical_conditions', ''),
            'country': request.form.get('country', ''),
        }
        
        # Validate required fields
        required_fields = ['name', 'age', 'gender', 'height', 'weight', 'fitness_goal', 'activity_level', 'workout_days', 'workout_duration']
        for field in required_fields:
            if not user_info[field] or str(user_info[field]).strip() == "":
                return jsonify({'error': f"Please fill in the {field.replace('_', ' ').title()} field."}), 400

        # Generate plans
        workout_plan = generator.generate_workout_plan(user_info)
        diet_plan = generator.generate_diet_plan(user_info)
        
        # Clean up the formatting
        workout_plan = clean_plan_text(workout_plan)
        diet_plan = clean_plan_text(diet_plan)
        
        return render_template('results.html', 
                             user_info=user_info,
                             workout_plan=workout_plan,
                             diet_plan=diet_plan,
                             timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    except ValueError:
        return jsonify({'error': 'Please check number fields (Age, Height, Weight, Workout Days, Workout Duration) contain valid numbers.'}), 400
    except Exception as e:
        return jsonify({'error': f'Error generating plans: {str(e)}'}), 500

@app.route('/save', methods=['POST'])
def save_plans():
    try:
        workout_plan = request.form.get('workout_plan')
        diet_plan = request.form.get('diet_plan')
        name = request.form.get('name')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create plans directory if it doesn't exist
        if not os.path.exists('plans'):
            os.makedirs('plans')
        
        # Save workout plan to workouts.sql
        with open(f"plans/workouts.sql", 'a', encoding='utf-8') as f:
            f.write(f"-- Workout Plan for {name}\n")
            f.write(f"-- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-- " + "="*48 + "\n")
            f.write(f"-- {workout_plan.replace(chr(10), chr(10) + '-- ')}\n\n")
        
        # Save diet plan to diets.sql
        with open(f"plans/diets.sql", 'a', encoding='utf-8') as f:
            f.write(f"-- Diet Plan for {name}\n")
            f.write(f"-- Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-- " + "="*48 + "\n")
            f.write(f"-- {diet_plan.replace(chr(10), chr(10) + '-- ')}\n\n")
        
        return jsonify({'success': 'Plans saved successfully! Files: workouts.sql and diets.sql'})
    
    except Exception as e:
        return jsonify({'error': f'Error saving files: {str(e)}'}), 500

# Route to handle Chrome DevTools requests (prevents 404 errors in logs)
@app.route('/.well-known/appspecific/com.chrome.devtools.json')
def chrome_devtools():
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)