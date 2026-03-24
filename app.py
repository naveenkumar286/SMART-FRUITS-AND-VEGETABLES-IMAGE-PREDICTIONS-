import os
import time
import re
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, Response, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import io
import base64
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError:
    px = go = pio = None
from string import Template
from difflib import get_close_matches
import json
import sqlite3
from functools import wraps

# Keras / TensorFlow
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.layers import DepthwiseConv2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'upload_images'), exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.environ.get('FLASK_SECRET', 'fallback-secret-key-change-in-prod')

# Database initialization
DB_PATH = os.path.join(BASE_DIR, 'users.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database on startup
with app.app_context():
    init_db()

# Ensure upload dirs exist on Render
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Custom DepthwiseConv2D to handle compatibility
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# Load model once at startup
try:
    model = load_model(os.path.join(BASE_DIR, 'FV.h5'), custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
except Exception as e:
    model = None

def create_recipes_csv_if_not_exists():
    """Create recipes.csv if it doesn't exist"""
    recipes_csv_path = os.path.join(BASE_DIR, 'recipes.csv')
    if not os.path.exists(recipes_csv_path):
        # Sample recipes data
        recipes_data = [
            {
                'food_name': 'Apple',
                'ingredients': '4 large Honeycrisp apples (peeled and cored)|2 tablespoons unsalted European butter|1/4 cup pure maple syrup|1 teaspoon ground Ceylon cinnamon',
                'cooking_process': 'Prepare mise en place: Core and slice apples into uniform 1/4-inch wedges|Heat butter in a heavy-bottomed stainless steel pan over medium heat|Add apple wedges in a single layer, avoid overcrowding|Sauté for 3-4 minutes without stirring to develop caramelization',
                'cooking_time': '12-15 minutes',
                'difficulty': 'Intermediate',
                'chef_tips': 'Choose firm apples that hold their shape. Avoid overcooking to maintain texture.'
            },
            {
                'food_name': 'Banana',
                'ingredients': '3 very ripe bananas (with brown spots)|2 large free-range eggs|1 cup all-purpose flour (sifted)|1/2 cup whole milk (room temperature)',
                'cooking_process': 'Mash bananas in a large mixing bowl until mostly smooth|Whisk eggs in a separate bowl until well beaten|Combine mashed bananas with beaten eggs|Add milk, sugar, vanilla extract, and whisk until incorporated',
                'cooking_time': '20-25 minutes',
                'difficulty': 'Intermediate',
                'chef_tips': 'Ripe bananas provide natural sweetness. Control heat to prevent burning.'
            }
        ]
        
        df = pd.DataFrame(recipes_data)
        df.to_csv(recipes_csv_path, index=False)
        print(f"Created default recipes.csv with {len(recipes_data)} recipes")

# Create CSV on startup
create_recipes_csv_if_not_exists()

# Load nutrition CSVs
try:
    fruits_df = pd.read_csv(os.path.join(BASE_DIR, 'fruits.csv'))
except Exception:
    fruits_df = None

try:
    vegetables_df = pd.read_csv(os.path.join(BASE_DIR, 'vegetables.csv'))
except Exception:
    vegetables_df = None

# Labels and categories
labels = {0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
          7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
          14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce',
          19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple',
          26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn',
          32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path):
    if model is None:
        raise RuntimeError('Model not loaded')
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int("".join(str(x) for x in y_class))
    res = labels.get(y, 'Unknown')
    return res.capitalize()

def get_nutrition_info(prediction):
    try:
        if not prediction:
            return None
        
        # Search in fruits CSV first
        if fruits_df is not None:
            fruit_match = fruits_df[fruits_df['name'].str.contains(prediction, case=False, na=False)]
            if not fruit_match.empty:
                row = fruit_match.iloc[0]
                return {
                    'energy': row['energy (kcal/kJ)'],
                    'protein': f"{row['protein (g)']}g",
                    'carbs': f"{row['carbohydrates (g)']}g",
                    'fiber': f"{row['fiber (g)']}g",
                    'vitamin_c': f"{row['vitamin C (mg)']}mg",
                    'calcium': f"{row['calcium (mg)']}mg"
                }
        
        # Search in vegetables CSV if not found in fruits
        if vegetables_df is not None:
            veg_match = vegetables_df[vegetables_df['name'].str.contains(prediction, case=False, na=False)]
            if not veg_match.empty:
                row = veg_match.iloc[0]
                return {
                    'energy': row['energy (kcal/kJ)'],
                    'protein': f"{row['protein (g)']}g",
                    'carbs': f"{row['carbohydrates (g)']}g",
                    'fiber': f"{row['fiber (g)']}g",
                    'vitamin_c': f"{row['vitamin C (mg)']}mg",
                    'calcium': f"{row['calcium (mg)']}mg"
                }
        
        return None
    except Exception:
        return None

def get_food_recipes(food_name):
    """Get 3 recipes for the given food item"""
    try:
        recipes_csv_path = os.path.join(BASE_DIR, 'recipes.csv')
        if os.path.exists(recipes_csv_path):
            recipes_df = pd.read_csv(recipes_csv_path)
            # Search for recipes matching the food name (case insensitive)
            recipe_matches = recipes_df[recipes_df['food_name'].str.contains(food_name, case=False, na=False)]
            
            recipes_list = []
            for _, row in recipe_matches.iterrows():
                recipes_list.append({
                    'recipe_name': row['recipe_name'],
                    'ingredients': row['ingredients'].split('|'),
                    'cooking_process': row['cooking_process'].split('|'),
                    'cooking_time': row['cooking_time'],
                    'difficulty': row['difficulty']
                })
            
            if recipes_list:
                return recipes_list
    except Exception as e:
        print(f"Error loading recipes from CSV: {e}")
    
    # Default recipes if not found
    return [{
        'recipe_name': f'Simple {food_name} Recipe',
        'ingredients': [f'1 pound {food_name}', '2 tbsp olive oil', 'Salt and pepper'],
        'cooking_process': [f'Prepare {food_name}', 'Season with salt and pepper', 'Cook until tender'],
        'cooking_time': '15 minutes',
        'difficulty': 'Easy'
    }]

@app.route('/chart_image')
def chart_image():
    pred = request.args.get('pred')
    chart_type = request.args.get('chart', 'bar')
    try:
        nutrition = get_nutrition_info(pred)
        if not nutrition:
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, 'No nutrition data', horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return send_file(buf, mimetype='image/png')

        labels_chart = ['Protein', 'Carbs', 'Fiber']
        values = [float(nutrition['protein'].replace('g','')), 
                  float(nutrition['carbs'].replace('g','')), 
                  float(nutrition['fiber'].replace('g',''))]

        fig = plt.figure(figsize=(4, 3))
        if chart_type == 'pie':
            try:
                filtered = [(l, v) for l, v in zip(labels_chart, values) if v and v > 0]
                if not filtered:
                    raise ValueError('No data for pie')
                labs, vals = zip(*filtered)
                plt.pie(vals, labels=labs, autopct='%1.1f%%', startangle=90)
                plt.title(f'{pred} nutrition (%)')
            except Exception:
                plt.text(0.5, 0.5, 'Chart unavailable', horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
        else:
            try:
                plt.bar(labels_chart, values, color=['#ff9999','#66b3ff','#99ff99'])
                plt.ylabel('Amount (g)')
                plt.title(f'{pred} nutrition')
            except Exception:
                plt.text(0.5, 0.5, 'Chart unavailable', horizontalalignment='center', verticalalignment='center')
                plt.axis('off')

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype='image/png')
    except Exception:
        return Response('Error generating chart', status=500)

@app.route('/plotly_chart')
def plotly_chart():
    pred = request.args.get('pred')
    chart_type = request.args.get('chart', 'pie')
    try:
        nutrition = get_nutrition_info(pred)
        if not nutrition:
            return jsonify({'error': 'No nutrition data'})
        
        nutrition_data = {
            'Nutrient': ['Protein', 'Carbs', 'Fiber'],
            'Value': [float(nutrition['protein'].replace('g','')), 
                     float(nutrition['carbs'].replace('g','')), 
                     float(nutrition['fiber'].replace('g',''))]
        }
        
        if chart_type == 'pie':
            fig = px.pie(pd.DataFrame(nutrition_data), values='Value', names='Nutrient', 
                        title=f"Nutrition Breakdown - {pred}")
        else:
            fig = go.Figure(data=[go.Surface(
                z=[nutrition_data['Value']],
                colorscale='Plasma'
            )])
            fig.update_layout(title=f"3D Nutrition - {pred}", height=400)
        
        return fig.to_json()
    except Exception:
        return jsonify({'error': 'Chart generation failed'})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form.get('username')
        password = request.form.get('password')
        
        if not username_or_email or not password:
            flash('All fields are required!', 'error')
            return render_template('login.html')
        
        # Check admin credentials
        if username_or_email == 'naveen@admin' and password == 'naveen@2006':
            session['user_id'] = 'admin'
            session['username'] = 'naveen@admin'
            session['is_admin'] = True
            session['login_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_users'))
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT id, username, email, password FROM users WHERE username = ? OR email = ?', 
                 (username_or_email, username_or_email))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['email'] = user[2]
            session['is_admin'] = False
            session['login_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username/email or password!', 'error')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return render_template('signup.html')
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Check if username exists
        c.execute('SELECT id FROM users WHERE username = ?', (username,))
        if c.fetchone():
            flash('Username already exists!', 'error')
            conn.close()
            return render_template('signup.html')
        
        # Check if email exists
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        if c.fetchone():
            flash('Email already exists!', 'error')
            conn.close()
            return render_template('signup.html')
        
        # Create user
        hashed_password = generate_password_hash(password)
        try:
            c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)', 
                     (username, email, hashed_password))
            conn.commit()
            flash('Account created successfully!', 'success')
        except sqlite3.Error as e:
            flash('Error creating account!', 'error')
        finally:
            conn.close()
        
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))

@app.route('/admin_users')
@login_required
def admin_users():
    if not session.get('is_admin'):
        return redirect(url_for('index'))
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, username, email, created_at FROM users ORDER BY created_at DESC')
    users = c.fetchall()
    conn.close()
    
    return render_template('admin_users.html', users=users)

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if 'food_history' not in session:
        session['food_history'] = []
    
    results = []
    errors = []

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part in the request', 'danger')
            return redirect(request.url)

        files = request.files.getlist('image')
        if not files or files == [None]:
            flash('No files selected', 'danger')
            return redirect(request.url)

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                name, ext = os.path.splitext(filename)
                unique_name = f"{name}_{int(time.time()*1000)}{ext}"
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
                
                try:
                    file.save(save_path)
                except Exception:
                    errors.append(f"Failed to save {filename}")
                    continue

                entry = {
                    'filename': unique_name,
                    'url': url_for('static', filename=f'uploads/{unique_name}'),
                    'predicted': None,
                    'category': None,
                    'error': None
                }

                try:
                    pred = prepare_image(save_path)
                    # Clean up file after prediction to save disk space
                    try:
                        os.remove(save_path)
                        entry['url'] = None
                    except Exception:
                        pass
                    entry['predicted'] = pred
                    entry['category'] = 'Vegetable' if pred in vegetables else 'Fruit'
                    
                    # Add to session history
                    session['food_history'].append({
                        'food': pred,
                        'category': entry['category'],
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
                    })
                    session.modified = True
                    
                except Exception:
                    entry['error'] = 'Prediction failed'
                    errors.append(f'Prediction failed for {filename}')
                    results.append(entry)
                    continue

                # Get nutrition info and recipes
                try:
                    nutrition = get_nutrition_info(pred)
                    entry['nutrition'] = nutrition
                    
                    # Get recipe suggestions (3 recipes per food)
                    recipes = get_food_recipes(pred)
                    entry['recipes'] = recipes
                    
                except Exception as e:
                    print(f"Error getting food info: {e}")
                    entry['nutrition'] = None
                    entry['recipes'] = None

                results.append(entry)
            else:
                errors.append(f'Unsupported file: {file.filename}')

        if not results and errors:
            flash('No valid images were processed', 'warning')

    return render_template('index.html', results=results, errors=errors)

@app.route('/meal_details/<day>')
@login_required
def meal_details(day):
    weekly_plan = session.get('weekly_plan')
    if weekly_plan and day in weekly_plan:
        day_plan = weekly_plan[day]
        return render_template('meal_details.html', day=day, meal_plan=day_plan)
    return redirect(url_for('meal_plate'))

@app.route('/meal_plate', methods=['GET', 'POST'])
@login_required
def meal_plate():
    if 'meal_history' not in session:
        session['meal_history'] = []
    if 'progress_data' not in session:
        session['progress_data'] = []
    
    meal_plan = session.get('meal_plan')
    bmi_result = session.get('bmi_result')
    weekly_plan = session.get('weekly_plan')
    macro_breakdown = session.get('macro_breakdown')
    meal_form_data = session.get('meal_form_data', {})
    
    # Auto-suggest goal based on BMI if available
    suggested_goal = None
    bmi_recommendation = None
    if bmi_result and 'bmi' in bmi_result:
        bmi_value = bmi_result['bmi']
        if bmi_value < 18.5:
            suggested_goal = 'weight_gain'
            bmi_recommendation = {
                'status': 'Underweight',
                'message': 'Based on your BMI, we recommend weight gain foods rich in healthy calories',
                'foods': 'High-calorie Tamil Nadu foods like Pongal, Ghee Rice, Parotta, and traditional sweets'
            }
        elif bmi_value >= 25:
            suggested_goal = 'weight_loss'
            bmi_recommendation = {
                'status': 'Overweight/Obese',
                'message': 'Based on your BMI, we recommend weight loss foods with controlled portions',
                'foods': 'Low-calorie options like Idli, Ragi Dosa, Sambar, and steamed vegetables'
            }
        else:
            suggested_goal = 'general_fitness'
            bmi_recommendation = {
                'status': 'Normal Weight',
                'message': 'Based on your BMI, we recommend balanced maintenance foods',
                'foods': 'Balanced Tamil Nadu meals with Dosa, Curd Rice, Fish Curry, and variety of vegetables'
            }
    
    # Get recipes for current meal plan
    recipes = []
    if meal_plan:
        all_foods = set()
        for meal_type in ['breakfast', 'lunch', 'dinner', 'snack']:
            if meal_type in meal_plan:
                for item in meal_plan[meal_type]:
                    all_foods.add(item['name'])
        
        for food in all_foods:
            recipe = generate_dynamic_recipe(food)
            if recipe:
                recipes.append(recipe)
    
    if request.method == 'POST':
        # BMI calculation
        if 'height' in request.form and 'weight' in request.form:
            try:
                height = float(request.form.get('height'))
                weight = float(request.form.get('weight'))
                age = int(request.form.get('age', 25))
                gender = request.form.get('gender', 'male')
                activity = request.form.get('activity', 'moderate')
                
                bmi = weight / ((height/100) ** 2)
                
                if gender == 'male':
                    bmr = 10 * weight + 6.25 * height - 5 * age + 5
                else:
                    bmr = 10 * weight + 6.25 * height - 5 * age - 161
                
                activity_multipliers = {'sedentary': 1.2, 'light': 1.375, 'moderate': 1.55, 'active': 1.725, 'very_active': 1.9}
                tdee = bmr * activity_multipliers.get(activity, 1.55)
                
                if bmi < 18.5:
                    category, advice, goal_suggestion = 'Underweight', 'Focus on weight gain with +500 calorie surplus', 'weight_gain'
                    target_calories = int(tdee + 500)
                    food_recommendation = 'High-calorie Tamil Nadu foods: Pongal with Ghee, Parotta, Ghee Rice, Vada, traditional sweets'
                elif 18.5 <= bmi < 25:
                    category, advice, goal_suggestion = 'Normal', 'Maintain current weight with balanced diet', 'general_fitness'
                    target_calories = int(tdee)
                    food_recommendation = 'Balanced Tamil Nadu meals: Dosa with Sambar, Curd Rice, Fish Curry, variety of vegetables'
                elif 25 <= bmi < 30:
                    category, advice, goal_suggestion = 'Overweight', 'Focus on weight loss with -500 calorie deficit', 'weight_loss'
                    target_calories = int(tdee - 500)
                    food_recommendation = 'Low-calorie Tamil Nadu foods: Idli with Sambar, Ragi Dosa, steamed vegetables, Rasam'
                else:
                    category, advice, goal_suggestion = 'Obese', 'Start with -750 calorie deficit', 'weight_loss'
                    target_calories = int(tdee - 750)
                    food_recommendation = 'Very low-calorie options: Idli without oil, vegetable Sambar, steamed Poriyal, buttermilk'
                
                bmi_result = {
                    'bmi': round(bmi, 1), 'category': category, 'advice': advice, 'goal_suggestion': goal_suggestion,
                    'bmr': int(bmr), 'tdee': int(tdee), 'target_calories': target_calories,
                    'height': height, 'weight': weight, 'age': age, 'gender': gender, 'activity': activity,
                    'food_recommendation': food_recommendation
                }
                
                session['bmi_result'] = bmi_result
                session['progress_data'].append({'date': datetime.now().strftime('%Y-%m-%d'), 'weight': weight, 'bmi': round(bmi, 1)})
                
                # Auto-generate meal plan based on BMI
                auto_meal_plan = generate_meal_plan(goal_suggestion, 'veg', 'tamil', 'medium')
                session['auto_meal_suggestion'] = auto_meal_plan
                session.modified = True
                
            except:
                bmi_result = {'error': 'Invalid input'}
        
        # Meal plan generation
        elif 'goal' in request.form:
            goal = request.form.get('goal')
            preference = request.form.get('preference')
            cuisine = request.form.get('cuisine')
            budget = request.form.get('budget')
            plan_type = request.form.get('plan_type', 'daily')
            
            meal_form_data = {'goal': goal, 'preference': preference, 'cuisine': cuisine, 'budget': budget, 'plan_type': plan_type}
            session['meal_form_data'] = meal_form_data
            
            if plan_type == 'weekly':
                weekly_plan = generate_weekly_meal_plan(goal, preference, cuisine, budget)
                session['weekly_plan'] = weekly_plan
                session['meal_plan'] = None
                session['macro_breakdown'] = None
                session['meal_history'].append({
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'type': 'weekly',
                    'goal': goal,
                    'plan': weekly_plan
                })
            else:
                meal_plan = generate_meal_plan(goal, preference, cuisine, budget)
                macro_breakdown = calculate_macro_breakdown(meal_plan)
                session['meal_plan'] = meal_plan
                session['macro_breakdown'] = macro_breakdown
                session['weekly_plan'] = None
                session['meal_history'].append({
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'type': 'daily',
                    'goal': goal,
                    'plan': meal_plan
                })
            session.modified = True
    
    return render_template('meal_plate.html', 
                          meal_plan=meal_plan, 
                          bmi_result=bmi_result, 
                          weekly_plan=weekly_plan, 
                          macro_breakdown=macro_breakdown,
                          meal_form_data=meal_form_data, 
                          recipes=recipes,
                          meal_history=session.get('meal_history', [])[-5:],
                          progress_data=session.get('progress_data', [])[-10:],
                          suggested_goal=suggested_goal,
                          bmi_recommendation=bmi_recommendation,
                          auto_meal_suggestion=session.get('auto_meal_suggestion'))

def calculate_macro_breakdown(meal_plan):
    if not meal_plan:
        return None
    total_cal = meal_plan['totals']['calories']
    total_protein = meal_plan['totals']['protein']
    total_carbs = meal_plan['totals']['carbs']
    total_fat = meal_plan['totals']['fat']
    
    protein_cal = total_protein * 4
    carbs_cal = total_carbs * 4
    fat_cal = total_fat * 9
    
    return {
        'protein_percent': round((protein_cal / total_cal) * 100) if total_cal > 0 else 0,
        'carbs_percent': round((carbs_cal / total_cal) * 100) if total_cal > 0 else 0,
        'fat_percent': round((fat_cal / total_cal) * 100) if total_cal > 0 else 0,
        'protein_cal': int(protein_cal),
        'carbs_cal': int(carbs_cal),
        'fat_cal': int(fat_cal)
    }

def generate_weekly_meal_plan(goal, preference, cuisine, budget):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly = {}
    
    for day in days:
        weekly[day] = generate_meal_plan(goal, preference, cuisine, budget)
    
    return weekly

def generate_meal_plan(goal, preference, cuisine, budget):
    # Professional Tamil Nadu and Indian meal templates
    meals = {
        'weight_loss': {
            'breakfast': [
                {'name': 'Idli with Sambar', 'portion': '3 pieces', 'calories': 150, 'protein': 6, 'carbs': 28, 'fat': 2},
                {'name': 'Coconut Chutney', 'portion': '2 tbsp', 'calories': 45, 'protein': 1, 'carbs': 3, 'fat': 4},
                {'name': 'Filter Coffee', 'portion': '1 cup', 'calories': 25, 'protein': 1, 'carbs': 3, 'fat': 1}
            ],
            'lunch': [
                {'name': 'Brown Rice', 'portion': '1 cup', 'calories': 220, 'protein': 5, 'carbs': 45, 'fat': 2},
                {'name': 'Rasam', 'portion': '1 bowl', 'calories': 80, 'protein': 3, 'carbs': 12, 'fat': 2},
                {'name': 'Poriyal (Beans)', 'portion': '1 cup', 'calories': 90, 'protein': 4, 'carbs': 15, 'fat': 3},
                {'name': 'Mor Kuzhambu', 'portion': '1 bowl', 'calories': 120, 'protein': 6, 'carbs': 8, 'fat': 8}
            ],
            'dinner': [
                {'name': 'Ragi Dosa', 'portion': '2 pieces', 'calories': 140, 'protein': 5, 'carbs': 25, 'fat': 3},
                {'name': 'Sambar', 'portion': '1 bowl', 'calories': 100, 'protein': 6, 'carbs': 15, 'fat': 3},
                {'name': 'Tomato Chutney', 'portion': '2 tbsp', 'calories': 35, 'protein': 1, 'carbs': 5, 'fat': 2}
            ],
            'snack': [{'name': 'Sundal (Chickpea)', 'portion': '1/2 cup', 'calories': 110, 'protein': 6, 'carbs': 18, 'fat': 2}],
            'swap': 'Replace white rice with brown rice or millets',
            'hydration': 'Start with warm water and lemon, drink buttermilk',
            'mistake': 'Eating too much coconut-based gravies'
        },
        'weight_gain': {
            'breakfast': [
                {'name': 'Pongal with Ghee', 'portion': '1 bowl', 'calories': 320, 'protein': 8, 'carbs': 45, 'fat': 12},
                {'name': 'Vada', 'portion': '2 pieces', 'calories': 180, 'protein': 6, 'carbs': 20, 'fat': 9},
                {'name': 'Filter Coffee with Milk', 'portion': '1 cup', 'calories': 80, 'protein': 4, 'carbs': 8, 'fat': 4}
            ],
            'lunch': [
                {'name': 'Ghee Rice', 'portion': '2 cups', 'calories': 450, 'protein': 8, 'carbs': 75, 'fat': 15},
                {'name': 'Chicken Chettinad', 'portion': '150g', 'calories': 280, 'protein': 25, 'carbs': 8, 'fat': 18},
                {'name': 'Kootu (Mixed Dal)', 'portion': '1 bowl', 'calories': 160, 'protein': 10, 'carbs': 22, 'fat': 5},
                {'name': 'Appalam', 'portion': '2 pieces', 'calories': 60, 'protein': 2, 'carbs': 8, 'fat': 2}
            ],
            'dinner': [
                {'name': 'Parotta', 'portion': '3 pieces', 'calories': 420, 'protein': 10, 'carbs': 65, 'fat': 15},
                {'name': 'Mutton Curry', 'portion': '1 bowl', 'calories': 350, 'protein': 28, 'carbs': 6, 'fat': 25},
                {'name': 'Raita', 'portion': '1 bowl', 'calories': 90, 'protein': 5, 'carbs': 8, 'fat': 5}
            ],
            'snack': [{'name': 'Banana Bajji', 'portion': '3 pieces', 'calories': 220, 'protein': 4, 'carbs': 35, 'fat': 8}],
            'swap': 'Add more ghee and coconut milk to increase calories',
            'hydration': 'Drink coconut water and buttermilk',
            'mistake': 'Not eating enough traditional sweets and ghee'
        },
        'general_fitness': {
            'breakfast': [
                {'name': 'Dosa with Potato Masala', 'portion': '2 pieces', 'calories': 280, 'protein': 8, 'carbs': 45, 'fat': 8},
                {'name': 'Sambar', 'portion': '1 bowl', 'calories': 100, 'protein': 6, 'carbs': 15, 'fat': 3},
                {'name': 'Coconut Chutney', 'portion': '2 tbsp', 'calories': 45, 'protein': 1, 'carbs': 3, 'fat': 4}
            ],
            'lunch': [
                {'name': 'Curd Rice', 'portion': '1 cup', 'calories': 200, 'protein': 8, 'carbs': 35, 'fat': 4},
                {'name': 'Fish Curry (Tamil Style)', 'portion': '150g', 'calories': 220, 'protein': 25, 'carbs': 5, 'fat': 12},
                {'name': 'Keerai Masiyal', 'portion': '1 cup', 'calories': 80, 'protein': 5, 'carbs': 10, 'fat': 3},
                {'name': 'Pickle', 'portion': '1 tsp', 'calories': 15, 'protein': 0, 'carbs': 2, 'fat': 1}
            ],
            'dinner': [
                {'name': 'Chapati', 'portion': '3 pieces', 'calories': 210, 'protein': 6, 'carbs': 42, 'fat': 3},
                {'name': 'Dal Tadka', 'portion': '1 bowl', 'calories': 140, 'protein': 10, 'carbs': 20, 'fat': 4},
                {'name': 'Vegetable Curry', 'portion': '1 cup', 'calories': 100, 'protein': 3, 'carbs': 15, 'fat': 4}
            ],
            'snack': [{'name': 'Murukku', 'portion': '5 pieces', 'calories': 120, 'protein': 3, 'carbs': 18, 'fat': 4}],
            'swap': 'Include variety of millets like ragi, kambu',
            'hydration': 'Drink plenty of water, coconut water, buttermilk',
            'mistake': 'Not including enough variety in vegetables'
        }
    }
    
    # Adjust for vegetarian preference with Tamil Nadu alternatives
    selected_plan = meals.get(goal, meals['general_fitness'])
    
    if preference == 'veg':
        for meal in ['breakfast', 'lunch', 'dinner']:
            for item in selected_plan[meal]:
                if 'chicken chettinad' in item['name'].lower():
                    item['name'] = 'Paneer Chettinad'
                    item['protein'] = 18
                    item['calories'] = 250
                elif 'mutton curry' in item['name'].lower():
                    item['name'] = 'Kathirikai Kuzhambu'
                    item['protein'] = 8
                    item['calories'] = 180
                    item['fat'] = 12
                elif 'fish curry' in item['name'].lower():
                    item['name'] = 'Vendakkai Sambar'
                    item['protein'] = 8
                    item['calories'] = 120
                    item['fat'] = 6
                elif any(meat in item['name'].lower() for meat in ['chicken', 'mutton', 'fish']):
                    item['name'] = item['name'].replace('Chicken', 'Paneer').replace('Mutton', 'Mixed Vegetable').replace('Fish', 'Vegetable')
                    item['protein'] = max(8, item['protein'] - 8)
    
    # Calculate totals
    total_calories = sum(item['calories'] for meal in ['breakfast', 'lunch', 'dinner', 'snack'] for item in selected_plan[meal])
    total_protein = sum(item['protein'] for meal in ['breakfast', 'lunch', 'dinner', 'snack'] for item in selected_plan[meal])
    total_carbs = sum(item['carbs'] for meal in ['breakfast', 'lunch', 'dinner', 'snack'] for item in selected_plan[meal])
    total_fat = sum(item['fat'] for meal in ['breakfast', 'lunch', 'dinner', 'snack'] for item in selected_plan[meal])
    
    selected_plan['totals'] = {
        'calories': total_calories,
        'protein': total_protein,
        'carbs': total_carbs,
        'fat': total_fat
    }
    
    return selected_plan

@app.route('/analytics')
@login_required
def analytics():
    food_history = session.get('food_history', [])
    meal_history = session.get('meal_history', [])
    progress_data = session.get('progress_data', [])
    workout_history = session.get('workout_history', [])
    
    # Calculate analytics data
    analytics_data = calculate_analytics(food_history, meal_history, progress_data, workout_history)
    
    return render_template('analytics.html', 
                         food_history=food_history,
                         analytics_data=analytics_data)

def calculate_analytics(food_history, meal_history, progress_data, workout_history):
    from collections import Counter
    
    # Food category analysis
    categories = [item.get('category', 'Unknown') for item in food_history]
    category_counts = dict(Counter(categories))
    
    # Most scanned foods
    foods = [item.get('food', 'Unknown') for item in food_history]
    food_counts = dict(Counter(foods).most_common(10))
    
    # Health metrics
    total_scans = len(food_history)
    fruit_scans = sum(1 for item in food_history if item.get('category') == 'Fruit')
    vegetable_scans = sum(1 for item in food_history if item.get('category') == 'Vegetable')
    
    # Calculate health score
    health_score = 0
    if total_scans > 0:
        healthy_ratio = (fruit_scans + vegetable_scans) / total_scans
        health_score = min(100, int(healthy_ratio * 100))
    
    return {
        'category_counts': category_counts,
        'food_counts': food_counts,
        'total_scans': total_scans,
        'fruit_scans': fruit_scans,
        'vegetable_scans': vegetable_scans,
        'meal_plans_created': len(meal_history),
        'total_workouts': len(workout_history),
        'health_score': health_score
    }

@app.route('/recipes', methods=['GET', 'POST'])
@login_required
def recipes():
    meal_plan = session.get('meal_plan')
    selected_recipe = None
    search_query = None
    
    if request.method == 'POST':
        search_query = request.form.get('search_food', '').strip()
        if search_query:
            selected_recipe = get_recipe_for_food(search_query)
    
    # Get all food items from current meal plan
    meal_foods = []
    if meal_plan:
        for meal_type in ['breakfast', 'lunch', 'dinner', 'snack']:
            if meal_type in meal_plan:
                for item in meal_plan[meal_type]:
                    meal_foods.append(item['name'])
    
    return render_template('recipes_advanced.html', 
                         meal_foods=meal_foods, 
                         selected_recipe=selected_recipe,
                         search_query=search_query)

def generate_dynamic_recipe(food_name):
    """Generate dynamic recipe using AI-like logic"""
    import random
    
    # Base ingredients and cooking methods
    base_ingredients = {
        'spices': ['salt', 'black pepper', 'turmeric', 'cumin powder', 'coriander powder', 'garam masala', 'red chili powder'],
        'aromatics': ['onion', 'garlic', 'ginger', 'green chilies', 'curry leaves'],
        'oils': ['olive oil', 'coconut oil', 'ghee', 'vegetable oil'],
        'herbs': ['fresh coriander', 'mint leaves', 'basil'],
        'liquids': ['water', 'coconut milk', 'vegetable broth', 'tomato puree']
    }
    
    cooking_methods = {
        'rice': ['boil', 'steam', 'sauté'],
        'dal': ['pressure cook', 'simmer', 'boil'],
        'vegetables': ['sauté', 'steam', 'roast', 'stir-fry'],
        'chicken': ['grill', 'curry', 'roast', 'stir-fry'],
        'paneer': ['pan-fry', 'curry', 'grill'],
        'eggs': ['scramble', 'boil', 'fry', 'poach'],
        'oats': ['cook', 'steam', 'boil']
    }
    
    # Determine food category
    food_lower = food_name.lower()
    if any(grain in food_lower for grain in ['rice', 'oats', 'bread', 'roti']):
        category = 'grain'
    elif any(protein in food_lower for protein in ['chicken', 'fish', 'egg', 'paneer', 'dal']):
        category = 'protein'
    elif any(veg in food_lower for veg in ['vegetable', 'curry', 'spinach', 'potato']):
        category = 'vegetable'
    else:
        category = 'general'
    
    # Generate ingredients list
    ingredients = [f"1 cup {food_name.lower()}"] if 'cup' not in food_name.lower() else [food_name]
    
    # Add random base ingredients
    ingredients.extend(random.sample(base_ingredients['aromatics'], 2))
    ingredients.extend(random.sample(base_ingredients['spices'], 3))
    ingredients.append(random.choice(base_ingredients['oils']))
    ingredients.append(random.choice(base_ingredients['liquids']))
    
    if category == 'protein':
        ingredients.extend(['1 tsp ginger-garlic paste', '2 tomatoes chopped'])
    elif category == 'vegetable':
        ingredients.extend(['1 tsp mustard seeds', '1/2 cup coconut'])
    
    # Generate cooking steps
    steps = [
        f"Wash and prepare {food_name.lower()} properly",
        "Heat oil in a heavy-bottomed pan over medium heat",
        "Add aromatics and sauté until fragrant (2-3 minutes)",
        "Add spices and cook for 1 minute until aromatic"
    ]
    
    if category == 'protein':
        steps.extend([
            f"Add {food_name.lower()} and cook until lightly browned",
            "Add tomatoes and cook until they break down",
            "Add liquid and simmer for 15-20 minutes",
            "Adjust seasoning and garnish with fresh herbs"
        ])
    elif category == 'grain':
        steps.extend([
            f"Add {food_name.lower()} and toast for 2 minutes",
            "Add hot liquid gradually while stirring",
            "Cook until tender and liquid is absorbed",
            "Let it rest for 5 minutes before serving"
        ])
    else:
        steps.extend([
            f"Add {food_name.lower()} and mix well",
            "Cook covered for 10-15 minutes until tender",
            "Stir occasionally and add liquid if needed",
            "Garnish and serve hot"
        ])
    
    return {
        'name': f"Homestyle {food_name.title()}",
        'ingredients': ingredients,
        'steps': steps,
        'prep_time': f"{random.randint(10, 20)} mins",
        'cook_time': f"{random.randint(15, 30)} mins",
        'serves': f"{random.randint(2, 4)} people",
        'difficulty': random.choice(['Easy', 'Medium']),
        'calories': random.randint(150, 350)
    }

def get_recipe_for_food(food_name):
    comprehensive_recipes = {
        # Breakfast Items
        'steel cut oats with berries': {
            'name': 'Steel Cut Oats with Berries',
            'prep_time': '5 mins', 'cook_time': '15 mins', 'serves': '1 person', 'difficulty': 'Easy',
            'cuisine': 'Healthy', 'calories': 180,
            'nutrition': {'protein': '7g', 'carbs': '32g', 'fat': '4g', 'fiber': '6g'},
            'ingredients': ['1 cup steel cut oats', '2 cups water', '1/2 cup mixed berries', '1 tbsp honey', 'Pinch of salt'],
            'steps': ['Boil water with salt', 'Add oats and simmer 15 mins', 'Stir occasionally', 'Top with berries and honey'],
            'chef_tips': ['Soak oats overnight for faster cooking'], 'variations': ['Add nuts for protein'],
            'storage': 'Refrigerate up to 3 days', 'health_benefits': 'High fiber, antioxidants from berries'
        },
        'greek yogurt': {
            'name': 'Greek Yogurt Parfait',
            'prep_time': '5 mins', 'cook_time': '0 mins', 'serves': '1 person', 'difficulty': 'Easy',
            'cuisine': 'Mediterranean', 'calories': 100,
            'nutrition': {'protein': '15g', 'carbs': '6g', 'fat': '0g', 'fiber': '0g'},
            'ingredients': ['150g Greek yogurt', '1 tbsp honey', '1/4 cup granola', 'Fresh fruits'],
            'steps': ['Layer yogurt in bowl', 'Drizzle honey', 'Add granola', 'Top with fruits'],
            'chef_tips': ['Use thick Greek yogurt for best texture'], 'variations': ['Add chia seeds'],
            'storage': 'Best consumed fresh', 'health_benefits': 'High protein, probiotics'
        },
        'whole grain toast': {
            'name': 'Avocado Whole Grain Toast',
            'prep_time': '5 mins', 'cook_time': '3 mins', 'serves': '1 person', 'difficulty': 'Easy',
            'cuisine': 'Modern', 'calories': 160,
            'nutrition': {'protein': '6g', 'carbs': '30g', 'fat': '3g', 'fiber': '5g'},
            'ingredients': ['2 slices whole grain bread', '1 ripe avocado', 'Salt and pepper', 'Lemon juice'],
            'steps': ['Toast bread until golden', 'Mash avocado with lemon', 'Season with salt and pepper', 'Spread on toast'],
            'chef_tips': ['Choose ripe but firm avocado'], 'variations': ['Add tomato slices'],
            'storage': 'Consume immediately', 'health_benefits': 'Healthy fats, fiber'
        },
        'scrambled eggs': {
            'name': 'Fluffy Scrambled Eggs',
            'prep_time': '2 mins', 'cook_time': '5 mins', 'serves': '1 person', 'difficulty': 'Easy',
            'cuisine': 'International', 'calories': 140,
            'nutrition': {'protein': '12g', 'carbs': '1g', 'fat': '10g', 'fiber': '0g'},
            'ingredients': ['2 large eggs', '2 tbsp milk', '1 tbsp butter', 'Salt and pepper'],
            'steps': ['Beat eggs with milk', 'Heat butter in pan', 'Add eggs, stir gently', 'Cook until just set'],
            'chef_tips': ['Low heat for creamiest texture'], 'variations': ['Add cheese or herbs'],
            'storage': 'Best served immediately', 'health_benefits': 'Complete protein, vitamins'
        },
        'stuffed paratha with ghee': {
            'name': 'Aloo Stuffed Paratha',
            'prep_time': '20 mins', 'cook_time': '15 mins', 'serves': '2 people', 'difficulty': 'Medium',
            'cuisine': 'Indian', 'calories': 420,
            'nutrition': {'protein': '10g', 'carbs': '50g', 'fat': '20g', 'fiber': '4g'},
            'ingredients': ['2 cups wheat flour', '3 boiled potatoes', '1 tsp cumin seeds', '2 tbsp ghee', 'Spices'],
            'steps': ['Make dough with flour', 'Prepare spiced potato filling', 'Stuff and roll paratha', 'Cook with ghee'],
            'chef_tips': ['Keep dough soft for easy rolling'], 'variations': ['Try paneer or cauliflower filling'],
            'storage': 'Best served hot', 'health_benefits': 'Energy from carbs, healthy fats'
        },
        
        # Lunch Items
        'brown rice bowl': {
            'name': 'Nutritious Brown Rice Bowl',
            'prep_time': '10 mins', 'cook_time': '25 mins', 'serves': '2 people', 'difficulty': 'Easy',
            'cuisine': 'Healthy', 'calories': 220,
            'nutrition': {'protein': '5g', 'carbs': '45g', 'fat': '2g', 'fiber': '4g'},
            'ingredients': ['1 cup brown rice', '2 cups water', '1 tsp salt', '1 tbsp olive oil'],
            'steps': ['Rinse rice thoroughly', 'Boil water with salt', 'Add rice, simmer 25 mins', 'Fluff with fork'],
            'chef_tips': ['Soak rice for 30 mins for better texture'], 'variations': ['Add vegetables while cooking'],
            'storage': 'Refrigerate up to 4 days', 'health_benefits': 'Whole grain, fiber, B vitamins'
        },
        'tandoori chicken': {
            'name': 'Tandoori Chicken',
            'prep_time': '30 mins', 'cook_time': '20 mins', 'serves': '2 people', 'difficulty': 'Medium',
            'cuisine': 'Indian', 'calories': 200,
            'nutrition': {'protein': '30g', 'carbs': '3g', 'fat': '8g', 'fiber': '1g'},
            'ingredients': ['500g chicken', '1 cup yogurt', '2 tbsp tandoori masala', '1 tbsp ginger-garlic paste', 'Lemon juice'],
            'steps': ['Marinate chicken in yogurt and spices', 'Let sit 30 mins', 'Grill or bake at 200°C', 'Cook 20 mins until done'],
            'chef_tips': ['Marinate longer for better flavor'], 'variations': ['Use chicken thighs for juiciness'],
            'storage': 'Refrigerate up to 3 days', 'health_benefits': 'High protein, low carb'
        },
        'tandoori paneer tikka': {
            'name': 'Tandoori Paneer Tikka',
            'prep_time': '20 mins', 'cook_time': '15 mins', 'serves': '2 people', 'difficulty': 'Medium',
            'cuisine': 'Indian', 'calories': 180,
            'nutrition': {'protein': '18g', 'carbs': '5g', 'fat': '12g', 'fiber': '1g'},
            'ingredients': ['250g paneer cubes', '1/2 cup yogurt', '1 tbsp tandoori masala', 'Bell peppers', 'Onions'],
            'steps': ['Cut paneer and vegetables', 'Marinate in yogurt and spices', 'Thread on skewers', 'Grill until golden'],
            'chef_tips': ['Don\'t over-marinate paneer'], 'variations': ['Add different vegetables'],
            'storage': 'Best served fresh', 'health_benefits': 'Vegetarian protein, calcium'
        },
        'chana dal': {
            'name': 'Chana Dal Curry',
            'prep_time': '10 mins', 'cook_time': '30 mins', 'serves': '3 people', 'difficulty': 'Easy',
            'cuisine': 'Indian', 'calories': 140,
            'nutrition': {'protein': '10g', 'carbs': '22g', 'fat': '3g', 'fiber': '8g'},
            'ingredients': ['1 cup chana dal', '2 cups water', '1 onion', '2 tomatoes', 'Spices', '1 tbsp oil'],
            'steps': ['Wash and soak dal', 'Pressure cook with water', 'Prepare tempering', 'Mix and simmer'],
            'chef_tips': ['Soak dal for faster cooking'], 'variations': ['Add vegetables for nutrition'],
            'storage': 'Refrigerate up to 4 days', 'health_benefits': 'Plant protein, fiber, folate'
        },
        'seasonal vegetables': {
            'name': 'Mixed Seasonal Vegetables',
            'prep_time': '15 mins', 'cook_time': '15 mins', 'serves': '2 people', 'difficulty': 'Easy',
            'cuisine': 'Indian', 'calories': 70,
            'nutrition': {'protein': '3g', 'carbs': '14g', 'fat': '1g', 'fiber': '5g'},
            'ingredients': ['2 cups mixed vegetables', '1 tbsp oil', '1 tsp cumin seeds', 'Turmeric', 'Salt'],
            'steps': ['Chop vegetables uniformly', 'Heat oil, add cumin', 'Add vegetables and spices', 'Cook until tender'],
            'chef_tips': ['Don\'t overcook to retain nutrients'], 'variations': ['Use seasonal produce'],
            'storage': 'Best consumed fresh', 'health_benefits': 'Vitamins, minerals, antioxidants'
        },
        
        # Dinner Items
        'chapati': {
            'name': 'Soft Whole Wheat Chapati',
            'prep_time': '20 mins', 'cook_time': '15 mins', 'serves': '4 people', 'difficulty': 'Medium',
            'cuisine': 'Indian', 'calories': 70,
            'nutrition': {'protein': '2g', 'carbs': '14g', 'fat': '1g', 'fiber': '2g'},
            'ingredients': ['2 cups whole wheat flour', '3/4 cup water', '1/2 tsp salt', '1 tsp oil'],
            'steps': ['Mix flour, salt, oil', 'Add water gradually', 'Knead soft dough', 'Roll and cook on tawa'],
            'chef_tips': ['Keep dough covered to prevent drying'], 'variations': ['Add herbs to dough'],
            'storage': 'Store in cloth, consume within day', 'health_benefits': 'Whole grain, fiber'
        },
        'mixed vegetable curry': {
            'name': 'Mixed Vegetable Curry',
            'prep_time': '15 mins', 'cook_time': '20 mins', 'serves': '3 people', 'difficulty': 'Easy',
            'cuisine': 'Indian', 'calories': 120,
            'nutrition': {'protein': '4g', 'carbs': '18g', 'fat': '5g', 'fiber': '6g'},
            'ingredients': ['3 cups mixed vegetables', '1 onion', '2 tomatoes', 'Coconut milk', 'Spices', '2 tbsp oil'],
            'steps': ['Sauté onions until golden', 'Add tomatoes and spices', 'Add vegetables and coconut milk', 'Simmer until cooked'],
            'chef_tips': ['Cut vegetables uniformly for even cooking'], 'variations': ['Add protein like paneer'],
            'storage': 'Refrigerate up to 3 days', 'health_benefits': 'Rich in vitamins and fiber'
        },
        'fresh curd': {
            'name': 'Homemade Fresh Curd',
            'prep_time': '5 mins', 'cook_time': '0 mins', 'serves': '2 people', 'difficulty': 'Easy',
            'cuisine': 'Indian', 'calories': 80,
            'nutrition': {'protein': '5g', 'carbs': '6g', 'fat': '4g', 'fiber': '0g'},
            'ingredients': ['1 cup fresh curd', '1 tsp roasted cumin powder', 'Salt to taste', 'Mint leaves'],
            'steps': ['Whisk curd until smooth', 'Add cumin powder and salt', 'Garnish with mint', 'Serve chilled'],
            'chef_tips': ['Use fresh, thick curd'], 'variations': ['Add chopped cucumber'],
            'storage': 'Consume within 2 days', 'health_benefits': 'Probiotics, calcium, protein'
        },
        
        # Snacks
        'seasonal fruit bowl': {
            'name': 'Fresh Seasonal Fruit Bowl',
            'prep_time': '10 mins', 'cook_time': '0 mins', 'serves': '1 person', 'difficulty': 'Easy',
            'cuisine': 'Healthy', 'calories': 120,
            'nutrition': {'protein': '2g', 'carbs': '30g', 'fat': '0g', 'fiber': '5g'},
            'ingredients': ['1 cup mixed seasonal fruits', '1 tsp lemon juice', '1 tsp honey', 'Mint leaves'],
            'steps': ['Wash and chop fruits', 'Mix in a bowl', 'Add lemon juice and honey', 'Garnish with mint'],
            'chef_tips': ['Use ripe, fresh fruits'], 'variations': ['Add nuts for protein'],
            'storage': 'Best consumed immediately', 'health_benefits': 'Vitamins, antioxidants, natural sugars'
        },
        'mixed nuts & dates': {
            'name': 'Energy Mix - Nuts & Dates',
            'prep_time': '5 mins', 'cook_time': '0 mins', 'serves': '1 person', 'difficulty': 'Easy',
            'cuisine': 'Healthy', 'calories': 320,
            'nutrition': {'protein': '10g', 'carbs': '35g', 'fat': '18g', 'fiber': '6g'},
            'ingredients': ['1/4 cup almonds', '1/4 cup walnuts', '4-5 dates', '2 tbsp raisins'],
            'steps': ['Chop dates if large', 'Mix all ingredients', 'Store in airtight container', 'Consume as needed'],
            'chef_tips': ['Soak almonds overnight for better digestion'], 'variations': ['Add different nuts'],
            'storage': 'Store in cool, dry place', 'health_benefits': 'Healthy fats, natural energy'
        }
    }
    
    # Search for recipe by food name (case insensitive)
    food_key = food_name.lower().strip()
    
    # Direct match
    if food_key in comprehensive_recipes:
        return comprehensive_recipes[food_key]
    
    # Partial match
    for key, recipe in comprehensive_recipes.items():
        if food_key in key or any(word in key for word in food_key.split()):
            return recipe
    
    # If no specific recipe found, generate dynamic recipe
    return generate_dynamic_recipe(food_name)

@app.route('/predict_camera', methods=['POST'])
@login_required
def predict_camera():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data'}), 400

        # Strip base64 header and decode
        b64 = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(b64)

        # Save to temp file, predict, then delete
        tmp_path = os.path.join(UPLOAD_FOLDER, f'cam_{int(time.time()*1000)}.jpg')
        with open(tmp_path, 'wb') as f:
            f.write(img_bytes)

        try:
            pred = prepare_image(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        category = 'Vegetable' if pred in vegetables else 'Fruit'
        nutrition = get_nutrition_info(pred)
        recipes  = get_food_recipes(pred)

        if 'food_history' not in session:
            session['food_history'] = []
        session['food_history'].append({
            'food': pred, 'category': category,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M')
        })
        session.modified = True

        return jsonify({
            'predicted': pred,
            'category': category,
            'nutrition': nutrition,
            'recipes': recipes
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/bmi')
def bmi():
    return render_template('bmi.html')

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if 'user_profile' not in session:
        session['user_profile'] = {'name': 'Guest User', 'email': '', 'phone': '', 'allergies': '', 'diet_preference': 'veg', 'fitness_goal': 'general_fitness'}
    
    if request.method == 'POST':
        session['user_profile'] = {
            'name': request.form.get('name', 'Guest User'),
            'email': request.form.get('email', ''),
            'phone': request.form.get('phone', ''),
            'allergies': request.form.get('allergies', ''),
            'diet_preference': request.form.get('diet_preference', 'veg'),
            'fitness_goal': request.form.get('fitness_goal', 'general_fitness')
        }
        session.modified = True
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('profile.html', profile=session['user_profile'])

@app.route('/workout', methods=['GET', 'POST'])
@login_required
def workout():
    if 'workout_history' not in session:
        session['workout_history'] = []
    
    workout_plan = None
    
    if request.method == 'POST':
        goal = request.form.get('workout_goal')
        level = request.form.get('fitness_level')
        duration = request.form.get('duration')
        
        workout_plan = generate_workout_plan(goal, level, duration)
        
        session['workout_history'].append({
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'goal': goal,
            'level': level,
            'duration': duration
        })
        session.modified = True
    
    return render_template('workout.html', workout_plan=workout_plan, workout_history=session.get('workout_history', [])[-5:], get_exercise_icon=get_exercise_icon)

def get_exercise_icon(exercise_name):
    # Convert exercise name to image filename format
    img_name = exercise_name.lower().replace(' ', '_').replace('-', '_') + '.png'
    # Return local image path
    return url_for('static', filename=f'exercise_images/{img_name}')

def generate_workout_plan(goal, level, duration):
    workouts = {
        'chest_fat': {
            'beginner': {
                '15': [
                    {'exercise': 'Wall Push-ups', 'sets': 2, 'reps': 8, 'rest': '45 sec', 'calories': 15},
                    {'exercise': 'Incline Push-ups', 'sets': 2, 'reps': 6, 'rest': '45 sec', 'calories': 12},
                    {'exercise': 'Arm Circles', 'sets': 2, 'reps': 15, 'rest': '30 sec', 'calories': 8},
                    {'exercise': 'Chest Stretch', 'sets': 2, 'reps': '30 sec', 'rest': '15 sec', 'calories': 5}
                ],
                '30': [
                    {'exercise': 'Push-ups', 'sets': 3, 'reps': 12, 'rest': '60 sec', 'calories': 30},
                    {'exercise': 'Incline Push-ups', 'sets': 3, 'reps': 10, 'rest': '60 sec', 'calories': 25},
                    {'exercise': 'Chest Dips', 'sets': 3, 'reps': 8, 'rest': '60 sec', 'calories': 28},
                    {'exercise': 'Wide Push-ups', 'sets': 2, 'reps': 10, 'rest': '60 sec', 'calories': 22},
                    {'exercise': 'Chest Fly', 'sets': 2, 'reps': 12, 'rest': '60 sec', 'calories': 18}
                ],
                '45': [
                    {'exercise': 'Push-ups', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 40},
                    {'exercise': 'Incline Push-ups', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 35},
                    {'exercise': 'Chest Dips', 'sets': 4, 'reps': 10, 'rest': '60 sec', 'calories': 38},
                    {'exercise': 'Diamond Push-ups', 'sets': 3, 'reps': 8, 'rest': '60 sec', 'calories': 32},
                    {'exercise': 'Wide Push-ups', 'sets': 3, 'reps': 12, 'rest': '60 sec', 'calories': 28},
                    {'exercise': 'Chest Fly', 'sets': 3, 'reps': 15, 'rest': '60 sec', 'calories': 25},
                    {'exercise': 'Push-up Hold', 'sets': 2, 'reps': '20 sec', 'rest': '45 sec', 'calories': 15}
                ],
                '60': [
                    {'exercise': 'Push-ups', 'sets': 5, 'reps': 15, 'rest': '90 sec', 'calories': 50},
                    {'exercise': 'Incline Push-ups', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 35},
                    {'exercise': 'Chest Dips', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 45},
                    {'exercise': 'Diamond Push-ups', 'sets': 4, 'reps': 10, 'rest': '60 sec', 'calories': 40},
                    {'exercise': 'Chest Fly', 'sets': 3, 'reps': 12, 'rest': '60 sec', 'calories': 30},
                    {'exercise': 'Wide Push-ups', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 35},
                    {'exercise': 'Decline Push-ups', 'sets': 3, 'reps': 10, 'rest': '60 sec', 'calories': 32},
                    {'exercise': 'Push-up to T', 'sets': 3, 'reps': 8, 'rest': '60 sec', 'calories': 28},
                    {'exercise': 'Chest Squeeze', 'sets': 2, 'reps': 15, 'rest': '45 sec', 'calories': 18}
                ]
            },
            'intermediate': {
                '15': [
                    {'exercise': 'Push-ups', 'sets': 3, 'reps': 12, 'rest': '45 sec', 'calories': 25},
                    {'exercise': 'Chest Dips', 'sets': 2, 'reps': 10, 'rest': '45 sec', 'calories': 20}
                ],
                '30': [
                    {'exercise': 'Push-ups', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 40},
                    {'exercise': 'Chest Dips', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 45},
                    {'exercise': 'Diamond Push-ups', 'sets': 3, 'reps': 10, 'rest': '60 sec', 'calories': 35}
                ],
                '45': [
                    {'exercise': 'Push-ups', 'sets': 5, 'reps': 18, 'rest': '60 sec', 'calories': 55},
                    {'exercise': 'Chest Dips', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 50},
                    {'exercise': 'Diamond Push-ups', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 45},
                    {'exercise': 'Decline Push-ups', 'sets': 3, 'reps': 10, 'rest': '60 sec', 'calories': 40}
                ],
                '60': [
                    {'exercise': 'Push-ups', 'sets': 6, 'reps': 20, 'rest': '90 sec', 'calories': 70},
                    {'exercise': 'Chest Dips', 'sets': 5, 'reps': 15, 'rest': '60 sec', 'calories': 60},
                    {'exercise': 'Diamond Push-ups', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 50},
                    {'exercise': 'Decline Push-ups', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 48},
                    {'exercise': 'Chest Fly', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 42}
                ]
            },
            'advanced': {
                '15': [
                    {'exercise': 'Diamond Push-ups', 'sets': 3, 'reps': 15, 'rest': '30 sec', 'calories': 35},
                    {'exercise': 'Decline Push-ups', 'sets': 3, 'reps': 12, 'rest': '30 sec', 'calories': 30}
                ],
                '30': [
                    {'exercise': 'One-arm Push-ups', 'sets': 4, 'reps': 8, 'rest': '60 sec', 'calories': 50},
                    {'exercise': 'Archer Push-ups', 'sets': 4, 'reps': 10, 'rest': '60 sec', 'calories': 45},
                    {'exercise': 'Explosive Push-ups', 'sets': 3, 'reps': 12, 'rest': '60 sec', 'calories': 40}
                ],
                '45': [
                    {'exercise': 'One-arm Push-ups', 'sets': 5, 'reps': 10, 'rest': '90 sec', 'calories': 65},
                    {'exercise': 'Archer Push-ups', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 55},
                    {'exercise': 'Explosive Push-ups', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 50},
                    {'exercise': 'Handstand Push-ups', 'sets': 3, 'reps': 8, 'rest': '90 sec', 'calories': 45}
                ],
                '60': [
                    {'exercise': 'One-arm Push-ups', 'sets': 6, 'reps': 12, 'rest': '90 sec', 'calories': 80},
                    {'exercise': 'Archer Push-ups', 'sets': 5, 'reps': 15, 'rest': '60 sec', 'calories': 70},
                    {'exercise': 'Explosive Push-ups', 'sets': 5, 'reps': 18, 'rest': '60 sec', 'calories': 65},
                    {'exercise': 'Handstand Push-ups', 'sets': 4, 'reps': 10, 'rest': '90 sec', 'calories': 60},
                    {'exercise': 'Planche Push-ups', 'sets': 3, 'reps': 6, 'rest': '120 sec', 'calories': 45}
                ]
            }
        },
        'belly_fat': {
            'beginner': {
                '15': [
                    {'exercise': 'Crunches', 'sets': 2, 'reps': 10, 'rest': '30 sec', 'calories': 12},
                    {'exercise': 'Plank Hold', 'sets': 2, 'reps': '20 sec', 'rest': '30 sec', 'calories': 10},
                    {'exercise': 'Knee Raises', 'sets': 2, 'reps': 8, 'rest': '30 sec', 'calories': 8},
                    {'exercise': 'Standing Side Bends', 'sets': 2, 'reps': 10, 'rest': '30 sec', 'calories': 6}
                ],
                '30': [
                    {'exercise': 'Crunches', 'sets': 3, 'reps': 15, 'rest': '45 sec', 'calories': 20},
                    {'exercise': 'Leg Raises', 'sets': 3, 'reps': 12, 'rest': '45 sec', 'calories': 25},
                    {'exercise': 'Plank Hold', 'sets': 3, 'reps': '30 sec', 'rest': '30 sec', 'calories': 18},
                    {'exercise': 'Bicycle Crunches', 'sets': 3, 'reps': 20, 'rest': '45 sec', 'calories': 22},
                    {'exercise': 'Side Plank', 'sets': 2, 'reps': '15 sec', 'rest': '30 sec', 'calories': 12}
                ],
                '45': [
                    {'exercise': 'Crunches', 'sets': 4, 'reps': 18, 'rest': '45 sec', 'calories': 28},
                    {'exercise': 'Leg Raises', 'sets': 4, 'reps': 15, 'rest': '45 sec', 'calories': 35},
                    {'exercise': 'Plank Hold', 'sets': 4, 'reps': '45 sec', 'rest': '30 sec', 'calories': 25},
                    {'exercise': 'Bicycle Crunches', 'sets': 3, 'reps': 20, 'rest': '45 sec', 'calories': 22}
                ],
                '60': [
                    {'exercise': 'Crunches', 'sets': 5, 'reps': 20, 'rest': '60 sec', 'calories': 35},
                    {'exercise': 'Leg Raises', 'sets': 4, 'reps': 18, 'rest': '45 sec', 'calories': 40},
                    {'exercise': 'Plank Hold', 'sets': 4, 'reps': '60 sec', 'rest': '30 sec', 'calories': 30},
                    {'exercise': 'Bicycle Crunches', 'sets': 4, 'reps': 25, 'rest': '45 sec', 'calories': 28},
                    {'exercise': 'Russian Twists', 'sets': 3, 'reps': 20, 'rest': '45 sec', 'calories': 22}
                ]
            },
            'intermediate': {
                '15': [
                    {'exercise': 'Bicycle Crunches', 'sets': 3, 'reps': 15, 'rest': '30 sec', 'calories': 18},
                    {'exercise': 'Plank Hold', 'sets': 2, 'reps': '45 sec', 'rest': '30 sec', 'calories': 15}
                ],
                '30': [
                    {'exercise': 'Mountain Climbers', 'sets': 4, 'reps': 20, 'rest': '45 sec', 'calories': 35},
                    {'exercise': 'Russian Twists', 'sets': 4, 'reps': 20, 'rest': '45 sec', 'calories': 28},
                    {'exercise': 'Dead Bug', 'sets': 3, 'reps': 15, 'rest': '45 sec', 'calories': 22}
                ],
                '45': [
                    {'exercise': 'Mountain Climbers', 'sets': 5, 'reps': 25, 'rest': '45 sec', 'calories': 45},
                    {'exercise': 'Russian Twists', 'sets': 4, 'reps': 25, 'rest': '45 sec', 'calories': 35},
                    {'exercise': 'Dead Bug', 'sets': 4, 'reps': 18, 'rest': '45 sec', 'calories': 28},
                    {'exercise': 'Hollow Body Hold', 'sets': 3, 'reps': '30 sec', 'rest': '60 sec', 'calories': 25}
                ],
                '60': [
                    {'exercise': 'Mountain Climbers', 'sets': 6, 'reps': 30, 'rest': '60 sec', 'calories': 60},
                    {'exercise': 'Russian Twists', 'sets': 5, 'reps': 30, 'rest': '45 sec', 'calories': 42},
                    {'exercise': 'Dead Bug', 'sets': 4, 'reps': 20, 'rest': '45 sec', 'calories': 32},
                    {'exercise': 'Hollow Body Hold', 'sets': 4, 'reps': '45 sec', 'rest': '60 sec', 'calories': 35},
                    {'exercise': 'V-ups', 'sets': 3, 'reps': 15, 'rest': '60 sec', 'calories': 28}
                ]
            },
            'advanced': {
                '15': [
                    {'exercise': 'Dragon Flag', 'sets': 3, 'reps': 8, 'rest': '60 sec', 'calories': 25},
                    {'exercise': 'L-sit Hold', 'sets': 3, 'reps': '15 sec', 'rest': '60 sec', 'calories': 20}
                ],
                '30': [
                    {'exercise': 'Dragon Flag', 'sets': 4, 'reps': 10, 'rest': '90 sec', 'calories': 40},
                    {'exercise': 'Human Flag', 'sets': 3, 'reps': '10 sec', 'rest': '90 sec', 'calories': 35},
                    {'exercise': 'Hanging Leg Raises', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 32}
                ],
                '45': [
                    {'exercise': 'Dragon Flag', 'sets': 5, 'reps': 12, 'rest': '90 sec', 'calories': 55},
                    {'exercise': 'Human Flag', 'sets': 4, 'reps': '15 sec', 'rest': '90 sec', 'calories': 48},
                    {'exercise': 'Hanging Leg Raises', 'sets': 4, 'reps': 18, 'rest': '60 sec', 'calories': 40},
                    {'exercise': 'Windshield Wipers', 'sets': 3, 'reps': 12, 'rest': '60 sec', 'calories': 35}
                ],
                '60': [
                    {'exercise': 'Dragon Flag', 'sets': 6, 'reps': 15, 'rest': '90 sec', 'calories': 70},
                    {'exercise': 'Human Flag', 'sets': 5, 'reps': '20 sec', 'rest': '90 sec', 'calories': 60},
                    {'exercise': 'Hanging Leg Raises', 'sets': 5, 'reps': 20, 'rest': '60 sec', 'calories': 50},
                    {'exercise': 'Windshield Wipers', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 45},
                    {'exercise': 'Front Lever Hold', 'sets': 3, 'reps': '10 sec', 'rest': '120 sec', 'calories': 40}
                ]
            }
        },
        'general_fitness': {
            'beginner': {
                '15': [
                    {'exercise': 'Walking', 'sets': 1, 'reps': '5 min', 'rest': '0', 'calories': 20},
                    {'exercise': 'Wall Push-ups', 'sets': 2, 'reps': 8, 'rest': '45 sec', 'calories': 15},
                    {'exercise': 'Arm Swings', 'sets': 2, 'reps': 10, 'rest': '30 sec', 'calories': 8},
                    {'exercise': 'Calf Raises', 'sets': 2, 'reps': 12, 'rest': '30 sec', 'calories': 10},
                    {'exercise': 'Gentle Stretching', 'sets': 1, 'reps': '3 min', 'rest': '0', 'calories': 5}
                ],
                '30': [
                    {'exercise': 'Walking', 'sets': 1, 'reps': '10 min', 'rest': '0', 'calories': 40},
                    {'exercise': 'Bodyweight Squats', 'sets': 3, 'reps': 12, 'rest': '45 sec', 'calories': 25},
                    {'exercise': 'Wall Push-ups', 'sets': 3, 'reps': 10, 'rest': '45 sec', 'calories': 20},
                    {'exercise': 'Lunges', 'sets': 2, 'reps': 10, 'rest': '45 sec', 'calories': 18},
                    {'exercise': 'Arm Circles', 'sets': 2, 'reps': 15, 'rest': '30 sec', 'calories': 10},
                    {'exercise': 'Standing Knee Lifts', 'sets': 2, 'reps': 12, 'rest': '30 sec', 'calories': 12}
                ],
                '45': [
                    {'exercise': 'Brisk Walking', 'sets': 1, 'reps': '15 min', 'rest': '0', 'calories': 60},
                    {'exercise': 'Bodyweight Squats', 'sets': 4, 'reps': 15, 'rest': '45 sec', 'calories': 35},
                    {'exercise': 'Push-ups', 'sets': 3, 'reps': 10, 'rest': '60 sec', 'calories': 25},
                    {'exercise': 'Lunges', 'sets': 3, 'reps': 12, 'rest': '45 sec', 'calories': 30},
                    {'exercise': 'Glute Bridges', 'sets': 3, 'reps': 15, 'rest': '45 sec', 'calories': 20},
                    {'exercise': 'Modified Burpees', 'sets': 2, 'reps': 8, 'rest': '60 sec', 'calories': 22},
                    {'exercise': 'Standing Calf Raises', 'sets': 3, 'reps': 20, 'rest': '30 sec', 'calories': 15},
                    {'exercise': 'Tricep Dips', 'sets': 2, 'reps': 10, 'rest': '45 sec', 'calories': 18}
                ],
                '60': [
                    {'exercise': 'Jogging', 'sets': 1, 'reps': '20 min', 'rest': '0', 'calories': 100},
                    {'exercise': 'Bodyweight Squats', 'sets': 4, 'reps': 18, 'rest': '45 sec', 'calories': 40},
                    {'exercise': 'Push-ups', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 35},
                    {'exercise': 'Lunges', 'sets': 4, 'reps': 15, 'rest': '45 sec', 'calories': 38},
                    {'exercise': 'Plank Hold', 'sets': 3, 'reps': '30 sec', 'rest': '30 sec', 'calories': 18},
                    {'exercise': 'Glute Bridges', 'sets': 4, 'reps': 20, 'rest': '45 sec', 'calories': 28},
                    {'exercise': 'Modified Burpees', 'sets': 3, 'reps': 10, 'rest': '60 sec', 'calories': 32},
                    {'exercise': 'Standing Calf Raises', 'sets': 4, 'reps': 25, 'rest': '30 sec', 'calories': 20},
                    {'exercise': 'Tricep Dips', 'sets': 3, 'reps': 12, 'rest': '45 sec', 'calories': 25},
                    {'exercise': 'Step-ups', 'sets': 3, 'reps': 15, 'rest': '45 sec', 'calories': 22}
                ]
            },
            'intermediate': {
                '15': [
                    {'exercise': 'Jumping Jacks', 'sets': 3, 'reps': 20, 'rest': '30 sec', 'calories': 25},
                    {'exercise': 'Push-ups', 'sets': 3, 'reps': 12, 'rest': '45 sec', 'calories': 22}
                ],
                '30': [
                    {'exercise': 'Burpees', 'sets': 4, 'reps': 10, 'rest': '60 sec', 'calories': 45},
                    {'exercise': 'Jump Squats', 'sets': 4, 'reps': 15, 'rest': '45 sec', 'calories': 40},
                    {'exercise': 'Mountain Climbers', 'sets': 3, 'reps': 20, 'rest': '45 sec', 'calories': 30},
                    {'exercise': 'Push-up to T', 'sets': 3, 'reps': 12, 'rest': '60 sec', 'calories': 28},
                    {'exercise': 'Squat Thrusts', 'sets': 3, 'reps': 10, 'rest': '45 sec', 'calories': 25},
                    {'exercise': 'Plank Jacks', 'sets': 3, 'reps': 15, 'rest': '45 sec', 'calories': 22}
                ],
                '45': [
                    {'exercise': 'Burpees', 'sets': 5, 'reps': 12, 'rest': '60 sec', 'calories': 60},
                    {'exercise': 'Jump Squats', 'sets': 4, 'reps': 18, 'rest': '45 sec', 'calories': 48},
                    {'exercise': 'Mountain Climbers', 'sets': 4, 'reps': 25, 'rest': '45 sec', 'calories': 40},
                    {'exercise': 'High Knees', 'sets': 3, 'reps': '30 sec', 'rest': '30 sec', 'calories': 25},
                    {'exercise': 'Push-up to T', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 35},
                    {'exercise': 'Squat Thrusts', 'sets': 4, 'reps': 12, 'rest': '45 sec', 'calories': 32},
                    {'exercise': 'Plank Jacks', 'sets': 4, 'reps': 20, 'rest': '45 sec', 'calories': 28},
                    {'exercise': 'Star Jumps', 'sets': 3, 'reps': 15, 'rest': '45 sec', 'calories': 22},
                    {'exercise': 'Bear Crawl', 'sets': 2, 'reps': '15 steps', 'rest': '60 sec', 'calories': 20}
                ],
                '60': [
                    {'exercise': 'Burpees', 'sets': 6, 'reps': 15, 'rest': '60 sec', 'calories': 80},
                    {'exercise': 'Jump Squats', 'sets': 5, 'reps': 20, 'rest': '45 sec', 'calories': 60},
                    {'exercise': 'Mountain Climbers', 'sets': 5, 'reps': 30, 'rest': '45 sec', 'calories': 50},
                    {'exercise': 'High Knees', 'sets': 4, 'reps': '45 sec', 'rest': '30 sec', 'calories': 35},
                    {'exercise': 'Bear Crawl', 'sets': 3, 'reps': '20 steps', 'rest': '60 sec', 'calories': 30},
                    {'exercise': 'Push-up to T', 'sets': 5, 'reps': 18, 'rest': '60 sec', 'calories': 42},
                    {'exercise': 'Squat Thrusts', 'sets': 5, 'reps': 15, 'rest': '45 sec', 'calories': 38},
                    {'exercise': 'Plank Jacks', 'sets': 5, 'reps': 25, 'rest': '45 sec', 'calories': 35},
                    {'exercise': 'Star Jumps', 'sets': 4, 'reps': 20, 'rest': '45 sec', 'calories': 32},
                    {'exercise': 'Lateral Lunges', 'sets': 4, 'reps': 15, 'rest': '45 sec', 'calories': 28},
                    {'exercise': 'Inchworms', 'sets': 3, 'reps': 10, 'rest': '60 sec', 'calories': 25}
                ]
            },
            'advanced': {
                '15': [
                    {'exercise': 'Pistol Squats', 'sets': 3, 'reps': 8, 'rest': '60 sec', 'calories': 30},
                    {'exercise': 'Handstand Push-ups', 'sets': 3, 'reps': 6, 'rest': '90 sec', 'calories': 25}
                ],
                '30': [
                    {'exercise': 'Muscle-ups', 'sets': 4, 'reps': 6, 'rest': '90 sec', 'calories': 50},
                    {'exercise': 'Pistol Squats', 'sets': 4, 'reps': 10, 'rest': '60 sec', 'calories': 45},
                    {'exercise': 'Handstand Push-ups', 'sets': 3, 'reps': 8, 'rest': '90 sec', 'calories': 35}
                ],
                '45': [
                    {'exercise': 'Muscle-ups', 'sets': 5, 'reps': 8, 'rest': '90 sec', 'calories': 65},
                    {'exercise': 'Pistol Squats', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 55},
                    {'exercise': 'Handstand Push-ups', 'sets': 4, 'reps': 10, 'rest': '90 sec', 'calories': 48},
                    {'exercise': 'Human Flag', 'sets': 3, 'reps': '15 sec', 'rest': '120 sec', 'calories': 40}
                ],
                '60': [
                    {'exercise': 'Muscle-ups', 'sets': 6, 'reps': 10, 'rest': '90 sec', 'calories': 85},
                    {'exercise': 'Pistol Squats', 'sets': 5, 'reps': 15, 'rest': '60 sec', 'calories': 70},
                    {'exercise': 'Handstand Push-ups', 'sets': 5, 'reps': 12, 'rest': '90 sec', 'calories': 60},
                    {'exercise': 'Human Flag', 'sets': 4, 'reps': '20 sec', 'rest': '120 sec', 'calories': 55},
                    {'exercise': 'Planche Push-ups', 'sets': 3, 'reps': 8, 'rest': '120 sec', 'calories': 45}
                ]
            }
        },
        'muscle_gain': {
            'beginner': {
                '15': [
                    {'exercise': 'Push-ups', 'sets': 3, 'reps': 10, 'rest': '60 sec', 'calories': 20},
                    {'exercise': 'Bodyweight Squats', 'sets': 3, 'reps': 12, 'rest': '60 sec', 'calories': 25}
                ],
                '30': [
                    {'exercise': 'Push-ups', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 35},
                    {'exercise': 'Bodyweight Squats', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 40},
                    {'exercise': 'Pike Push-ups', 'sets': 3, 'reps': 8, 'rest': '60 sec', 'calories': 25},
                    {'exercise': 'Glute Bridges', 'sets': 3, 'reps': 15, 'rest': '45 sec', 'calories': 20},
                    {'exercise': 'Tricep Dips', 'sets': 3, 'reps': 10, 'rest': '60 sec', 'calories': 22},
                    {'exercise': 'Wall Sit', 'sets': 2, 'reps': '30 sec', 'rest': '60 sec', 'calories': 15}
                ],
                '45': [
                    {'exercise': 'Push-ups', 'sets': 5, 'reps': 15, 'rest': '90 sec', 'calories': 50},
                    {'exercise': 'Bodyweight Squats', 'sets': 4, 'reps': 18, 'rest': '60 sec', 'calories': 48},
                    {'exercise': 'Pike Push-ups', 'sets': 4, 'reps': 10, 'rest': '60 sec', 'calories': 35},
                    {'exercise': 'Lunges', 'sets': 3, 'reps': 12, 'rest': '60 sec', 'calories': 30}
                ],
                '60': [
                    {'exercise': 'Push-ups', 'sets': 6, 'reps': 18, 'rest': '90 sec', 'calories': 65},
                    {'exercise': 'Bodyweight Squats', 'sets': 5, 'reps': 20, 'rest': '60 sec', 'calories': 60},
                    {'exercise': 'Pike Push-ups', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 42},
                    {'exercise': 'Lunges', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 40},
                    {'exercise': 'Tricep Dips', 'sets': 3, 'reps': 10, 'rest': '60 sec', 'calories': 28}
                ]
            },
            'intermediate': {
                '15': [
                    {'exercise': 'Diamond Push-ups', 'sets': 3, 'reps': 10, 'rest': '60 sec', 'calories': 28},
                    {'exercise': 'Jump Squats', 'sets': 3, 'reps': 12, 'rest': '60 sec', 'calories': 30}
                ],
                '30': [
                    {'exercise': 'Diamond Push-ups', 'sets': 4, 'reps': 12, 'rest': '60 sec', 'calories': 40},
                    {'exercise': 'Jump Squats', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 45},
                    {'exercise': 'Handstand Push-ups', 'sets': 3, 'reps': 6, 'rest': '90 sec', 'calories': 35}
                ],
                '45': [
                    {'exercise': 'Diamond Push-ups', 'sets': 5, 'reps': 15, 'rest': '60 sec', 'calories': 55},
                    {'exercise': 'Jump Squats', 'sets': 4, 'reps': 18, 'rest': '60 sec', 'calories': 55},
                    {'exercise': 'Handstand Push-ups', 'sets': 4, 'reps': 8, 'rest': '90 sec', 'calories': 48},
                    {'exercise': 'Bulgarian Split Squats', 'sets': 3, 'reps': 12, 'rest': '60 sec', 'calories': 38}
                ],
                '60': [
                    {'exercise': 'Diamond Push-ups', 'sets': 6, 'reps': 18, 'rest': '60 sec', 'calories': 70},
                    {'exercise': 'Jump Squats', 'sets': 5, 'reps': 20, 'rest': '60 sec', 'calories': 65},
                    {'exercise': 'Handstand Push-ups', 'sets': 4, 'reps': 10, 'rest': '90 sec', 'calories': 55},
                    {'exercise': 'Bulgarian Split Squats', 'sets': 4, 'reps': 15, 'rest': '60 sec', 'calories': 50},
                    {'exercise': 'Archer Push-ups', 'sets': 3, 'reps': 8, 'rest': '90 sec', 'calories': 40}
                ]
            },
            'advanced': {
                '15': [
                    {'exercise': 'One-arm Push-ups', 'sets': 3, 'reps': 6, 'rest': '90 sec', 'calories': 35},
                    {'exercise': 'Pistol Squats', 'sets': 3, 'reps': 8, 'rest': '90 sec', 'calories': 30}
                ],
                '30': [
                    {'exercise': 'One-arm Push-ups', 'sets': 4, 'reps': 8, 'rest': '90 sec', 'calories': 50},
                    {'exercise': 'Pistol Squats', 'sets': 4, 'reps': 10, 'rest': '90 sec', 'calories': 45},
                    {'exercise': 'Muscle-ups', 'sets': 3, 'reps': 5, 'rest': '120 sec', 'calories': 40}
                ],
                '45': [
                    {'exercise': 'One-arm Push-ups', 'sets': 5, 'reps': 10, 'rest': '90 sec', 'calories': 65},
                    {'exercise': 'Pistol Squats', 'sets': 4, 'reps': 12, 'rest': '90 sec', 'calories': 55},
                    {'exercise': 'Muscle-ups', 'sets': 4, 'reps': 6, 'rest': '120 sec', 'calories': 50},
                    {'exercise': 'Planche Push-ups', 'sets': 3, 'reps': 5, 'rest': '120 sec', 'calories': 40}
                ],
                '60': [
                    {'exercise': 'One-arm Push-ups', 'sets': 6, 'reps': 12, 'rest': '90 sec', 'calories': 80},
                    {'exercise': 'Pistol Squats', 'sets': 5, 'reps': 15, 'rest': '90 sec', 'calories': 70},
                    {'exercise': 'Muscle-ups', 'sets': 5, 'reps': 8, 'rest': '120 sec', 'calories': 65},
                    {'exercise': 'Planche Push-ups', 'sets': 4, 'reps': 6, 'rest': '120 sec', 'calories': 55},
                    {'exercise': 'Human Flag', 'sets': 3, 'reps': '15 sec', 'rest': '120 sec', 'calories': 45}
                ]
            }
        },
        'cardio': {
            'beginner': {
                '15': [
                    {'exercise': 'Marching in Place', 'sets': 1, 'reps': '5 min', 'rest': '0', 'calories': 25},
                    {'exercise': 'Arm Circles', 'sets': 2, 'reps': '30 sec', 'rest': '30 sec', 'calories': 10}
                ],
                '30': [
                    {'exercise': 'Walking', 'sets': 1, 'reps': '15 min', 'rest': '0', 'calories': 60},
                    {'exercise': 'Jumping Jacks', 'sets': 3, 'reps': 20, 'rest': '60 sec', 'calories': 30},
                    {'exercise': 'Step-ups', 'sets': 3, 'reps': 15, 'rest': '60 sec', 'calories': 25},
                    {'exercise': 'Arm Swings', 'sets': 3, 'reps': 20, 'rest': '30 sec', 'calories': 15},
                    {'exercise': 'Marching in Place', 'sets': 2, 'reps': '60 sec', 'rest': '30 sec', 'calories': 12},
                    {'exercise': 'Side Steps', 'sets': 3, 'reps': 15, 'rest': '30 sec', 'calories': 18}
                ],
                '45': [
                    {'exercise': 'Brisk Walking', 'sets': 1, 'reps': '20 min', 'rest': '0', 'calories': 80},
                    {'exercise': 'Jumping Jacks', 'sets': 4, 'reps': 25, 'rest': '60 sec', 'calories': 40},
                    {'exercise': 'Step-ups', 'sets': 4, 'reps': 18, 'rest': '60 sec', 'calories': 35},
                    {'exercise': 'High Knees', 'sets': 3, 'reps': '30 sec', 'rest': '30 sec', 'calories': 25},
                    {'exercise': 'Butt Kicks', 'sets': 3, 'reps': '30 sec', 'rest': '30 sec', 'calories': 22},
                    {'exercise': 'Side Steps', 'sets': 4, 'reps': 20, 'rest': '30 sec', 'calories': 25},
                    {'exercise': 'Arm Circles', 'sets': 3, 'reps': 30, 'rest': '30 sec', 'calories': 18},
                    {'exercise': 'Knee Lifts', 'sets': 3, 'reps': 15, 'rest': '30 sec', 'calories': 15}
                ],
                '60': [
                    {'exercise': 'Jogging', 'sets': 1, 'reps': '25 min', 'rest': '0', 'calories': 125},
                    {'exercise': 'Jumping Jacks', 'sets': 5, 'reps': 30, 'rest': '60 sec', 'calories': 50},
                    {'exercise': 'Step-ups', 'sets': 4, 'reps': 20, 'rest': '60 sec', 'calories': 40},
                    {'exercise': 'High Knees', 'sets': 4, 'reps': '45 sec', 'rest': '30 sec', 'calories': 35},
                    {'exercise': 'Butt Kicks', 'sets': 3, 'reps': '30 sec', 'rest': '30 sec', 'calories': 20},
                    {'exercise': 'Side Steps', 'sets': 5, 'reps': 25, 'rest': '30 sec', 'calories': 32},
                    {'exercise': 'Arm Circles', 'sets': 4, 'reps': 40, 'rest': '30 sec', 'calories': 25},
                    {'exercise': 'Knee Lifts', 'sets': 4, 'reps': 20, 'rest': '30 sec', 'calories': 22},
                    {'exercise': 'Leg Swings', 'sets': 3, 'reps': 15, 'rest': '30 sec', 'calories': 18},
                    {'exercise': 'Calf Raises', 'sets': 4, 'reps': 25, 'rest': '30 sec', 'calories': 20}
                ]
            },
            'intermediate': {
                '15': [
                    {'exercise': 'Burpees', 'sets': 3, 'reps': 8, 'rest': '60 sec', 'calories': 35},
                    {'exercise': 'Mountain Climbers', 'sets': 3, 'reps': 15, 'rest': '45 sec', 'calories': 25}
                ],
                '30': [
                    {'exercise': 'Running', 'sets': 1, 'reps': '15 min', 'rest': '0', 'calories': 90},
                    {'exercise': 'Burpees', 'sets': 4, 'reps': 10, 'rest': '60 sec', 'calories': 45},
                    {'exercise': 'Mountain Climbers', 'sets': 4, 'reps': 20, 'rest': '45 sec', 'calories': 35}
                ],
                '45': [
                    {'exercise': 'Running', 'sets': 1, 'reps': '20 min', 'rest': '0', 'calories': 120},
                    {'exercise': 'Burpees', 'sets': 5, 'reps': 12, 'rest': '60 sec', 'calories': 60},
                    {'exercise': 'Mountain Climbers', 'sets': 4, 'reps': 25, 'rest': '45 sec', 'calories': 40},
                    {'exercise': 'Jump Rope', 'sets': 3, 'reps': '60 sec', 'rest': '60 sec', 'calories': 45}
                ],
                '60': [
                    {'exercise': 'Running', 'sets': 1, 'reps': '25 min', 'rest': '0', 'calories': 150},
                    {'exercise': 'Burpees', 'sets': 6, 'reps': 15, 'rest': '60 sec', 'calories': 80},
                    {'exercise': 'Mountain Climbers', 'sets': 5, 'reps': 30, 'rest': '45 sec', 'calories': 50},
                    {'exercise': 'Jump Rope', 'sets': 4, 'reps': '90 sec', 'rest': '60 sec', 'calories': 60},
                    {'exercise': 'Box Jumps', 'sets': 3, 'reps': 12, 'rest': '90 sec', 'calories': 40}
                ]
            },
            'advanced': {
                '15': [
                    {'exercise': 'Sprint Intervals', 'sets': 5, 'reps': '30 sec', 'rest': '30 sec', 'calories': 50},
                    {'exercise': 'Burpee Box Jumps', 'sets': 3, 'reps': 8, 'rest': '60 sec', 'calories': 45}
                ],
                '30': [
                    {'exercise': 'HIIT Running', 'sets': 1, 'reps': '15 min', 'rest': '0', 'calories': 120},
                    {'exercise': 'Burpee Box Jumps', 'sets': 4, 'reps': 10, 'rest': '60 sec', 'calories': 60},
                    {'exercise': 'Battle Ropes', 'sets': 3, 'reps': '45 sec', 'rest': '60 sec', 'calories': 50}
                ],
                '45': [
                    {'exercise': 'HIIT Running', 'sets': 1, 'reps': '20 min', 'rest': '0', 'calories': 160},
                    {'exercise': 'Burpee Box Jumps', 'sets': 5, 'reps': 12, 'rest': '60 sec', 'calories': 75},
                    {'exercise': 'Battle Ropes', 'sets': 4, 'reps': '60 sec', 'rest': '60 sec', 'calories': 65},
                    {'exercise': 'Plyometric Push-ups', 'sets': 3, 'reps': 10, 'rest': '90 sec', 'calories': 40}
                ],
                '60': [
                    {'exercise': 'HIIT Running', 'sets': 1, 'reps': '25 min', 'rest': '0', 'calories': 200},
                    {'exercise': 'Burpee Box Jumps', 'sets': 6, 'reps': 15, 'rest': '60 sec', 'calories': 95},
                    {'exercise': 'Battle Ropes', 'sets': 5, 'reps': '90 sec', 'rest': '60 sec', 'calories': 85},
                    {'exercise': 'Plyometric Push-ups', 'sets': 4, 'reps': 12, 'rest': '90 sec', 'calories': 55},
                    {'exercise': 'Turkish Get-ups', 'sets': 3, 'reps': 8, 'rest': '90 sec', 'calories': 45}
                ]
            }
        }
    }
    
    # Get workout plan with fallback logic
    goal_workouts = workouts.get(goal, workouts['general_fitness'])
    level_workouts = goal_workouts.get(level, goal_workouts.get('beginner', {}))
    plan = level_workouts.get(duration, level_workouts.get('30', []))
    
    # If still no plan found, use general fitness beginner 30min as ultimate fallback
    if not plan:
        plan = workouts['general_fitness']['beginner']['30']
    
    total_calories = sum(ex['calories'] for ex in plan)
    
    tips = {
        'chest_fat': 'Focus on chest exercises with cardio. Maintain calorie deficit for fat loss!',
        'belly_fat': 'Core exercises + cardio + clean diet = flat belly. Stay consistent!',
        'general_fitness': 'Balance strength, cardio, and flexibility. Consistency beats intensity!'
    }
    
    return {
        'exercises': plan,
        'total_calories': total_calories,
        'tip': tips.get(goal, 'Stay hydrated and listen to your body!'),
        'goal': goal,
        'level': level,
        'duration': duration
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)