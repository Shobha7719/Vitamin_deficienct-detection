import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
# Ensure you have TensorFlow and Keras installed, or comment out if not using AI model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uuid # For unique filenames
from datetime import datetime # For tracking registration time

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['SECRET_KEY'] = '020679' # !! IMPORTANT: CHANGE THIS TO A STRONG, RANDOM KEY !!
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- User Management (Flask-Login and SQLite) ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

DATABASE = 'users.db'

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    return conn

def init_db():
    """Initializes the database by creating the users table if it doesn't exist."""
    with app.app_context(): # Ensure app context for database operations outside of requests
        conn = get_db_connection()
        conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                registered_at TEXT NOT NULL
            );
        ''')
        conn.commit()
        conn.close()

class User(UserMixin):
    """User class for Flask-Login."""
    def __init__(self, id, username, email, password):
        self.id = id
        self.username = username
        self.email = email
        self.password = password

    def get_id(self):
        """Returns the user ID, required by Flask-Login."""
        return str(self.id)

    @staticmethod
    def get(user_id):
        """Retrieves a user by their ID."""
        conn = get_db_connection()
        user_data = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
        conn.close()
        if user_data:
            return User(user_data['id'], user_data['username'], user_data['email'], user_data['password'])
        return None

    @staticmethod
    def get_by_username(username):
        """Retrieves a user by their username."""
        conn = get_db_connection()
        user_data = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        if user_data:
            return User(user_data['id'], user_data['username'], user_data['email'], user_data['password'])
        return None

    @staticmethod
    def get_by_email(email):
        """Retrieves a user by their email address."""
        conn = get_db_connection()
        user_data = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user_data:
            return User(user_data['id'], user_data['username'], user_data['email'], user_data['password'])
        return None

@login_manager.user_loader
def load_user(user_id):
    """Callback for Flask-Login to load a user from the database."""
    return User.get(user_id)

# Initialize the database on application startup
init_db()

# --- Load the AI Model ---
model = None # Initialize model to None
try:
    model = load_model('models/MobileNet_VD_Model.h5')
    print("MobileNet_VD_Model.h5 loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Consider what to do if the model fails to load:
    # - Disable prediction features
    # - Show an error message to the user
    # For now, `model` remains `None`, and `predict_image` handles it.

# Define class indices (ensure this matches your training)
# IMPORTANT: Replace these with the actual class indices from your training script's val_generator.class_indices
CLASS_INDICES = {
    'Vitamin A deficiency': 0,
    'Vitamin B-12 deficiency': 1,
    'Vitamin B1 deficiency': 2,
    'Vitamin B2 deficiency': 3,
    'Vitamin B3 deficiency': 4,
    'Vitamin B9 deficiency': 5,
    'Vitamin C deficiency': 6,
    'Vitamin D deficiency': 7,
    'Vitamin E deficiency': 8,
    'Vitamin K deficiency': 9,
    'zinc, iron, biotin, or protein deficiency': 10
}
# Reverse mapping for prediction output
CLASS_LABELS = {v: k for k, v in CLASS_INDICES.items()}
TARGET_SIZE = (128, 128) # Ensure this matches your model's expected input size

def predict_image(img_path):
    """
    Performs a prediction on an image using the loaded Keras model.
    Args:
        img_path (str): The file path to the image.
    Returns:
        tuple: (predicted_class, confidence, status)
    """
    if model is None:
        return "Model not loaded", 0.0, "error"
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=TARGET_SIZE)
        img_array = image.img_to_array(img) / 255.0 # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension: (1, height, width, 3)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_index]

        predicted_class = CLASS_LABELS.get(predicted_index, "Unknown")

        return predicted_class, float(confidence), "success" # Convert numpy float to Python float for JSON
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction Error", 0.0, f"error: {e}"

# --- Vitamin Data Structure for Description Pages ---
# This dictionary will hold all the detailed information for each vitamin.
# Use concise keys like 'vitamin_a', 'vitamin_b12' to match URL parameters.
VITAMIN_DATA = {
    "vitamin_a": {
        "name": "Vitamin A",
        "overview": "Essential for vision, immune function, and skin health.",
        "description": "Vitamin A is a fat-soluble vitamin crucial for maintaining healthy vision, especially in low light. It also supports immune function, cell growth, and skin integrity.",
        "symptoms": [
            "Night blindness (Nyctalopia)",
            "Dry eyes (Xerophthalmia)",
            "Poor immune function (frequent infections)",
            "Dry, scaly skin",
            "Follicular hyperkeratosis (bumpy skin)",
            "Stunted growth in children"
        ],
        "remedies": {
            "home": [
                "Increase intake of vitamin A-rich foods: carrots, sweet potatoes, spinach, kale, liver, eggs, dairy products.",
                "Ensure balanced diet with healthy fats for absorption."
            ],
            "herbal": [
                "While not direct 'herbal' remedies for acute deficiency, some herbs like spirulina or moringa are nutrient-dense and can support overall health."
            ],
            "medical": [
                "Vitamin A supplements (under medical supervision due to toxicity risk).",
                "Retinoid medications (for severe skin conditions related to deficiency)."
            ]
        },
        "extreme_effects": "Severe, prolonged Vitamin A deficiency can lead to permanent blindness (corneal scarring), increased susceptibility to infections, and even death, particularly in children.",
        "expert": "Ophthalmologist (for vision issues), General Physician, Nutritionist/Dietitian."
    },
    "vitamin_b12": {
        "name": "Vitamin B-12",
        "overview": "Vital for nerve function, red blood cell formation, and DNA synthesis.",
        "description": "Vitamin B12 (cobalamin) is a water-soluble vitamin involved in the metabolism of every cell of the human body. It plays a key role in the normal functioning of the brain and nervous system, and in the formation of red blood cells.",
        "symptoms": [
            "Fatigue and weakness",
            "Pale skin",
            "Sore tongue (glossitis)",
            "Tingling or numbness (paresthesia)",
            "Difficulty walking and balance problems",
            "Memory loss or cognitive difficulties",
            "Anemia (macrocytic anemia)"
        ],
        "remedies": {
            "home": [
                "Consume B12-rich foods: meat, fish, poultry, eggs, dairy products, fortified cereals/plant milks.",
                "For vegetarians/vegans, consider fortified foods or supplements."
            ],
            "herbal": [
                "No direct herbal sources of B12. Some fermented foods *may* contain trace amounts but are not reliable sources."
            ],
            "medical": [
                "B12 injections (for severe deficiency or malabsorption issues).",
                "Oral B12 supplements (cyanocobalamin or methylcobalamin).",
                "Nasal gel forms."
            ]
        },
        "extreme_effects": "Untreated severe B12 deficiency can lead to irreversible neurological damage, severe anemia (megaloblastic anemia), heart problems, and significant cognitive decline.",
        "expert": "Hematologist (for anemia), Neurologist (for nerve issues), General Physician, Gastroenterologist (for absorption issues)."
    },
    "vitamin_c": {
        "name": "Vitamin C",
        "overview": "Crucial for immune system, collagen production, and antioxidant protection.",
        "description": "Vitamin C (ascorbic acid) is a water-soluble vitamin and a powerful antioxidant. It's essential for collagen synthesis (important for skin, bones, and blood vessels), immune function, and wound healing.",
        "symptoms": [
            "Fatigue and irritability",
            "Swollen, bleeding gums",
            "Joint pain and swelling",
            "Easy bruising",
            "Poor wound healing",
            "Dry skin and hair",
            "Anemia (in severe cases)"
        ],
        "remedies": {
            "home": [
                "Increase intake of fruits and vegetables: citrus fruits, berries, bell peppers, broccoli, kiwi, tomatoes."
            ],
            "herbal": [
                "Rosehips, Amla (Indian Gooseberry), Acerola cherry are rich in natural Vitamin C and can be consumed in various forms."
            ],
            "medical": [
                "Vitamin C supplements (oral)."
            ]
        },
        "extreme_effects": "Severe and prolonged Vitamin C deficiency leads to scurvy, characterized by widespread bleeding, anemia, exhaustion, swelling, and eventual death if untreated.",
        "expert": "General Physician, Dermatologist (for skin issues), Dentist (for gum issues)."
    },
    "vitamin_d": {
        "name": "Vitamin D",
        "overview": "Essential for bone health, calcium absorption, and immune system regulation.",
        "description": "Vitamin D is a fat-soluble vitamin that helps your body absorb calcium and phosphorus, crucial for building and maintaining strong bones. It also plays a role in immune function, muscle function, and cell growth.",
        "symptoms": [
            "Bone pain and muscle weakness",
            "Fatigue",
            "Frequent infections",
            "Mood changes (depression)",
            "Hair loss (in some cases)",
            "Children: Rickets (soft, weak bones)",
            "Adults: Osteomalacia (softening of bones)"
        ],
        "remedies": {
            "home": [
                "Safe sun exposure (10-30 minutes, 3 times a week, depending on skin type and location).",
                "Consume Vitamin D-rich foods: fatty fish (salmon, mackerel), fortified milk/cereals, egg yolks, some mushrooms."
            ],
            "herbal": [
                "No significant herbal sources of Vitamin D directly, but some sun-dried herbs might have trace amounts."
            ],
            "medical": [
                "Vitamin D supplements (D3 cholecalciferol is generally preferred).",
                "High-dose prescription Vitamin D for severe deficiencies."
            ]
        },
        "extreme_effects": "Extreme Vitamin D deficiency can lead to severe bone deformities (rickets in children, osteomalacia in adults), increased risk of fractures, and may contribute to chronic diseases.",
        "expert": "Endocrinologist, Orthopedist (for bone issues), General Physician."
    },
    "other_b_vitamins": {
        "name": "Other B Vitamins (B1, B2, B3, B9, etc.)",
        "overview": "A group of vitamins vital for energy metabolism, neurological function, and cell health.",
        "description": "The B-vitamin complex includes B1 (Thiamine), B2 (Riboflavin), B3 (Niacin), B5 (Pantothenic Acid), B6 (Pyridoxine), B7 (Biotin - also listed under minerals for hair/skin context), B9 (Folate), and B12 (Cobalamin). Each plays a unique, but often interconnected, role in converting food into energy, nerve function, and red blood cell production.",
        "symptoms": [
            "B1 (Thiamine): Beriberi (neurological and cardiovascular issues), fatigue, muscle weakness.",
            "B2 (Riboflavin): Sore throat, cracks at mouth corners (angular cheilitis), skin rashes, eye irritation.",
            "B3 (Niacin): Pellagra (Dermatitis, Diarrhea, Dementia), fatigue.",
            "B9 (Folate/Folic Acid): Fatigue, weakness, mouth sores, megaloblastic anemia, birth defects (neural tube)."
        ],
        "remedies": {
            "home": [
                "B1: Whole grains, pork, fish, legumes, nuts.",
                "B2: Dairy, eggs, leafy greens, lean meats.",
                "B3: Meat, poultry, fish, nuts, legumes, fortified grains.",
                "B9: Leafy green vegetables, legumes, citrus fruits, fortified grains."
            ],
            "herbal": [
                "Nutrient-rich herbs like nettle or dandelion can provide some B vitamins, but specific deficiencies often require concentrated sources."
            ],
            "medical": [
                "B-complex supplements or individual B vitamin supplements, depending on the specific deficiency diagnosed."
            ]
        },
        "extreme_effects": "Extreme deficiencies of individual B vitamins can lead to severe and potentially life-threatening conditions like Wernicke-Korsakoff syndrome (B1), severe neurological damage, and severe birth defects (B9).",
        "expert": "General Physician, Neurologist, Hematologist, Nutritionist/Dietitian."
    },
    "minerals_proteins": {
        "name": "Mineral/Protein Deficiencies (Zinc, Iron, Biotin, Protein)",
        "overview": "Key for immune response, oxygen transport, healthy hair/skin, and overall body structure and function.",
        "description": "Beyond vitamins, minerals like Iron, Zinc, and Biotin, along with adequate protein intake, are vital. Iron is for oxygen transport, Zinc for immune function and wound healing, Biotin for metabolism and hair/skin/nail health, and Protein for building and repairing tissues.",
        "symptoms": [
            "Iron Deficiency: Fatigue, weakness, pale skin, shortness of breath, brittle nails, restless legs (anemia).",
            "Zinc Deficiency: Poor immune function, slow wound healing, hair loss, loss of taste/smell, skin lesions.",
            "Biotin Deficiency: Hair loss, brittle nails, skin rash (dermatitis), neurological symptoms.",
            "Protein Deficiency: Muscle wasting, edema (swelling), weakness, fatigue, poor growth in children, impaired immune function."
        ],
        "remedies": {
            "home": [
                "Iron: Red meat, poultry, fish, beans, lentils, spinach (with Vitamin C for absorption).",
                "Zinc: Meat, shellfish, legumes, nuts, seeds, dairy.",
                "Biotin: Eggs, nuts, seeds, sweet potatoes, mushrooms, avocados.",
                "Protein: Lean meats, poultry, fish, eggs, dairy, legumes, tofu, quinoa."
            ],
            "herbal": [
                "Nettle (for iron), ginger (for absorption support), and other nutrient-dense plants can be supportive but not primary solutions for severe deficiencies."
            ],
            "medical": [
                "Iron supplements (ferrous sulfate/gluconate), Zinc supplements, Biotin supplements, Protein powders/supplements (in cases of insufficient dietary intake)."
            ]
        },
        "extreme_effects": "Extreme deficiencies can lead to severe anemia (Iron), impaired growth and immune collapse (Zinc), severe dermatitis and neurological issues (Biotin), and Kwashiorkor/Marasmus (Protein), which are life-threatening forms of malnutrition.",
        "expert": "General Physician, Hematologist (for Iron), Dermatologist (for Zinc/Biotin skin/hair issues), Nutritionist/Dietitian."
    },
}

# --- Routes ---

@app.route('/')
@app.route('/home')
def home():
    """Renders the main home page."""
    return render_template('home.html')

@app.route('/about')
def about():
    """Renders the About Us page."""
    return render_template('about.html')

@app.route('/how_it_works')
def how_it_works():
    """Renders the 'How It Works' page."""
    return render_template('how_it_works.html')

@app.route('/description')
def description():
    """Renders the main description page with clickable vitamin cards."""
    # You might want to pass VITAMIN_DATA or a subset if needed for rendering the cards
    return render_template('description.html')

@app.route('/deficiency/<string:vitamin_name>')
def deficiency_detail(vitamin_name):
    """Renders the detailed information page for a specific vitamin deficiency."""
    # Retrieve the vitamin data based on the URL parameter
    vitamin = VITAMIN_DATA.get(vitamin_name)
    if not vitamin:
        # Handle case where vitamin_name is not found, e.g., redirect or show error
        flash("The requested vitamin information was not found.", 'danger')
        return redirect(url_for('description')) # Redirect back to the main description page
    return render_template('deficiency_detail.html', vitamin=vitamin)

@app.route('/contact_us')
def contact_us():
    """Renders the Contact Us page."""
    return render_template('contact_us.html')

@app.route('/methodology')
def methodology():
    """Renders the Methodology page."""
    return render_template('methodology.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if not username or not email or not password or not confirm_password:
            flash('All fields are required!', 'danger')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        conn = get_db_connection()
        user_by_username = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        user_by_email = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()

        if user_by_username:
            flash('Username already taken!', 'danger')
        elif user_by_email:
            flash('Email already registered!', 'danger')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            registered_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute('INSERT INTO users (username, email, password, registered_at) VALUES (?, ?, ?, ?)',
                         (username, email, hashed_password, registered_at))
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            conn.close() # Close connection after successful commit
            return redirect(url_for('login'))
        conn.close() # Close connection if there was an error and redirect
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        user_data = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user_data:
            user = User(user_data['id'], user_data['username'], user_data['email'], user_data['password'])
            if check_password_hash(user.password, password):
                login_user(user)
                flash('Logged in successfully!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password.', 'danger')
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Handles user logout."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the user dashboard."""
    return render_template('dashboard.html', user=current_user)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Handles image upload for prediction."""
    prediction_result = None
    image_path = None

    if request.method == 'POST':
        if 'image_upload' not in request.files:
            flash('No file part', 'warning')
            return redirect(request.url)

        file = request.files['image_upload']
        if file.filename == '':
            flash('No selected file', 'warning')
            return redirect(request.url)

        if file:
            # Generate a unique filename to prevent conflicts
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_path = url_for('static', filename=f'uploads/{filename}')

            # Perform prediction
            predicted_class, confidence, status = predict_image(filepath)
            if status == "success":
                prediction_result = {
                    'class': predicted_class,
                    'confidence': f"{confidence*100:.2f}%"
                }
            else:
                flash(f"Error during prediction: {predicted_class}", 'danger')
                # Optionally remove the uploaded file if prediction failed
                # os.remove(filepath)

    return render_template('predict.html', prediction_result=prediction_result, image_path=image_path)

@app.route('/predict_camera', methods=['POST'])
@login_required
def predict_camera():
    """API endpoint to handle image capture from webcam for prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    if file:
        # Generate a unique filename (assuming PNG from webcam)
        filename = str(uuid.uuid4()) + '.png'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predicted_class, confidence, status = predict_image(filepath)

        if status == "success":
            return jsonify({
                'predicted_class': predicted_class,
                'confidence': f"{confidence*100:.2f}%",
                'image_url': url_for('static', filename=f'uploads/{filename}')
            })
        else:
            # If prediction fails, return a 500 error with the specific error message
            return jsonify({'error': predicted_class}), 500
    return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    # Ensure the database is initialized before running the app
    # This init_db() call outside app_context is generally safer for first run.
    # It's already handled by `with app.app_context():` above, but a double check is fine.
    # init_db() # Can be commented out if you are sure init_db is called reliably.
    app.run(debug=True) # Set debug=False for production