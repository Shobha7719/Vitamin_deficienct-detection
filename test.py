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
import urllib.parse 
import cv2

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
    model = load_model('vitamin_deficiency_model.h5')
    print("vitamin_deficiency_model.h5 loaded successfully.")
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



class_map={
    'Vitamin A deficiency':'vitamin_a',
    'Vitamin B-12 deficiency':'vitamin_b12',
    'Vitamin B1 deficiency': 'vitamin_b1',
    'Vitamin B2 deficiency': 'vitamin_b2',
    'Vitamin B3 deficiency': 'vitamin_b3',
    'Vitamin B9 deficiency': 'vitamin_b9',
    'Vitamin C deficiency': 'vitamin_c',
    'Vitamin D deficiency': 'vitamin_d',
    'Vitamin E deficiency': 'vitamin_e',
    'Vitamin K deficiency': 'vitamin_k',
    'zinc, iron, biotin, or protein deficiency':'minerals_proteins'

}
# Reverse mapping for prediction output
CLASS_LABELS = {v: k for k, v in CLASS_INDICES.items()}
TARGET_SIZE = (224, 224) # Ensure this matches your model's expected input size

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
        "description": "Vitamin A is a fat-soluble vitamin crucial for maintaining healthy vision, especially in low light. It also supports immune function, cell growth, and skin integrity. It exists in two main forms: preformed Vitamin A (retinol) from animal products and provitamin A carotenoids (like beta-carotene) from plant-based foods.",
        "symptoms": [
            "Night blindness (Nyctalopia)",
            "Dry eyes (Xerophthalmia) leading to corneal damage",
            "Poor immune function (frequent infections, especially respiratory)",
            "Dry, scaly skin and follicular hyperkeratosis (bumpy skin)",
            "Stunted growth in children",
            "Delayed wound healing"
        ],
        "common_foods": [
            "Carrots, sweet potatoes, pumpkin, butternut squash (rich in beta-carotene)",
            "Spinach, kale, dark leafy greens",
            "Liver (especially beef liver)",
            "Eggs, dairy products (milk, cheese), fortified cereals",
            "Fish liver oils"
        ],
        "dri_adult": "900 mcg RAE (Retinol Activity Equivalents) for men, 700 mcg RAE for women (or 3,000 IU)",
        "risks_of_excess": "Blurred vision, nausea, dizziness, headaches, muscle pain, skin peeling, liver damage, birth defects (during pregnancy). Chronic high intake can lead to hypervitaminosis A.",
        "remedies": {
            "home": [
                "Increase intake of vitamin A-rich foods (as listed above).",
                "Ensure balanced diet with healthy fats for absorption (as Vitamin A is fat-soluble)."
            ],
            "herbal": [
                "While not direct 'herbal' remedies for acute deficiency, some nutrient-dense plants like spirulina or moringa are high in beta-carotene and can support overall health."
            ],
            "medical": [
                "Vitamin A supplements (under medical supervision due to toxicity risk).",
                "Retinoid medications (e.g., for severe skin conditions related to deficiency, but this is different from dietary supplementation)."
            ]
        },
        "extreme_effects": "Severe, prolonged Vitamin A deficiency is a leading cause of preventable blindness worldwide (due to permanent corneal scarring, xerophthalmia, keratomalacia). It also significantly increases susceptibility to severe infections and can be fatal, particularly in children.",
        "expert": "Ophthalmologist (for vision issues), General Physician, Nutritionist/Dietitian."
    },
    "vitamin_b1": {
        "name": "Vitamin B1 (Thiamine)",
        "overview": "Essential for energy metabolism, nerve function, and carbohydrate breakdown.",
        "description": "Thiamine is a water-soluble B vitamin that plays a critical role in energy metabolism, converting carbohydrates into energy. It's also vital for nerve function, muscle contraction, and heart function.",
        "symptoms": [
            "Fatigue and weakness",
            "Irritability and poor memory",
            "Muscle weakness and aches",
            "Tingling or numbness in hands/feet (peripheral neuropathy)",
            "Edema (swelling) of legs",
            "Rapid heart rate",
            "In severe cases: Beriberi (wet beriberi affects heart, dry beriberi affects nerves), Wernicke-Korsakoff Syndrome (severe brain disorder)."
        ],
        "common_foods": [
            "Pork, beef, poultry, fish",
            "Whole grains (brown rice, whole wheat bread, oats)",
            "Legumes (beans, lentils)",
            "Nuts and seeds",
            "Fortified breads, cereals, and pasta"
        ],
        "dri_adult": "1.2 mg for men, 1.1 mg for women",
        "risks_of_excess": "Thiamine is water-soluble, so excess is usually excreted. No known toxicity from food or oral supplements.",
        "remedies": {
            "home": [
                "Increase intake of thiamine-rich foods.",
                "Avoid excessive alcohol consumption (which depletes thiamine)."
            ],
            "herbal": [
                "Nutrient-rich herbs like nettle, spirulina, and various whole-food plant sources can contribute but are not a primary treatment for deficiency."
            ],
            "medical": [
                "Oral thiamine supplements.",
                "Thiamine injections (for severe deficiency or malabsorption)."
            ]
        },
        "extreme_effects": "Severe B1 deficiency leads to Beriberi (affecting cardiovascular or nervous systems) or Wernicke-Korsakoff Syndrome, a severe neurological disorder causing confusion, ataxia, and memory impairment.",
        "expert": "General Physician, Neurologist, Cardiologist."
    },
    "vitamin_b2": {
        "name": "Vitamin B2 (Riboflavin)",
        "overview": "Crucial for energy production, cellular function, and metabolism of fats/drugs.",
        "description": "Riboflavin is a water-soluble B vitamin essential for numerous cellular processes, including energy production, cellular growth, and the metabolism of fats, drugs, and steroids. It also helps convert other B vitamins into usable forms.",
        "symptoms": [
            "Sore throat and swelling of the mucous membranes",
            "Cracks and sores at the corners of the mouth (angular cheilitis)",
            "Inflammation of the tongue (glossitis)",
            "Skin rashes (especially around nose, mouth, and scrotum/labia)",
            "Eye fatigue, sensitivity to light (photophobia), blurred vision",
            "Anemia"
        ],
        "common_foods": [
            "Dairy products (milk, yogurt, cheese)",
            "Eggs",
            "Lean meats, poultry, fish",
            "Fortified cereals and breads",
            "Almonds, mushrooms, spinach"
        ],
        "dri_adult": "1.3 mg for men, 1.1 mg for women",
        "risks_of_excess": "Riboflavin is water-soluble, so excess is usually excreted. No known toxicity from food or oral supplements. High doses may cause bright yellow urine.",
        "remedies": {
            "home": [
                "Increase intake of riboflavin-rich foods.",
                "Store foods properly as riboflavin is sensitive to light."
            ],
            "herbal": [
                "Brewer's yeast, spirulina, and other whole-food plant sources can contribute."
            ],
            "medical": [
                "Oral riboflavin supplements."
            ]
        },
        "extreme_effects": "Severe B2 deficiency can impair the metabolism of other B vitamins and lead to severe skin conditions, chronic fatigue, and neurological abnormalities.",
        "expert": "General Physician, Dermatologist, Ophthalmologist."
    },
    "vitamin_b3": {
        "name": "Vitamin B3 (Niacin)",
        "overview": "Important for energy metabolism, DNA repair, and antioxidant activity.",
        "description": "Niacin is a water-soluble B vitamin that plays a vital role in converting food into energy, supporting digestive system function, and maintaining healthy skin and nerves. It exists in two primary forms: nicotinic acid and nicotinamide.",
        "symptoms": [
            "Fatigue and weakness",
            "Digestive issues (indigestion, nausea, vomiting, diarrhea)",
            "Skin problems: Dermatitis (dark, scaly rash, especially on sun-exposed areas)",
            "Neurological symptoms: Headaches, apathy, memory loss, depression",
            "In severe cases: Pellagra (characterized by the '3 Ds': Dermatitis, Diarrhea, Dementia, and potentially Death if untreated)."
        ],
        "common_foods": [
            "Meat (beef, pork), poultry (chicken, turkey)",
            "Fish (tuna, salmon)",
            "Peanuts, mushrooms, avocados",
            "Legumes",
            "Fortified cereals and breads"
        ],
        "dri_adult": "16 mg NE (Niacin Equivalents) for men, 14 mg NE for women",
        "risks_of_excess": "High doses (especially nicotinic acid) can cause 'niacin flush' (redness, itching, tingling), liver damage, gastrointestinal upset, and impaired glucose tolerance.",
        "remedies": {
            "home": [
                "Increase intake of niacin-rich foods.",
                "Ensure sufficient tryptophan intake, as the body can convert tryptophan into niacin."
            ],
            "herbal": [
                "Some whole grains and legumes are good sources of niacin and its precursor, tryptophan."
            ],
            "medical": [
                "Oral niacin supplements (nicotinamide form to avoid flush, or timed-release nicotinic acid under medical supervision for cholesterol management)."
            ]
        },
        "extreme_effects": "Untreated severe B3 deficiency leads to Pellagra, a debilitating disease impacting the skin, digestive system, and nervous system, which can be fatal.",
        "expert": "General Physician, Dermatologist, Gastroenterologist, Neurologist."
    },
    "vitamin_b9": {
        "name": "Vitamin B9 (Folate / Folic Acid)",
        "overview": "Critical for cell growth, DNA synthesis, and red blood cell formation.",
        "description": "Folate (the natural form found in foods) and Folic Acid (the synthetic form used in supplements and fortified foods) are water-soluble B vitamins essential for DNA synthesis and repair, cell growth, and the formation of red blood cells. It's particularly crucial during periods of rapid growth, such as pregnancy and infancy.",
        "symptoms": [
            "Fatigue, weakness, lethargy",
            "Pale skin",
            "Sore tongue and mouth sores",
            "Headaches and irritability",
            "Digestive issues (diarrhea)",
            "In severe cases: Megaloblastic anemia (large, immature red blood cells)",
            "In pregnancy: Increased risk of neural tube defects in the baby."
        ],
        "common_foods": [
            "Leafy green vegetables (spinach, kale, romaine lettuce)",
            "Legumes (beans, lentils, chickpeas)",
            "Asparagus, Brussels sprouts, broccoli",
            "Citrus fruits",
            "Avocado",
            "Fortified cereals, breads, and pasta (with folic acid)"
        ],
        "dri_adult": "400 mcg DFE (Dietary Folate Equivalents) for adults; 600 mcg DFE for pregnant women",
        "risks_of_excess": "High intake of *folic acid* (not natural folate) can mask a Vitamin B12 deficiency, potentially leading to irreversible neurological damage if B12 deficiency remains undiagnosed and untreated.",
        "remedies": {
            "home": [
                "Increase intake of folate-rich foods (as listed above).",
                "Cook foods gently as folate is heat-sensitive."
            ],
            "herbal": [
                "Nutrient-rich herbs like alfalfa, dandelion, and nettle can provide some folate."
            ],
            "medical": [
                "Folic acid supplements (especially recommended for women of childbearing age and during pregnancy).",
                "Blood tests to diagnose specific type of anemia."
            ]
        },
        "extreme_effects": "Severe B9 deficiency leads to megaloblastic anemia. In pregnancy, it significantly increases the risk of neural tube defects (e.g., spina bifida, anencephaly) in the developing fetus.",
        "expert": "General Physician, Hematologist (for anemia), Obstetrician/Gynecologist (for pregnancy-related issues)."
    },
    "vitamin_b12": {
        "name": "Vitamin B12 (Cobalamin)",
        "overview": "Vital for nerve function, red blood cell formation, and DNA synthesis.",
        "description": "Vitamin B12 (cobalamin) is a water-soluble vitamin involved in the metabolism of every cell of the human body. It plays a key role in the normal functioning of the brain and nervous system, and in the formation of red blood cells. Unlike other B vitamins, B12 requires intrinsic factor for absorption in the gut, making malabsorption a common cause of deficiency.",
        "symptoms": [
            "Fatigue and weakness, especially with exertion",
            "Pale or yellowish skin",
            "Sore, red tongue (glossitis)",
            "Tingling, numbness, or 'pins and needles' sensations (paresthesia) in hands/feet",
            "Difficulty walking and balance problems (ataxia)",
            "Memory loss, cognitive difficulties, confusion, depression",
            "Anemia (macrocytic or megaloblastic anemia)",
            "Vision problems"
        ],
        "common_foods": [
            "Meat (beef, lamb, pork)",
            "Poultry (chicken, turkey)",
            "Fish (salmon, tuna, cod)",
            "Eggs",
            "Dairy products (milk, yogurt, cheese)",
            "Fortified cereals and plant milks (e.g., soy milk, almond milk)"
        ],
        "dri_adult": "2.4 mcg for adults",
        "risks_of_excess": "B12 is water-soluble and generally considered very safe. No known toxicity from high doses, as excess is excreted.",
        "remedies": {
            "home": [
                "Consume B12-rich foods. For vegetarians/vegans, consistent intake of fortified foods or reliable supplements is essential."
            ],
            "herbal": [
                "No direct reliable herbal sources of B12. Some fermented foods *may* contain trace amounts but are not reliable sources for addressing deficiency."
            ],
            "medical": [
                "B12 injections (for severe deficiency, pernicious anemia, or malabsorption issues).",
                "High-dose oral B12 supplements (cyanocobalamin or methylcobalamin) can be effective even with some malabsorption.",
                "Nasal gel forms."
            ]
        },
        "extreme_effects": "Untreated severe B12 deficiency can lead to irreversible neurological damage, severe anemia (megaloblastic anemia), heart problems, and significant cognitive decline, including dementia-like symptoms.",
        "expert": "Hematologist (for anemia), Neurologist (for nerve issues), General Physician, Gastroenterologist (for absorption issues like pernicious anemia or Crohn's disease)."
    },
    "vitamin_c": {
        "name": "Vitamin C",
        "overview": "Crucial for immune system, collagen production, and antioxidant protection.",
        "description": "Vitamin C (ascorbic acid) is a water-soluble vitamin and a powerful antioxidant. It's essential for collagen synthesis (important for skin, bones, tendons, ligaments, and blood vessels), immune function, wound healing, and iron absorption. It also protects cells from damage by free radicals.",
        "symptoms": [
            "Fatigue, weakness, and irritability",
            "Swollen, purple, spongy, and bleeding gums",
            "Loose teeth",
            "Joint pain and swelling",
            "Easy bruising and poor wound healing",
            "Dry skin and hair with 'corkscrew' hairs",
            "Anemia (often due to impaired iron absorption)",
            "Petechiae (small red spots on the skin from burst capillaries)"
        ],
        "common_foods": [
            "Citrus fruits (oranges, lemons, grapefruit)",
            "Berries (strawberries, blueberries, raspberries)",
            "Bell peppers (especially red and yellow)",
            "Broccoli, Brussels sprouts",
            "Kiwi, tomatoes, cantaloupe"
        ],
        "dri_adult": "90 mg for men, 75 mg for women (higher for smokers)",
        "risks_of_excess": "Generally low toxicity. High doses can cause gastrointestinal upset (diarrhea, nausea, cramps), and in individuals with hemochromatosis, can lead to iron overload. May increase risk of kidney stones in susceptible individuals.",
        "remedies": {
            "home": [
                "Increase intake of Vitamin C-rich fruits and vegetables (as listed above).",
                "Consume raw or lightly cooked foods, as heat can destroy Vitamin C."
            ],
            "herbal": [
                "Rosehips, Amla (Indian Gooseberry), Acerola cherry, and Camu Camu are exceptionally rich natural sources of Vitamin C and can be consumed in various forms (powder, tea, fresh)."
            ],
            "medical": [
                "Oral Vitamin C supplements (ascorbic acid or buffered forms for sensitive stomachs)."
            ]
        },
        "extreme_effects": "Severe and prolonged Vitamin C deficiency leads to Scurvy, a serious and potentially fatal disease characterized by widespread bleeding, anemia, exhaustion, swelling, and eventual death if untreated. Historically common in sailors on long voyages without fresh produce.",
        "expert": "General Physician, Dermatologist (for skin issues), Dentist (for gum issues), Hematologist (for anemia)."
    },
    "vitamin_d": {
        "name": "Vitamin D",
        "overview": "Essential for bone health, calcium absorption, and immune system regulation.",
        "description": "Vitamin D is a fat-soluble vitamin that acts like a hormone. It's primarily produced in the skin upon exposure to sunlight. It's crucial for helping your body absorb calcium and phosphorus, which are vital for building and maintaining strong bones. It also plays a significant role in immune function, cell growth, and muscle function.",
        "symptoms": [
            "Bone pain and muscle weakness",
            "Fatigue and general aches",
            "Frequent infections due to weakened immune system",
            "Mood changes (depression, SAD)",
            "Hair loss (in some cases of severe deficiency)",
            "Children: Rickets (soft, weak bones leading to bowed legs, delayed growth, bone deformities)",
            "Adults: Osteomalacia (softening of bones, leading to bone pain and increased fracture risk)"
        ],
        "common_foods": [
            "Fatty fish (salmon, mackerel, tuna, cod liver oil)",
            "Fortified foods: Milk, plant milks (almond, soy, oat), orange juice, cereals",
            "Egg yolks",
            "Some mushrooms (especially UV-exposed ones)"
        ],
        "dri_adult": "600-800 IU (International Units) for adults (higher for older adults, often 1000-2000 IU or more for optimal levels)",
        "risks_of_excess": "Vitamin D toxicity (hypervitaminosis D) is rare but serious, usually from excessive supplementation, not sun exposure or food. Symptoms include nausea, vomiting, weakness, frequent urination, kidney problems, and hypercalcemia (high blood calcium) leading to calcium deposits in soft tissues.",
        "remedies": {
            "home": [
                "Safe sun exposure (10-30 minutes, 3 times a week, varying with skin type, latitude, and time of day).",
                "Consume Vitamin D-rich foods (as listed above)."
            ],
            "herbal": [
                "No significant herbal sources of Vitamin D directly. Some sun-dried herbs might have trace amounts, but direct sun exposure and dietary sources are key."
            ],
            "medical": [
                "Vitamin D supplements (D3 cholecalciferol is generally preferred and more effective).",
                "High-dose prescription Vitamin D for severe deficiencies (often 50,000 IU weekly/monthly)."
            ]
        },
        "extreme_effects": "Extreme Vitamin D deficiency can lead to severe bone deformities (rickets in children, osteomalacia in adults), severe muscle weakness, increased risk of fractures, and may contribute to chronic diseases like osteoporosis, heart disease, and certain cancers.",
        "expert": "Endocrinologist, Orthopedist (for bone issues), General Physician, Nephrologist (for kidney issues related to calcium)."
    },
    "vitamin_e": {
        "name": "Vitamin E",
        "overview": "A powerful antioxidant, important for immune function and skin health.",
        "description": "Vitamin E is a fat-soluble vitamin that acts as a powerful antioxidant, protecting cells from damage caused by harmful molecules called free radicals. It's crucial for immune function, cell signaling, gene expression, and maintaining healthy skin and eyes.",
        "symptoms": [
            "Muscle weakness",
            "Vision problems (impaired vision, retinopathy)",
            "Neuropathy (nerve damage, leading to numbness, tingling, loss of sensation)",
            "Ataxia (loss of control of body movements, difficulty with coordination and balance)",
            "Weakened immune system (increased susceptibility to infections)",
            "Hemolytic anemia (destruction of red blood cells, rare in adults but seen in premature infants)"
        ],
        "common_foods": [
            "Nuts (almonds, hazelnuts, peanuts)",
            "Seeds (sunflower seeds, pumpkin seeds)",
            "Vegetable oils (wheat germ oil, sunflower oil, safflower oil, soybean oil, corn oil)",
            "Leafy green vegetables (spinach, broccoli)",
            "Avocado, mango, kiwi",
            "Fortified cereals"
        ],
        "dri_adult": "15 mg (or 22.4 IU) of alpha-tocopherol for adults",
        "risks_of_excess": "Generally low toxicity. High doses, especially from supplements, can increase the risk of bleeding (anticoagulant effect), especially in individuals taking blood thinners like warfarin. May also cause nausea, diarrhea, and fatigue.",
        "remedies": {
            "home": [
                "Increase intake of Vitamin E-rich foods (as listed above)."
            ],
            "herbal": [
                "Some herbs and plant oils are rich in Vitamin E, supporting overall intake. Examples include wheat germ and sunflower seeds."
            ],
            "medical": [
                "Vitamin E supplements (usually in cases of chronic fat malabsorption disorders, certain genetic disorders, or very premature infants). Always consult a doctor before taking high-dose supplements, especially if on blood thinners."
            ]
        },
        "extreme_effects": "Severe Vitamin E deficiency is rare in healthy individuals but can lead to chronic neurological problems including impaired balance and coordination (ataxia), muscle weakness (myopathy), visual disturbances (retinopathy), and impaired immune response. It is often linked to underlying genetic disorders or fat malabsorption conditions.",
        "expert": "General Physician, Neurologist, Gastroenterologist (for malabsorption issues)."
    },
    "vitamin_k": {
        "name": "Vitamin K",
        "overview": "Essential for blood clotting and bone health.",
        "description": "Vitamin K is a fat-soluble vitamin vital for blood coagulation (clotting) and bone metabolism. It plays a key role in producing active forms of proteins needed for blood clotting (like prothrombin) and for binding calcium in bones and other tissues (e.g., osteocalcin). It exists as K1 (phylloquinone) from plants and K2 (menaquinones) from animal products and bacterial synthesis.",
        "symptoms": [
            "Easy bruising and excessive bleeding from wounds, punctures, or injection sites",
            "Heavy menstrual periods",
            "Bleeding from the gums or nosebleeds",
            "Blood in urine or stool (melena)",
            "Newborns: Vitamin K Deficiency Bleeding (VKDB, previously hemorrhagic disease of the newborn)",
            "Long-term deficiency: Increased risk of bone fractures and osteoporosis, arterial calcification."
        ],
        "common_foods": [
            "Vitamin K1: Leafy green vegetables (kale, spinach, collard greens, Swiss chard, broccoli, Brussels sprouts, cabbage, lettuce)",
            "Vitamin K2: Natto (fermented soybeans), some cheeses, egg yolks, liver, fermented foods (in smaller amounts)."
        ],
        "dri_adult": "120 mcg for men, 90 mcg for women",
        "risks_of_excess": "No known toxicity from high doses of Vitamin K1 or K2 from food or supplements in healthy individuals. However, Vitamin K can interfere with anticoagulant medications like warfarin (Coumadin), reducing their effectiveness.",
        "remedies": {
            "home": [
                "Increase intake of Vitamin K-rich foods, especially leafy green vegetables.",
                "Ensure sufficient intake of healthy fats for absorption (as Vitamin K is fat-soluble)."
            ],
            "herbal": [
                "Green tea, parsley, and alfalfa are some herbal sources of Vitamin K."
            ],
            "medical": [
                "Vitamin K supplements (phytonadione for K1, menaquinone for K2 - always under medical supervision, especially if on blood thinners).",
                "Vitamin K injections for severe bleeding disorders or in newborns to prevent VKDB."
            ]
        },
        "extreme_effects": "Extreme Vitamin K deficiency can lead to severe, life-threatening bleeding (hemorrhage) due to impaired blood clotting, especially in newborns (VKDB) and individuals with malabsorption issues or those taking certain medications. Long-term deficiency contributes to poor bone mineralization and increased fracture risk.",
        "expert": "Hematologist (for bleeding disorders), General Physician."
    },
    "minerals_proteins": {
        "name": "Mineral/Protein Deficiencies (Iron, Zinc, Biotin, Protein)",
        "overview": "Key for immune response, oxygen transport, healthy hair/skin, and overall body structure and function.",
        "description": "Beyond vitamins, essential minerals like Iron and Zinc, and the B-vitamin Biotin (often grouped with minerals due to its common symptoms), along with adequate protein intake, are vital. Iron is fundamental for oxygen transport in blood; Zinc supports immune function, wound healing, and taste/smell; Biotin is crucial for metabolism and the health of hair, skin, and nails; and Protein is the building block for all tissues, enzymes, and hormones.",
        "symptoms": [
            "Iron Deficiency (Anemia): Fatigue, weakness, pale skin, shortness of breath, dizziness, brittle nails, restless legs, pica (craving non-food items).",
            "Zinc Deficiency: Poor immune function (frequent infections), slow wound healing, hair loss, loss of taste/smell, appetite loss, skin lesions (dermatitis), impaired growth in children.",
            "Biotin Deficiency: Hair loss (alopecia), brittle nails, red, scaly rash around eyes, nose, mouth, and genital area (dermatitis), neurological symptoms (depression, lethargy, hallucinations), muscle pain.",
            "Protein Deficiency: Muscle wasting and weakness, edema (swelling, especially in legs and abdomen), fatigue, stunted growth in children, impaired immune function, brittle hair and nails, skin changes."
        ],
        "common_foods": [
            "Iron: Red meat, poultry, fish, beans, lentils, spinach (with Vitamin C for absorption).",
            "Zinc: Oysters (very high), red meat, poultry, beans, nuts, fortified cereals, dairy products.",
            "Biotin: Eggs (cooked), nuts, seeds, sweet potatoes, mushrooms, avocados.",
            "Protein: Lean meats, poultry, fish, eggs, dairy, legumes, tofu, quinoa."
        ],
        "dri_adult": {
            "Iron": "8 mg for men, 18 mg for pre-menopausal women (due to menstrual losses)",
            "Zinc": "11 mg for men, 8 mg for women",
            "Biotin": "30 mcg for adults",
            "Protein": "0.8 grams per kilogram of body weight (e.g., 56g for a 70kg person), higher for active individuals, pregnant women, etc."
        },
        "risks_of_excess": {
            "Iron": "Can cause gastrointestinal upset, liver damage, heart problems, and other organ damage, especially in individuals with hemochromatosis. Acute overdose is toxic.",
            "Zinc": "Nausea, vomiting, diarrhea, abdominal cramps, copper deficiency (due to competition for absorption), impaired immune function.",
            "Biotin": "Generally no known toxicity from high doses.",
            "Protein": "Can stress kidneys in those with kidney disease, potential for dehydration if water intake is not adequate, may displace other essential nutrients if relied on excessively."
        },
        "remedies": {
            "home": [
                "Iron: Consume iron-rich foods, pair plant-based iron with Vitamin C for better absorption.",
                "Zinc: Consume zinc-rich foods.",
                "Biotin: Consume biotin-rich foods.",
                "Protein: Ensure diverse protein sources, adequate total caloric intake."
            ],
            "herbal": [
                "Nettle (for iron), ginger (for absorption support), and various nutrient-dense plants can be supportive but not primary solutions for severe deficiencies. Horsetail is sometimes used for biotin-like benefits."
            ],
            "medical": [
                "Iron supplements (ferrous sulfate/gluconate), Zinc supplements, Biotin supplements, Protein powders/supplements (in cases of insufficient dietary intake or increased needs). Intravenous iron for severe anemia."
            ]
        },
        "extreme_effects": "Extreme deficiencies can lead to severe anemia (Iron), impaired growth and immune collapse (Zinc), severe dermatitis and neurological issues (Biotin), and Kwashiorkor/Marasmus (Protein), which are life-threatening forms of malnutrition with severe impacts on physical and cognitive development.",
        "expert": "General Physician, Hematologist (for Iron), Dermatologist (for Zinc/Biotin skin/hair issues), Nutritionist/Dietitian, Nephrologist (for protein metabolism issues)."
    }
}

# --- Routes ---

@app.route('/')
@app.route('/home')
def home():
    # If user not logged in, force login first and preserve the requested path
    if not current_user.is_authenticated:
        return redirect(url_for('login', next=request.path))
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
        # flash("The requested vitamin information was not found.", 'danger')
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
        return redirect(url_for('home'))
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
                return redirect(url_for('home'))
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
    deficiency_details_url = None # Initialize for prediction details link

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
            vitamin_key=class_map.get(predicted_class)
          

   
            if status == "success":
                prediction_result = {
                    'class': predicted_class,
                    'confidence': f"{confidence*100:.2f}%"
                }
                
            else:
                flash(f"Error during prediction: {predicted_class}", 'danger')
                # Optionally remove the uploaded file if prediction failed
                # os.remove(filepath)
            return render_template('predict.html', prediction_result=prediction_result, image_path=image_path,deficiency_details=vitamin_key)
            

    return render_template('predict.html')

@app.route('/predict_camera', methods=['POST'])
@login_required
def predict_camera():
    """Handle image upload from webcam and predict vitamin deficiency."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    if file:
        try:
            # Save the uploaded file
            filename = str(uuid.uuid4()) + '.png'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load and preprocess image
            img_size = 224
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = img / 255.0  # normalize

            # Predict
            predictions = model.predict(img)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_class = CLASS_LABELS[predicted_index]
            confidence = np.max(predictions)

            # Map to vitamin key
            vitamin_key = class_map.get(predicted_class)
            know_more_link = None
            if vitamin_key:
                know_more_link = url_for('deficiency_detail', vitamin_name=urllib.parse.quote(vitamin_key))

            return jsonify({
                'predicted_class': predicted_class,
                'confidence': f"{confidence * 100:.2f}%",
                'image_url': url_for('static', filename=f'uploads/{filename}'),
                'deficiency_details': vitamin_key,
                'know_more_link': know_more_link
            })

        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    # Ensure the database is initialized before running the app
    # This init_db() call outside app_context is generally safer for first run.
    # It's already handled by `with app.app_context():` above, but a double check is fine.
    # init_db() # Can be commented out if you are sure init_db is called reliably.
    app.run(debug=True) # Set debug=False for production