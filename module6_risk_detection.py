import pandas as pd
import requests
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "r9Rgn4hgPvXSJZaNcFCStO9p59unwsSOk2XRsfMH"

# -----------------------------
# FETCH NUTRIENTS FROM USDA API
# -----------------------------
def get_food_nutrients(food_name):
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&api_key={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception:
        return None

    try:
        food = data['foods'][0]
        nutrients = food['foodNutrients']

        result = {
            "carbs": 0,
            "sugar": 0,
            "protein": 0,
            "fat": 0
        }

        for n in nutrients:
            name = n['nutrientName']

            if "Carbohydrate" in name:
                result["carbs"] = n['value']
            elif "Sugars" in name:
                result["sugar"] = n['value']
            elif "Protein" in name:
                result["protein"] = n['value']
            elif "Total lipid" in name:
                result["fat"] = n['value']

        return result

    except:
        return None

# -----------------------------
# RULE ENGINE
# -----------------------------

# Known drug-food interactions
DRUG_FOOD_INTERACTIONS = {
    "warfarin": ["spinach", "kale", "broccoli", "brussels sprouts", "green tea"],
    "metformin": ["alcohol", "sugary foods"],
    "aspirin": ["fish oil", "garlic", "ginkgo"],
    "atorvastatin": ["grapefruit", "pomelo"],
    "lisinopril": ["banana", "potassium supplements"],
    "ciprofloxacin": ["milk", "yogurt", "cheese", "calcium"],
    "tetracycline": ["milk", "cheese", "yogurt", "iron"],
    "ibuprofen": ["alcohol"],
    "levothyroxine": ["soy", "walnuts", "coffee"],
    "digoxin": ["licorice", "bran"],
}

# Disease-nutrient risk thresholds (generic)
DISEASE_NUTRIENT_RULES = {
    "diabetes": {"carbs": (30, 50), "sugar": (15, 25)},
    "bp": {"fat": (20, 30), "sodium": (1500, 2300)},
    "hypertension": {"fat": (20, 30), "sodium": (1500, 2300)},
    "blood pressure": {"fat": (20, 30), "sodium": (1500, 2300)},
    "heart": {"fat": (15, 25), "cholesterol": (200, 300)},
    "heart disease": {"fat": (15, 25), "cholesterol": (200, 300)},
    "cardiovascular": {"fat": (15, 25), "cholesterol": (200, 300)},
    "obesity": {"carbs": (40, 60), "fat": (20, 35), "sugar": (20, 30)},
    "kidney": {"protein": (30, 50), "sodium": (1500, 2300)},
    "kidney disease": {"protein": (30, 50), "sodium": (1500, 2300)},
    "liver": {"fat": (20, 35), "protein": (30, 50)},
    "liver disease": {"fat": (20, 35), "protein": (30, 50)},
    "gout": {"protein": (25, 40)},
    "celiac": {"carbs": (40, 60)},
}

# Safe food recommendations per disease category
SAFE_FOODS_DB = {
    "diabetes": ["Broccoli", "Spinach", "Cucumber", "Beans", "Oats", "Almonds", "Avocado", "Fish", "Eggs", "Greek Yogurt"],
    "bp": ["Banana", "Spinach", "Oats", "Beans", "Carrot", "Sweet Potato", "Berries", "Beets", "Garlic", "Fish"],
    "hypertension": ["Banana", "Spinach", "Oats", "Beans", "Carrot", "Sweet Potato", "Berries", "Beets", "Garlic", "Fish"],
    "blood pressure": ["Banana", "Spinach", "Oats", "Beans", "Carrot", "Sweet Potato", "Berries", "Beets", "Garlic", "Fish"],
    "heart": ["Oats", "Salmon", "Avocado", "Almonds", "Walnuts", "Olive Oil", "Beans", "Berries", "Spinach", "Tomato"],
    "heart disease": ["Oats", "Salmon", "Avocado", "Almonds", "Walnuts", "Olive Oil", "Beans", "Berries", "Spinach", "Tomato"],
    "cardiovascular": ["Oats", "Salmon", "Avocado", "Almonds", "Walnuts", "Olive Oil", "Beans", "Berries", "Spinach", "Tomato"],
    "obesity": ["Cucumber", "Spinach", "Broccoli", "Carrot", "Apple", "Berries", "Chicken Breast", "Fish", "Egg Whites", "Green Tea"],
    "kidney": ["Apple", "Cabbage", "Cauliflower", "Garlic", "Onion", "Bell Pepper", "Blueberries", "Cranberries", "Fish", "Egg Whites"],
    "kidney disease": ["Apple", "Cabbage", "Cauliflower", "Garlic", "Onion", "Bell Pepper", "Blueberries", "Cranberries", "Fish", "Egg Whites"],
    "liver": ["Oats", "Green Tea", "Garlic", "Berries", "Grapefruit", "Olive Oil", "Beetroot", "Broccoli", "Nuts", "Fish"],
    "liver disease": ["Oats", "Green Tea", "Garlic", "Berries", "Grapefruit", "Olive Oil", "Beetroot", "Broccoli", "Nuts", "Fish"],
    "gout": ["Cherries", "Vitamin C foods", "Low-fat Dairy", "Vegetables", "Whole Grains", "Water", "Coffee", "Citrus Fruits", "Tofu", "Eggs"],
    "celiac": ["Rice", "Quinoa", "Corn", "Potato", "Fruits", "Vegetables", "Meat", "Fish", "Eggs", "Beans"],
    "healthy": ["Spinach", "Broccoli", "Carrot", "Apple", "Banana", "Oats", "Rice", "Fish", "Chicken", "Yogurt"],
}

def rule_engine(disease, medicine, nutrients, food):
    carbs = nutrients.get("carbs", 0)
    sugar = nutrients.get("sugar", 0)
    fat = nutrients.get("fat", 0)
    protein = nutrients.get("protein", 0)

    risk = None
    reason = None

    # Check drug-food interactions
    med_key = medicine.strip().lower()
    food_key = food.strip().lower()
    if med_key in DRUG_FOOD_INTERACTIONS:
        for bad_food in DRUG_FOOD_INTERACTIONS[med_key]:
            if bad_food in food_key or food_key in bad_food:
                return 2, f"Known drug-food interaction: {medicine} ↔ {food}"

    # Check disease-nutrient rules (dynamic lookup)
    disease_key = disease.strip().lower()
    if disease_key in DISEASE_NUTRIENT_RULES:
        rules = DISEASE_NUTRIENT_RULES[disease_key]
        nutrient_map = {"carbs": carbs, "sugar": sugar, "fat": fat, "protein": protein}
        for nutrient_name, (moderate_thresh, high_thresh) in rules.items():
            val = nutrient_map.get(nutrient_name, 0)
            if val > high_thresh:
                return 2, f"Very high {nutrient_name} ({val}g) for {disease}"
            elif val > moderate_thresh:
                if risk is None or risk < 1:
                    risk = 1
                    reason = f"Moderate {nutrient_name} ({val}g) for {disease}"

    # General high-value warnings for unknown diseases
    if risk is None and disease_key not in ["healthy", "none", ""]:
        if fat > 30:
            return 1, f"High fat content ({fat}g) — may not be ideal for {disease}"
        if sugar > 25:
            return 1, f"High sugar content ({sugar}g) — may not be ideal for {disease}"

    return risk, reason


def get_safe_foods(disease):
    """Return safe food recommendations for a given disease."""
    disease_key = disease.strip().lower()
    # Try exact match first, then partial match
    if disease_key in SAFE_FOODS_DB:
        return SAFE_FOODS_DB[disease_key]
    for key, foods in SAFE_FOODS_DB.items():
        if disease_key in key or key in disease_key:
            return foods
    return SAFE_FOODS_DB.get("healthy", [])

# -----------------------------
# TRAIN MODEL (DUMMY REALISTIC DATA)
# -----------------------------
def train_model():
    data = {
        "disease": [
            "Diabetes","Diabetes","Diabetes",
            "BP","BP","Heart","Healthy"
        ],
        "carbs": [60,35,15,10,5,20,12],
        "sugar": [30,15,5,2,1,3,5],
        "protein": [3,5,10,15,10,20,5],
        "fat": [1,2,5,35,15,30,5],
        "risk": [2,1,0,2,1,2,0]
    }

    df = pd.DataFrame(data)

    le = LabelEncoder()
    df["disease"] = le.fit_transform(df["disease"])

    X = df[["disease","carbs","sugar","protein","fat"]]
    y = df["risk"]

    model = XGBClassifier(n_estimators=50, max_depth=3)
    model.fit(X, y)

    return model, le
