
# from flask import Flask, request, jsonify, render_template
# import joblib
# import pandas as pd
# import os

# import warnings
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# app = Flask(__name__, static_folder='static', template_folder='templates')

# # Load model and scaler using relative paths
# model = joblib.load("model.pkl")


# # Mappings
# occupation_mapping = {
#     'Software Engineer': 0, 'Doctor': 1, 'Sales Representative': 2, 'Teacher': 3,
#     'Nurse': 4, 'Engineer': 5, 'Accountant': 6, 'Scientist': 7, 'Lawyer': 8,
#     'Salesperson': 9, 'Manager': 10, 'Student': 11, 'Athlete': 12, 'Artist': 13
# }
# gender_mapping = {'Female': 0, 'Male': 1}

# # Helper functions
# def calculate_bmi(weight, height_cm):
#     height_m = height_cm / 100
#     return weight / (height_m ** 2)

# def get_bmi_category(bmi):
#     if bmi < 18.5:
#         return "Underweight"
#     elif 18.5 <= bmi < 25:
#         return "Normal"
#     else:
#         return "Overweight"

# def predict_sleep_disorder(score, stress, duration):
#     if score < 4 or (stress > 7 and duration < 5):
#         return "🚨 High risk of sleep disorder. Consult a specialist."
#     elif score < 6:
#         return "⚠️ Moderate risk. Consider improving sleep hygiene."
#     return "✅ Low risk. Maintain healthy sleep habits."

# # Routes
# @app.route("/")
# def home():
#     return render_template("home.html")

# @app.route("/about")
# def about():
#     return render_template("about.html")

# # @app.route("/contact")
# # def contact():
# #     return render_template("contact.html")


# @app.route("/prediction", methods=["GET" ,"POST"])
# def predict():
#     if request.method == "GET":
#         return render_template("prediction.html")  # This serves the prediction form page
    
#     data = request.get_json()

#     try:
#         age = int(data["age"])
#         gender = data["gender"]
#         occupation = data["occupation"]
#         sleep_duration = float(data["sleepDuration"])
#         activity = int(data["activityLevel"])
#         stress = int(data["stressLevel"])
#         steps = int(data["steps"])
#         weight = float(data["weight"])
#         height = float(data["height"])
#         systolic = int(data["systolic"])
#         diastolic = int(data["diastolic"])
#         heart_rate = int(data["heartRate"])

#         bmi = calculate_bmi(weight, height)
#         bmi_category = get_bmi_category(bmi)

#         user_df = pd.DataFrame([{
#             'Gender': gender_mapping[gender],
#             'Age': age,
#             'Occupation': occupation_mapping[occupation],
#             'Sleep Duration': sleep_duration,
#             'Physical Activity Level': activity,
#             'Stress Level': stress,
#             'Systolic BP': systolic,
#             'Diastolic BP': diastolic,
#             'Heart Rate': heart_rate,
#             'Daily Steps': steps
#         }])

#         # scaled_input = scaler.transform(user_df)
#         # sleep_score = model.predict(scaled_input)[0]
#         # disorder_risk = predict_sleep_disorder(sleep_score, stress, sleep_duration)

#         return jsonify({
#             "bmi": round(bmi, 2),
#             "bmi_category": bmi_category,
#             "sleep_score": round(sleep_score, 1),
#             "disorder_risk": disorder_risk
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 400

# # Run app
# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model and feature names
try:
    model = joblib.load('model.pkl')
    with open('model_features.pkl', 'rb') as f:
        model_features = pickle.load(f)
except FileNotFoundError as e:
    raise Exception(f"Model files not found: {str(e)}")

# Mappings
occupation_mapping = {
    'Software Engineer': 0, 'Doctor': 1, 'Sales Representative': 2, 'Teacher': 3,
    'Nurse': 4, 'Engineer': 5, 'Accountant': 6, 'Scientist': 7, 'Lawyer': 8,
    'Salesperson': 9, 'Manager': 10, 'Student': 11, 'Athlete': 12, 'Artist': 13
}
gender_mapping = {'Female': 0, 'Male': 1}


# Helper functions
def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return weight / (height_m ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    return "Overweight"

def predict_sleep_disorder(score, stress, duration):
    if score < 4 or (stress > 7 and duration < 5):
        return "🚨 High risk of sleep disorder. Consult a specialist."
    elif score < 6:
        return "⚠️ Moderate risk. Consider improving sleep hygiene."
    return "✅ Low risk. Maintain healthy sleep habits."

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/prediction", methods=["GET", "POST"])
@app.route("/prediction", methods=["GET", "POST"])
def predict():
    print("hello saar unnara?")
    if request.method == "GET":
        return render_template("prediction.html")
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    try:
        # Validate and extract inputs
        required_fields = [
            "age", "sleepDuration", "activityLevel", "stressLevel",
            "steps", "weight", "height", "systolic", "diastolic", "heartRate",
            "gender", "occupation"
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Convert and validate inputs
        try:
            age = int(data["age"])
            gender = gender_mapping[data["gender"]]
            occupation = occupation_mapping[data["occupation"]]
            sleep_duration = float(data["sleepDuration"])
            activity = int(data["activityLevel"])
            stress = int(data["stressLevel"])
            steps = int(data["steps"])
            weight = float(data["weight"])
            height = float(data["height"])
            systolic = int(data["systolic"])
            diastolic = int(data["diastolic"])
            heart_rate = int(data["heartRate"])
        except (ValueError, KeyError) as e:
            return jsonify({"error": f"Invalid input value: {str(e)}"}), 400

        # Calculate derived features
        bmi = calculate_bmi(weight, height)
        bmi_category = get_bmi_category(bmi)
        
        # Create input DataFrame with exactly the features the model expects
        input_data = {
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Systolic BP': systolic,
            'Diastolic BP': diastolic, 
            'Physical Activity Level': activity,
            'Stress Level': stress,
            'Heart Rate': heart_rate,
            'Daily Steps': steps,
            'Sleep_Consistency': 0,  # Placeholder - should be calculated or provided
            'Activity_Stress_Interaction': activity * stress  # Example interaction
        }
        
        # Ensure we only include features the model expects
        filtered_input = {k: input_data[k] for k in model_features if k in input_data}
        user_df = pd.DataFrame([filtered_input], columns=model_features)

        print(model_features)
        print(user_df)

        # Make prediction
        sleep_score = model.predict(user_df)[0]
        disorder_risk = predict_sleep_disorder(sleep_score, stress, sleep_duration)

        return jsonify({
            "bmi": round(bmi, 2),
            "bmi_category": bmi_category,
            "sleep_score": round(sleep_score, 1),
            "disorder_risk": disorder_risk
        })

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
    # After loading the model, check features
print("Model expects these features:", model_features)
print("Model feature importance:", model.feature_importances_)  # If available