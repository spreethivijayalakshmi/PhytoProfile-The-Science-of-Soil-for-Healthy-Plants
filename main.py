from flask import Flask, render_template, request
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load models and other necessary files
try:
    model = tf.keras.models.load_model(r"C:\Users\91989\Downloads\classification_model (1).h5", compile=False)
    scaler = joblib.load(r'C:\Users\91989\Downloads\scaler.pkl')
    label_encoder = joblib.load(r'C:\Users\91989\Downloads\label_encoder.pkl')
    generator_model = tf.keras.models.load_model(r"C:\Users\91989\Downloads\generator_epoch_5000 (1).h5", compile=False)
except Exception as e:
    print(f"Error loading models: {e}")
    raise e


# Home page route
@app.route('/')
def home():
    return render_template('index.html')


# Route to generate new profiles

@app.route('/generate', methods=['POST'])
def generate_new_profiles():
    try:
        input_dim = 9
        random_conditions = np.random.rand(1, input_dim)
        generated_profiles = generator_model.predict(random_conditions)
        generated_profiles_original_scale = scaler.inverse_transform(generated_profiles)
        generated_values = generated_profiles_original_scale[0].tolist()

        print(f"Generated values: {generated_values}")  # Debugging line to check if the route is triggered

        # Return the index page with the generated values filled in the form
        return render_template('index.html', generated_values=generated_values)
    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")


# Route to classify and evaluate soil
@app.route('/classify', methods=['POST'])
def classify_and_evaluate():
    try:
        # Get data from form
        pH = float(request.form['pH'])
        soil_ec = float(request.form['soil_ec'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        urea = float(request.form['urea'])
        tsp = float(request.form['tsp'])
        mop = float(request.form['mop'])
        moisture = float(request.form['moisture'])
        temperature = float(request.form['temperature'])

        # Prepare input data for classification
        inputs = [pH, soil_ec, phosphorus, potassium, urea, tsp, mop, moisture, temperature]
        input_data = pd.DataFrame([inputs], columns=['pH', 'Soil EC', 'Phosphorus', 'Potassium',
                                                     'Urea', 'T.S.P', 'M.O.P', 'Moisture', 'Temperature'])
        input_data_scaled = scaler.transform(input_data)

        # Get predictions
        predictions = model.predict(input_data_scaled)
        predicted_class = np.argmax(predictions, axis=1)
        plant_type = label_encoder.inverse_transform(predicted_class)[0]

        # Soil quality evaluation
        evaluation, conclusion = evaluate_soil(pH, soil_ec, phosphorus, potassium, urea, tsp, mop, moisture,
                                               temperature)

        return render_template('index.html', plant_type=plant_type, evaluation=evaluation, conclusion=conclusion)

    except ValueError as ve:
        return render_template('index.html', error=f"Input Error: {str(ve)}")
    except Exception as e:
        return render_template('index.html', error=f"Error during classification/evaluation: {str(e)}")


# Function to evaluate soil parameters
def evaluate_soil(pH, soil_ec, phosphorus, potassium, urea, tsp, mop, moisture, temperature):
    evaluation = {}
    evaluation['pH'] = 'Good' if 6.0 <= pH <= 7.5 else 'Bad'
    evaluation['Soil EC'] = 'Good' if 0.1 <= soil_ec <= 0.5 else 'Bad'
    evaluation['Phosphorus'] = 'Good' if 10 <= phosphorus <= 40 else 'Bad'
    evaluation['Potassium'] = 'Good' if 120 <= potassium <= 200 else 'Bad'
    evaluation['Urea'] = 'Good' if 20 <= urea <= 50 else 'Bad'
    evaluation['T.S.P'] = 'Good' if 10 <= tsp <= 30 else 'Bad'
    evaluation['M.O.P'] = 'Good' if 20 <= mop <= 60 else 'Bad'
    evaluation['Moisture'] = 'Good' if 60 <= moisture <= 80 else 'Bad'
    evaluation['Temperature'] = 'Good' if 15 <= temperature <= 35 else 'Bad'

    good_count = sum(1 for result in evaluation.values() if result == 'Good')
    bad_count = len(evaluation) - good_count

    if bad_count == 0:
        conclusion = "This soil is ideal for plant growth and meets all recommended criteria."
    elif bad_count <= 2:
        conclusion = "This soil is mostly good, but attention is required for the following parameters: " + \
                     ", ".join([key for key, result in evaluation.items() if result == 'Bad']) + "."
    else:
        conclusion = "This soil has several issues and requires significant improvement, especially for: " + \
                     ", ".join([key for key, result in evaluation.items() if result == 'Bad']) + "."

    return evaluation, conclusion


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
