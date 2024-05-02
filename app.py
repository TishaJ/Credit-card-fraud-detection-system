from flask import Flask, request, render_template, jsonify
import os
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessor for Use Case 1
def load_model_and_preprocessor():
    models_directory = os.path.join("models")
    model_path = os.path.join(models_directory, "uc1_model.pickle")
    preprocessor_path = os.path.join(models_directory, "uc1_preprocessor.pickle")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)

    return model, preprocessor

model_uc1, preprocessor_uc1 = load_model_and_preprocessor()

@app.route('/')
def index():
    # Render the payment.html template
    return render_template('payment.html')

@app.route('/predict', methods=['GET','POST'])

def predict():
    try:
        # Retrieve form data
        street = request.form.get('street')
        city = request.form.get('city')
        state = request.form.get('state')
        Zip_code = request.form.get('Zip_code')

        # Check for missing data
        if not all([street, city, state, Zip_code]):
            raise ValueError("Missing one or more required fields.")

        # Create a DataFrame with the received data
        input_data = pd.DataFrame({
            "Street": [street],
            "City": [city],
            "State": [state],
            "Zip_Code": [Zip_code]
        })

        # Preprocess the input and make prediction
        preprocessed_input = preprocessor_uc1.transform(input_data)
        prediction = model_uc1.predict(preprocessed_input)

        # Determine prediction result
        result = "fraudulent" if prediction[0] == 1 else "legitimate"

        # Return the prediction as a JSON response
        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
