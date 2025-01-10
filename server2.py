from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('saved_model.keras')

# Define one-hot encoding for product IDs
unique_product_ids = [0, 1, 2]  # Replace with actual unique product IDs
one_hot_product_ids = {
    pid: [1 if i == idx else 0 for i in range(len(unique_product_ids))]
    for idx, pid in enumerate(unique_product_ids)
}

# Preprocess input data
def preprocess_input(data):
    product_id = data['product_id']
    days_since_last_restock = data['days_since_last_restock']
    expiry_in_days = data['expiry_in_days']

    # One-hot encode the product ID
    if product_id not in one_hot_product_ids:
        raise ValueError(f"Invalid product_id: {product_id}")

    one_hot_encoded_product_id = one_hot_product_ids[product_id]

    # Combine the one-hot encoding with numeric features
    input_features = one_hot_encoded_product_id + [days_since_last_restock, expiry_in_days]
    return np.array(input_features).reshape(1, -1)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from POST request
        data = request.get_json()

        # Preprocess the input data
        input_data = preprocess_input(data)

        # Make prediction
        prediction = model.predict(input_data)

        # Convert the prediction to a standard Python float
        response = {'prediction': int(prediction[0][0])}
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# @app.route('/predictStock', methods=['POST'])
# def predictStock():
#     try:
#         # Get input data from POST request
#         data = request.get_json()

#         # Preprocess the input data
#         input_data = preprocess_input(data)

#         # Make prediction
#         prediction = model.predict(input_data)

#         # Convert the prediction to a standard Python float
#         response = {'prediction': int(prediction[0][0])}
#         return jsonify(response), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

@app.route('/predictStock', methods=['POST'])
def predictStock():
    try:
        # Get input data from POST request
        data = request.get_json()

        # Validate input
        if not all(key in data for key in ('expiry_days', 'total_days', 'consumption')):
            return jsonify({'error': "Missing required keys. 'expiry_days', 'total_days', and 'consumption' are required."}), 400
        
        # Extract values
        expiry_days = data['expiry_days']
        total_days = data['total_days']
        consumption = data['consumption']

        # Validate input values
        if total_days <= 0:
            return jsonify({'error': "'total_days' must be greater than 0."}), 400

        # Compute restock
        restock = expiry_days * (consumption / total_days)

        # Create response
        response = {'restock': round(restock, 2)}  # Rounded to 2 decimal places for clarity
        return jsonify(response), 200
    except Exception as e:
        # Return error response if an exception occurs
        return jsonify({'error': str(e)}), 400
 
# Run the Flask app
PORT = 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)
