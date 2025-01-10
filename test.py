import tensorflow as tf
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('./saved_model.h5')

# Sample test data (you can modify this as needed)
test_data = [
    {"product_id": 3, "days_since_last_restock": 12, "expiry_in_days": 40}
]

# Map product_id to one-hot encoding
unique_product_ids = [1, 2, 3]  # Adjust based on the training data
one_hot_product_ids = {
    pid: [1 if i == idx else 0 for i in range(len(unique_product_ids))]
    for idx, pid in enumerate(unique_product_ids)
}

# Prepare the test inputs
test_inputs = np.array([
    one_hot_product_ids[d["product_id"]] + [d["days_since_last_restock"], d["expiry_in_days"]]
    for d in test_data
])

# Normalize the test inputs (use the same normalization logic as during training)
def normalize(arr, original_data):
    max_values = np.max(original_data, axis=0)
    min_values = np.min(original_data, axis=0)
    return (arr - min_values) / (max_values - min_values)

# Example of original training data used for normalization during training
original_inputs = [
    [1, 0, 0, 5, 25],  # Example row with one-hot and features (training data sample)
    # Add more rows based on your actual training data structure
]
original_inputs = np.array(original_inputs)
normalized_test_inputs = normalize(test_inputs, original_inputs)

# Make predictions
predictions = model.predict(normalized_test_inputs)

# Display predictions
for i, prediction in enumerate(predictions):
    print(f"Test Data {i+1}: Predicted Quantity: {prediction[0]:.2f}")
