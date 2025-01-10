import tensorflow as tf
import numpy as np
import pandas as pd

# Load and preprocess the dataset
data = pd.read_csv('big_demand_forecast_dataset2.csv')

# Map product_id to one-hot encoding
unique_product_ids = list(data["product_id"].unique())
one_hot_product_ids = {
    pid: [1 if i == idx else 0 for i in range(len(unique_product_ids))]
    for idx, pid in enumerate(unique_product_ids)
}

# Prepare the dataset
inputs = np.array([
    one_hot_product_ids[row["product_id"]] + [row["days_since_last_restock"], row["expiry_in_days"]]
    for _, row in data.iterrows()
])
outputs = np.array([[row["quantity"]] for _, row in data.iterrows()])

print("Inputs shape:", inputs.shape)
print("Outputs shape:", outputs.shape)

# Normalize inputs and outputs
def normalize(arr):
    return (arr - np.min(arr, axis=0)) / (np.max(arr, axis=0) - np.min(arr, axis=0))

normalized_inputs = normalize(inputs)
normalized_outputs = normalize(outputs)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(normalized_inputs.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='mean_squared_error',
              metrics=['mae'])

# Train the model
print("Training the model...")
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    normalized_inputs,
    normalized_outputs,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

print("Training complete.")

# Save the model
model.save('./saved_model.keras')
print("Model saved to ./saved_model.h5")