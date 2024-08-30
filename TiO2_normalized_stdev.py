import numpy as np
import os
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt  # Import Matplotlib for plotting


# Define the mapping from 3-bit sequences with current mean values and standard deviations
base_current_map = {
    '000': {'mean': 8.85316E-06, 'std': 8.58608E-09},
    '001': {'mean': 9.18208E-06, 'std': 1.80349E-08},
    '010': {'mean': 8.99725E-06, 'std': 1.19153E-08},
    '011': {'mean': 9.13353E-06, 'std': 1.02241E-08},
    '100': {'mean': 9.03669E-06, 'std': 1.36718E-08},
    '101': {'mean': 9.07613E-06, 'std': 1.93405E-08},
    '110': {'mean': 8.96579E-06, 'std': 1.64907E-08},
    '111': {'mean': 8.49867E-06, 'std': 8.40118E-09}
}

def read_and_transform_data(file_path, num_samples=100):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    currents = []
    for line in lines:
        key = line.strip()
        if key in normalized_base_current_map:
            # Generate 'num_samples' current values for each line based on its mean and std deviation
            sample_currents = np.random.normal(
                base_current_map[key]['mean'],
                base_current_map[key]['std'],
                num_samples
            )
            # Normalize these sample currents
            normalized_samples = (sample_currents - min_current) / (max_current - min_current)
            currents.extend(normalized_samples)
    return currents


# Calculate the minimum and maximum values from the map
min_current = min(base_current_map.values())
max_current = max(base_current_map.values())

# Normalize the current values to range from 0 to 1
normalized_base_current_map = {key: (value - min_current) / (max_current - min_current) for key, value in base_current_map.items()}

def load_data(directory):
    features = []
    labels = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            currents = read_and_transform_data(file_path)
            label = int(re.search(r'label_(\d+)_', file_name).group(1))
            features.append(currents)
            labels.append(label)
    return np.array(features), np.array(labels)

X, y = load_data(r'C:\Users\97843\Desktop\MNIST_2\Processed_01')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (80%) and temporary (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Split the temporary set into validation (10%) and test (10%) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = Sequential([
    Input(shape=(192,)),
    Dense(10, activation='softmax', kernel_regularizer=l2(0.001))
])

model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and store the history
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
