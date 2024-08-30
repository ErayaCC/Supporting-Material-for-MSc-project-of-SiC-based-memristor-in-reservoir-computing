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

# Define the mapping from 3-bit sequences
base_current_map = {
    '000': 8.85316E-06,
    '001': 9.18208E-06,
    '010': 8.99725E-06,
    '011': 9.13353E-06,
    '100': 9.03669E-06,
    '101': 9.07613E-06,
    '110': 8.96579E-06,
    '111': 8.49867E-06
}

# Calculate the minimum and maximum values from the map
min_current = min(base_current_map.values())
max_current = max(base_current_map.values())

# Normalize the current values to range from 0 to 1
normalized_base_current_map = {key: (value - min_current) / (max_current - min_current) for key, value in base_current_map.items()}

def read_and_transform_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    currents = [normalized_base_current_map[line.strip()]
                for line in lines if line.strip() in normalized_base_current_map]
    return currents


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

X, y = load_data(r'C:\Users\97843\Desktop\Dissertation\MNIST_2\Processed_01')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (80%) and temporary (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Split the temporary set into validation (10%) and test (10%) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Learning rate sweep
learning_rates = np.logspace(-5, -2, num=30)  # 10 points per decade
accuracies = []

for lr in learning_rates:
    model = Sequential([
        Input(shape=(192,)),
        Dense(10, activation='softmax', kernel_regularizer=l2(0.001))
    ])

    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model and store the history
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test)
    accuracies.append(accuracy)
    print(f"Learning Rate: {lr:.5f} - Test Accuracy: {accuracy * 100:.2f}%")

# Plotting learning rates against accuracies
plt.semilogx(learning_rates, accuracies)
plt.title('Model Accuracy vs. Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
