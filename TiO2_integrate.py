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

# Define the three different maps
base_current_map_FC = {
    '000': 8.85316E-06,
    '001': 9.18208E-06,
    '010': 8.99725E-06,
    '011': 9.13353E-06,
    '100': 9.03669E-06,
    '101': 9.07613E-06,
    '110': 8.96579E-06,
    '111': 8.49867E-06
}

base_current_map_FC1 = {
    '000': 8.85316E-06,
    '001': 9.05712E-06,
    '010': 8.97872E-06,
    '011': 8.98716E-06,
    '100': 9.02512E-06,
    '101': 8.95224E-06,
    '110': 8.94168E-06,
    '111': 8.36722E-06
}

base_current_map_FC2 = {
    '000': 8.8532E-06,
    '001': 9.0360E-06,
    '010': 8.9706E-06,
    '011': 8.9595E-06,
    '100': 9.0243E-06,
    '101': 8.9275E-06,
    '110': 8.9312E-06,
    '111': 8.3380E-06
}

def read_and_transform_data(file_path, base_current_map):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    currents = [base_current_map[line.strip()]
                for line in lines if line.strip() in base_current_map]
    return currents

def load_data(directory, map1, map2, map3):
    features = []
    labels = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            currents1 = read_and_transform_data(file_path, map1)
            currents2 = read_and_transform_data(file_path, map2)
            currents3 = read_and_transform_data(file_path, map3)
            currents_total = currents1 + currents2 + currents3 # Concatenate currents from all maps to form one feature vector
            label = int(re.search(r'label_(\d+)_', file_name).group(1))
            features.append(currents_total)
            labels.append(label)
    return np.array(features), np.array(labels)

X, y = load_data(r'C:\Users\97843\Desktop\Dissertation\TiO2\MNIST_2\Processed_20', base_current_map_FC, base_current_map_FC1, base_current_map_FC2)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training (80%) and temporary (20%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Split the temporary set into validation (10%) and test (10%) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = Sequential([
    Input(shape=(576,)),
    Dense(10, activation='softmax', kernel_regularizer=l2(0.001))
])

model.compile(optimizer=Adam(learning_rate=0.00149), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
