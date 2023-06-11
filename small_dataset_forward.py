import numpy as np
import pandas as pd
import time

pd.set_option("display.precision", 10)

def load_data(filename):
    # Initialize empty lists to store the data
    classes = []
    features = []

    # Read the data from the file
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespaces
            if line:
                data = line.split()  # Split the line into individual values
                class_value = float(data[0])  # Parse the class value as a float
                feature_values = [float(value) for value in data[1:]]  # Parse the feature values as floats

                # Append the class and feature values to the respective lists
                classes.append(class_value)
                features.append(feature_values)

    return classes, features


def calculate_distance(data, feature_indices, i, j):
    distance = 0
    for feature_idx in feature_indices:
        distance += abs(data.iloc[i, feature_idx] - data.iloc[j, feature_idx])
    return distance


def evaluate_accuracy(data, feature_indices):
    num_rows = data.shape[0]
    accuracy_counter = 0

    for i in range(num_rows):
        predicted_class_idx = 0
        nearest_value = np.inf

        for j in range(num_rows):
            if j != i:
                distance = calculate_distance(data, feature_indices, i, j)

                if distance <= nearest_value:
                    nearest_value = distance
                    predicted_class_idx = j

        if data.iloc[i, 0] == data.iloc[predicted_class_idx, 0]:
            accuracy_counter += 1

    predict_accuracy = float(accuracy_counter) / num_rows
    return predict_accuracy


def display(selected_features, feature_accuracy, time_measurement):
    features_str = ', '.join(map(str, selected_features))
    accuracy_percentage = round(feature_accuracy * 100, 1)
    print(f"Selected Features: {features_str} | Accuracy: {accuracy_percentage}% | Time: {round(time_measurement, 2)}ms")


def forward_selection(data):
    selected_features = []
    best_accuracy = 0
    num_features = data.shape[1]

    for i in range(1, num_features):
        if i not in selected_features:
            current_features = selected_features + [i]
            start_time = time.time() * 1000
            feature_accuracy = evaluate_accuracy(data, current_features)
            end_time = time.time() * 1000
            time_measurement = end_time - start_time
            display(current_features, feature_accuracy, time_measurement)

            if feature_accuracy > best_accuracy:
                best_feature = i
                best_accuracy = feature_accuracy

    selected_features.append(best_feature)
    display(selected_features, best_accuracy, time_measurement)

    return selected_features


# Load the data from the file
filename = "/Users/grohan/Documents/CS170_small_Data__26.txt"  # Replace with the actual filename
classes, features = load_data(filename)

# Convert the loaded data into a pandas DataFrame
data = pd.DataFrame(features)
data.insert(0, "Class", classes)

# Select the best feature
best_feature_indices = forward_selection(data)

# Perform forward selection
print("Forward Selection:")
selected_features = []
selected_features.append(best_feature_indices)
best_accuracy = 0
num_features = data.shape[1]

for i in range(1, num_features):
    if i not in selected_features:
        current_features = selected_features + [i]
        start_time = time.time() * 1000
        feature_accuracy = evaluate_accuracy(data, current_features)
        end_time = time.time() * 1000
        time_measurement = end_time - start_time
        display(current_features, feature_accuracy, time_measurement)

        if feature_accuracy > best_accuracy:
            best_feature = i
            best_accuracy = feature_accuracy

    selected_features.append(best_feature)
    display(selected_features, best_accuracy, time_measurement)
