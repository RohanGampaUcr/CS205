import numpy as np
import pandas as pd
from time import perf_counter

pd.set_option("display.precision", 10)

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
                
                if distance < nearest_value:
                    nearest_value = distance
                    predicted_class_idx = j
        
        if data.iloc[i, 0] == data.iloc[predicted_class_idx, 0]:
            accuracy_counter += 1
    
    predict_accuracy = float(accuracy_counter) / num_rows
    return predict_accuracy

def display(selected_features, feature_accuracy, time_measurement):
    features_str = ', '.join(map(str, selected_features))
    accuracy_percentage = round(feature_accuracy * 100, 2)
    print(f"Selected Features: {features_str} | Accuracy: {accuracy_percentage}% | Time: {round(time_measurement, 2)}ms")

def backward_elimination(data):
    selected_features = list(range(1, data.shape[1]))
    best_accuracy = evaluate_accuracy(data, selected_features)
    display(selected_features, best_accuracy, 0)
    
    while len(selected_features) > 1:
        time_measurements = []
        feature_accuracies = []
        
        for feature_index in selected_features:
            remaining_features = selected_features.copy()
            remaining_features.remove(feature_index)
            
            start_time = perf_counter() * 1000
            feature_accuracy = evaluate_accuracy(data, remaining_features)
            end_time = perf_counter() * 1000
            time_measurement = end_time - start_time
            
            time_measurements.append(time_measurement)
            feature_accuracies.append(feature_accuracy)
        
        best_feature_index = np.argmax(feature_accuracies)
        best_accuracy = feature_accuracies[best_feature_index]
        best_time_measurement = time_measurements[best_feature_index]
        
        if best_accuracy > evaluate_accuracy(data, selected_features):
            selected_features.pop(best_feature_index)
            display(selected_features, best_accuracy, best_time_measurement)
        else:
            break
    
    display(selected_features, best_accuracy, 0)
    return selected_features

def load_data(filename):
    classes = []
    features = []
    
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if line:
                data = line.split()
                class_value = float(data[0])
                feature_values = [float(value) for value in data[1:]]
                classes.append(class_value)
                features.append(feature_values)
    
    return classes, features

# Load the data from the file
filename = "/Users/grohan/Documents/CS170_small_Data__26.txt"  # Replace with the actual filename
classes, features = load_data(filename)

# Convert the loaded data into a pandas DataFrame
data = pd.DataFrame(features)
data.insert(0, "Class", classes)

# Perform backward elimination
print("Backward Elimination:")
selected_features = backward_elimination(data)
