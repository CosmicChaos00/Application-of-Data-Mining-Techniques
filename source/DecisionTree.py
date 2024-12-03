import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Load and prepare the data
data = load_iris()
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])

X = dataset.copy()
y = data['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Function to add noise to the training data
def add_noise(X, noise_level):
    # Generate random noise and add to the original data
    noise = np.random.normal(loc=0.0, scale=noise_level / 100.0 * X.std(axis=0), size=X.shape)
    X_noisy = X + noise
    return X_noisy

# Train and evaluate the model for different noise levels
# Function to add noise
def add_noise(X, noise_level):
    noise = np.random.normal(0, noise_level / 100.0 * X.std(axis=0), X.shape)
    return X + noise

# Initialize the classifier without noise and evaluate
clf_norm = DecisionTreeClassifier(random_state=0, max_depth=3, min_samples_leaf=5)
clf_norm.fit(X_train, y_train)

# Cross-validation on clean training data
scores = cross_val_score(clf_norm, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validated accuracy with 0% noise: {scores.mean():.2f}")

# Accuracy on the test set
y_norm_predict = clf_norm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_norm_predict)
print(f"Accuracy with 0% noise: {accuracy:.2f}")

# Train and evaluate the classifier for different noise levels
noise_levels = [1, 10, 30]

for level in noise_levels:
    print(f"\nTraining model with {level}% noise added to training data...")

    # Add noise to the scaled training data
    X_train_noisy = add_noise(X_train_scaled, level)

    # Train the model on noisy data using DecisionTreeClassifier
    clf = DecisionTreeClassifier(random_state=12, max_depth=3, min_samples_leaf=5)
    clf.fit(X_train_noisy, y_train)

    # Cross-validation on noisy data
    scores = cross_val_score(clf, X_train_noisy, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validated accuracy with {level}% noise: {scores.mean():.2f}")

    # Predict on the original test data (without noise)
    y_predict = clf.predict(X_test_scaled)

    # Compute Accuracy between actual and predicted values
    accuracy = accuracy_score(y_test, y_predict)
    print(f"Accuracy with {level}% noise: {accuracy:.2f}")