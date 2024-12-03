import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the Iris dataset
data = load_iris()
X = data['data']
y = data['target']

# One-hot encode the target variable for neural networks
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=0)

# Prepare y labels for the Decision Tree (not one-hot encoded)
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train_labels)
y_pred = clf.predict(X_test)
accuracy_dt = accuracy_score(y_test_labels, y_pred)
print(f"Decision Tree Test Accuracy: {accuracy_dt:.2f}")

# Cross-validation for Decision Tree
k = 5  # Number of folds for cross-validation
kf = KFold(n_splits=k, shuffle=True, random_state=0)
cv_scores_dt = cross_val_score(clf, X_scaled, y, cv=kf)
print(f"Decision Tree Cross-Validated Accuracy: {np.mean(cv_scores_dt):.2f}")

# Function to build a shallow neural network model
def build_shallow_nn(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to build a deep neural network model
def build_deep_nn(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the shallow neural network
shallow_nn = build_shallow_nn(X_train.shape[1])
shallow_nn.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
loss_shallow, accuracy_shallow = shallow_nn.evaluate(X_test, y_test, verbose=0)
print(f"Shallow Neural Network Test Accuracy: {accuracy_shallow:.2f}")

# Train and evaluate the deep neural network
deep_nn = build_deep_nn(X_train.shape[1])
deep_nn.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
loss_deep, accuracy_deep = deep_nn.evaluate(X_test, y_test, verbose=0)
print(f"Deep Neural Network Test Accuracy: {accuracy_deep:.2f}")

# Cross-validation for shallow and deep neural networks
shallow_nn_scores = []
deep_nn_scores = []

for train_index, test_index in kf.split(X_scaled):
    X_train_fold, X_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y_encoded[train_index], y_encoded[test_index]

    # Shallow Neural Network Cross-Validation
    shallow_nn = build_shallow_nn(X_train_fold.shape[1])
    shallow_nn.fit(X_train_fold, y_train_fold, epochs=50, batch_size=16, verbose=0)
    _, accuracy_shallow_fold = shallow_nn.evaluate(X_test_fold, y_test_fold, verbose=0)
    shallow_nn_scores.append(accuracy_shallow_fold)

    # Deep Neural Network Cross-Validation
    deep_nn = build_deep_nn(X_train_fold.shape[1])
    deep_nn.fit(X_train_fold, y_train_fold, epochs=50, batch_size=16, verbose=0)
    _, accuracy_deep_fold = deep_nn.evaluate(X_test_fold, y_test_fold, verbose=0)
    deep_nn_scores.append(accuracy_deep_fold)

print(f"Shallow Neural Network Cross-Validated Accuracy: {np.mean(shallow_nn_scores):.2f}")
print(f"Deep Neural Network Cross-Validated Accuracy: {np.mean(deep_nn_scores):.2f}")
