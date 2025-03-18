import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# Load the dataset with error handling
try:
    data = np.loadtxt('data_012.txt', delimiter=',')
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

X = data[:, :2]  # Features (first two columns)
y = data[:, 2].astype(int)  # Labels (last column) (ensure integer format)

# Data visualization
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Class 0', alpha=0.6)
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Class 1', alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Dataset Visualization')
plt.show()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Neural network model
model = MLPClassifier(hidden_layer_sizes=(8, 4),  # Two hidden layers (8 neurons, 4 neurons)
                      activation='logistic',
                      max_iter=1500,  
                      learning_rate_init=0.001,  
                      random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Decision boundary visualization
plt.figure(figsize=(8,6))
plot_decision_regions(X_scaled, y, clf=model, legend=2)
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.title('Decision Boundary')
plt.show()

# Save the trained model
with open('AI03_Chad.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as AI03_Chad.pkl")
