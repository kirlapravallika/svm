import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_classification
from sklearn.preprocessing import StandardScaler

# ==============================
# Part 1: Ring-shaped data (RBF)
# ==============================
X_ring, y_ring = make_circles(n_samples=500, factor=0.4, noise=0.05, random_state=42)
X_ring = StandardScaler().fit_transform(X_ring)

svm_rbf = SVC(kernel='rbf', gamma='auto', C=1.0)
svm_rbf.fit(X_ring, y_ring)

# ==============================
# Part 2: Effect of C
# ==============================
X_lin, y_lin = make_classification(
    n_samples=100, n_features=2, n_redundant=0,
    n_informative=2, class_sep=1.5, random_state=42
)
X_lin = StandardScaler().fit_transform(X_lin)

svm_c1 = SVC(kernel='linear', C=1.0)
svm_c10 = SVC(kernel='linear', C=10.0)

svm_c1.fit(X_lin, y_lin)
svm_c10.fit(X_lin, y_lin)

print("Support vectors with C=1.0 :", svm_c1.n_support_)
print("Support vectors with C=10.0:", svm_c10.n_support_)

# ==============================
# Part 3: Support vector property
# ==============================
print("Support vectors depend only on boundary points")

# ==============================
# Part 4: Feature scaling example
# ==============================
X_unscaled = np.array([[0.2, 2000], [0.8, 8000], [0.4, 5000]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)

print("Unscaled features:\n", X_unscaled)
print("Scaled features:\n", X_scaled)

# ==============================
# Part 5: Decision function
# ==============================
w = np.array([0.5, -1.2])
b = -0.3
x = np.array([4, 3])

decision_value = np.dot(w, x) + b
prediction = "Positive" if decision_value > 0 else "Negative"

print("Decision value:", decision_value)
print("Predicted class:", prediction)
