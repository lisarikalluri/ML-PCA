import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np


wine = load_wine()
X, y = wine.data, wine.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


logistic = LogisticRegression(max_iter=10000, random_state=42)
logistic.fit(X_train_scaled, y_train)
y_pred = logistic.predict(X_test_scaled)


baseline_accuracy = accuracy_score(y_test, y_pred)
baseline_precision = precision_score(y_test, y_pred, average='macro')
baseline_recall = recall_score(y_test, y_pred, average='macro')

print("Baseline Model (No PCA) Performance:")
print(f"Accuracy: {baseline_accuracy:.4f}")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall: {baseline_recall:.4f}")


components = [2, 5, 10] 
accuracy_list = []

for n in components:

    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
   
    logistic_pca = LogisticRegression(max_iter=10000, random_state=42)
    logistic_pca.fit(X_train_pca, y_train)
    y_pred_pca = logistic_pca.predict(X_test_pca)
    

    accuracy = accuracy_score(y_test, y_pred_pca)
    accuracy_list.append(accuracy)
    
    print(f"\nPCA with {n} Components:")
    print(f"Accuracy: {accuracy:.4f}")


plt.figure(figsize=(8, 6))
plt.plot(components, accuracy_list, marker='o', linestyle='--', color='b')
plt.axhline(y=baseline_accuracy, color='r', linestyle='-', label='Baseline Accuracy')
plt.title("Accuracy vs. Number of PCA Components")
plt.xlabel("Number of Components")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


