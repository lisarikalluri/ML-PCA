import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

sns.pairplot(df, hue="species", markers=["o", "s", "D"])
plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(iris.data)

scaled_df = pd.DataFrame(scaled_data, columns=iris.feature_names)
import numpy as np

cov_matrix = np.cov(scaled_data.T)
print("Covariance Matrix:\n", cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

pca_data = scaled_data.dot(sorted_eigenvectors[:, :2])

pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
pca_df['species'] = iris.target


plt.figure(figsize=(8,6))
sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df, palette='Set1', markers=["o", "s", "D"])
plt.title('PCA: Iris Dataset')
plt.show()

