# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Step 1: Summary of the data
# You can print shapes, unique labels, and sample images to get an overview
print("Shape of training data:", x_train.shape)
print("Unique labels in training data:", np.unique(y_train))
# Additional summary steps can be performed based on your specific needs

# Step 2: Reduce data dimensionality using PCA or t-SNE
# For example, using PCA
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(-1, 28 * 28))
pca = PCA(n_components=2)  # You can choose the number of components
x_train_pca = pca.fit_transform(x_train_scaled)

# Alternatively, using t-SNE
#tsne = TSNE(n_components=2, random_state=42)
#x_train_tsne = tsne.fit_transform(x_train_scaled)

# Step 3: Visualize the reduced dataset
# Visualize the reduced data using scatter plots or other visualization techniques
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.5, s=10)
plt.title('PCA Reduced Fashion-MNIST Dataset')
plt.show()

# Step 4: Cluster the dataset using KMeans
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(x_train_pca)

# Evaluate clustering results with classification labels
# You can use confusion matrix or other metrics
conf_matrix = confusion_matrix(y_train, cluster_labels)
print("Confusion Matrix:\n", conf_matrix)

# Step 5: Split the dataset into training and testing
x_train, x_val, y_train, y_val = train_test_split(x_train_pca, y_train, test_size=0.2, random_state=42)

# Step 6: Perform classification and evaluate results
# Using a simple classifier like Random Forest as an example
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(x_val)

# Evaluate classification results
accuracy = accuracy_score(y_val, y_pred)
print("Classification Accuracy:", accuracy)

# Additional steps can include hyperparameter tuning, cross-validation, etc.
