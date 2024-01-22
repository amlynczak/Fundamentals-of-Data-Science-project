import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def cluster_and_classify(data, y_train, labels=None, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(data)

    if labels is None:
        labels = np.zeros(len(data))

    classifiers = {}
    for cluster_label in range(n_clusters):
        cluster_data = data[cluster_labels == cluster_label]
        cluster_labels_true = labels[cluster_labels == cluster_label]

        X_train, X_test, y_train_cluster, y_test = train_test_split(
            cluster_data, cluster_labels_true, test_size=0.2, random_state=42
        )

        classifier = DecisionTreeClassifier(random_state=42)
        classifier.fit(X_train, y_train_cluster)
        classifiers[cluster_label] = classifier

        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Cluster {cluster_label} - Classifier Accuracy: {accuracy}")

    plt.figure(figsize=(10, 6))
    for cluster_label, classifier in classifiers.items():
        plt.subplot(2, 5, cluster_label + 1)
        plot_tree(classifier, filled=True, feature_names=[f'Feature {i}' for i in range(data.shape[1])], class_names=[str(i) for i in np.unique(labels)])
        plt.title(f"Cluster {cluster_label}")

    # Confusion matrix for clustering quality assessment
    matrix = confusion_matrix(y_train, cluster_labels)
    print("Confusion matrix:")
    print(matrix)

    plt.tight_layout()
    plt.show()

    return classifiers, cluster_labels
