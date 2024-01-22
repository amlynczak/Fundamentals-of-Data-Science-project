import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist

from solutions.reduce_dimen import reduction
from solutions.visualisation import visualisation
from solutions.clustering import cluster_and_classify
from solutions.split import split_dataset
from solutions.classification import classification_RFC, classification_LR, classification_KNN, classification_NB

def preprocess_data(X):
    X_flat = X.reshape(X.shape[0], -1)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_flat.reshape(-1, 28*28))

    return x_train_scaled

def main():
    #downloading the fashion MNIST dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train_scaled = preprocess_data(x_train)

    #1 summary of the data - report

    #2 reducing the data dimensionality
    data_red_2d, data_red_3d = reduction(x_train_scaled)

    #3 visualising the reduced data
    visualisation(data_red_2d, data_red_3d, y_train, "solutions/img/visualisation")

    #4 clustering the data
    cluster_and_classify(data_red_2d, y_train)

    #5 splitting the data into train and test
    x_train_split, X_test_split, y_train_split, y_test_split = split_dataset(x_train_scaled, y_train)

    #6 performing classification and evaluating its result
    classification_KNN(x_train_split, X_test_split, y_train_split, y_test_split)
    classification_RFC(x_train_split, X_test_split, y_train_split, y_test_split)
    classification_LR(x_train_split, X_test_split, y_train_split, y_test_split)
    classification_NB(x_train_split, X_test_split, y_train_split, y_test_split)

if __name__ == "__main__":
    main()
