from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report

#classifiation using K-Nearest Neighbours
def classification_KNN(X_train, X_test, y_train, y_test):
    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nKNeighborsClassifier:\n")
    print(f"Accuracy: {accuracy:.2f}\n")
    print("Classification Report:\n", report)
    print("\n")

#classification using Random Forest Classifier
def classification_RFC(X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nRnadomForestClassifier:\n")
    print(f"Accuracy: {accuracy:.2f}\n")
    print("Classification Report:\n", report)
    print("\n")

#classification using Logistic Regression
def classification_LR(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression(random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nLogisticRegression:\n")
    print(f"Accuracy: {accuracy:.2f}\n")
    print("Classification Report:\n", report)
    print("\n")

#classification using Naive Bytes
def classification_NB(X_train, X_test, y_train, y_test):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nGaussianNB:\n")
    print(f"Accuracy: {accuracy:.2f}\n")
    print("Classification Report:\n", report)
    print("\n")