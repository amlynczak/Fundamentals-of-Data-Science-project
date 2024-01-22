from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def classification(X_train, X_test, y_train, y_test):
    classifier = SVC(random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)