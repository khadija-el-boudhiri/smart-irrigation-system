from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, f1, matrix, report