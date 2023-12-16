import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import utils

def train_logistic_regression(X, y):
    logreg = LogisticRegression(max_iter=10000)  # Increase max_iter to avoid convergence warning
    logreg.fit(X, y)
    return logreg

def evaluate_logistic_regression(model, X, y, dataset_name):
    pred_prob = model.predict_proba(X)[:, 1]  # Probability of the positive class
    pred_binary = np.round(pred_prob)
    
    mse = mean_squared_error(y, pred_binary)
    mae = mean_absolute_error(y, pred_binary)
    accuracy = accuracy_score(y, pred_binary)

    print(f"{dataset_name} Evaluation:")
    print(f"Mean squared error: {mse}")
    print(f"Mean absolute error: {mae}")
    print(f"Accuracy: {accuracy}")

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = utils.getData("red")
    
    # Training Logistic Regression
    logreg_model = train_logistic_regression(X_train, y_train)
    pred_train_prob = logreg_model.predict_proba(X_train)[:, 1]  # Probability of the positive class
    pred_train_binary = np.round(pred_train_prob)
    
    print("Training:")
    evaluate_logistic_regression(logreg_model, X_train, y_train, "Training")

    # Validation
    best_err = float('inf')
    best_logreg_model = None
    num_batches = 10  # Replace with your actual number of batches
    batch_size = len(X_val) // num_batches

    for i in range(num_batches):
        X_batch = X_val[i * batch_size:(i + 1) * batch_size]
        y_batch = y_val[i * batch_size:(i + 1) * batch_size]

        logreg_model = train_logistic_regression(X_batch, y_batch)
        pred_val_prob = logreg_model.predict_proba(X_batch)[:, 1]  # Probability of the positive class
        pred_val_binary = np.round(pred_val_prob)

        err = mean_squared_error(y_batch, pred_val_binary)
        if err < best_err:
            best_err = err
            best_logreg_model = logreg_model

    # Testing
    print("\nTesting:")
    evaluate_logistic_regression(best_logreg_model, X_test, y_test, "Testing")

if __name__ == "__main__":
    main()
    
