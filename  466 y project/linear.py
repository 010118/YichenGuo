import numpy as np
import math
import utils
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from matplotlib import pyplot as plt

def train_model(X_train, y_train):
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    return LR

def evaluate_model(model, X, y, dataset_name):
    pred = model.predict(X)
    pred_rounded = np.rint(pred)
    y_rounded = np.rint(np.array(y))

    mse = mean_squared_error(y_rounded, pred_rounded)
    mae = mean_absolute_error(y_rounded, pred_rounded)
    accuracy = utils.accuracy(pred_rounded, y_rounded)

    print(f"{dataset_name} Evaluation:")
    print(f"Mean squared error: {mse}")
    print(f"Mean absolute error: {mae}")
    print(f"Accuracy: {accuracy}")
    if dataset_name == "Testing":
        print("Close Accuracy (+-1 score): " + str(utils.close_accuracy(pred_rounded, y_rounded)))

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = utils.getData("red")
    
    # Training
    model = train_model(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_train_rounded = np.rint(pred_train)
    y_train_rounded = np.rint(np.array(y_train))
    
    print("Training:")
    evaluate_model(model, X_train, y_train, "Training")

    # Validation
    best_err = float('inf')
    best_model = None
    num_batches = 10  # Replace with your actual number of batches
    batch_size = len(X_val) // num_batches

    for i in range(num_batches):
        X_batch = X_val[i * batch_size:(i + 1) * batch_size]
        y_batch = y_val[i * batch_size:(i + 1) * batch_size]

        model = train_model(X_batch, y_batch)
        pred_val = model.predict(X_batch)
        pred_val_rounded = np.rint(pred_val)
        y_batch_rounded = np.rint(np.array(y_batch))

        err = mean_squared_error(y_batch_rounded, pred_val_rounded)
        if err < best_err:
            best_err = err
            best_model = model

    # Testing
    print("\nTesting:")
    evaluate_model(best_model, X_test, y_test, "Testing")

if __name__ == "__main__":
    main()